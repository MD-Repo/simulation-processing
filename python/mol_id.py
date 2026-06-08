#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openbabel-wheel",
#     "MDAnalysis>=2.0",
# ]
# ///
"""
mol_id.py — Two utilities for small-molecule identification:

  1) Extract a ligand from PDB/GRO files and produce a canonical SMILES string.
  2) Given a SMILES string, look up the common molecule name via PubChem.

Usage (with uv — resolves dependencies automatically from the header above):
    uv run mol_id.py smiles-from-structure prod.pdb
    uv run mol_id.py smiles-from-structure minimal.gro
    uv run mol_id.py smiles-from-structure run.tpr
    uv run mol_id.py name-from-smiles "NC(=O)c1ccc[n+](c1)[C@@H]2O[C@H](CO[P](O)(O)=O)[C@@H](O)[C@H]2O"
    uv run mol_id.py both system.gro

File loading is delegated to MDAnalysis, so any format MDAnalysis understands
works: PDB, GRO, GROMACS .top/.itp/.tpr, Amber .prmtop/.parm7, CHARMM .psf,
mmCIF, etc. Format is sniffed from content (catches text files with misleading
extensions); for binary files the extension is used. The ligand residue is
auto-detected by elimination (skips amino acids, nucleotides, water, and ions).
Pass --format or --resname to override either.

Best inputs are those that carry BOTH connectivity AND 3D coordinates — .tpr,
.pdb (with CONECT records), or topology+coords combined. A coordinate-only
file (.gro) still works (bonds inferred from geometry). A topology-only text
file (.top/.itp without coordinates) will give correct atoms and bonds but may
under-perceive aromaticity, since OpenBabel relies on planar geometry to
identify aromatic rings — that can defeat downstream PubChem name lookup.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import urllib.parse
import urllib.request
from typing import Optional

import warnings

import numpy as np
from openbabel import openbabel as ob

# Suppress Open Babel's C-level stderr warnings (e.g. "unusual valence" in InChI code).
ob.obErrorLog.SetOutputLevel(ob.obError)

# MDAnalysis prints a forest of harmless warnings on import; quiet them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import MDAnalysis as mda


# ---------------------------------------------------------------------------
# Part 1: Structure file → canonical SMILES
# ---------------------------------------------------------------------------

# Residue names to ignore when auto-detecting the ligand.
_AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Common protonation / tautomer variants (Amber, CHARMM)
    "HIE", "HID", "HIP", "HSD", "HSE", "HSP",
    "ASH", "GLH", "LYN", "CYM", "CYX",
})

_NUCLEOTIDES = frozenset({
    "A", "C", "G", "T", "U",
    "DA", "DC", "DG", "DT", "DU",
    "RA", "RC", "RG", "RT", "RU",
    "ADE", "CYT", "GUA", "THY", "URA",
})

_WATERS = frozenset({
    "HOH", "WAT", "H2O", "TIP", "TIP3", "TIP3P", "TIP4", "TIP4P", "TIP5",
    "SOL", "T3P", "T4P", "SPC", "SPCE",
    "OPC",          # OPC 4-site water model (AMBER)
    # GROMACS writes 4-char residue names one column early; MDAnalysis reads
    # the last 3 chars. These are the misread forms of common water models:
    "IP3",          # TIP3 misread
    "IP4",          # TIP4 / TIP4P misread
    "IP5",          # TIP5 misread
})

# Common lipid residue names.  Also includes the 3-char suffixes that appear
# when GROMACS writes a 4-char lipid name (e.g. POPC) one column early and
# MDAnalysis reads columns 18-20, dropping the first character.
_LIPIDS = frozenset({
    # Phosphatidylcholines
    "DPPC", "POPC", "DOPC", "DLPC", "DMPC", "DSPC",
    # Phosphatidylethanolamines
    "POPE", "DPPE", "DOPE", "DLPE", "DMPE", "DSPE",
    # Phosphatidylglycerols
    "POPG", "DPPG", "DLPG", "DMPG", "DSPG",
    # Phosphatidylserines / phosphatidic acids
    "DOPS", "DPPS", "DOPA",
    # Cholesterol / sphingolipids
    "CHOL", "CHL1", "CER", "PSM",
    # Lysophospholipids / other
    "DHPC",
    # GROMACS 4-char misreads (last 3 chars of the real name):
    "PPC",   # DPPC
    # OPC already in _WATERS (covers POPC → OPC too)
    "OPE",   # POPE / DOPE
    "LPC",   # DLPC
    "MPC",   # DMPC
    # SPC already in _WATERS (covers DSPC → SPC)
    "OPG",   # POPG
    "PPE",   # DPPE
    "PPG",   # DPPG
    "HOL",   # CHOL
    "HL1",   # CHL1
    "OPS",   # DOPS
    "OPA",   # DOPA
    "HPC",   # DHPC
})

# Monatomic ions, by element symbol or common force-field alias.
_ION_NAMES = frozenset({
    "NA", "K", "MG", "CA", "CL", "ZN", "FE", "MN", "CU", "BR", "I", "IOD",
    "LI", "RB", "CS", "F", "CO", "NI", "HG", "CD", "BA", "SR", "AL", "SE",
    "SOD", "POT", "CLA", "MGY",
})


def _is_skipped_residue(resname: str) -> bool:
    """True if this residue name is part of the protein/nucleic/solvent/ion/lipid background."""
    rn = (resname or "").upper()
    if rn in _AMINO_ACIDS or rn in _NUCLEOTIDES or rn in _WATERS or rn in _LIPIDS:
        return True
    # Strip charge/multiplicity decorations: "Na+", "MG2+", "CL-", etc.
    stripped = rn.rstrip("+-0123456789")
    return stripped in _ION_NAMES


# Map our friendly format names (and common file extensions) to the
# topology_format strings MDAnalysis expects.
_FMT_TO_MDA = {
    "pdb":     "PDB",
    "gro":     "GRO",
    "top":     "ITP",     # MDAnalysis treats GROMACS .top/.itp text files identically
    "itp":     "ITP",
    "tpr":     "TPR",
    "psf":     "PSF",
    "cif":     "MMCIF",
    "mmcif":   "MMCIF",
    "prmtop":  "TOP",     # Amber parm
    "parm7":   "TOP",
}


def detect_format(path: str) -> Optional[str]:
    """
    Sniff the input file's format. Returns an MDAnalysis topology_format
    name (e.g. "PDB", "GRO", "ITP", "TPR"), or None to let MDAnalysis
    auto-detect from the file extension.

    Content-based for text files (catches a topology saved with a misleading
    .gro extension); for binary files (e.g. .tpr) falls back to extension via
    `_FMT_TO_MDA`.
    """
    with open(path, "rb") as f:
        head_bytes = f.read(4096)
    if not head_bytes:
        return None

    # If a meaningful fraction of bytes are non-text, treat the file as binary.
    text_bytes = sum(1 for b in head_bytes if 9 <= b <= 126 or b in (10, 13))
    if text_bytes / len(head_bytes) < 0.85:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        return _FMT_TO_MDA.get(ext)

    head = head_bytes.decode("utf-8", errors="replace").splitlines()

    if any(re.match(r"^\s*\[\s*(defaults|atomtypes|moleculetype|system|molecules)\s*\]", L) for L in head):
        return "ITP"  # GROMACS topology / .itp text format

    pdb_records = ("ATOM  ", "HETATM", "HEADER", "CRYST1", "MODEL ", "REMARK", "TITLE ", "COMPND")
    if any(L.startswith(pdb_records) for L in head):
        return "PDB"

    if len(head) >= 2:
        try:
            int(head[1].strip())
            return "GRO"
        except ValueError:
            pass

    return None  # let MDAnalysis attempt its own detection from the extension


def _resolve_user_fmt(fmt: Optional[str]) -> Optional[str]:
    """Translate a CLI --format value (e.g. 'top', 'tpr') to MDAnalysis's name."""
    if not fmt:
        return None
    return _FMT_TO_MDA.get(fmt.lower(), fmt.upper())


def load_universe(path: str, fmt: Optional[str] = None):
    """Load a structure/topology file as an MDAnalysis Universe."""
    mda_fmt = _resolve_user_fmt(fmt) if fmt else detect_format(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mda_fmt:
            return mda.Universe(path, topology_format=mda_fmt)
        return mda.Universe(path)


def find_ligand_resnames(universe) -> dict:
    """{resname: atom_count} for non-background residues in a Universe."""
    counts: dict = {}
    if not hasattr(universe.atoms, "resnames"):
        return counts
    names, totals = np.unique(universe.atoms.resnames, return_counts=True)
    for n, c in zip(names, totals):
        n_str = str(n)
        if not _is_skipped_residue(n_str):
            counts[n_str] = int(c)
    return counts


def resolve_target_resnames(universe, path: str, resname: Optional[str]) -> list:
    """
    Decide which residue name(s) to extract from a Universe.

    - If `resname` is given AND atoms with that name exist in the file → use
      it alone.
    - If `resname` is given but NOT present → fall back to every non-background
      residue in the file (matches prior PDB extraction semantics).
    - If `resname` is None → return every non-background residue.

    Always returns a non-empty list, or raises ValueError.
    """
    present = set(map(str, np.unique(universe.atoms.resnames))) if hasattr(universe.atoms, "resnames") else set()
    if resname is not None and resname in present:
        return [resname]

    candidates = find_ligand_resnames(universe)
    if resname is not None and not candidates:
        raise ValueError(
            f"No atoms with residue name '{resname}' in {path}, and no "
            f"non-background residues to fall back on."
        )
    if not candidates:
        raise ValueError(
            f"No ligand-like residue found in {path} after filtering out amino "
            f"acids, nucleotides, water, and ions. Pass --resname explicitly."
        )

    return list(candidates.keys())


# Leading element letter is ambiguity-free for these — atom names like "HG21",
# "C5R", "ND2", "N1" should always be parsed as their first letter (H, C, N, O,
# P, S) rather than as a 2-letter element (Hg, Cr, Nd, Na). MDAnalysis's ITP
# parser in particular sometimes maps GAFF atom *types* (like "na" = amine N)
# straight into the element column, which is why we can't trust the element
# field blindly.
_ORGANIC_LETTERS = {"H", "C", "N", "O", "P", "S"}


def _atomic_num_for(atom) -> int:
    """Atomic number for an MDAnalysis Atom, robust to bogus element fields."""
    name = (getattr(atom, "name", "") or "").strip()
    elem = (getattr(atom, "element", "") or "").strip()

    # If MDA's element agrees with the atom name's leading letters, trust it.
    if elem and name and name.upper().startswith(elem.upper()):
        n = ob.GetAtomicNum(elem.capitalize())
        if n > 0:
            return n

    # Otherwise extract from the atom name's leading alpha prefix.
    leading = ""
    for c in name:
        if c.isalpha():
            leading += c
        else:
            break
    if leading:
        if leading[0].upper() in _ORGANIC_LETTERS:
            return ob.GetAtomicNum(leading[0].upper())
        for length in (2, 1):
            if len(leading) >= length:
                n = ob.GetAtomicNum(leading[:length].capitalize())
                if n > 0:
                    return n

    # Last resort: trust the element field even if the name was empty.
    if elem:
        n = ob.GetAtomicNum(elem.capitalize())
        if n > 0:
            return n
    return 0


def _mol_summary(mol) -> dict:
    """Build the standard SMILES/formula/InChIKey summary for an OBMol."""
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    smiles = conv.WriteString(mol).strip().split("\t")[0]
    conv.SetOutFormat("inchikey")
    inchikey = conv.WriteString(mol).strip()
    return {
        "smiles": smiles,
        "formula": mol.GetFormula(),
        "num_atoms": mol.NumAtoms(),
        "num_heavy_atoms": mol.NumHvyAtoms(),
        "charge": mol.GetTotalCharge(),
        "inchikey": inchikey,
    }


def _smiles_from_coords(sel) -> dict:
    """
    Coordinates-available path: serialize the selection to a temp PDB and let
    OpenBabel's PDB reader handle it. This reuses OB's well-tested geometry-
    based bond and aromaticity perception.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.write(tmp_path)
        conv = ob.OBConversion()
        conv.SetInFormat("pdb")
        mol = ob.OBMol()
        conv.ReadFile(mol, tmp_path)
        if mol.NumBonds() == 0:
            mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        return _mol_summary(mol)
    finally:
        os.unlink(tmp_path)


def _smiles_from_topology(sel) -> dict:
    """
    Topology-only path (no coordinates): build OBMol directly from atoms and
    bonds. Aromaticity may be under-perceived since OpenBabel relies on
    geometry to identify aromatic rings.
    """
    has_bonds = False
    try:
        has_bonds = len(sel.bonds) > 0
    except (mda.exceptions.NoDataError, AttributeError):
        pass

    mol = ob.OBMol()
    mol.BeginModify()
    idx_map: dict = {}
    for atom in sel:
        a = mol.NewAtom()
        a.SetAtomicNum(_atomic_num_for(atom))
        idx_map[int(atom.index)] = a.GetIdx()

    if has_bonds:
        sel_set = set(idx_map.keys())
        for bond in sel.bonds:
            i_idx = int(bond.atoms[0].index)
            j_idx = int(bond.atoms[1].index)
            if i_idx in sel_set and j_idx in sel_set:
                mol.AddBond(idx_map[i_idx], idx_map[j_idx], 1)
    mol.EndModify()
    mol.PerceiveBondOrders()
    return _mol_summary(mol)


def universe_to_smiles(universe, resname: str) -> dict:
    """
    Extract atoms with `resname` from `universe` and return canonical SMILES
    plus summary. Dispatches to a coordinate-aware path when possible (which
    gives the best aromaticity perception) or a topology-only fallback.
    """
    resnames = universe.atoms.resnames
    mask = resnames == resname
    if not mask.any():
        mask = np.char.upper(resnames.astype(str)) == resname.upper()
    all_match = universe.atoms[mask]
    if len(all_match) == 0:
        raise ValueError(f"No atoms with residue name '{resname}' in the file.")

    # If multiple copies of the same residue name are present (e.g., many lipid
    # molecules in a full simulation box), use only the first residue instance so
    # that SMILES represents a single molecule rather than all copies concatenated.
    unique_resix = np.unique(all_match.resindices)
    sel = all_match[all_match.resindices == unique_resix[0]] if len(unique_resix) > 1 else all_match

    has_coords = False
    try:
        pos = sel.positions
        has_coords = pos is not None and len(pos) == len(sel)
    except (mda.exceptions.NoDataError, AttributeError):
        has_coords = False

    return _smiles_from_coords(sel) if has_coords else _smiles_from_topology(sel)


def structure_to_smiles(
    path: str,
    fmt: Optional[str] = None,
    resname: Optional[str] = None,
) -> list:
    """
    Universal pipeline: load any MDAnalysis-supported file (PDB, GRO, GROMACS
    .top/.tpr, PSF, mmCIF, Amber prmtop, ...), identify the ligand residue(s),
    and return a list of canonical-SMILES summaries (one per distinct ligand
    residue), each tagged with its `resname`.

    If `fmt` is None, format is sniffed from file content (with extension as
    fallback for binary inputs). If `resname` is None, every non-background
    residue is processed; if given, only that residue is processed (with a
    fallback to all non-background residues if it isn't present in the file).
    """
    universe = load_universe(path, fmt)
    target_resnames = resolve_target_resnames(universe, path, resname)
    results = []
    for rn in target_resnames:
        result = universe_to_smiles(universe, rn)
        result["resname"] = rn
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Part 2: SMILES → molecule name via Wikidata + PubChem
# ---------------------------------------------------------------------------

# Polite, identifying User-Agent for free-service calls (Wikidata requires this).
_USER_AGENT = "mol_id.py (small-molecule identification utility)"


def _http_get_json(
    url: str, headers: Optional[dict] = None, retries: int = 2
) -> Optional[dict]:
    """
    Fetch JSON from a URL with simple retry/backoff. Returns None if every
    attempt fails. PubChem in particular sometimes returns 503/PUGREST.busy
    when called in quick succession.
    """
    hdrs = {"Accept": "application/json", "User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
            else:
                return None
    return None




def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Compute the InChIKey for a SMILES string via OpenBabel."""
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    conv.SetOutFormat("inchikey")
    mol = ob.OBMol()
    if not conv.ReadString(mol, smiles):
        return None
    key = conv.WriteString(mol).strip()
    return key or None


def query_wikidata_by_inchikey(inchikey: str) -> Optional[str]:
    """
    Look up a chemical compound on Wikidata by InChIKey (property P235) and
    return its English label, e.g. "ATP" or "adenosine 2',5'-bisphosphate".
    Returns None if no Wikidata entry has this InChIKey.
    """
    sparql = (
        'SELECT ?itemLabel WHERE { '
        f'?item wdt:P235 "{inchikey}" . '
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } '
        '} LIMIT 1'
    )
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
        {"query": sparql, "format": "json"}
    )
    data = _http_get_json(url)
    if not data:
        return None
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    label = bindings[0].get("itemLabel", {}).get("value", "").strip()
    # If Wikidata has no English label it returns the QID (e.g. "Q12345").
    if not label or re.match(r"^Q\d+$", label):
        return None
    return label


def _unichem_refs(inchikey: str) -> list:
    """Return UniChem cross-reference list for an InChIKey, or []."""
    url = f"https://www.ebi.ac.uk/unichem/rest/inchikey/{urllib.parse.quote(inchikey)}"
    data = _http_get_json(url)
    return data if isinstance(data, list) else []


def query_pdbe_by_inchikey(inchikey: str) -> Optional[str]:
    """
    Look up an InChIKey via UniChem to find a PDBe compound ID (source 5),
    then query the PDBe compound summary API for a curated name.
    Only covers compounds that appear in at least one PDB structure.
    """
    refs = _unichem_refs(inchikey)
    pdbe_id = next((x["src_compound_id"] for x in refs if x.get("src_id") == "5"), None)
    if not pdbe_id:
        return None
    data = _http_get_json(
        f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{pdbe_id}"
    )
    if not data:
        return None
    entries = list(data.values())
    if not entries or not entries[0]:
        return None
    name = entries[0][0].get("name", "").strip()
    return name or None




# --- Heuristic ranker for PubChem synonyms ---------------------------------

# CAS number pattern: e.g. "3805-37-6"
_CAS_RE = re.compile(r"^\d{1,7}-\d{1,2}-\d$")
# PDB ID pattern: exactly 4 chars, first is a digit, rest alphanumeric, e.g. "8hvp"
_PDB_ID_RE = re.compile(r"^\d[A-Za-z0-9]{3}$")
# InChIKey: 14 uppercase letters, dash, 10 uppercase letters, dash, 1 uppercase letter
_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def is_junk_synonym(name: str) -> bool:
    """
    True if `name` looks like a database code or IUPAC monster rather than a
    real common name. PubChem synonym lists are ordered by frequency of use,
    so we don't try to rank readability — we just filter junk and trust order.
    """
    n = (name or "").strip()
    if not n:
        return True
    if _CAS_RE.match(n):
        return True                    # CAS number, e.g. "3805-37-6"
    if _PDB_ID_RE.match(n):
        return True                    # PDB ID, e.g. "8hvp"
    if _INCHIKEY_RE.match(n):
        return True                    # InChIKey masquerading as a synonym
    if ":" in n:
        return True                    # registry IDs: CHEBI:..., MeSH:..., RefChem:...
    if len(n) > 80:
        return True                    # IUPAC behemoths
    if " " not in n and len(n) > 30:
        return True                    # long no-space tokens: peptide notations, vendor codes
    # Code-like: no spaces, embedded digits and letters, all-caps or all-lowercase.
    # Catches uppercase codes ("A2P5P", "CHEMBL123") and lowercase catalog numbers
    # ("orb1702635"). Mixed-case names and pure acronyms without digits survive.
    if (
        " " not in n
        and any(c.isdigit() for c in n)
        and any(c.isalpha() for c in n)
        and (n.upper() == n or n.lower() == n)
    ):
        return True
    return False


def first_acceptable_synonym(synonyms: list) -> Optional[str]:
    """Return the first non-junk synonym (PubChem orders by relevance)."""
    for s in synonyms or []:
        if not is_junk_synonym(s):
            return s
    return None


def smiles_to_name(smiles: str) -> dict:
    """
    Resolve a SMILES string to a human-friendly molecule name.

    Strategy:
      1. Compute InChIKey from the SMILES.
      2. Try Wikidata by InChIKey — curated common names.
      3. Query PubChem synonyms (junk-filtered: strips CAS numbers, PDB IDs,
         registry codes, catalog numbers, and long no-space tokens like peptide
         sequence notations).
      4. Fall back to Wikidata label even if code-like.
      5. Try PDBe compound summary via UniChem — curated names for anything
         that has appeared in a PDB structure (e.g. "HYDROXYETHYLENE-BASED INHIBITOR").
      6. Last resort: PubChem Title field (usually the IUPAC name).

    Returns a dict including `best_name` (the chosen display name),
    `name_source` (where it came from), plus the raw evidence (iupac_name,
    pubchem_title, synonyms, cid, inchikey, pdbe_name) for transparency.
    """
    result: dict = {"smiles_input": smiles}

    inchikey = smiles_to_inchikey(smiles)
    if inchikey:
        result["inchikey"] = inchikey

    # --- 1. Wikidata by InChIKey ---
    wikidata_label = query_wikidata_by_inchikey(inchikey) if inchikey else None
    if wikidata_label:
        result["wikidata_label"] = wikidata_label

    # --- 2. PubChem: try SMILES, then InChIKey ---
    encoded_smi = urllib.parse.quote(smiles, safe="")
    url_props_smi = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded_smi}/property/IUPACName,Title,MolecularFormula,Charge/JSON"
    )
    data = _http_get_json(url_props_smi)
    if data is None and inchikey:
        url_props_key = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
            f"{inchikey}/property/IUPACName,Title,MolecularFormula,Charge/JSON"
        )
        data = _http_get_json(url_props_key)

    pubchem_synonyms: list = []
    pubchem_title: Optional[str] = None
    if data and "PropertyTable" in data:
        props = data["PropertyTable"]["Properties"][0]
        cid = props.get("CID")
        result["cid"] = cid
        result["iupac_name"] = props.get("IUPACName", "unknown")
        pubchem_title = props.get("Title")
        result["pubchem_title"] = pubchem_title or "unknown"
        result["formula"] = props.get("MolecularFormula", "")
        result["charge"] = props.get("Charge", 0)

        if cid:
            syn_data = _http_get_json(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{cid}/synonyms/JSON"
            )
            if syn_data and "InformationList" in syn_data:
                pubchem_synonyms = (
                    syn_data["InformationList"]["Information"][0].get("Synonym", [])
                )
                result["synonyms"] = pubchem_synonyms[:10]

    # --- 3. PDBe and ChEBI lookups via UniChem ---
    pdbe_name = query_pdbe_by_inchikey(inchikey) if inchikey else None
    if pdbe_name:
        result["pdbe_name"] = pdbe_name


    # --- 4. Pick a display name ---
    # Priority: Wikidata (readable) → PubChem synonym (junk-filtered) →
    #           Wikidata (code-like) → CCD → ChEBI → PubChem title
    best_name: Optional[str] = None
    name_source: Optional[str] = None

    if wikidata_label and any(c.islower() for c in wikidata_label) and not is_junk_synonym(wikidata_label):
        best_name = wikidata_label
        name_source = "wikidata"

    if best_name is None:
        syn = first_acceptable_synonym(pubchem_synonyms)
        if syn:
            best_name = syn
            name_source = "pubchem_synonyms"

    if best_name is None and wikidata_label and not is_junk_synonym(wikidata_label):
        best_name = wikidata_label
        name_source = "wikidata"

    if best_name is None and pdbe_name:
        best_name = pdbe_name
        name_source = "pdbe"

    if best_name is None and pubchem_title:
        best_name = pubchem_title
        name_source = "pubchem_title"

    result["best_name"] = best_name if best_name is not None else "unknown"
    if name_source:
        result["name_source"] = name_source

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _SmartHelpFormatter(argparse.HelpFormatter):
    """
    Wraps single-paragraph description text to terminal width, but preserves
    any text that already contains explicit newlines (e.g. our hand-formatted
    epilog blocks of examples) verbatim.
    """
    def _fill_text(self, text, width, indent):
        if "\n" in text.strip():
            return "".join(indent + line for line in text.splitlines(keepends=True))
        return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def main():
    top_description = (
        "Identify a small molecule from a structure/topology file "
        "(PDB, GRO, GROMACS .top/.itp/.tpr, CHARMM .psf, mmCIF, Amber prmtop, "
        "etc.) and/or look up its common name via Wikidata + PubChem."
    )
    top_epilog = (
        "Examples:\n"
        "  uv run mol_id.py smiles-from-structure prod.pdb\n"
        "  uv run mol_id.py smiles-from-structure run.tpr\n"
        "  uv run mol_id.py both system.gro\n"
        "  uv run mol_id.py name-from-smiles \"CN1C=NC2=C1C(=O)N(C(=O)N2C)C\"\n"
        "\n"
        "Best results come from inputs that carry both connectivity and 3D\n"
        "coordinates (.tpr, .pdb with CONECT records, or any file paired with\n"
        "coords). Coordinate-only files (.gro) work — bonds are inferred from\n"
        "geometry. Topology-only text files (.top/.itp without coords) work but\n"
        "may under-perceive aromaticity, since OpenBabel uses planar geometry to\n"
        "identify aromatic rings."
    )

    parser = argparse.ArgumentParser(
        prog="mol_id.py",
        description=top_description,
        epilog=top_epilog,
        formatter_class=_SmartHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def _add_verbose_arg(p):
        p.add_argument(
            "-v", "--verbose",
            action="store_true",
            default=False,
            help="Print full JSON output instead of the concise default.",
        )

    def _add_struct_args(p):
        p.add_argument(
            "path",
            help=(
                "Path to a structure or topology file. Format is auto-detected from "
                "content (text files) or extension (binary files). Supported: "
                "PDB, GRO, GROMACS .top/.itp/.tpr, CHARMM .psf, mmCIF, Amber "
                ".prmtop/.parm7, and anything else MDAnalysis can read."
            ),
        )
        p.add_argument(
            "--format", dest="fmt",
            choices=["pdb", "gro", "top", "itp", "tpr", "psf", "cif", "mmcif", "prmtop", "parm7"],
            default=None,
            help="Force input file format instead of sniffing from content / extension.",
        )
        p.add_argument(
            "--resname", default=None,
            help=(
                "Residue name of the ligand. If omitted, auto-detected by elimination "
                "(skips amino acids, nucleotides, water, and ions)."
            ),
        )

    p1 = sub.add_parser(
        "smiles-from-structure",
        help="Extract ligand from a structure/topology file and produce canonical SMILES.",
        description=(
            "Read the input file, isolate the ligand residue, and emit canonical "
            "SMILES + InChIKey + atom/bond/charge summary as JSON on stdout."
        ),
        epilog=(
            "Examples:\n"
            "  uv run mol_id.py smiles-from-structure prod.pdb\n"
            "  uv run mol_id.py smiles-from-structure run.tpr --resname LIG\n"
            "  uv run mol_id.py smiles-from-structure system.top --format itp"
        ),
        formatter_class=_SmartHelpFormatter,
    )
    _add_struct_args(p1)
    _add_verbose_arg(p1)

    p2 = sub.add_parser(
        "name-from-smiles",
        help="Look up molecule name from a SMILES string (requires internet).",
        description=(
            "Resolve a SMILES string to a human-readable name using Wikidata "
            "(by InChIKey) and PubChem (with synonym ranking that filters out "
            "CAS numbers, registry IDs, and IUPAC monsters)."
        ),
        epilog=(
            "Example:\n"
            "  uv run mol_id.py name-from-smiles \"CN1C=NC2=C1C(=O)N(C(=O)N2C)C\""
        ),
        formatter_class=_SmartHelpFormatter,
    )
    p2.add_argument("smiles", help="SMILES string (quote it to protect special characters from the shell).")
    _add_verbose_arg(p2)

    p3 = sub.add_parser(
        "both",
        help="Extract SMILES from a structure/topology, then look up the name.",
        description=(
            "Run smiles-from-structure followed by name-from-smiles for each "
            "ligand found. Emits a JSON list of {structure, name} objects. "
            "Useful when you have a structure file and want a human-friendly "
            "name for whatever ligand(s) it contains."
        ),
        epilog=(
            "Example:\n"
            "  uv run mol_id.py both prod.tpr -o result.json"
        ),
        formatter_class=_SmartHelpFormatter,
    )
    _add_struct_args(p3)
    p3.add_argument(
        "-o", "--outfile",
        help="Write full JSON output to this file instead of stdout.",
    )
    _add_verbose_arg(p3)

    args = parser.parse_args()

    if args.command == "smiles-from-structure":
        results = structure_to_smiles(args.path, fmt=args.fmt, resname=args.resname)
        if args.verbose:
            print(json.dumps(results, indent=2))
        else:
            print("\n".join(sorted([r["smiles"] for r in results])))

    elif args.command == "name-from-smiles":
        result = smiles_to_name(args.smiles)
        if args.verbose:
            print(json.dumps(result, indent=2))
        else:
            print(result["best_name"])

    elif args.command == "both":
        struct_results = structure_to_smiles(args.path, fmt=args.fmt, resname=args.resname)
        combined = [
            {"structure": sr, "name": smiles_to_name(sr["smiles"])}
            for sr in struct_results
        ]
        if args.outfile:
            with open(args.outfile, "wt") as fh:
                json.dump(combined, fh, indent=2)
        elif args.verbose:
            print(json.dumps(combined, indent=2))
        else:
            for entry in combined:
                resname = entry["structure"]["resname"]
                best = entry["name"]["best_name"]
                print(f"{resname}: {best}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        # Expected user-facing failures (no ligand found, bad format, etc.)
        # — print a clean error rather than a Python traceback.
        print(f"mol_id: {e}", file=sys.stderr)
        sys.exit(1)
