#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openbabel-wheel",
#     "MDAnalysis>=2.0",
#     "rdkit>=2026.3.3",
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
from typing import NamedTuple, Optional

import warnings

import numpy as np
from openbabel import openbabel as ob
from rdkit import Chem, RDLogger

# Suppress Open Babel's C-level stderr warnings (e.g. "unusual valence" in InChI code).
ob.obErrorLog.SetOutputLevel(ob.obError)

# InChI and InChIKey come from RDKit, which bundles InChI 1.07.3, rather than
# from OpenBabel, which still bundles 1.04 from 2011. OpenBabel keeps every
# other job here: reading the coordinates, perceiving bonds, writing canonical
# SMILES. See compare_smiles.to_inchi() for the measurement behind the split.
RDLogger.DisableLog("rdApp.*")


def _inchikey_from_smiles(smiles: str) -> Optional[str]:
    """The standard InChIKey for a SMILES, or None if neither toolkit will.

    RDKit (InChI 1.07.3) unless it refuses the molecule, then OpenBabel (1.04).

    The fallback is not optional here. RDKit checks valence and OpenBabel does
    not, and these SMILES come from bonds perceived off simulated coordinates,
    which produces things like a neutral four-bonded nitrogen routinely --
    measured 2026-09-09, RDKit rejects 1,963 of 6,659 inferred structures,
    29.5%. Returning no key for those would empty a field that is populated
    today and is what MDR-55 matched sibling simulations on.

    Which version produced a given key therefore varies by row. That is the
    argument for recording it alongside the value; until there is a column for
    it, prefer the newer and never fail over it.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        key = Chem.MolToInchiKey(mol)
        if key:
            return key

    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    conv.SetOutFormat("inchikey")
    ob_mol = ob.OBMol()
    if not conv.ReadString(ob_mol, smiles):
        return None
    return conv.WriteString(ob_mol).strip() or None


# MDAnalysis prints a forest of harmless warnings on import; quiet them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import MDAnalysis as mda


# ---------------------------------------------------------------------------
# Part 1: Structure file → canonical SMILES
# ---------------------------------------------------------------------------

# Residue names to ignore when auto-detecting the ligand.
_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        # Common protonation / tautomer variants (Amber, CHARMM)
        "HIE",
        "HID",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
        "ASH",
        "GLH",
        "LYN",
        "CYM",
        "CYX",
    }
)

_NUCLEOTIDES = frozenset(
    {
        "A",
        "C",
        "G",
        "T",
        "U",
        "DA",
        "DC",
        "DG",
        "DT",
        "DU",
        "RA",
        "RC",
        "RG",
        "RT",
        "RU",
        "ADE",
        "CYT",
        "GUA",
        "THY",
        "URA",
    }
)

_WATERS = frozenset(
    {
        "HOH",
        "WAT",
        "H2O",
        "TIP",
        "TIP3",
        "TIP3P",
        "TIP4",
        "TIP4P",
        "TIP5",
        "SOL",
        "T3P",
        "T4P",
        "SPC",
        "SPCE",
        "OPC",  # OPC 4-site water model (AMBER)
        # GROMACS writes 4-char residue names one column early; MDAnalysis reads
        # the last 3 chars. These are the misread forms of common water models:
        "IP3",  # TIP3 misread
        "IP4",  # TIP4 / TIP4P misread
        "IP5",  # TIP5 misread
    }
)

# Common lipid residue names.  Also includes the 3-char suffixes that appear
# when GROMACS writes a 4-char lipid name (e.g. POPC) one column early and
# MDAnalysis reads columns 18-20, dropping the first character.
_LIPIDS = frozenset(
    {
        # Phosphatidylcholines
        "DPPC",
        "POPC",
        "DOPC",
        "DLPC",
        "DMPC",
        "DSPC",
        # Phosphatidylethanolamines
        "POPE",
        "DPPE",
        "DOPE",
        "DLPE",
        "DMPE",
        "DSPE",
        # Phosphatidylglycerols
        "POPG",
        "DPPG",
        "DLPG",
        "DMPG",
        "DSPG",
        # Phosphatidylserines / phosphatidic acids
        "DOPS",
        "DPPS",
        "DOPA",
        # Cholesterol / sphingolipids
        "CHOL",
        "CHL1",
        "CER",
        "PSM",
        # Lysophospholipids / other
        "DHPC",
        # GROMACS 4-char misreads (last 3 chars of the real name):
        "PPC",  # DPPC
        # OPC already in _WATERS (covers POPC → OPC too)
        "OPE",  # POPE / DOPE
        "LPC",  # DLPC
        "MPC",  # DMPC
        # SPC already in _WATERS (covers DSPC → SPC)
        "OPG",  # POPG
        "PPE",  # DPPE
        "PPG",  # DPPG
        "HOL",  # CHOL
        "HL1",  # CHL1
        "OPS",  # DOPS
        "OPA",  # DOPA
        "HPC",  # DHPC
    }
)

# Monatomic ions, by element symbol or common force-field alias.
_ION_NAMES = frozenset(
    {
        "NA",
        "K",
        "MG",
        "CA",
        "CL",
        "ZN",
        "FE",
        "MN",
        "CU",
        "BR",
        "I",
        "IOD",
        "LI",
        "RB",
        "CS",
        "F",
        "CO",
        "NI",
        "HG",
        "CD",
        "BA",
        "SR",
        "AL",
        "SE",
        "SOD",
        "POT",
        "CLA",
        "MGY",
        # CHARMM's names for the rest of its monatomic ions.
        "CAL",
        "CES",
        "LIT",
        "RUB",
        "BAR",
        "CAD",
    }
)


def _is_skipped_residue(resname: str) -> bool:
    """True if this residue name is part of the protein/nucleic/solvent/ion/lipid background."""
    rn = (resname or "").upper()
    if (
        rn in _AMINO_ACIDS
        or rn in _NUCLEOTIDES
        or rn in _WATERS
        or rn in _LIPIDS
    ):
        return True
    # Strip charge/multiplicity decorations: "Na+", "MG2+", "CL-", etc.
    stripped = rn.rstrip("+-0123456789")
    return stripped in _ION_NAMES


# Map our friendly format names (and common file extensions) to the
# topology_format strings MDAnalysis expects.
_FMT_TO_MDA = {
    "pdb": "PDB",
    "gro": "GRO",
    "top": "ITP",  # MDAnalysis treats GROMACS .top/.itp text files identically
    "itp": "ITP",
    "tpr": "TPR",
    "psf": "PSF",
    "cif": "MMCIF",
    "mmcif": "MMCIF",
    "prmtop": "TOP",  # Amber parm
    "parm7": "TOP",
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

    if any(
        re.match(
            r"^\s*\[\s*(defaults|atomtypes|moleculetype|system|molecules)\s*\]",
            L,
        )
        for L in head
    ):
        return "ITP"  # GROMACS topology / .itp text format

    pdb_records = (
        "ATOM  ",
        "HETATM",
        "HEADER",
        "CRYST1",
        "MODEL ",
        "REMARK",
        "TITLE ",
        "COMPND",
    )
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


def find_ligand_resnames(universe, notes: Optional[list] = None) -> dict:
    """{resname: atom_count} for non-background residues in a Universe.

    A residue name is background by the lists above, or when every residue of
    that name is joined into a polymer (see _polymer_residues) and carries no
    molecule attached to it that can be released (see _attached_group). Each
    name passed over that way is said in `notes`, for the submitter.
    """
    counts: dict = {}
    if not hasattr(universe.atoms, "resnames"):
        return counts
    names, totals = np.unique(universe.atoms.resnames, return_counts=True)
    for n, c in zip(names, totals):
        n_str = str(n)
        if not _is_skipped_residue(n_str):
            counts[n_str] = int(c)
    linked = _polymer_residues(universe, counts)
    for n in list(counts):
        residues = universe.residues[universe.residues.resnames == n]
        if not (len(residues) and all(r.resindex in linked for r in residues)):
            continue
        first = residues[0]
        where = f"{n} {first.resid}"
        if not _has_coords(first.atoms):
            del counts[n]
            _note(
                notes,
                f"Residue {where} is joined into a polymer chain, so it was "
                f"not taken for a ligand; with no coordinates, a molecule "
                f"bonded to its side chain could not be looked for.",
            )
            continue
        attached = _attached_group(first.atoms)
        if attached is None:
            del counts[n]
            _note(
                notes,
                f"Residue {where} is joined into a polymer chain, so it was "
                f"not taken for a ligand.",
            )
        elif attached.mol is None:
            del counts[n]
            _note(
                notes,
                f"Residue {where} carries a group of {attached.num_heavy} "
                f"heavy atoms covalently bound at {n} {attached.join}, which "
                f"was not recorded as a ligand: the bond is not one that "
                f"hydrolysis would break, so the molecule it came from is not "
                f"known. Declare the ligand in the metadata if it is one.",
            )
    return counts


def _note(notes: Optional[list], text: str) -> None:
    """Add `text` to `notes`, once."""
    if notes is not None and text not in notes:
        notes.append(text)


# Atom-name pairs that join a residue into a polymer, the residue's own atom
# first: a peptide bond from either end, a phosphodiester from either end.
_BACKBONE_LINKS = (
    ("N", "C"),
    ("C", "N"),
    ("P", "O3'"),
    ("P", "O3*"),
    ("O3'", "P"),
    ("O3*", "P"),
)
# A peptide C-N is 1.33 A and a phosphodiester P-O3' 1.61 A; the same atoms
# unbonded do not come within 2.5 A.
_LINK_CUTOFF = 1.9
# Atoms per block of the backbone-link search, which computes every distance
# (see _backbone_links): a block of 1,024 against 20,000 partners is 160 MB.
_LINK_SEARCH_BLOCK = 1024


def _polymer_residues(universe, resnames) -> set:
    """
    Resindices of the residues named in `resnames` that are joined by a
    peptide or phosphodiester bond into a chain holding at least one standard
    amino acid or nucleotide: residues of a polymer, not molecules beside it.

    A ligand is chosen by elimination, and the lists of standard residues can
    never name every residue a chain may hold. The Gla domain of a
    NAMD-simulated coagulation protein carries nine gamma-carboxyglutamates
    (CGU). Taken for a ligand, the first was cut from its chain and read as
    2-[(2S)-2-amino-3-oxopropyl]propanedioic acid -- its backbone carbonyl an
    aldehyde, its carboxylates given hydrogens -- and, with no ligand
    declared, published as the simulation's ligand. Across the processed
    corpus the same befalls AIB, pyroglutamate (PCA), homoarginine (HRG), the
    retinal-bound lysine (LYR) and an acetyl cap (ACE).

    What makes a residue part of a chain is its backbone bond, whatever its
    name. A ligand bonded to the protein through a side chain -- a covalent
    inhibitor on a cysteine's sulfur or a lysine's NZ -- is not joined this
    way and stays a ligand. So does a ligand that is itself a chain of
    nonstandard residues, cyclosporin's for one, bonded to no standard
    residue. Bonds come from the topology where it states them and from
    distances in the frame, both: a file whose CONECT records cover only its
    ligand still has its peptide bonds found. With neither, nothing is
    excluded. The answer for a residue does not depend on which other names
    are asked about.
    """
    names = {str(n) for n in resnames}
    atoms = universe.atoms
    if (
        not len(atoms)
        or not np.isin(atoms.resnames.astype(str), list(names)).any()
    ):
        return set()

    edges = _backbone_links(universe)
    parent: dict = {}

    def find(r):
        while parent.setdefault(r, r) != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for x, y in edges:
        parent[find(x)] = find(y)

    resnames_by_ix = universe.residues.resnames.astype(str)
    anchored = {
        find(r)
        for r in list(parent)
        if resnames_by_ix[r] in _AMINO_ACIDS
        or resnames_by_ix[r] in _NUCLEOTIDES
    }
    return {
        int(r)
        for r in list(parent)
        if resnames_by_ix[r] in names and find(r) in anchored
    }


def _backbone_links(universe) -> set:
    """(resindex, resindex) for each backbone bond between two residues, from
    the topology's stated bonds and from distances in the frame."""
    atoms = universe.atoms
    names = atoms.names.astype(str)
    resix = atoms.resindices
    edges: set = set()

    try:
        bonded = atoms.bonds.indices if len(atoms.bonds) else None
    except (mda.exceptions.NoDataError, AttributeError):
        bonded = None
    if bonded is not None:
        i, j = bonded[:, 0], bonded[:, 1]
        for own, other in _BACKBONE_LINKS:
            for a, b in ((i, j), (j, i)):
                hit = (
                    (names[a] == own)
                    & (names[b] == other)
                    & (resix[a] != resix[b])
                )
                edges.update(
                    zip(resix[a[hit]].tolist(), resix[b[hit]].tolist())
                )

    try:
        atoms.positions
    except (mda.exceptions.NoDataError, AttributeError):
        return edges
    from MDAnalysis.lib.distances import capped_distance

    box = universe.dimensions
    if box is None or not np.all(box[:3] > 0):
        box = None
    # Every distance, block by block. In a triclinic box MDAnalysis 2.10's
    # nsgrid and pkdtree searches both drop pairs well inside the cutoff:
    # MDR00020894, two RNA strands in a truncated octahedron, lost 7 of its
    # 335 backbone links to nsgrid, among them the O3'-P bond (1.56 A) of a
    # strand's first residue (G5), which was then reported as a ligand.
    # Bonded pairs placed at random outside such a cell are missed at about 2%
    # by nsgrid and 1% by pkdtree, wrapped into the cell or not; in a cubic
    # box, and with bruteforce in any box, none are.
    for own, other in _BACKBONE_LINKS:
        a = atoms[names == own]
        b = atoms[names == other]
        if len(a) == 0 or len(b) == 0:
            continue
        for start in range(0, len(a), _LINK_SEARCH_BLOCK):
            block = a[start : start + _LINK_SEARCH_BLOCK]
            pairs = capped_distance(
                block.positions,
                b.positions,
                _LINK_CUTOFF,
                box=box,
                method="bruteforce",
                return_distances=False,
            )
            for i, j in pairs:
                if block[i].resindex != b[j].resindex:
                    edges.add((int(block[i].resindex), int(b[j].resindex)))
    return edges


def resolve_target_resnames(
    universe, path: str, resname: Optional[str], notes: Optional[list] = None
) -> list:
    """
    Decide which residue name(s) to extract from a Universe.

    - If `resname` is given AND atoms with that name exist in the file → use
      it alone.
    - If `resname` is given but NOT present → fall back to every non-background
      residue in the file (matches prior PDB extraction semantics).
    - If `resname` is None → return every non-background residue.

    Always returns a non-empty list, or raises ValueError.
    """
    present = (
        set(map(str, np.unique(universe.atoms.resnames)))
        if hasattr(universe.atoms, "resnames")
        else set()
    )
    if resname is not None and resname in present:
        return [resname]

    candidates = find_ligand_resnames(universe, notes)
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
    inchikey = _inchikey_from_smiles(smiles) or ""
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
    mol = _read_coords(sel)
    return _mol_summary(_bond_orders_from_hydrogens(mol) or mol)


def _read_coords(sel):
    """The selection as an OBMol, its bonds and bond orders read by
    OpenBabel from the coordinates. Atoms keep the selection's order."""
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
        return mol
    finally:
        os.unlink(tmp_path)


# An amino acid's backbone atoms, by the names every force field gives them.
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT", "OT1", "OT2"})
# N, O, S, Se: where an amino acid's side chain can be joined to something.
_SIDE_CHAIN_HETEROATOMS = frozenset({7, 8, 16, 34})
# An attached group this size or larger is a molecule on the residue: retinal
# (20 heavy atoms past the lysine's NZ), PLP (15), biotin (15), farnesyl and
# myristoyl (15), lipoyl (11), palmitoyl (17). Smaller ones are the residue's
# own modification: a phosphate, an acetyl, a methyl, a guanidino,
# gamma-carboxyglutamate's second carboxylate, and the two residues the
# genetic code itself carries on a lysine, hypusine's aminohydroxybutyl (6)
# and pyrrolysine's methylpyrroline-carbonyl (8).
_MIN_ATTACHED_HEAVY_ATOMS = 10


class _Attached(NamedTuple):
    """A group found on a chain residue's side chain. `mol` is the molecule
    released from it, or None when the bond is not one hydrolysis breaks."""

    mol: Optional["ob.OBMol"]
    join: str
    num_heavy: int


def _attached_group(sel):
    """
    A molecule attached to a chain residue through its side chain, as the
    molecule free of it, with the side-chain atom's name, or None when the
    residue carries none.

    Some residues of a chain carry a molecule bonded to them: rhodopsin's
    retinal on a lysine (LYR), a palmitoyl on a cysteine (CYSP). The residue is
    part of the chain (_polymer_residues), but the molecule on it is what a
    reader would call the ligand. The amino acid is taken to run out from CA,
    through carbons, to the first N, O, S or Se; a group of at least
    _MIN_ATTACHED_HEAVY_ATOMS heavy atoms hanging from that atom by a single
    bond is the molecule.

    Its bond orders are read with the side-chain atom still in place and a
    hydrogen standing in for the rest of the amino acid, so that an all-atom
    group is a whole molecule to _bond_orders_from_hydrogens. It is then
    reported as released from the residue by hydrolysis: the side-chain atom
    becomes an oxygen with the same bond to the group. A retinal Schiff base
    (C=NZ) gives retinal (C=O); a palmitoyl thioester (C(=O)-SG) gives
    palmitic acid; biotin's amide on NZ gives biotin.

    That holds only for a bond hydrolysis breaks: a double bond to the
    side-chain atom (an imine), or a single one to an acyl carbon (a
    thioester, ester or amide). A group joined any other way -- a thioether
    like farnesyl's, a Michael adduct on a cysteine -- would come back
    hydroxylated, a molecule no one put there, so its `mol` is None.
    """
    names = [str(n) for n in sel.names]
    if "CA" not in names:
        return None
    try:
        mol = _read_coords(sel)
    except Exception:
        return None
    atoms = list(ob.OBMolAtomIter(mol))
    if len(atoms) != len(names):
        return None
    heavy = [a.GetAtomicNum() != 1 for a in atoms]
    adj = {
        i: [n.GetIdx() - 1 for n in ob.OBAtomAtomIter(a) if heavy[n.GetIdx() - 1]]
        for i, a in enumerate(atoms)
        if heavy[i]
    }
    ca = names.index("CA")
    backbone = {i for i, n in enumerate(names) if heavy[i] and n in _BACKBONE_ATOMS}

    # Carbons out from CA, stopping at the first heteroatoms.
    amino, stack, joins = {ca}, [ca], []
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j in amino or j in backbone:
                continue
            amino.add(j)
            if atoms[j].GetAtomicNum() in _SIDE_CHAIN_HETEROATOMS:
                joins.append(j)
            else:
                stack.append(j)

    best = None
    for x in joins:
        for y in adj[x]:
            if y in amino or y in backbone:
                continue
            group, stack = {y}, [y]
            while stack:
                i = stack.pop()
                for j in adj[i]:
                    if j != x and j not in group:
                        group.add(j)
                        stack.append(j)
            if group & (amino | backbone):
                continue
            if len(group) >= _MIN_ATTACHED_HEAVY_ATOMS and (
                best is None or len(group) > len(best[2])
            ):
                best = (x, y, group)
    if best is None:
        return None
    x, y, group = best

    # The group, its hydrogens, the side-chain atom and its hydrogens, and a
    # hydrogen where each of that atom's amino-acid neighbours was.
    g = ob.OBMol(mol)
    gat = list(ob.OBMolAtomIter(g))
    keep = set(group) | {x}
    keep |= {
        i
        for i, a in enumerate(atoms)
        if not heavy[i]
        and any((n.GetIdx() - 1) in keep for n in ob.OBAtomAtomIter(a))
    }
    xa, ya = gat[x], gat[y]
    for w in adj[x]:
        if w in group:
            continue
        here = np.array([xa.GetX(), xa.GetY(), xa.GetZ()])
        there = np.array([gat[w].GetX(), gat[w].GetY(), gat[w].GetZ()])
        at = here + (there - here) / np.linalg.norm(there - here)
        h = g.NewAtom()
        h.SetAtomicNum(1)
        h.SetVector(*map(float, at))
        g.AddBond(xa.GetIdx(), h.GetIdx(), 1)
    for i in sorted(set(range(len(gat))) - keep, reverse=True):
        g.DeleteAtom(gat[i])
    xi, yi = xa.GetIdx(), ya.GetIdx()

    settled = _bond_orders_from_hydrogens(g) or g
    if not _hydrolysable(settled, xi, yi):
        return _Attached(None, names[x], len(group))
    xs = settled.GetAtom(xi)
    for h in [n for n in ob.OBAtomAtomIter(xs) if n.GetAtomicNum() == 1]:
        settled.DeleteAtom(h)
    xs = settled.GetAtom(xi)
    xs.SetAtomicNum(8)
    xs.SetFormalCharge(0)
    ob.OBAtomAssignTypicalImplicitHydrogens(xs)
    return _Attached(settled, names[x], len(group))


def _hydrolysable(mol, xi: int, yi: int) -> bool:
    """Whether the bond from side-chain atom `xi` to the group's atom `yi`
    (OpenBabel indices) is one hydrolysis breaks: double (an imine), or single
    to a carbon double-bonded to an O, S or N (an acyl)."""
    x, y = mol.GetAtom(xi), mol.GetAtom(yi)
    bond = mol.GetBond(x, y)
    if bond is None:
        return False
    if bond.GetBondOrder() == 2:
        return True
    return y.GetAtomicNum() == 6 and any(
        b.GetBondOrder() == 2 and b.GetNbrAtom(y).GetAtomicNum() in (7, 8, 16)
        for b in ob.OBAtomBondIter(y)
        if b.GetNbrAtom(y).GetIdx() != xi
    )


# A double or triple bond the hydrogens call for must be shorter in the frame
# than a single bond between the same two elements, by at least this much.
# This refuses what a united-atom residue forces, double bonds at full
# single-bond length; it cannot tell a strained double bond from a
# conjugated single one. In the processed frames genuine C=N run to 1.41 A
# and C=C to 1.47 A, 0.05-0.10 under the summed radii, where a thioester's
# C-S sits at 1.74 A, 0.07 under: a margin of 0.10 lost fourteen correct
# readings to refuse that one.
_MULTIPLE_BOND_MARGIN = 0.05

# The search for bond orders is exponential in the atoms whose valence it can
# choose. Real ligands settle in under 100 iterations; a polyphosphate -- every
# phosphate oxygen a candidate for the double bond or the charge -- never
# settles, and without a bound held a worker for hours. Past this it gives up
# in about 0.1 s and OpenBabel's reading stands.
_MAX_BOND_ORDER_ITERATIONS = 10000


def _bond_orders_from_hydrogens(mol) -> Optional["ob.OBMol"]:
    """
    Bond orders settled by valence from the explicit hydrogens, on OpenBabel's
    connectivity, or None to keep OpenBabel's geometric reading.

    PerceiveBondOrders types each atom by its bond angles, and a simulated frame
    bends them. Vorapaxar's vinyl carbons, 1.36 A apart with three neighbours
    each, came out of a NAMD frame with angles OpenBabel took for sp3: the bond
    was made single and each carbon given an implicit hydrogen, C29H35FN2O4 for
    C29H33FN2O4. An all-atom simulation states every hydrogen, and with every
    hydrogen in place the valences leave one assignment of bond orders; RDKit's
    DetermineBondOrders finds it without reading an angle.

    The total charge is tried as the file's formal charges state it, then one
    either side, and the first assignment that holds is kept. It is refused --
    and OpenBabel's reading stands -- when it leaves a radical, charges a
    carbon, charges an oxygen positively, charges any atom by more than one,
    or calls a bond double or triple that the frame holds at single-bond
    length outside an aromatic ring. Those are what a residue with an open
    valence is otherwise forced into: a united-atom residue, whose carbons
    carry their hydrogens implicitly, or a residue cut from a polymer, whose
    backbone carbonyl read as an acylium (C#[O+]) and whose pyroglutamate came
    out a +2 ion. A residue with no hydrogens at all is left to OpenBabel, as
    is one whose bond orders the search cannot settle within
    _MAX_BOND_ORDER_ITERATIONS.
    """
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    atoms = list(ob.OBMolAtomIter(mol))
    if not any(a.GetAtomicNum() == 1 for a in atoms):
        return None

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    stated = 0
    for a in atoms:
        ra = Chem.Atom(a.GetAtomicNum())
        ra.SetFormalCharge(a.GetFormalCharge())
        ra.SetNoImplicit(True)
        stated += a.GetFormalCharge()
        i = rw.AddAtom(ra)
        conf.SetAtomPosition(i, Point3D(a.GetX(), a.GetY(), a.GetZ()))
    for b in ob.OBMolBondIter(mol):
        i, j = b.GetBeginAtomIdx() - 1, b.GetEndAtomIdx() - 1
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    rw.AddConformer(conf, assignId=True)

    for charge in (stated, stated - 1, stated + 1):
        m = Chem.Mol(rw)
        try:
            rdDetermineBonds.DetermineBondOrders(
                m, charge=charge, maxIterations=_MAX_BOND_ORDER_ITERATIONS
            )
            Chem.SanitizeMol(m)
        except Exception:
            continue
        if any(_implausible(a) for a in m.GetAtoms()):
            continue
        if not _multiple_bonds_are_short(m):
            continue

        Chem.Kekulize(m, clearAromaticFlags=True)
        conv = ob.OBConversion()
        conv.SetInFormat("mol")
        out = ob.OBMol()
        if conv.ReadString(out, Chem.MolToMolBlock(m)):
            return out
    return None


def _implausible(atom) -> bool:
    """A radical or a charge no simulated ligand carries: on carbon, positive
    on oxygen, positive on a double-bonded sulfur, or more than one on any
    atom.

    A double-bonded S+ is a charge-separated form, not a molecule: with a
    thioacid's sulfur listed before its oxygen the valence search returns
    C(=[SH+])[O-] for C(=O)S. A sulfonium's S+ has three single bonds and
    stands."""
    q = atom.GetFormalCharge()
    return bool(
        atom.GetNumRadicalElectrons()
        or abs(q) > 1
        or (q and atom.GetAtomicNum() == 6)
        or (q > 0 and atom.GetAtomicNum() == 8)
        or (
            q > 0
            and atom.GetAtomicNum() == 16
            and any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())
        )
    )


def _multiple_bonds_are_short(m) -> bool:
    """Whether every double or triple bond of `m` outside an aromatic ring is
    shorter in its conformer than a single bond between the same elements."""
    table = Chem.GetPeriodicTable()
    pos = m.GetConformer().GetPositions()
    for bond in m.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.SINGLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        single = sum(
            table.GetRcovalent(m.GetAtomWithIdx(k).GetAtomicNum())
            for k in (i, j)
        )
        if np.linalg.norm(pos[i] - pos[j]) > single - _MULTIPLE_BOND_MARGIN:
            return False
    return True


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


def universe_to_smiles(
    universe, resname: str, notes: Optional[list] = None
) -> dict:
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
        raise ValueError(
            f"No atoms with residue name '{resname}' in the file."
        )

    # If multiple copies of the same residue name are present (e.g., many lipid
    # molecules in a full simulation box), use only the first residue instance so
    # that SMILES represents a single molecule rather than all copies concatenated.
    # A copy joined into a polymer is passed over while a free one exists.
    unique_resix = np.unique(all_match.resindices)
    linked = _polymer_residues(universe, {str(resname)})
    free = [r for r in unique_resix if int(r) not in linked]
    if free:
        unique_resix = np.array(free)
    sel = all_match[all_match.resindices == unique_resix[0]]

    has_coords = _has_coords(sel)

    # A chain residue is read for the molecule attached to it, where one is.
    if has_coords and int(unique_resix[0]) in linked:
        attached = _attached_group(sel)
        if attached is not None and attached.mol is not None:
            result = _mol_summary(attached.mol)
            result["cut_from"] = f"{resname} {attached.join}"
            _note(
                notes,
                f"The ligand {result['formula']} was found covalently bound "
                f"to residue {resname} {sel.residues[0].resid} at "
                f"{attached.join}, part of a polymer chain. It is recorded as "
                f"the molecule hydrolysis would release, not as bound.",
            )
            return result

    return (
        _smiles_from_coords(sel) if has_coords else _smiles_from_topology(sel)
    )


def _has_coords(sel) -> bool:
    try:
        pos = sel.positions
        return pos is not None and len(pos) == len(sel)
    except (mda.exceptions.NoDataError, AttributeError):
        return False


def structure_to_smiles(
    path: str,
    fmt: Optional[str] = None,
    resname: Optional[str] = None,
    notes: Optional[list] = None,
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

    What the submitter should hear of -- a residue passed over as part of a
    chain, a ligand cut from one -- is added to `notes`.
    """
    universe = load_universe(path, fmt)
    target_resnames = resolve_target_resnames(universe, path, resname, notes)
    results = []
    for rn in target_resnames:
        result = universe_to_smiles(universe, rn, notes)
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
    """Compute the InChIKey for a SMILES string via RDKit (InChI 1.07.3).

    This key is the lookup term for Wikidata and PDBe below, and those index
    standard InChIKeys, so the version that generates it matters more here than
    anywhere else in this file.
    """

    return _inchikey_from_smiles(smiles)


def query_wikidata_by_inchikey(inchikey: str) -> Optional[str]:
    """
    Look up a chemical compound on Wikidata by InChIKey (property P235) and
    return its English label, e.g. "ATP" or "adenosine 2',5'-bisphosphate".
    Returns None if no Wikidata entry has this InChIKey.
    """
    sparql = (
        "SELECT ?itemLabel WHERE { "
        f'?item wdt:P235 "{inchikey}" . '
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } '
        "} LIMIT 1"
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
    pdbe_id = next(
        (x["src_compound_id"] for x in refs if x.get("src_id") == "5"), None
    )
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
        return True  # CAS number, e.g. "3805-37-6"
    if _PDB_ID_RE.match(n):
        return True  # PDB ID, e.g. "8hvp"
    if _INCHIKEY_RE.match(n):
        return True  # InChIKey masquerading as a synonym
    if ":" in n:
        return True  # registry IDs: CHEBI:..., MeSH:..., RefChem:...
    if len(n) > 80:
        return True  # IUPAC behemoths
    if " " not in n and len(n) > 30:
        return True  # long no-space tokens: peptide notations, vendor codes
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
    wikidata_label = (
        query_wikidata_by_inchikey(inchikey) if inchikey else None
    )
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
                pubchem_synonyms = syn_data["InformationList"]["Information"][
                    0
                ].get("Synonym", [])
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

    if (
        wikidata_label
        and any(c.islower() for c in wikidata_label)
        and not is_junk_synonym(wikidata_label)
    ):
        best_name = wikidata_label
        name_source = "wikidata"

    if best_name is None:
        syn = first_acceptable_synonym(pubchem_synonyms)
        if syn:
            best_name = syn
            name_source = "pubchem_synonyms"

    if (
        best_name is None
        and wikidata_label
        and not is_junk_synonym(wikidata_label)
    ):
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
            return "".join(
                indent + line for line in text.splitlines(keepends=True)
            )
        return textwrap.fill(
            text, width, initial_indent=indent, subsequent_indent=indent
        )


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
        '  uv run mol_id.py name-from-smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"\n'
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
    sub = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND"
    )

    def _add_verbose_arg(p):
        p.add_argument(
            "-v",
            "--verbose",
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
            "--format",
            dest="fmt",
            choices=[
                "pdb",
                "gro",
                "top",
                "itp",
                "tpr",
                "psf",
                "cif",
                "mmcif",
                "prmtop",
                "parm7",
            ],
            default=None,
            help="Force input file format instead of sniffing from content / extension.",
        )
        p.add_argument(
            "--resname",
            default=None,
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
            '  uv run mol_id.py name-from-smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"'
        ),
        formatter_class=_SmartHelpFormatter,
    )
    p2.add_argument(
        "smiles",
        help="SMILES string (quote it to protect special characters from the shell).",
    )
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
            "Example:\n" "  uv run mol_id.py both prod.tpr -o result.json"
        ),
        formatter_class=_SmartHelpFormatter,
    )
    _add_struct_args(p3)
    p3.add_argument(
        "-o",
        "--outfile",
        help="Write full JSON output to this file instead of stdout.",
    )
    _add_verbose_arg(p3)

    args = parser.parse_args()
    notes: list = []
    try:
        _run(args, notes)
    finally:
        # One line each on stderr, for mdr-process to pass on as warnings.
        for note in notes:
            print(f"{NOTE_MARKER}{note}", file=sys.stderr)


# The prefix of each note's line on stderr. mdr-process reads it.
NOTE_MARKER = "[mdrepo] note="


def _run(args, notes: list) -> None:
    if args.command == "smiles-from-structure":
        results = structure_to_smiles(
            args.path, fmt=args.fmt, resname=args.resname, notes=notes
        )
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
        struct_results = structure_to_smiles(
            args.path, fmt=args.fmt, resname=args.resname, notes=notes
        )
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
