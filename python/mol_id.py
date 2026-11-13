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
    residue: it is read as one peptide (_spanning_molecule). Bonds come from the topology where it states them and from
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


def _canonical_smiles(mol) -> str:
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    return conv.WriteString(mol).strip().split("\t")[0]


def _neutral_form(smiles: str) -> Optional["ob.OBMol"]:
    """The molecule of `smiles` with its charges neutralised by adding and
    removing protons, every hydrogen explicit, or None when that changes
    nothing or RDKit will not read the SMILES.

    A carboxylate or a phosphate gains its proton and an ammonium loses one.
    A charge no proton can settle is left as it is: a quaternary nitrogen,
    with one anion kept to balance it (phosphocholine stays a zwitterion), a
    nitro group, an N-oxide.
    """
    from rdkit.Chem.MolStandardize import rdMolStandardize

    m = Chem.MolFromSmiles(smiles)
    if m is None or not any(a.GetFormalCharge() for a in m.GetAtoms()):
        return None
    try:
        neutral = rdMolStandardize.Uncharger().uncharge(m)
    except Exception:
        return None
    if Chem.MolToSmiles(neutral) == Chem.MolToSmiles(m):
        return None
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    out = ob.OBMol()
    if not conv.ReadString(out, Chem.MolToSmiles(neutral)):
        return None
    out.AddHydrogens()
    return out


def _mol_summary(mol) -> dict:
    """The SMILES/formula/InChIKey summary for an OBMol, as the molecule's
    neutral form.

    A ligand is recorded neutral whatever charge the simulation holds it at
    (Travis, 2026-11-12): the record says which molecule it is, not its
    protonation state at the simulation's pH. When the structure holds it
    charged, `simulated_smiles`, `simulated_formula` and `simulated_charge`
    say how, so that a declared ligand can be compared with what is in the
    file and not only with the form recorded.
    """
    simulated = _canonical_smiles(mol)
    neutral = _neutral_form(simulated)
    if neutral is None:
        return {
            "smiles": simulated,
            "formula": mol.GetFormula(),
            "num_atoms": mol.NumAtoms(),
            "num_heavy_atoms": mol.NumHvyAtoms(),
            "charge": mol.GetTotalCharge(),
            "inchikey": _inchikey_from_smiles(simulated) or "",
        }
    smiles = _canonical_smiles(neutral)
    return {
        "smiles": smiles,
        "formula": neutral.GetFormula(),
        "num_atoms": neutral.NumAtoms(),
        "num_heavy_atoms": neutral.NumHvyAtoms(),
        "charge": neutral.GetTotalCharge(),
        "inchikey": _inchikey_from_smiles(smiles) or "",
        "simulated_smiles": simulated,
        "simulated_formula": mol.GetFormula(),
        "simulated_charge": mol.GetTotalCharge(),
    }


def _smiles_from_coords(sel) -> dict:
    """
    Coordinates-available path: serialize the selection to a temp PDB and let
    OpenBabel's PDB reader handle it. This reuses OB's well-tested geometry-
    based bond and aromaticity perception.
    """
    mol = _read_coords(sel)
    return _mol_summary(_bond_orders_from_hydrogens(mol) or mol)


# Two atoms this close (A) are bonded for the purpose of making a molecule
# whole: every covalent bond between the elements a ligand holds (C-I 2.14,
# S-S 2.05) and nothing a periodic image could mimic.
_WHOLE_BOND_CUTOFF = 2.2


def _whole_positions(sel):
    """`sel`'s positions with its molecule made whole across the periodic
    boundary, or None when there is no box or nothing to move.

    An XTC stores atoms wrapped into the cell, not molecules, and the
    TPR-less GROMACS path (cpptraj with the coordinate PDB as topology)
    keeps them so. OpenBabel then perceives bonds from the coordinates and
    reads a ligand split across the box as two fragments with the ring
    destroyed and the valences filled: ibuprofen as
    CC(=C)[C@H](C(=O)O)C.CC(CC(=C)C)C, C13H24O2 (review F6).

    Atoms within `_WHOLE_BOND_CUTOFF` under the minimum image are taken as
    bonded and walked from the first atom of each connected piece, each
    atom placed in the image nearest the atom it was reached from. That
    makes a molecule of any size whole (a chain longer than half the box
    included, which a single shift to the image nearest one atom would
    break). An atom reached from nothing keeps its place.
    """
    box = sel.universe.dimensions
    if box is None or not np.all(box[:3] > 0) or len(sel) < 2:
        return None
    from MDAnalysis.lib.distances import capped_distance, minimize_vectors

    pos = sel.positions
    pairs = capped_distance(
        pos,
        pos,
        _WHOLE_BOND_CUTOFF,
        box=box,
        method="bruteforce",
        return_distances=False,
    )
    adjacent = [[] for _ in range(len(pos))]
    for i, j in pairs:
        if i != j:
            adjacent[i].append(int(j))
            adjacent[j].append(int(i))

    out = pos.copy()
    placed = np.zeros(len(pos), dtype=bool)
    for start in range(len(pos)):
        if placed[start]:
            continue
        placed[start] = True
        queue = [start]
        while queue:
            i = queue.pop()
            for j in adjacent[i]:
                if placed[j]:
                    continue
                step = minimize_vectors((pos[j] - out[i]).reshape(1, 3), box)[0]
                out[j] = out[i] + step
                placed[j] = True
                queue.append(j)
    if np.allclose(out, pos):
        return None
    return out


def _read_coords(sel):
    """The selection as an OBMol, its bonds and bond orders read by
    OpenBabel from the coordinates. Atoms keep the selection's order. The
    molecule is written whole (`_whole_positions`); the selection's own
    positions are left as they were."""
    fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    whole = _whole_positions(sel)
    original = sel.positions.copy() if whole is not None else None
    try:
        if whole is not None:
            sel.positions = whole
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
        if original is not None:
            sel.positions = original
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

# A residue with explicit polar hydrogens but united-atom CH groups lets the
# search make up the missing hydrogens with bond orders, and in a ring or a
# vinyl group, already short, the margin above passes them: hydroquinone with
# its ring CH united came back OC1=C=C=C(O)C#C1, every ring bond ~1.40 A.
# What it forces is always a triple bond or a carbon with two double bonds,
# and a real one is far shorter than a single bond: C#C 1.20 A and C#N 1.16
# under sums of 1.52 and 1.47, an allene's or a ketene's C=C 1.31. These must
# be under a single bond by this much.
_TRIPLE_BOND_MARGIN = 0.25
_CUMULATED_BOND_MARGIN = 0.20
# No ligand has a triple bond in a ring this small: cyclooctyne, the smallest
# cycloalkyne that can be kept in a bottle, has eight.
_MAX_RING_WITH_NO_TRIPLE_BOND = 7

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

    No file says what a ligand's total charge is (see _charges_to_try), so
    every charge from 0 to +-_MAX_SEARCHED_CHARGE is tried, and of the
    assignments that hold the one with the fewest charged atoms is kept, then
    the one of lowest total charge. An assignment is refused when it leaves a
    radical, charges a carbon, charges an oxygen positively, charges any atom
    by more than one, puts two double bonds on a charged ring nitrogen, reads
    a nitro group as N([O-])[O-], calls
    a bond double or triple that the frame holds at single-bond length, calls
    a bond triple or cumulated that the frame holds at double- or
    aromatic-bond length, or puts a triple bond in a ring of seven atoms or
    fewer. When every charge is refused OpenBabel's reading stands. Those are
    what a residue
    with an open valence is otherwise forced into: a united-atom residue,
    whose carbons carry their hydrogens implicitly (all of them, or beside
    explicit polar hydrogens), or a residue cut from a polymer, whose
    backbone carbonyl read as an acylium (C#[O+]) and whose pyroglutamate came
    out a +2 ion. A residue with no hydrogens at all is left to OpenBabel, as
    is one whose bond orders the search cannot settle within
    _MAX_BOND_ORDER_ITERATIONS.

    The search depends on atom order. With a sulfur listed ahead of the atoms
    that should carry the double bonds, it returns charge-separated forms --
    a thiazole as C=[S+]...[O-] -- which the checks refuse; when every charge
    has been refused, or the assignment kept still has charged atoms, the
    search runs once more with the sulfurs listed last (see _sulfur_last) and
    the better of the two is kept.
    """
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

    orders = [None]
    if (sulfur_last := _sulfur_last(rw)) is not None:
        orders.append(sulfur_last)
    # The better of the two orders by the same rule as within one: fewest
    # charged atoms, then lowest total charge. An order that settles with no
    # charged atom cannot be bettered and the other is not searched.
    best, best_rank = None, None
    for order in orders:
        m = _settled_bond_orders(rw, stated, order)
        if m is None:
            continue
        rank = _charge_rank(m)
        if best_rank is None or rank < best_rank:
            best, best_rank = m, rank
        if best_rank[0] == 0:
            break
    if best is None:
        return None
    Chem.Kekulize(best, clearAromaticFlags=True)
    conv = ob.OBConversion()
    conv.SetInFormat("mol")
    out = ob.OBMol()
    if conv.ReadString(out, Chem.MolToMolBlock(best)):
        return out
    return None


def _charge_rank(m) -> tuple:
    """(charged atoms, magnitude of the total charge): the lower the better."""
    charges = [a.GetFormalCharge() for a in m.GetAtoms()]
    return (sum(1 for q in charges if q), abs(sum(charges)))


# The largest total charge, either sign, the bond-order search tries. ATP is
# -4, spermine +4, IP3 -6; past this a ligand is a polyphosphate the search
# does not settle anyway (phytate, -12).
_MAX_SEARCHED_CHARGE = 6


def _charges_to_try(stated: int) -> list:
    """Total charges for the bond-order search, smallest magnitude first and
    the negative of a pair before the positive.

    `stated` is the sum of the file's formal charges, and it is 0 for every
    ligand: cpptraj and gmx write no charges into a PDB, and psf_elements
    writes them for monatomic ions only. Tried at 0 and +-1 alone, anything
    of charge 2 or more never had its real charge tried, and the valences
    were then met some other way: AMP(2-) came back with its adenine as
    [N+]1=CN=C2C1=[N+]=CN=C2N, ATP(4-) with a dihydro-adenine, spermine(4+)
    with four neutral four-valent nitrogens (review F2). A stated charge
    outside the range is tried as well.
    """
    charges = set(range(-_MAX_SEARCHED_CHARGE, _MAX_SEARCHED_CHARGE + 1))
    charges.add(stated)
    return sorted(charges, key=lambda q: (abs(q), q > 0))


def _settled_bond_orders(rw, stated: int, order):
    """The bond-order assignment for `rw` that holds with the fewest charged
    atoms and, among those, the lowest total charge (4 before 6), over the
    charges of _charges_to_try; None when none holds. Its atoms are in
    `order` while RDKit searches (None for their own order) and in their own
    order after.

    With every hydrogen explicit the true charge is the one that needs no
    charge separation: at -2 ATP's valences are met with six charged atoms,
    at -4 with four. An assignment at total charge q has at least |q|
    charged atoms, so once one is held with n, no charge of magnitude n or
    more can better it and the search stops: at 0 for a neutral molecule.
    """
    from rdkit.Chem import rdDetermineBonds

    best, best_rank = None, None
    for charge in _charges_to_try(stated):
        if best_rank is not None and abs(charge) >= best_rank[0]:
            break
        m = Chem.Mol(rw)
        if order is not None:
            m = Chem.RenumberAtoms(m, order)
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
        rank = _charge_rank(m)
        if best_rank is None or rank < best_rank:
            best, best_rank = m, rank
    if best is not None and order is not None:
        back = [0] * len(order)
        for new, old in enumerate(order):
            back[old] = new
        best = Chem.RenumberAtoms(best, back)
    return best


def _sulfur_last(m) -> Optional[list]:
    """
    An atom order with every sulfur after every other atom, or None when that
    is the order already.

    DetermineBondOrders depends on atom order. Given a sulfur ahead of the
    atoms that should take the double bonds, it puts one on the sulfur and
    balances it with charges elsewhere: seven thiazole-bearing ligands in the
    processed corpus came out in charge-separated forms with a C=[S+], all
    refused by _implausible, so OpenBabel's reading stood: a nitro group
    RDKit rejects, or a thiazole given hydrogens it does not have. With the
    sulfurs last the same search finds the thiazole -- dasatinib with its
    piperazine protonated, an aminothiazole with its nitro group as
    [N+](=O)[O-].
    """
    sulfur = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 16]
    rest = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 16]
    order = rest + sulfur
    return None if order == list(range(m.GetNumAtoms())) else order


def _implausible(atom) -> bool:
    """A radical or a charge no simulated ligand carries: on carbon, positive
    on oxygen, positive on a double-bonded sulfur, positive on a ring
    nitrogen with two double bonds, or more than one on any atom.

    A ring =[N+]= is what the search makes of a tetrazolate tried at the
    wrong sign: losartan's came back C1=[N+]=NN=N1, a cation, for an anion
    (review F2). Outside a ring it is an azide or a diazo group and stands.

    A neutral nitrogen with two O- is a nitro group read two electrons over,
    N([O-])[O-] for [N+](=O)[O-]: where the search cannot place the nitro's
    own charges at 0 (it depends on atom order) it settles this at -2, and
    it would be recorded as a dihydroxylamine (MDR00000544, MDR00020883).

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
        or (
            q == 0
            and atom.GetAtomicNum() == 7
            and sum(
                n.GetAtomicNum() == 8 and n.GetFormalCharge() < 0
                for n in atom.GetNeighbors()
            )
            > 1
        )
        or (
            q > 0
            and atom.GetAtomicNum() == 7
            and atom.IsInRing()
            and sum(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds())
            > 1
        )
    )


def _multiple_bonds_are_short(m) -> bool:
    """Whether every double or triple bond of `m` is shorter in its conformer
    than a single bond between the same elements: by _TRIPLE_BOND_MARGIN for
    a triple bond, _CUMULATED_BOND_MARGIN for a double bond on a carbon with
    two, and _MULTIPLE_BOND_MARGIN for the rest. A triple bond in a ring of
    _MAX_RING_WITH_NO_TRIPLE_BOND atoms or fewer is refused at any length.

    The bonds are read from a Kekule copy. RDKit's aromaticity model takes a
    ring of alternating triple bonds for aromatic, and aromatic bonds used
    to be passed over here: 2-pyridone with its ring CH united and its NH
    explicit came back [H]n1c#cc#cc1=O, uracil O=C1NC#CC(=O)N1, pyrrole
    N1C#CC#C1, each triple bond 1.34-1.40 A long and never measured (review
    F3).

    One thing is still passed over: the plain double bonds of a ring RDKit
    calls aromatic, which a Kekule structure places on every other bond by
    choice, not by length. A frame stretches them past the margin:
    vorapaxar's NAMD frame holds one aromatic C-C at 1.50 A, and measuring
    it lost the whole reading. Their triple and cumulated bonds are measured.
    """
    aromatic = {b.GetIdx() for b in m.GetBonds() if b.GetIsAromatic()}
    m = Chem.Mol(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    table = Chem.GetPeriodicTable()
    pos = m.GetConformer().GetPositions()
    rings = m.GetRingInfo()

    def cumulated(atom) -> bool:
        return atom.GetAtomicNum() == 6 and (
            sum(
                b.GetBondType() == Chem.BondType.DOUBLE
                for b in atom.GetBonds()
            )
            > 1
        )

    for bond in m.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            continue
        if bond.GetBondType() == Chem.BondType.TRIPLE and any(
            rings.IsBondInRingOfSize(bond.GetIdx(), size)
            for size in range(3, _MAX_RING_WITH_NO_TRIPLE_BOND + 1)
        ):
            return False
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        single = sum(
            table.GetRcovalent(m.GetAtomWithIdx(k).GetAtomicNum())
            for k in (i, j)
        )
        if bond.GetBondType() == Chem.BondType.TRIPLE:
            margin = _TRIPLE_BOND_MARGIN
        elif cumulated(bond.GetBeginAtom()) or cumulated(bond.GetEndAtom()):
            margin = _CUMULATED_BOND_MARGIN
        elif bond.GetIdx() in aromatic:
            continue
        else:
            margin = _MULTIPLE_BOND_MARGIN
        if np.linalg.norm(pos[i] - pos[j]) > single - margin:
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
    sel, in_chain = _first_copy(universe, resname)
    has_coords = _has_coords(sel)

    # A chain residue is read for the molecule attached to it, where one is.
    if has_coords and in_chain:
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


def _first_copy(universe, resname: str):
    """(the atoms of the first residue named `resname`, whether it is joined
    into a polymer chain).

    If multiple copies of the same residue name are present (e.g., many lipid
    molecules in a full simulation box), only the first is read, so that the
    SMILES represents a single molecule rather than all copies concatenated:
    the record says which ligands a simulation holds, not how many of each
    (Travis, 2026-11-12). A copy joined into a polymer is passed over while
    a free one exists.
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
    unique_resix = np.unique(all_match.resindices)
    linked = _polymer_residues(universe, {str(resname)})
    free = [r for r in unique_resix if int(r) not in linked]
    if free:
        unique_resix = np.array(free)
    first = int(unique_resix[0])
    return all_match[all_match.resindices == first], first in linked


# The elements whose bonds make one molecule of several residues. A metal is
# left out on purpose: an imidazole on a heme's iron, 2.0 A away, is a ligand
# beside the heme, not a part of it.
_COVALENT_ELEMENTS = frozenset({5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53})
# Two atoms of different residues are bonded when closer than the sum of
# their covalent radii plus this (A). A glycosidic C-O is 1.43 against
# 1.72, a disulfide 2.05 against 2.40; a hydrogen bond's O...O, 2.5 at its
# shortest, is far outside 1.62.
_BOND_TOLERANCE = 0.3
# Residues that end a peptide and are not residues of it.
_PEPTIDE_CAPS = frozenset({"ACE", "FOR", "NME", "NMA", "NH2", "NHE", "CT3"})
_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    "HIE": "H", "HID": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K", "CYM": "C", "CYX": "C",
}  # fmt: skip


class _Spanning(NamedTuple):
    """A molecule made of several residues. `residues` are in chain order
    when `peptide`, in file order otherwise."""

    residues: list
    peptide: bool
    cyclic: bool


def _bonded_residues(universe, start: int, eligible: np.ndarray) -> list:
    """Resindices of the residues covalently bonded, directly or through each
    other, to residue `start`, itself included, in file order. Only residues
    marked in `eligible` (by resindex) are walked into.

    Bonds are the topology's where it states them and, with coordinates,
    those found by distance under the minimum image (_BOND_TOLERANCE), both,
    as for backbone links.
    """
    atoms = universe.atoms
    resix = atoms.resindices
    found = {int(start)}

    stated: dict = {}
    try:
        bonded = atoms.bonds.indices if len(atoms.bonds) else None
    except (mda.exceptions.NoDataError, AttributeError):
        bonded = None
    if bonded is not None:
        a, b = resix[bonded[:, 0]], resix[bonded[:, 1]]
        cross = (a != b) & eligible[a] & eligible[b]
        for x, y in zip(a[cross].tolist(), b[cross].tolist()):
            stated.setdefault(x, set()).add(y)
            stated.setdefault(y, set()).add(x)

    pool = None
    if _has_coords(atoms):
        from MDAnalysis.lib.distances import capped_distance

        table = Chem.GetPeriodicTable()
        candidates = atoms[eligible[resix]]
        numbers = np.array([_atomic_num_for(a) for a in candidates])
        keep = np.isin(numbers, list(_COVALENT_ELEMENTS))
        pool = candidates[keep]
        radii = np.array([table.GetRcovalent(int(n)) for n in numbers[keep]])
        pool_resix = pool.resindices
        box = universe.dimensions
        if box is None or not np.all(box[:3] > 0):
            box = None

    queue = [int(start)]
    while queue:
        r = queue.pop()
        near = set(stated.get(r, ()))
        if pool is not None and len(pool):
            mine = np.flatnonzero(pool_resix == r)
            if len(mine):
                pairs = capped_distance(
                    pool.positions[mine],
                    pool.positions,
                    2 * radii.max() + _BOND_TOLERANCE,
                    box=box,
                    method="bruteforce",
                    return_distances=True,
                )
                for (i, j), d in zip(*pairs):
                    if (
                        pool_resix[j] != r
                        and d < radii[mine[i]] + radii[j] + _BOND_TOLERANCE
                    ):
                        near.add(int(pool_resix[j]))
        for other in near - found:
            found.add(other)
            queue.append(other)
    return sorted(found)


def _peptide_order(universe, residues: list):
    """(`residues` in order from N- to C-terminus, whether the chain closes on
    itself), or None when they are not one peptide.

    They are one when the peptide bonds among them, each residue's C to the
    next one's N, run through every residue once: a single path, or a single
    ring (cyclosporin). Other bonds between them, a disulfide or a side-chain
    bridge, do not matter.
    """
    from MDAnalysis.lib.distances import capped_distance

    members = set(residues)
    atoms = universe.atoms[np.isin(universe.atoms.resindices, residues)]
    names = atoms.names.astype(str)
    c, n = atoms[names == "C"], atoms[names == "N"]
    following: dict = {}

    def link(a, b) -> bool:
        if a == b or a not in members or b not in members:
            return True
        if following.get(a, b) != b:
            return False
        following[a] = b
        return True

    try:
        for bond in atoms.bonds:
            x, y = bond.atoms
            for own, other in ((x, y), (y, x)):
                if own.name == "C" and other.name == "N":
                    if not link(int(own.resindex), int(other.resindex)):
                        return None
    except (mda.exceptions.NoDataError, AttributeError):
        pass
    if _has_coords(atoms) and len(c) and len(n):
        box = universe.dimensions
        if box is None or not np.all(box[:3] > 0):
            box = None
        pairs = capped_distance(
            c.positions,
            n.positions,
            _LINK_CUTOFF,
            box=box,
            method="bruteforce",
            return_distances=False,
        )
        for i, j in pairs:
            if not link(int(c[i].resindex), int(n[j].resindex)):
                return None

    preceded = set(following.values())
    if len(preceded) != len(following):
        return None
    starts = [r for r in residues if r not in preceded]
    cyclic = not starts and len(following) == len(residues)
    if cyclic:
        starts = [min(residues)]
    if len(starts) != 1:
        return None
    order = [starts[0]]
    while order[-1] in following and following[order[-1]] != order[0]:
        order.append(following[order[-1]])
    if len(order) != len(residues):
        return None
    return order, cyclic


def _spanning_molecule(universe, resname: str) -> Optional[_Spanning]:
    """The molecule of several residues that the first `resname` residue is
    part of, or None when that residue is a molecule by itself.

    A ligand used to be one residue, whatever it was bonded to. A
    tetrapeptide written DAL-SAR-DAL-MLE came back as three ligands, the
    D-alanine an aminoaldehyde C[C@@H](C=O)N with its open valences filled
    and its second copy dropped; POPC written the Lipid21 way as a palmitic
    aldehyde, an oleic aldehyde and a glycerophosphocholine (review F5).

    The other residues walked into are those that are not background by
    name and not joined into a polymer chain, so a ligand on a cysteine's
    sulfur is still one residue here.
    """
    sel, in_chain = _first_copy(universe, resname)
    if in_chain:
        return None
    resnames = universe.residues.resnames.astype(str)
    eligible = np.array([not _is_skipped_residue(n) for n in resnames])
    if not eligible.any():
        return None
    names = {n for n in np.unique(resnames[eligible])}
    for r in _polymer_residues(universe, names):
        eligible[r] = False
    start = int(sel.resindices[0])
    eligible[start] = True
    residues = _bonded_residues(universe, start, eligible)
    if len(residues) < 2:
        return None
    ordered = _peptide_order(universe, residues)
    if ordered is None:
        return _Spanning(residues, False, False)
    return _Spanning(ordered[0], True, ordered[1])


def _peptide_summary(universe, spanning: _Spanning) -> dict:
    """A peptide ligand as mol_id reports it: no SMILES, a sequence.

    One letter per residue, X for any that is not a standard amino acid
    (Travis, 2026-11-12: DAL, SAR and MLE have no one-letter code, and X is
    the code for a residue that has none). `resnames` keeps what each X
    stands for, the caps included; `sequence` leaves the caps out.
    """
    residues = [universe.residues[r] for r in spanning.residues]
    resnames = [str(r.resname) for r in residues]
    body = list(resnames)
    if len(body) > 1 and body[0].upper() in _PEPTIDE_CAPS:
        body = body[1:]
    if len(body) > 1 and body[-1].upper() in _PEPTIDE_CAPS:
        body = body[:-1]
    return {
        "sequence": "".join(_ONE_LETTER.get(n.upper(), "X") for n in body),
        "sequence_type": "protein",
        "resnames": resnames,
        "resids": [int(r.resid) for r in residues],
        "cyclic": spanning.cyclic,
        "resname": resnames[0],
    }


def _residue_list(universe, residues: list) -> str:
    """`DAL 1, SAR 2 and MLE 4` for a note; the first five and a count past
    that."""
    labels = [
        f"{universe.residues[r].resname} {universe.residues[r].resid}"
        for r in residues
    ]
    if len(labels) > 5:
        return ", ".join(labels[:5]) + f" and {len(labels) - 5} more"
    return ", ".join(labels[:-1]) + " and " + labels[-1]


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
    peptides: Optional[list] = None,
) -> list:
    """
    Universal pipeline: load any MDAnalysis-supported file (PDB, GRO, GROMACS
    .top/.tpr, PSF, mmCIF, Amber prmtop, ...), identify the ligand residue(s),
    and return a list of canonical-SMILES summaries (one per distinct ligand
    residue), each tagged with its `resname`.

    A ligand made of several residues bonded to each other is one molecule,
    not one per residue name (see _spanning_molecule). If it is a peptide it
    is a ligand by sequence, not by SMILES: its summary (_peptide_summary) is
    added to `peptides` and is not in the list returned. Anything else of
    several residues -- a glycan, a lipid written as head and tails -- is
    refused for now and said in `notes` (Travis, 2026-11-12).

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
    seen: set = set()
    for rn in target_resnames:
        spanning = _spanning_molecule(universe, rn)
        if spanning is not None:
            key = frozenset(spanning.residues)
            if key in seen:
                continue
            seen.add(key)
            where = _residue_list(universe, spanning.residues)
            if not spanning.peptide:
                _note(
                    notes,
                    f"Residues {where} are bonded into one molecule that is "
                    f"not a peptide (a glycan, or a lipid written as several "
                    f"residues, for one). A ligand of several residues is "
                    f"not recorded unless it is a peptide.",
                )
                continue
            peptide = _peptide_summary(universe, spanning)
            if peptides is not None:
                peptides.append(peptide)
            _note(
                notes,
                f"Residues {where} are one peptide of "
                f"{len(peptide['sequence'])} residues, bonded to no other "
                f"chain: a peptide ligand, read by its sequence "
                f"({peptide['sequence']}; X is a residue with no one-letter "
                f"code), not residue by residue.",
            )
            continue
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
    peptides: list = []
    if args.command == "smiles-from-structure":
        results = structure_to_smiles(
            args.path,
            fmt=args.fmt,
            resname=args.resname,
            notes=notes,
            peptides=peptides,
        )
        if args.verbose:
            # Small molecules only: mdr-process's ligand_resnames reads this
            # list as one shape. A peptide ligand reaches it through `both`.
            print(json.dumps(results, indent=2))
        else:
            lines = sorted([r["smiles"] for r in results])
            lines += [f"peptide {p['sequence']}" for p in peptides]
            print("\n".join(lines))

    elif args.command == "name-from-smiles":
        result = smiles_to_name(args.smiles)
        if args.verbose:
            print(json.dumps(result, indent=2))
        else:
            print(result["best_name"])

    elif args.command == "both":
        struct_results = structure_to_smiles(
            args.path,
            fmt=args.fmt,
            resname=args.resname,
            notes=notes,
            peptides=peptides,
        )
        # A small molecule is {structure, name}; a peptide ligand is
        # {peptide} and has neither. mdr-process reads both shapes.
        combined = [
            {"structure": sr, "name": smiles_to_name(sr["smiles"])}
            for sr in struct_results
        ] + [{"peptide": p} for p in peptides]
        if args.outfile:
            with open(args.outfile, "wt") as fh:
                json.dump(combined, fh, indent=2)
        elif args.verbose:
            print(json.dumps(combined, indent=2))
        else:
            for entry in combined:
                if "peptide" in entry:
                    p = entry["peptide"]
                    print(f"{p['resname']}: peptide {p['sequence']}")
                    continue
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
