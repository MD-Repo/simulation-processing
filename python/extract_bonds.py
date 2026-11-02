#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-11-02
Purpose: The bonds of full.pdb's atoms, for the standard .psf mdr-process
         writes (graph only: atoms, elements, residues, chains, bonds)

The bonds come from the best topology the upload has, mapped onto full.pdb,
or are guessed from full.pdb's distances when it has none. Output: one line
per bond, two 1-based atom serials in full.pdb's order, and a JSON summary
saying where they came from.

Sources, best first. A source that cannot be read or does not map falls
through to the next one, with a warning, never to failure:

  1. .tpr (MDAnalysis). An uploaded .tpr beats the declared .top: the .top
     usually includes a force-field folder that was not uploaded
     (charmm36m.ff/forcefield.itp in all five sampled 2026-11-02).
  2. .psf, read here. MDAnalysis refuses BIOVIA's "!NBONDS" header and
     parmed reads such a file with no bonds at all, silently.
  3. AMBER prmtop/.parm7, and MISATO's production.top, which is a prmtop
     (recognised by its "%VERSION" first line, not by its extension).
  4. A GROMACS .top that carries everything it needs (ParmEd-written ones do).
  5. Guessed from full.pdb's distances.

full.pdb is often not the topology's atoms: the trajectory may be stripped
of water (and ions), and MISATO keeps heavy atoms only, in another order
within each residue. So residues are aligned in order, dropping topology
residues with no counterpart, and atoms are matched by name within each
residue. Every full.pdb atom must map, or that source is refused. Bonds are
kept between two mapped atoms; H-H (a rigid water's third constraint) is
not a bond.

The guess: two atoms are bonded when closer than 0.55 x the sum of their
van der Waals radii (MDAnalysis' rule), with S and Br at 2.0 A between two
heavy atoms (the defaults miss disulfides at 1.98-2.12 A, Met S-C at 1.94 A
and C-Br at 1.95-1.98 A). Never H-H, never an ion. Checked 2026-11-02 on 36
systems against 420,301 bonds read from their topologies: none missed, none
added. When a topology is used, the guess still runs as a cross-check, and
a disagreement is a warning.
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

warnings.filterwarnings("ignore")

Bond = Tuple[int, int]

# van der Waals radii (A). MDAnalysis' table, plus the ions it lacks.
VDW = {
    "H": 1.1,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.8,
    "S": 1.8,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.9,
    "B": 1.92,
    "SI": 2.1,
    "NA": 2.27,
    "K": 2.75,
    "MG": 1.73,
    "CA": 2.31,
    "ZN": 1.39,
    "FE": 1.94,
    "MN": 1.97,
    "CU": 1.4,
    "CO": 1.92,
    "NI": 1.63,
    "LI": 1.82,
    "CD": 1.58,
    "HG": 1.55,
    "AL": 1.84,
    "RB": 3.03,
    "CS": 3.43,
    "BA": 2.68,
    "SR": 2.49,
}
# Between two heavy atoms only; see the module docstring
VDW_HEAVY = {"S": 2.0, "BR": 2.0}
FUDGE = 0.55
MIN_BOND = 0.1
# A single-atom residue of one of these is an ion: bonded to nothing
IONS = {
    "NA",
    "K",
    "CL",
    "MG",
    "CA",
    "ZN",
    "LI",
    "RB",
    "CS",
    "BA",
    "SR",
    "BR",
    "I",
    "F",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "AL",
}
# Mapped bonds longer than this (minimum image) are reported as a warning
LONG_BOND = 2.2


class Args(NamedTuple):
    """Command-line arguments"""

    pdb: str
    topology: List[str]
    out: str
    summary: str


@dataclass
class PdbAtom:
    """One ATOM/HETATM record"""

    name: str
    resname: str
    element: str
    xyz: Tuple[float, float, float]


@dataclass
class Residue:
    """A residue: its name and (atom index, atom name) pairs"""

    name: str
    atoms: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class Topology:
    """What a source gives: residues in order and bonds by atom index"""

    num_atoms: int
    residues: List[Residue]
    bonds: List[Bond]


@dataclass
class Result:
    """The bonds of full.pdb and where they came from"""

    source: str
    source_file: Optional[str]
    bonds: List[Bond]
    num_atoms: int
    dropped: Dict[str, int] = field(default_factory=dict)
    guess_missing: Optional[int] = None
    guess_extra: Optional[int] = None
    long_bonds: int = 0
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="The bonds of full.pdb's atoms, for the standard .psf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p", "--pdb", help="full.pdb", metavar="FILE", required=True
    )

    parser.add_argument(
        "-t",
        "--topology",
        help="Uploaded files that may carry bonds (topology, .tpr, ...); "
        "others are ignored",
        metavar="FILE",
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "-o",
        "--out",
        help="Bonds, two 1-based atom serials per line",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--summary",
        help="JSON summary",
        metavar="FILE",
        required=True,
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pdb):
        parser.error(f'No such file "{args.pdb}"')

    return Args(
        pdb=args.pdb,
        topology=args.topology,
        out=args.out,
        summary=args.summary,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    try:
        result = extract(args.pdb, args.topology)
    except Exception as err:
        sys.exit(f"Cannot get bonds for {os.path.basename(args.pdb)}: {err}")

    with open(args.out, "wt") as fh:
        for i, j in result.bonds:
            print(i + 1, j + 1, sep="\t", file=fh)

    with open(args.summary, "wt") as fh:
        json.dump(summary(result), fh, indent=2)

    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    print(
        f"{len(result.bonds)} bonds for {result.num_atoms} atoms "
        f"from {result.source_file or result.source}"
    )


# --------------------------------------------------
def summary(result: Result) -> dict:
    """The JSON summary"""

    return {
        "source": result.source,
        "source_file": result.source_file,
        "num_atoms": result.num_atoms,
        "num_bonds": len(result.bonds),
        "dropped_topology_residues": result.dropped,
        "guess_missing": result.guess_missing,
        "guess_extra": result.guess_extra,
        "long_bonds": result.long_bonds,
        "warnings": result.warnings,
    }


# --------------------------------------------------
def extract(pdb_path: str, candidates: List[str]) -> Result:
    """Bonds for full.pdb from the best source that maps"""

    atoms, residues, box = read_pdb(pdb_path)
    if not atoms:
        raise ValueError("no ATOM/HETATM records")

    guessed = guess_bonds(atoms, residues, box)
    notes: List[str] = []

    for kind, path in rank_sources(candidates):
        name = os.path.basename(path)
        try:
            top = read_topology(kind, path)
            mapping = map_atoms(residues, top.residues)
        except Exception as err:
            notes.append(f"{name} not used: {err}")
            continue

        bonds = map_bonds(top.bonds, mapping, atoms)
        used = set(mapping.values())
        dropped = Counter(
            r.name
            for r in top.residues
            if not any(i in used for i, _ in r.atoms)
        )
        result = Result(
            source=kind,
            source_file=name,
            bonds=sorted(bonds),
            num_atoms=len(atoms),
            dropped=dict(dropped.most_common()),
            guess_missing=len(bonds - guessed),
            guess_extra=len(guessed - bonds),
            long_bonds=count_long(bonds, atoms, box),
            warnings=notes,
        )
        disagree = result.guess_missing + result.guess_extra
        if disagree > max(3, len(bonds) // 1000):
            result.warnings.append(
                f"{name}'s bonds and the distance guess disagree on "
                f"{disagree} ({result.guess_missing} not guessed, "
                f"{result.guess_extra} guessed only)"
            )
        if result.long_bonds:
            result.warnings.append(
                f"{result.long_bonds} bonds from {name} are longer than "
                f"{LONG_BOND} A in {os.path.basename(pdb_path)}"
            )
        return result

    notes.append("no topology with bonds; bonds guessed from distances")
    return Result(
        source="guessed",
        source_file=None,
        bonds=sorted(guessed),
        num_atoms=len(atoms),
        warnings=notes,
    )


# --------------------------------------------------
def rank_sources(paths: List[str]) -> List[Tuple[str, str]]:
    """The files that can carry bonds, best first, as (kind, path)"""

    rank = {"tpr": 0, "psf": 1, "prmtop": 2, "top": 3}
    found = []
    for path in paths:
        kind = source_kind(path)
        if kind and os.path.isfile(path):
            found.append((rank[kind], kind, path))
    return [(kind, path) for _, kind, path in sorted(found)]


# --------------------------------------------------
def source_kind(path: str) -> Optional[str]:
    """tpr, psf, prmtop (incl. a prmtop named .top), top, or None"""

    ext = os.path.splitext(path)[1].lower()
    if ext == ".tpr":
        return "tpr"
    if ext == ".psf":
        return "psf"
    if ext in (".prmtop", ".parm7"):
        return "prmtop"
    if ext == ".top":
        try:
            with open(path, "rb") as fh:
                head = fh.read(16)
        except OSError:
            return None
        return "prmtop" if head.startswith(b"%VERSION") else "top"
    return None


# --------------------------------------------------
def read_pdb(
    path: str,
) -> Tuple[List[PdbAtom], List[Residue], Optional[np.ndarray]]:
    """Atoms, residues (a change of chain, number or name starts one) and
    the CRYST1 box, if any"""

    atoms: List[PdbAtom] = []
    residues: List[Residue] = []
    box = None
    prev = None
    with open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("CRYST1") and box is None:
                try:
                    box = np.array(
                        [
                            float(line[6:15]),
                            float(line[15:24]),
                            float(line[24:33]),
                            float(line[33:40]),
                            float(line[40:47]),
                            float(line[47:54]),
                        ],
                        dtype=np.float32,
                    )
                except ValueError:
                    box = None
            elif line.startswith("ENDMDL"):
                break
            elif line.startswith(("ATOM", "HETATM")):
                name = line[12:16].strip()
                # cpptraj right-aligns the residue name in columns 17-20,
                # so a four-letter one (TIP3) starts a column early
                resname = line[16:21].strip()
                key = (line[21], line[22:27], resname)
                if key != prev:
                    residues.append(Residue(resname))
                    prev = key
                residues[-1].atoms.append((len(atoms), name))
                atoms.append(
                    PdbAtom(
                        name=name,
                        resname=resname,
                        element=element_of(line[76:78], name),
                        xyz=(
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ),
                    )
                )
    # A box of 1 A cubed is CRYST1's "no box"
    if box is not None and (box[:3] <= 1.0).any():
        box = None
    return atoms, residues, box


# --------------------------------------------------
def element_of(column: str, name: str) -> str:
    """The element column, upper case, or a guess from the atom name"""

    col = column.strip().upper()
    if col and col.isalpha():
        return col
    letters = name.lstrip("0123456789").upper()
    if letters[:2] in ("CL", "BR"):
        return letters[:2]
    return letters[:1]


# --------------------------------------------------
def read_topology(kind: str, path: str) -> Topology:
    """Residues and bonds from one source"""

    if kind == "psf":
        return read_psf(path)

    import MDAnalysis as mda

    if kind == "tpr":
        u = mda.Universe(path, topology_format="TPR")
    elif kind == "prmtop":
        u = mda.Universe(path, topology_format="PRMTOP")
    else:
        u = mda.Universe(path, topology_format="ITP", infer_system=True)

    residues = [
        Residue(r.resname, [(a.index, a.name) for a in r.atoms])
        for r in u.residues
    ]
    try:
        bonds = [(int(i), int(j)) for i, j in u.bonds.indices]
    except Exception:
        bonds = []
    if not bonds:
        raise ValueError("it holds no bonds")
    return Topology(len(u.atoms), residues, bonds)


# --------------------------------------------------
def read_psf(path: str) -> Topology:
    """A .psf's atoms and bonds. Takes "!NBOND" and BIOVIA's "!NBONDS"."""

    with open(path, "rt", errors="replace") as fh:
        lines = iter(fh.read().splitlines())

    def count(line: str, section: str) -> Optional[int]:
        f = line.split()
        if len(f) > 1 and f[1].startswith(section):
            return int(f[0])
        return None

    natom = next(
        (
            n
            for n in map(lambda l: count(l, "!NATOM"), lines)
            if n is not None
        ),
        None,
    )
    if natom is None:
        raise ValueError("no !NATOM section")

    residues: List[Residue] = []
    prev = None
    for i in range(natom):
        f = next(lines).split()
        if len(f) < 5:
            raise ValueError(f"atom {i + 1} has {len(f)} fields")
        key = (f[1], f[2], f[3])
        if key != prev:
            residues.append(Residue(f[3]))
            prev = key
        residues[-1].atoms.append((i, f[4]))

    nbond = next(
        (
            n
            for n in map(lambda l: count(l, "!NBOND"), lines)
            if n is not None
        ),
        0,
    )
    ids: List[int] = []
    for line in lines:
        # The next section's header ends a short bond list
        if len(ids) >= 2 * nbond or "!" in line:
            break
        try:
            ids.extend(int(x) - 1 for x in line.split())
        except ValueError:
            raise ValueError(
                f"a bond line is not atom numbers: {line.strip()!r}"
            )
    if len(ids) < 2 * nbond:
        raise ValueError(f"it states {nbond} bonds and lists {len(ids) // 2}")
    if nbond == 0:
        raise ValueError("it holds no bonds")
    bonds = list(zip(ids[0 : 2 * nbond : 2], ids[1 : 2 * nbond : 2]))
    if any(not (0 <= i < natom and 0 <= j < natom) for i, j in bonds):
        raise ValueError("a bond names an atom it does not have")
    return Topology(natom, residues, bonds)


# --------------------------------------------------
def atom_key(name: str) -> str:
    """An atom name as matched: 4 characters, with the old PDB style gmx
    trjconv writes for 4-character hydrogens (1HD1) turned back (HD11)"""

    name = name.strip()
    if len(name) == 4 and name[0].isdigit():
        name = name[1:] + name[0]
    return name[:4]


# --------------------------------------------------
def residue_key(name: str) -> str:
    """A residue name as compared: letters and digits, 3 characters"""

    return re.sub(r"[^A-Z0-9]", "", name.upper())[:3]


# --------------------------------------------------
def map_atoms(
    pdb_residues: List[Residue], top_residues: List[Residue]
) -> Dict[int, int]:
    """full.pdb atom index -> topology atom index. Residues are taken in
    order; a topology residue with no counterpart in the pdb is skipped
    (water, ions, ...). A pdb residue matches when every one of its atom
    names is in the topology residue, and the residue names agree (a
    single-atom residue) or it has more than one atom (HIS vs HIE)."""

    mapping: Dict[int, int] = {}
    t = 0
    for p, pres in enumerate(pdb_residues):
        want = [(i, atom_key(n)) for i, n in pres.atoms]
        while True:
            if t >= len(top_residues):
                raise ValueError(
                    f"residue {p + 1} of the pdb ({pres.name}, "
                    f"{len(want)} atoms) has no match in the topology"
                )
            tres = top_residues[t]
            t += 1
            names: Dict[str, int] = {}
            for i, n in tres.atoms:
                names.setdefault(atom_key(n), i)
            same = residue_key(tres.name) == residue_key(pres.name)
            if (same or len(want) > 1) and all(n in names for _, n in want):
                for i, n in want:
                    mapping[i] = names[n]
                break
    return mapping


# --------------------------------------------------
def map_bonds(
    top_bonds: List[Bond], mapping: Dict[int, int], atoms: List[PdbAtom]
) -> Set[Bond]:
    """The topology's bonds between two mapped atoms, as pdb indices; never
    H-H"""

    inv = {t: p for p, t in mapping.items()}
    bonds: Set[Bond] = set()
    for i, j in top_bonds:
        if i in inv and j in inv:
            a, b = sorted((inv[i], inv[j]))
            if a == b or (
                atoms[a].element == "H" and atoms[b].element == "H"
            ):
                continue
            bonds.add((a, b))
    return bonds


# --------------------------------------------------
def guess_bonds(
    atoms: List[PdbAtom], residues: List[Residue], box: Optional[np.ndarray]
) -> Set[Bond]:
    """Bonds from distances: closer than FUDGE x the sum of the two radii"""

    from MDAnalysis.lib.distances import capped_distance

    ion = np.zeros(len(atoms), dtype=bool)
    for res in residues:
        if len(res.atoms) == 1:
            i = res.atoms[0][0]
            if atoms[i].element in IONS:
                ion[i] = True

    elements = [a.element for a in atoms]
    radius = np.array([VDW.get(e, 1.7) for e in elements])
    heavy_radius = np.array(
        [VDW_HEAVY.get(e, r) for e, r in zip(elements, radius)]
    )
    is_h = np.array([e == "H" for e in elements])
    xyz = np.array([a.xyz for a in atoms], dtype=np.float32)

    cutoff = FUDGE * 2 * float(max(heavy_radius.max(), radius.max()))
    pairs, dist = capped_distance(
        xyz,
        xyz,
        max_cutoff=cutoff,
        min_cutoff=MIN_BOND,
        box=box,
        return_distances=True,
    )

    bonds: Set[Bond] = set()
    for (i, j), d in zip(pairs, dist):
        if i >= j or ion[i] or ion[j] or (is_h[i] and is_h[j]):
            continue
        if is_h[i] or is_h[j]:
            limit = FUDGE * (radius[i] + radius[j])
        else:
            limit = FUDGE * (heavy_radius[i] + heavy_radius[j])
        if d < limit:
            bonds.add((int(i), int(j)))
    return bonds


# --------------------------------------------------
def count_long(
    bonds: Set[Bond], atoms: List[PdbAtom], box: Optional[np.ndarray]
) -> int:
    """Bonds longer than LONG_BOND, minimum image"""

    if not bonds:
        return 0
    from MDAnalysis.lib.distances import calc_bonds

    xyz = np.array([a.xyz for a in atoms], dtype=np.float32)
    idx = np.array(sorted(bonds))
    lengths = calc_bonds(xyz[idx[:, 0]], xyz[idx[:, 1]], box=box)
    return int((lengths > LONG_BOND).sum())


# --------------------------------------------------
if __name__ == "__main__":
    main()
