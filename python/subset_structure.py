#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Cut a structure down to the atoms its trajectories hold, and write
         them as a PDB in their order

A release can hold trajectories of only part of its system: GROMACS can save
one group (compressed-x-grps = protein), and a release can strip the solvent
from its trajectories. With no structure of that part alone, the structure is
cut to the same atoms with an MDAnalysis selection and written as a PDB.

The release's topology, when given, plays one of two parts, told apart by its
atom count:

  * A topology of the whole system supplies the bonds among the selected
    atoms, which the caller hands to generate_topology.py, and a .tpr the
    coordinates too: the atoms are then read from it rather than from the
    structure, since it holds the coordinates the production run started
    from. A GROMACS .top reads only with the force-field files it includes,
    which a release seldom ships, so the bonds then come from the structure.

  * A topology of the saved atoms alone -- a stripped .psf beside the
    trajectories -- is the release's own account of those atoms. The atoms
    the selection picks are compared with it name by name and residue by
    residue, and the report says how many differ, so the caller can deposit
    it as released when none do rather than generate one.

The whole system selected from a file that is the release's own topology of
it (the .tpr itself, or a topology read beside the structure) is that
topology's atoms exactly, and the report says so. zenodo:17359970's runs are
each a .tpr with selection "all".

When the atoms were read from a .psf, or match the release's .psf of them and
the structure they came from had no elements, the PDB's element column is
filled from the .psf's masses and bonds (psf_elements.py).

Moved here from mdrepo-agent's internal/stage/subset.go with the rest of the
file-producing steps (mdrepo-agent#98).

Output: the PDB at --out, and one JSON object on stdout:

  {"atoms": N, "total": N,
   "bonds": null | [[i, j], ...]     0-based, in the PDB's order
   "released": bool,                 the topology is the release's own of these atoms
   "mismatches": N, "examples": [...],
   "from_topology": bool,            the atoms were read from the topology
   "topology_used": path | null,     the file taken as the topology: --topology,
                                     or the .tpr structure standing in for it
   "elements": null | {"from": PSF, "repartitioned": N, "ions": N}}

Exit 0 when written; 1 with the reason as the last line of stderr when a file
cannot be read, the selection holds no atoms, or the PDB does not hold the
selected atoms; 2 when MDAnalysis is not installed.
"""

import argparse
import json
import os
import sys
import warnings
from typing import NamedTuple, Optional

import psf_elements

# Topology formats MDAnalysis reads on their own, for the bonds and the atoms.
# A .top is not among them: it reads only with its includes.
SELF_CONTAINED = (".tpr", ".psf", ".prmtop", ".parm7")


class Args(NamedTuple):
    structure: str
    topology: Optional[str]
    selection: str
    out: str


class Report(NamedTuple):
    atoms: int
    total: int
    bonds: Optional[list]
    released: bool
    mismatches: int
    examples: list[str]
    from_topology: bool
    topology_used: Optional[str]
    elements: Optional[dict]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Write the atoms a selection picks from a structure as a PDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--structure",
        required=True,
        metavar="FILE",
        help="Structure holding the whole system's coordinates",
    )
    parser.add_argument(
        "-t",
        "--topology",
        metavar="FILE",
        help="The release's topology, of the whole system or of the saved atoms",
    )
    parser.add_argument(
        "-l", "--selection", required=True, metavar="SEL", help="MDAnalysis selection"
    )
    parser.add_argument(
        "-o", "--out", required=True, metavar="PDB", help="Structure to write"
    )
    a = parser.parse_args()
    return Args(a.structure, a.topology or None, a.selection, a.out)


# --------------------------------------------------
# --------------------------------------------------
def subset(structure: str, topology: Optional[str], selection: str, out: str) -> Report:
    """Write the atoms the selection picks as a PDB at out and report on them."""

    warnings.filterwarnings("ignore")
    import numpy as np
    import MDAnalysis as mda

    s, t = structure, topology or ""
    # A .tpr structure is its own topology, so the atoms' places in it are
    # known and it can be cut to them: with no topology named, or with one
    # MDAnalysis reads only with its includes (a .top beside the .tpr).
    if s.lower().endswith(".tpr") and not (t and t.lower().endswith(SELF_CONTAINED)):
        t = s
    top, whole, from_top = None, None, False
    if t and t != s and t.lower().endswith(SELF_CONTAINED):
        try:
            top = mda.Universe(t)
        except Exception as e:
            raise ValueError(
                f"the topology {os.path.basename(t)} could not be read: {e}"
            )
    if t.lower().endswith(".tpr") and (t == s or top is not None):
        u = top if top is not None else mda.Universe(t)
        whole, from_top = True, True
        if s != t:
            su = mda.Universe(s)
            if len(su.atoms) != len(u.atoms):
                whole, from_top = None, False
                u = su
    else:
        u = mda.Universe(s)
        if top is not None and len(top.atoms) == len(u.atoms):
            whole = True
            try:
                u = mda.Universe(t, s)
                from_top = True
            except Exception:
                pass

    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(
            f"the selection {selection!r} holds no atoms of {os.path.basename(s)}"
        )
    ag.write(out)

    bonds = None
    try:
        pos = {a: i for i, a in enumerate(ag.indices)}
        bonds = []
        for b in ag.bonds:
            i, j = b.indices
            if i in pos and j in pos:
                bonds.append(sorted([int(pos[i]), int(pos[j])]))
    except Exception:
        bonds = None

    released, mismatches, examples = False, 0, []
    if t and whole and (top is not None or t == s) and len(ag) == len(u.atoms):
        released = True
    if top is not None and not whole:
        ref = top.atoms
        released = True
        if len(ref) != len(ag):
            mismatches = abs(len(ref) - len(ag)) or 1
            examples = [
                f"the released topology holds {len(ref)} atoms and the selection {len(ag)}"
            ]
        else:
            d = np.nonzero((ag.names != ref.names) | (ag.resnames != ref.resnames))[0]
            mismatches = int(len(d))
            examples = [
                "atom %d: selected %s %s%d, released %s %s%d"
                % (
                    i + 1,
                    ag.names[i],
                    ag.resnames[i],
                    ag.resids[i],
                    ref.names[i],
                    ref.resnames[i],
                    ref.resids[i],
                )
                for i in d[:5]
            ]

    atoms = psf_elements.pdb_atom_count(out)
    if atoms != len(ag):
        raise ValueError(
            f"the subset PDB holds {atoms} atoms and the selection {len(ag)}"
        )

    elements = None
    if t and psf_elements.is_psf(t):
        if from_top:
            # The atoms are the .psf's atoms it names, in order.
            elements = psf_elements.fill_elements(out, t, ag.indices.tolist()).as_dict()
        elif released and mismatches == 0 and psf_elements.pdb_lacks_elements(out):
            # The atoms are the released .psf's, name for name: where the
            # structure they were read from had no elements, it gives them.
            elements = psf_elements.fill_elements(out, t).as_dict()

    return Report(
        atoms=atoms,
        total=len(u.atoms),
        bonds=bonds,
        released=released,
        mismatches=mismatches,
        examples=examples,
        from_topology=from_top,
        topology_used=t or None,
        elements=elements,
    )


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        sys.stderr.write(f"MDAnalysis is not installed in {sys.executable}\n")
        sys.exit(2)

    # Only what this run created is removed on failure, not a file that was
    # already there.
    existed = os.path.exists(args.out)
    try:
        report = subset(args.structure, args.topology, args.selection, args.out)
    except Exception as e:
        if not existed and os.path.exists(args.out):
            os.remove(args.out)
        # The reason is the last line of stderr, so it has to be one line.
        reason = " ".join(str(e).split())
        sys.stderr.write(
            f"selecting {args.selection!r} from {os.path.basename(args.structure)}: {reason}\n"
        )
        sys.exit(1)
    print(json.dumps(report._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
