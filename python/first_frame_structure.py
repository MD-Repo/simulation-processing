#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Write a structure file from a topology and the first frame of a
         trajectory, for a submission that ships no coordinates outside its
         trajectories

A CHARMM/NAMD release can be a .psf and its .dcd files and nothing else: the
topology holds the atoms and no coordinates, and the trajectories hold the
coordinates. MDRepo takes a structure file, so one is written from the two
with MDAnalysis, the way MDRepo's own processing reads them, and the topology
is deposited as released. 29 of the 69 records in the agent's discovery
corpus shipping a loose .psf and .dcd ship no coordinate file with them. A
GROMACS release can be the same in effect: a .tpr and .xtc of every atom
beside a water-stripped .pdb, as zenodo:17359970's six bilayer runs are;
MDAnalysis reads a .tpr for its atoms alone, so the same two files write it.

When the topology is a .psf, which carries no elements, the PDB's element
column is filled from the topology's masses and bonds (psf_elements.py)
rather than left for a reader to guess from the atoms' names.

Moved here from mdrepo-agent's internal/stage/firstframe.go with the rest of
the file-producing steps (mdrepo-agent#98).

Output: the PDB at --out, and one JSON object on stdout:

  {"atoms": N, "elements": null | {"from": PSF, "repartitioned": N, "ions": N}}

Exit 0 when written; 1 with the reason as the last line of stderr when the
files cannot be read or the written PDB does not hold the topology's atoms;
2 when MDAnalysis is not installed.
"""

import argparse
import json
import os
import sys
import warnings
from typing import NamedTuple

import psf_elements


class Args(NamedTuple):
    topology: str
    trajectory: str
    out: str


class Report(NamedTuple):
    atoms: int
    elements: dict | None


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Write a PDB from a topology and a trajectory's first frame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t", "--topology", required=True, metavar="FILE", help="Topology file"
    )
    parser.add_argument(
        "-f", "--trajectory", required=True, metavar="FILE", help="Trajectory file"
    )
    parser.add_argument(
        "-o", "--out", required=True, metavar="PDB", help="Structure to write"
    )
    a = parser.parse_args()
    return Args(a.topology, a.trajectory, a.out)


# --------------------------------------------------
def pdb_atom_count(path: str) -> int:
    """The ATOM and HETATM records of the first model."""

    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                n += 1
            elif line.startswith("ENDMDL"):
                break
    return n


# --------------------------------------------------
def write(topology: str, trajectory: str, out: str) -> Report:
    """Write the topology's atoms at the trajectory's first frame as a PDB."""

    warnings.filterwarnings("ignore")
    import MDAnalysis as mda

    u = mda.Universe(topology, trajectory)
    u.trajectory[0]
    u.atoms.write(out)

    atoms = pdb_atom_count(out)
    if atoms != len(u.atoms):
        raise ValueError(
            f"the structure written from the first frame holds {atoms} atoms "
            f"and the topology {len(u.atoms)}"
        )
    elements = None
    if psf_elements.is_psf(topology):
        elements = psf_elements.fill_elements(out, topology).as_dict()
    return Report(atoms, elements)


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        sys.stderr.write(f"MDAnalysis is not installed in {sys.executable}\n")
        sys.exit(2)

    try:
        report = write(args.topology, args.trajectory, args.out)
    except Exception as e:
        if os.path.exists(args.out):
            os.remove(args.out)
        # The reason is the last line of stderr, so it has to be one line.
        reason = " ".join(str(e).split())
        sys.stderr.write(
            f"writing the structure from {os.path.basename(args.topology)} and the "
            f"first frame of {os.path.basename(args.trajectory)}: {reason}\n"
        )
        sys.exit(1)
    print(json.dumps(report._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
