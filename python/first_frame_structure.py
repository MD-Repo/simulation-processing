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

A CHARMM-GUI "Chamber" .parm7 is AMBER's format holding a CHARMM force field,
and opens with %FLAG CTITLE where AMBER's own has %FLAG TITLE. MDAnalysis
refuses it outright; ParmEd reads it, and MDAnalysis builds the Universe from
the ParmEd structure. zenodo:6487373 is eleven such systems, each a .parm7 and
its .nc runs with no coordinate file.

Moved here from mdrepo-agent's internal/stage/firstframe.go with the rest of
the file-producing steps (mdrepo-agent#98).

Output: the PDB at --out, and one JSON object on stdout:

  {"atoms": N, "elements": null | {"from": PSF, "repartitioned": N, "ions": N},
   "chamber": bool}

"chamber" is true when the topology was a Chamber .parm7, read through ParmEd.

Exit 0 when written; 1 with the reason as the last line of stderr when the
files cannot be read or the written PDB does not hold the topology's atoms;
2 when MDAnalysis is not installed.
"""

import argparse
import json
import os
import sys
from typing import NamedTuple

import psf_elements
from universe import is_chamber, open_universe


class Args(NamedTuple):
    topology: str
    trajectory: str
    out: str


class Report(NamedTuple):
    atoms: int
    elements: dict | None
    chamber: bool


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
def write(topology: str, trajectory: str, out: str) -> Report:
    """Write the topology's atoms at the trajectory's first frame as a PDB."""

    chamber = is_chamber(topology)
    u = open_universe(topology, trajectory)
    u.trajectory[0]
    u.atoms.write(out)

    atoms = psf_elements.pdb_atom_count(out)
    if atoms != len(u.atoms):
        raise ValueError(
            f"the structure written from the first frame holds {atoms} atoms "
            f"and the topology {len(u.atoms)}"
        )
    elements = None
    if psf_elements.is_psf(topology):
        elements = psf_elements.fill_elements(out, topology).as_dict()
    return Report(atoms, elements, chamber)


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
        report = write(args.topology, args.trajectory, args.out)
    except Exception as e:
        if not existed and os.path.exists(args.out):
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
