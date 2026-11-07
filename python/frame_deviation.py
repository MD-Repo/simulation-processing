#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-11-07
Purpose: Measure how far a structure's coordinates are from a trajectory's
         first frame

When a structure is cut to the atoms a trajectory holds (subset_structure.py),
a match with the first frame atom for atom says the selection picked the
trajectory's atoms, not merely as many. Each difference is taken the short
way across the periodic box, since an atom wrapped to the other side is the
same atom. A miss is the caller's to disclose, not an error here: the first
saved frame need not be the coordinates the structure holds.

Moved here from mdrepo-agent's internal/stage/subset.go (mdrepo-agent#98):
everything that reads or writes a structure, topology or trajectory is a
command here.

Output: one JSON object on stdout:

  {"atoms": N, "deviation": A}     the largest distance, in Angstrom

Exit 0 when measured; 1 with the reason as the last line of stderr when the
files cannot be read together; 2 when MDAnalysis is not installed.
"""

import argparse
import json
import os
import sys
from typing import NamedTuple

from universe import open_universe


class Args(NamedTuple):
    structure: str
    trajectory: str


class Report(NamedTuple):
    atoms: int
    deviation: float


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Measure a structure against a trajectory's first frame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--structure", required=True, metavar="FILE", help="Structure"
    )
    parser.add_argument(
        "-f", "--trajectory", required=True, metavar="FILE", help="Trajectory"
    )
    a = parser.parse_args()
    return Args(a.structure, a.trajectory)


# --------------------------------------------------
def deviation(structure: str, trajectory: str) -> Report:
    """The largest distance between an atom of the structure and the same
    atom in the trajectory's first frame."""

    import numpy as np
    from MDAnalysis.lib.distances import minimize_vectors

    ref = open_universe(structure).atoms.positions.copy()
    u = open_universe(structure, trajectory)
    u.trajectory[0]
    d = u.atoms.positions - ref
    if u.dimensions is not None and all(u.dimensions[:3] > 0):
        d = minimize_vectors(d, u.dimensions)
    return Report(len(u.atoms), float(np.linalg.norm(d, axis=1).max()))


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        sys.stderr.write(f"MDAnalysis is not installed in {sys.executable}\n")
        sys.exit(2)

    try:
        report = deviation(args.structure, args.trajectory)
    except Exception as e:
        reason = " ".join(str(e).split())
        sys.stderr.write(
            f"reading {os.path.basename(args.structure)} with "
            f"{os.path.basename(args.trajectory)}: {reason}\n"
        )
        sys.exit(1)
    print(json.dumps(report._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
