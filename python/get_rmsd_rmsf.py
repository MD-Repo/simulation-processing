#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-02-04
Purpose: Rock the Casbah
"""

import argparse
import json
import MDAnalysis as mda
import sys
from MDAnalysis.analysis import rms
from typing import List, NamedTuple

MAX_RMSD = 500
MAX_RMSF = 500


class Args(NamedTuple):
    """Command-line arguments"""

    structure: str
    trajectory: str
    out_file: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Rock the Casbah",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--structure",
        help="Structure file",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--trajectory",
        help="Trajectory file",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--out-file",
        help="Output JSON file",
        metavar="FILE",
        default="rmsf_rmsd.json",
    )

    args = parser.parse_args()

    return Args(
        structure=args.structure,
        trajectory=args.trajectory,
        out_file=args.out_file,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    universe = mda.Universe(args.structure, args.trajectory)
    traj = universe.select_atoms("name CA")
    if len(traj) == 0:
        traj = universe.select_atoms("all")
        print("[!] No CA atoms found; using all atoms for RMSD/RMSF (CG model)")

    # sample values to ~1000 points
    def sample(vals: List[float]) -> List[float]:
        num = len(vals)
        sample_num = num // 1000
        return vals[::sample_num] if num > 1000 else vals

    rmsd_values = sample(rms.RMSD(traj).run().results["rmsd"].T[2].tolist())
    rmsf_values = sample(rms.RMSF(traj).run().results["rmsf"].tolist())

    for type_, max_allowed, vals in [
        ("RMSD", MAX_RMSD, rmsd_values),
        ("RMSF", MAX_RMSF, rmsf_values),
    ]:
        if not vals:
            sys.exit(f"Failed to get {type_} values!")

        vals_max = max(vals)
        if vals_max > max_allowed:
            print(
                f"Trajectory {type_} val {vals_max} greater than {max_allowed}",
                file=sys.stderr,
            )
            return

    out_fh = open(args.out_file, "wt")
    json.dump({"rmsd": rmsd_values, "rmsf": rmsf_values}, out_fh)
    print(f"Done, see '{args.out_file}'")


# --------------------------------------------------
if __name__ == "__main__":
    main()
