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
def sample(vals: List[float]) -> List[float]:
    """Decimate to ~1000 points, for storage and plotting"""

    num = len(vals)
    sample_num = num // 1000
    return vals[::sample_num] if num > 1000 else vals


# --------------------------------------------------
def check_ceiling(type_: str, max_allowed: float, vals: List[float]) -> None:
    """Refuse a trajectory whose RMSD or RMSF is not physical

    The ceiling is not a scientific bound, it is a corruption detector: every
    real trip has been 10^5 to 10^7, and what it catches is a trajectory whose
    frames are structurally broken -- runs of all-zero frames that, fitted to
    a reference, land at coordinates that saturate the XTC integer encoding.

    The headroom is smaller than it looks, so measure before lowering the bar.
    Nothing aligns the trajectory before rms.RMSF, so global drift is in the
    number: the healthy replicates of ticket 2339's system (sims 98340, 98342,
    98343) store RMSF maxima of 38.65, 39.07 and 39.49 against the 500, which
    is a factor of 12, not the two orders of magnitude a small well-behaved
    system would give.

    `vals` must be the FULL vector, not a decimated one. sample() strides to
    ~1000 points, and taking the max after it meant the check only ever looked
    at one value in `num // 1000` -- so a narrow spike sitting off the stride
    was never looked at. RMSF decimates on any protein over 1000 residues and
    RMSD on any run over 1000 frames, which is a lot of the archive.

    Ticket 2339 is what prompted this, though it is not an example of it: 20
    corrupt frames, 45001-45020 of 80,000, pinned at the XTC limit. RMSF is
    per-CA, and 251 CA atoms are under the cutoff, so its vector was never
    decimated and it caught them at 588,039. RMSD did not, and decimation is
    not the reason -- its full-vector max is 24.4 against 24.0 decimated.
    rms.RMSD superimposes each frame on the reference, so a frame whose atoms
    all sit at one saturated coordinate centres to the origin and scores 18.2,
    indistinguishable from ordinary motion. RMSD is structurally blind to this
    shape of corruption whatever order the max is taken in. What the order
    buys is the spike RMSD *can* see, and the RMSF spike on a large protein.

    Exits non-zero rather than returning. Refusing to write is the right call
    -- a value this far out means the trajectory is garbage, not that the
    ceiling is wrong -- but returning made the script exit 0, so the caller
    could only infer the refusal from a missing output file and reported
    "Failed to create rmsd_rmsf.json" while this explanation went to a
    discarded stderr. mdr-process already surfaces stderr on a non-zero exit
    (process.rs, get_rmsd_rmsf), so this is what puts the reason into the
    ticket log and last_error.
    """

    if not vals:
        sys.exit(f"Failed to get {type_} values!")

    vals_max = max(vals)
    if vals_max > max_allowed:
        sys.exit(f"Trajectory {type_} val {vals_max} greater than {max_allowed}")


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    universe = mda.Universe(args.structure, args.trajectory)
    traj = universe.select_atoms("name CA")
    if len(traj) == 0:
        traj = universe.select_atoms("all")
        print("[!] No CA atoms found; using all atoms for RMSD/RMSF (CG model)")

    # Check the ceilings against every value, and decimate only on the way
    # into the JSON. See check_ceiling() for why that order matters.
    rmsd_full = rms.RMSD(traj).run().results["rmsd"].T[2].tolist()
    rmsf_full = rms.RMSF(traj).run().results["rmsf"].tolist()

    check_ceiling("RMSD", MAX_RMSD, rmsd_full)
    check_ceiling("RMSF", MAX_RMSF, rmsf_full)

    out_fh = open(args.out_file, "wt")
    json.dump({"rmsd": sample(rmsd_full), "rmsf": sample(rmsf_full)}, out_fh)
    print(f"Done, see '{args.out_file}'")


# --------------------------------------------------
if __name__ == "__main__":
    main()
