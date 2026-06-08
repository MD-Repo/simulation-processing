#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2025-09-16
Purpose: Sample trajectory
"""

import argparse
import MDAnalysis as mda
from typing import NamedTuple

END_SAMPLE_FROM = 1000
TOTAL_FRAMES = 100


class Args(NamedTuple):
    """Command-line arguments"""

    trajectory: str
    structure: str
    outfile: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Sample trajectory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--trajectory",
        help="Trajectory file",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--structure",
        help="Structure file",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-o", "--outfile", help="Output file", metavar="FILE", required=True
    )

    args = parser.parse_args()

    return Args(
        trajectory=args.trajectory,
        structure=args.structure,
        outfile=args.outfile,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    trajectory = mda.Universe(args.structure, args.trajectory)
    num_frames = len(trajectory.trajectory)
    end_frame = min(num_frames, END_SAMPLE_FROM)
    sample_rate = end_frame // TOTAL_FRAMES if end_frame > TOTAL_FRAMES else 1
    sampled_frames = trajectory.trajectory[:end_frame][::sample_rate]
    trajectory.atoms.write(args.outfile, frames=sampled_frames)

    print(f"Wrote {num_frames} frames to '{args.outfile}'")


# --------------------------------------------------
if __name__ == "__main__":
    main()
