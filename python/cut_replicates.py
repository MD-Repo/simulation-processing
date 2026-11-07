#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Cut a trajectory of concatenated replicates into one file per
         replicate, of equal frame count

MDRepo counts a simulation's replicates as its trajectory files, one per
file. A release that concatenated its replicates into one file is deposited
as MDRepo reads it only after the file is cut. The pieces hold the first
frames first and the last frames last; a file whose frames do not divide
evenly is refused, since the pieces are the release's replicates only when
they are the same length.

Whole frames are copied byte for byte. A DCD's header is repeated with its
frame count set to the frames that follow it; an XTC or a TRR has no header
beyond each frame's own, so its frames are simply regrouped. No coordinate
is decoded, and none changes.

Moved here from mdrepo-agent's internal/traj Cut with the rest of the
file-producing steps (mdrepo-agent#98).

Output: one file per --out, in order, and one JSON object on stdout:

  {"format": "dcd", "frames": N, "frames_each": N, "pieces": [paths]}

Exit 0 when cut; 1 with the reason as the last line of stderr when the file
cannot be walked, is not a DCD, XTC or TRR, or does not divide.
"""

import argparse
import json
import os
import sys
from typing import NamedTuple

import trajframes as tf


class Args(NamedTuple):
    trajectory: str
    outs: list[str]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Cut a trajectory into equal pieces, one file each",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--trajectory", required=True, metavar="FILE", help="Trajectory to cut"
    )
    parser.add_argument(
        "-o", "--out", required=True, action="append", metavar="FILE",
        help="A piece to write, in order; give one per replicate",
    )  # fmt: skip
    a = parser.parse_args()
    if len(a.out) < 2:
        # Exit 1, not argparse's 2: a bad value is this run's refusal, and the
        # contract keeps 2 for an environment that lacks something.
        sys.stderr.write(f"cutting into {len(a.out)} pieces is not a cut" + "\n")
        sys.exit(1)
    return Args(a.trajectory, a.out)


# --------------------------------------------------
def cut(path: str, outs: list[str]) -> dict:
    run = tf.read_run(path)
    n = len(outs)
    frames = len(run.frames)
    if frames % n:
        raise tf.BadTrajectory(
            f"{run.name} holds {frames} frames, which do not divide into {n} equal pieces"
        )
    per = frames // n
    for i, dest in enumerate(outs):
        a, b = run.byte_range(i * per, (i + 1) * per)
        tf.write_piece(dest, run, per, [(path, a, b)])
    return {"format": run.format, "frames": frames, "frames_each": per, "pieces": outs}


# --------------------------------------------------
def main() -> None:
    args = get_args()
    existed = tf.existing(args.outs)
    try:
        report = cut(args.trajectory, args.outs)
    except Exception as e:
        tf.remove_fresh(args.outs, existed)
        sys.stderr.write(
            f"cutting {os.path.basename(args.trajectory)}: {' '.join(str(e).split())}\n"
        )
        sys.exit(1)
    print(json.dumps(report))


# --------------------------------------------------
if __name__ == "__main__":
    main()
