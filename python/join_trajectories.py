#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Join the consecutive segments of one run into one trajectory file

MDRepo counts a simulation's replicates as its trajectory files, one per
file. A release that saved one run in consecutive pieces (md.part0001.xtc,
md.part0002.xtc) is deposited as MDRepo reads it only after the pieces are
joined. A segment that begins with the frame the one before it ended on,
which a run continued from a checkpoint writes twice, has that frame left
out. The frame is told by its coordinates, which works in any format; a
matching time stamp alone is not enough, since a release whose stamps are
all placeholders stamps every frame alike. A segment whose stamps begin
before the previous one ends is refused: the segments are not in order, or
not one run. A DCD stamps nothing, so its order is taken as given.

Whole frames are copied byte for byte; a DCD's header is written once with
the joined frame count. No coordinate is decoded, and none changes.

Moved here from mdrepo-agent's internal/traj Join with the rest of the
file-producing steps (mdrepo-agent#98).

Output: the joined file at --out, and one JSON object on stdout:

  {"format": "xtc", "frames": N, "repeated": N}

Exit 0 when joined; 1 with the reason as the last line of stderr when a
segment cannot be walked, the segments are not one format or one layout, or
are not in order.
"""

import argparse
import json
import os
import sys
from typing import NamedTuple

import trajframes as tf


class Args(NamedTuple):
    trajectories: list[str]
    out: str


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Join consecutive segments of one run into one file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--trajectory", required=True, action="append", metavar="FILE",
        help="A segment, in order; give two or more",
    )  # fmt: skip
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="FILE",
        help="The joined trajectory to write",
    )
    a = parser.parse_args()
    if len(a.trajectory) < 2:
        parser.error(f"joining {len(a.trajectory)} files is not a join")
    return Args(a.trajectory, a.out)


# --------------------------------------------------
def join(paths: list[str], out: str) -> dict:
    runs, firsts, lasts = [], [], []
    for p in paths:
        run = tf.read_run(p)
        if runs:
            tf.joinable(runs[0], run)
        with open(p, "rb") as fh:
            firsts.append(run.digest(fh, 0))
            lasts.append(run.digest(fh, len(run.frames) - 1))
        runs.append(run)

    pieces, frames, repeated = [], 0, 0
    for i, run in enumerate(runs):
        first = 0
        if i > 0 and firsts[i] == lasts[i - 1]:
            first, repeated = 1, repeated + 1
        elif i > 0 and run.times is not None:
            prev = runs[i - 1].times
            assert prev is not None
            last, t = prev[-1], run.times[0]
            if t < last and not tf.same_time(t, last):
                raise tf.BadTrajectory(
                    f"{run.name} begins at {t:g} ps, before {runs[i - 1].name} ends at {last:g} ps, so the two "
                    "are not consecutive segments of one run in this order"
                )
        if first == len(run.frames):
            continue
        a, b = run.byte_range(first, len(run.frames))
        pieces.append((paths[i], a, b))
        frames += len(run.frames) - first
    tf.write_piece(out, runs[0], frames, pieces)
    return {"format": runs[0].format, "frames": frames, "repeated": repeated}


# --------------------------------------------------
def main() -> None:
    args = get_args()
    existed = tf.existing([args.out])
    try:
        report = join(args.trajectories, args.out)
    except Exception as e:
        tf.remove_fresh([args.out], existed)
        sys.stderr.write(
            f"joining the segments of {os.path.basename(args.out)}: {' '.join(str(e).split())}\n"
        )
        sys.exit(1)
    print(json.dumps(report))


# --------------------------------------------------
if __name__ == "__main__":
    main()
