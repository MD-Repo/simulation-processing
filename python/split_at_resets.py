#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Split a trajectory into one file per run where its clock returns
         to the start, leaving out frames written twice

A released file is often several runs end to end, and three different
things look alike as "the clock went back": a run that ends and the next
that begins at the start again; a frame written twice, as the last frame of
one work unit is at the head of the next; and a run resumed from a
checkpoint before the frame it last wrote. Telling them apart is what lets
a file of concatenated replicates be cut at the right places and nowhere
else. zenodo:14013285 holds 20 such files: in each, 85 Folding@home clones
of exactly 1 us end to end, a repeated frame every 5 ns and a few dozen
short overlaps, 18,105 backward steps in all that one count could not
explain.

The file is cut where its time returns to the first frame's. Within a
segment, a frame stamped with a time already kept is left out. A backward
step to a time never kept is refused, as is a DCD, which stamps no frame
times. A file with no reset, repeated stamp or overlap has nothing to split
at: the command writes nothing and says so, exit 0, and the caller deposits
the file as it is.

Every frame kept is copied byte for byte. Each piece is named by
--out-template with {k} the piece's number from 1 and {n} how many there
are; the caller renames them as it likes.

Moved here from mdrepo-agent's internal/traj SplitAtResets with the rest of
the file-producing steps (mdrepo-agent#98).

Output: the pieces, and one JSON object on stdout:

  {"format": "xtc", "split": true, "pieces": [paths], "frames": [N, ...],
   "dropped": N, "resets": N, "repeated": N, "overlaps": N}
  {"format": "xtc", "split": false, "reason": "...", "frames": N,
   "first_ps": t, "last_ps": t}                     nothing to split at

Exit 0 when split or when there was nothing to split; 1 with the reason as
the last line of stderr when the file cannot be walked, stamps no times, or
steps back to a time never kept.
"""

import argparse
import bisect
import json
import os
import sys
from typing import NamedTuple

import trajframes as tf


class Args(NamedTuple):
    trajectory: str
    template: str


class Segment(NamedTuple):
    first: int  # inclusive frame indices
    last: int
    start_ps: float
    end_ps: float


class Restarts(NamedTuple):
    resets: int
    repeated: int
    overlaps: int
    segments: list[Segment]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Split a trajectory into one file per run at its time resets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--trajectory", required=True, metavar="FILE", help="Trajectory to split"
    )
    parser.add_argument(
        "-o", "--out-template", required=True, metavar="TEMPLATE",
        help="Path of each piece, with {k} its number from 1 and {n} the count: out/run_{k}of{n}.xtc",
    )  # fmt: skip
    a = parser.parse_args()
    if "{k}" not in a.out_template:
        parser.error("--out-template needs {k}, or every piece would be one file")
    return Args(a.trajectory, a.out_template)


# --------------------------------------------------
def classify_restarts(times: list[float]) -> Restarts:
    """Walk the frame times. A repeated stamp is judged before a reset, so
    two frames at the start are a repeat and not a one-frame segment."""

    resets = repeated = overlaps = 0
    segments: list[Segment] = []
    if not times:
        return Restarts(0, 0, 0, [])
    start = times[0]
    first = 0
    for i in range(1, len(times)):
        t, prev = times[i], times[i - 1]
        if t > prev and not tf.same_time(t, prev):
            continue
        if tf.same_time(t, prev):
            repeated += 1
        elif tf.same_time(t, start):
            resets += 1
            segments.append(Segment(first, i - 1, times[first], times[i - 1]))
            first = i
        else:
            overlaps += 1
    segments.append(Segment(first, len(times) - 1, times[first], times[-1]))
    return Restarts(resets, repeated, overlaps, segments)


# --------------------------------------------------
def split(path: str, template: str) -> dict:
    run = tf.read_run(path)
    times = run.times
    if times is None:
        raise tf.BadTrajectory(
            f"{run.name} is a {run.format}, which stamps no frame times to split at"
        )
    rs = classify_restarts(times)
    if rs.resets == 0 and rs.repeated == 0 and rs.overlaps == 0:
        return {
            "format": run.format,
            "split": False,
            "reason": f"{run.name} has no time reset to split at: its {len(times)} frames run from "
            f"{times[0]:g} to {times[-1]:g} ps",
            "frames": len(times),
            "first_ps": times[0],
            "last_ps": times[-1],
        }

    plans = []  # (pieces, frames) per segment
    dropped = 0
    for seg in rs.segments:
        pieces: list[tuple[str, int, int]] = []
        frames = 0
        kept: list[float] = []
        last, have = 0.0, False
        start = -1

        def flush(to: int) -> None:
            nonlocal start
            if start >= 0:
                a, b = run.byte_range(start, to)
                pieces.append((path, a, b))
                start = -1

        for i in range(seg.first, seg.last + 1):
            t = times[i]
            at = bisect.bisect_left(kept, t)
            is_repeat = at < len(kept) and tf.same_time(t, kept[at])
            if not is_repeat and at > 0:
                is_repeat = tf.same_time(t, kept[at - 1])
            if is_repeat:
                dropped += 1
                flush(i)
                continue
            if have and t < last:
                raise tf.BadTrajectory(
                    f"{run.name} has a backward time step at frame {i} ({t:g} ps) that does not match a "
                    "previously kept frame"
                )
            last, have = t, True
            bisect.insort(kept, t)
            if start < 0:
                start = i
            frames += 1
        flush(seg.last + 1)
        plans.append((pieces, frames))

    n = len(plans)
    paths, counts = [], []
    for k, (pieces, frames) in enumerate(plans, 1):
        dest = template.format(k=k, n=n)
        tf.write_piece(dest, run, frames, pieces)
        paths.append(dest)
        counts.append(frames)
    return {
        "format": run.format,
        "split": True,
        "pieces": paths,
        "frames": counts,
        "dropped": dropped,
        "resets": rs.resets,
        "repeated": rs.repeated,
        "overlaps": rs.overlaps,
    }


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        report = split(args.trajectory, args.template)
    except Exception as e:
        sys.stderr.write(
            f"splitting {os.path.basename(args.trajectory)} at its time resets: {' '.join(str(e).split())}\n"
        )
        sys.exit(1)
    print(json.dumps(report))


# --------------------------------------------------
if __name__ == "__main__":
    main()
