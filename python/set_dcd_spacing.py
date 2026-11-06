#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Set the frame spacing a DCD's header declares

A DCD has no frame times, only a header time step and save interval, and
MDRepo derives its sampling from that one field. A rewriting tool leaves it
as a placeholder more often than not (a step of exactly 1.0, or a step and
interval that multiply to exactly 1 ps), and where the release states the
spacing, the release wins: the staged copy's header is set to it and the
rewrite disclosed. Only the time step field is written; the save interval
is read and kept, so the declared spacing is interval x step x AKMA.

Moved here from mdrepo-agent's internal/traj SetDCDSpacing with the rest of
the file-producing steps (mdrepo-agent#98).

Output: the header rewritten in place, and one JSON object on stdout:

  {"was_ps": <the spacing the header declared>, "placeholder": bool,
   "ps": <the spacing it declares now>}

Exit 0 when written; 1 with the reason as the last line of stderr when the
file is not a DCD this can read.
"""

import argparse
import json
import os
import struct
import sys
from typing import NamedTuple

import trajframes as tf


class Args(NamedTuple):
    trajectory: str
    ps: float


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Set the frame spacing a DCD's header declares",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--trajectory", required=True, metavar="DCD", help="DCD to rewrite"
    )
    parser.add_argument(
        "-p", "--ps", required=True, type=float, help="Frame spacing in picoseconds"
    )
    a = parser.parse_args()
    if not (a.ps > 0):
        parser.error(f"a frame spacing of {a.ps} ps cannot be written")
    return Args(a.trajectory, a.ps)


# --------------------------------------------------
def set_spacing(path: str, ps: float) -> dict:
    name = os.path.basename(path)
    with open(path, "r+b") as fh:
        h = tf.read_dcd_header(fh, name)
        was = h.declared_ps_per_frame()
        delta = ps / (max(h.nsavc, 1) * tf.AKMA_PS)
        fh.seek(tf.DCD_DELTA_OFFSET)
        fh.write(struct.pack(h.order + ("f" if h.charmm else "d"), delta))
        fh.flush()
        os.fsync(fh.fileno())
    return {"was_ps": was, "placeholder": h.placeholder(), "ps": ps}


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        report = set_spacing(args.trajectory, args.ps)
    except Exception as e:
        sys.stderr.write(
            f"{os.path.basename(args.trajectory)}: {' '.join(str(e).split())}\n"
        )
        sys.exit(1)
    print(json.dumps(report))


# --------------------------------------------------
if __name__ == "__main__":
    main()
