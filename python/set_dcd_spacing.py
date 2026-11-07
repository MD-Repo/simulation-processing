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
        # Exit 1, not argparse's 2: a bad value is this run's refusal, and the
        # contract keeps 2 for an environment that lacks something.
        sys.stderr.write(f"a frame spacing of {a.ps} ps cannot be written" + "\n")
        sys.exit(1)
    return Args(a.trajectory, a.ps)


# --------------------------------------------------
def set_spacing(path: str, ps: float) -> dict:
    name = os.path.basename(path)
    with open(path, "r+b") as fh:
        h = tf.read_dcd_header(fh, name)
        if h.nsavc <= 0:
            # Readers take the spacing as step x interval; with no interval
            # no time step can declare one, and a write would report a
            # spacing the file does not carry.
            raise tf.Refused(
                f"the header's save interval is {h.nsavc}, so no time step can declare a frame spacing"
            )
        was = h.declared_ps_per_frame()
        delta = ps / (h.nsavc * tf.AKMA_PS)
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
    except tf.Refused as e:
        print(json.dumps({"refused": str(e)}))
        sys.stderr.write(f"{os.path.basename(args.trajectory)}: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(
            f"{os.path.basename(args.trajectory)}: {' '.join(str(e).split())}\n"
        )
        sys.exit(1)
    print(json.dumps(report))


# --------------------------------------------------
if __name__ == "__main__":
    main()
