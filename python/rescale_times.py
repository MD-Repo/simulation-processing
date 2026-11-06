#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Multiply every frame's time stamp in a trajectory, in place

Some releases stamp their frames in the wrong unit: nanosecond values in a
field the format defines as picoseconds, so the interval read out of the
file is 1000x too small, or frames stamped exactly 1 ps apart, the default
every rewriting tool falls back to, against a spacing the release states.
MDRepo derives sampling from the trajectory itself, so depositing the bytes
unaltered would publish a time axis that is wrong; the staged copy is
corrected, and the release is never touched.

Only the time field of each frame is rewritten: the 4-byte big-endian float
at offset 12 of an XTC frame header, the real after a TRR frame's fixed
header, the time record of an AMBER NetCDF, the time dataset of an HDF5
file that stores it uncompressed. No frame changes length and no coordinate
block is read, let alone re-encoded, so the file stays byte-identical
everywhere except the time field. Offsets are collected before any write.

Refused, with the reason, and nothing written: a DCD, which states one time
step in its header and stamps no frame, so there is nothing per frame to
rescale and the header is a claim already known to be unreliable (see
set_dcd_spacing.py); a DTR, which stores each frame's time both in its
index and inside the frame; an HDF5 time dataset behind a compression
filter, or holding integers, or in a unit other than picoseconds. A refusal
is a fact about the file, which the caller discloses, not a failure.

Moved here from mdrepo-agent's internal/traj RescaleTimes with the rest of
the file-producing steps (mdrepo-agent#98).

Output: the file rewritten in place, and one JSON object on stdout:

  {"format": "xtc", "frames": N}            when rewritten
  {"format": "dcd", "refused": "<why>"}     when refused

Exit 0 when rewritten; 1 with the reason as the last line of stderr when
refused (the report still says so) or when the file is damaged; 2 when h5py
is needed and not installed.
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
    factor: float


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Multiply every frame's time stamp in place",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--trajectory",
        required=True,
        metavar="FILE",
        help="Trajectory to rewrite",
    )
    parser.add_argument(
        "-x",
        "--factor",
        required=True,
        type=float,
        help="Factor to multiply each stamp by",
    )
    a = parser.parse_args()
    if not (a.factor > 0) or a.factor == float("inf"):
        parser.error(f"frame times cannot be scaled by {a.factor}")
    return Args(a.trajectory, a.factor)


# --------------------------------------------------
def rescale(path: str, factor: float) -> dict:
    """Rewrite the stamps and report how many; raise Refused for a format or
    file whose stamps cannot be rewritten in place."""

    name = os.path.basename(path)
    fmt = tf.identify(path)
    if fmt == "dtr":
        raise tf.Refused(
            "a DTR stores each frame's time both in its timekeys index and inside the frame, "
            "so its times are not rewritten"
        )
    if fmt == "dcd":
        raise tf.Refused(
            "cannot rescale times in a DCD; only formats that stamp each frame with its own time can be rescaled"
        )
    if fmt is None:
        raise tf.BadTrajectory(f"{name} is not a trajectory this reads")

    if fmt == "hdf5":
        f, td = tf.hdf5_time_dataset(path)
        with f:
            n = int(td.shape[0]) if td.shape else 0
            td[...] = td[...] * factor
        return {"format": fmt, "frames": n}

    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        if fmt == "xtc":
            frames = tf.walk_xtc(fh, size, name)
        elif fmt == "trr":
            frames = tf.walk_trr(fh, size, name)
        else:
            frames = tf.netcdf_time_stamps(fh, size, name)
        for fr in frames:
            scaled = fr.time * factor
            fh.seek(fr.time_offset)
            fh.write(struct.pack(fr.order + ("d" if fr.wide else "f"), scaled))
        fh.flush()
        os.fsync(fh.fileno())
    return {"format": fmt, "frames": len(frames)}


# --------------------------------------------------
def main() -> None:
    args = get_args()
    name = os.path.basename(args.trajectory)
    try:
        if tf.identify(args.trajectory) == "hdf5":
            import h5py  # noqa: F401
    except ImportError:
        sys.stderr.write(f"h5py is not installed in {sys.executable}\n")
        sys.exit(2)
    try:
        report = rescale(args.trajectory, args.factor)
    except tf.Refused as e:
        print(json.dumps({"format": tf.identify(args.trajectory), "refused": str(e)}))
        sys.stderr.write(f"{name}: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"{name}: {' '.join(str(e).split())}\n")
        sys.exit(1)
    print(json.dumps(report))


# --------------------------------------------------
if __name__ == "__main__":
    main()
