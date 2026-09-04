#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2025-09-16
Purpose: Sample trajectory
"""

import argparse
import os
import sys
from typing import NamedTuple

import MDAnalysis as mda

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
def describe(path: str) -> str:
    """"<path> (N bytes)" or a note that it is missing, for error output."""

    try:
        return f"{path} ({os.path.getsize(path):,} bytes)"
    except OSError as e:
        return f"{path} (UNREADABLE: {e})"


# --------------------------------------------------
def cleanup(path: str) -> None:
    """Remove a partial output, ignoring the case where it is not there."""

    try:
        os.unlink(path)
    except OSError:
        pass


# --------------------------------------------------
def die(message: str, args: Args, num_frames: int = -1) -> None:
    """Exit non-zero with enough context to act on.

    WHY THIS EXISTS. mdr-process reports a failed child as
    "Command failed: {cmd:?}\n{stderr}", so whatever this writes to stderr
    is the ONLY account of what went wrong that reaches the driver's
    record. This script previously wrote nothing of its own: 1m0o failed
    here on 2026-08-29 at rep_9 and its record carries an empty stderr, so
    the cause was never recoverable. Name the inputs and the frame count,
    because the inputs are what differ between the reps that work and the
    one that does not.
    """

    print(f"sample_trajectory.py: {message}", file=sys.stderr)
    print(f"  structure : {describe(args.structure)}", file=sys.stderr)
    print(f"  trajectory: {describe(args.trajectory)}", file=sys.stderr)
    print(f"  outfile   : {args.outfile}", file=sys.stderr)
    if num_frames >= 0:
        print(f"  frames    : {num_frames}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    try:
        trajectory = mda.Universe(args.structure, args.trajectory)
    except Exception as e:
        die(f"failed to load universe: {type(e).__name__}: {e}", args)

    num_frames = len(trajectory.trajectory)

    # A 0-frame trajectory reaches .write() as an empty frame list, which
    # fails somewhere inside MDAnalysis rather than here. Catch it where it
    # can still be explained.
    if num_frames == 0:
        die("trajectory has 0 frames; nothing to sample", args, num_frames)

    end_frame = min(num_frames, END_SAMPLE_FROM)
    sample_rate = end_frame // TOTAL_FRAMES if end_frame > TOTAL_FRAMES else 1
    sampled_frames = trajectory.trajectory[:end_frame][::sample_rate]

    # Write to a temporary name and rename only on success.
    #
    # WHY. Reading a damaged frame can kill this process with a HARDWARE
    # SIGNAL, not an exception -- 1m0o rep_9 dies with SIGFPE on frame 94
    # of 95, so no `except` above can run and nothing is printed. By then
    # some frames are already on disk. mdr-process skips this step whenever
    # the output file exists (process.rs sample_trajectory, "Sampled
    # trajectory exists"), so the half-written file made the NEXT run skip
    # the sampling and report success: 1m0o's leftover sampled.xtc holds 94
    # frames instead of 95, and a re-run would have imported it.
    #
    # A rename is atomic on the same filesystem, so either the whole
    # sample is there or nothing is, and a crash always fails the same way
    # twice instead of turning into a quiet success.
    # The temporary name must KEEP the .xtc extension -- MDAnalysis picks
    # its writer from the extension, so "sampled.xtc.partial" fails with
    # "No writer found for format". Leading dot keeps it out of the way,
    # and it stays in the same directory so the rename is atomic.
    out_dir, out_name = os.path.split(args.outfile)
    root, ext = os.path.splitext(out_name)
    tmp_out = os.path.join(out_dir, f".{root}.partial{ext}")
    try:
        trajectory.atoms.write(tmp_out, frames=sampled_frames)
    except Exception as e:
        cleanup(tmp_out)
        die(f"failed to write sample: {type(e).__name__}: {e}", args, num_frames)

    try:
        os.replace(tmp_out, args.outfile)
    except OSError as e:
        cleanup(tmp_out)
        die(f"failed to move sample into place: {e}", args, num_frames)

    print(f"Wrote {num_frames} frames to '{args.outfile}'")


# --------------------------------------------------
if __name__ == "__main__":
    main()
