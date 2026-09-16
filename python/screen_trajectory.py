#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-09-16
Purpose: Refuse a trajectory whose frames are not simulation data, before
         cpptraj is asked to convert it

Two submissions have arrived carrying frames that hold no simulation data --
one with a block of exact zeros, one with what uninitialised memory looks like
when it is written to a file as if it were coordinates. Neither was caught by
anything upstream of conversion, and the conversion step is the worst place to
find out:

  * cpptraj can spin on such a frame for hours instead of failing. Ticket 2371
    held the queue for 2h52m on one frame of one replicate, and was found by
    hand rather than by any alarm.

  * The damage survives conversion. `IVM-fxr-a.nc` of that same ticket carries
    two frames whose damaged atoms are all solvent, so the stripped
    `minimal.xtc` that `get_rmsd_rmsf.py` measures is clean and the RMSF
    ceiling has nothing to fire on -- while the `full.xtc` written beside it
    holds a frame that crashes the XTC reader outright. That file is 932 MB
    and would have been published.

So the ceiling in `get_rmsd_rmsf.py` is not a substitute for this. It runs
after conversion, on the stripped selection, and it measures a consequence;
this runs before conversion, on every atom, and it measures the defect.

NetCDF, XTC, TRR and DCD are screened, which is 99.9% of the trajectory
replicates on record. NetCDF takes a direct path that can read the unit cell
without touching the coordinates; the rest go frame by frame through
MDAnalysis, which opens all three without needing a topology. Measured at
about 76 MB/s, so a 932 MB trajectory costs ~12s against a job of tens of
minutes.

XTC matters as much as NetCDF here even though no submission has arrived as
one, because it is what we WRITE. The full.xtc cpptraj produced from
IVM-fxr-a.nc is the file that crashes the reader, and `.mdc` submissions --
the largest population at 553,147 replicates -- are decompressed to XTC before
this runs, so screening XTC covers them too.

A file in a format named above that cannot be opened is REFUSED, not passed
through. The first draft of this passed it through on the grounds that an open
failure says more about the screener than the data -- and then a leaked test
stub broke every MDAnalysis Universe in the process, and every damaged
trajectory sailed through reporting "not screened" while the tests stayed
green. A guard that turns itself off quietly when its reader breaks is the
exact failure this whole exercise is about. Over-refusing is loud and gets
fixed; under-screening is silent and does not.

Only a format with no reader here at all -- anything outside the lists above --
passes through, and the note says so. If reading a file kills this process
outright the caller sees the signal and refuses too, which is also correct: a
trajectory that crashes its own format's reader must not reach cpptraj.

The second limit is that nothing here counts frames. A `.mdc` that lost most
of its trajectory to a silent truncation decompresses to a short, clean XTC
and passes. That needs a declared frame count to compare against, which is a
submission-format question rather than a screening one.

Kept deliberately in step with `check_amber.py` in MD-Repo/preflight-checks,
which is the contributor-facing tool: `scan_cell` and `scan_coordinates` are
the same two rules, and a submitter who runs that tool must not be told their
data is fine by one implementation and rejected by the other. Change one, change
both, and keep the two test corpora agreeing.
"""

import argparse
import os
import sys
import warnings
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.io import netcdf_file

# scipy warns on closing an mmap'd NetCDF while array views onto it still
# exist. That is exactly how the coordinates are read -- in chunks, copying
# each chunk -- so the warning describes intended use.
warnings.filterwarnings(
    "ignore",
    message="Cannot close a netcdf_file opened with mmap=True",
    category=RuntimeWarning,
)

# Frames are read in blocks so a large trajectory never lands in memory whole.
CHUNK_FRAMES = 256

# A coordinate this large is not a position. It is 0.1 mm, where a simulation
# box is a few hundred angstroms at most.
#
# The check matters because XTC cannot store NaN: writing one saturates the
# 32-bit integer encoding, and the value reads back as +/-21,474,836 A -- a
# finite, non-zero number that a finiteness test passes. That saturation is the
# signature the RMSD/RMSF ceiling sees after conversion, and it is the only
# trace a damaged frame leaves in an XTC. Without this, screening an XTC would
# find nothing.
MAX_ABS_COORD = 1.0e6

# How many frame numbers to name in an error before summarising the rest. The
# message goes into md_upload_instance_message, and a trajectory with thousands
# of damaged frames should not push everything else out of the record.
MAX_NAMED = 12

# NetCDF has a direct path: its cell arrays can be read without the
# coordinates, so --cell-only means something there and nowhere else.
NETCDF_SUFFIXES = (".nc", ".netcdf")

# Everything else goes frame by frame through MDAnalysis, which opens each of
# these without a topology file.
MDANALYSIS_SUFFIXES = (".xtc", ".trr", ".dcd")


class Args(NamedTuple):
    """Command-line arguments"""

    trajectory: str
    cell_only: bool


class Verdict(NamedTuple):
    """What the screen decided, and what it did to decide it"""

    problem: Optional[str]
    note: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Refuse a trajectory holding frames that are not data",
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
        "--cell-only",
        action="store_true",
        help="Check the unit cell only, NetCDF only; ignored for other formats",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.trajectory):
        parser.error(f'No such trajectory "{args.trajectory}"')

    return Args(trajectory=args.trajectory, cell_only=args.cell_only)


# --------------------------------------------------
def group_runs(indexes: List[int]) -> List[Tuple[int, int]]:
    """Group ascending frame indexes into contiguous runs"""

    grouped: List[Tuple[int, int]] = []
    for idx in indexes:
        if grouped and idx == grouped[-1][1] + 1:
            grouped[-1] = (grouped[-1][0], idx)
        else:
            grouped.append((idx, idx))

    return grouped


# --------------------------------------------------
def format_runs(runs: List[Tuple[int, int]]) -> str:
    """Render frame runs as compact ranges"""

    shown = [
        f"{start}-{end}" if start != end else f"{start}"
        for start, end in runs[:MAX_NAMED]
    ]
    if len(runs) > MAX_NAMED:
        shown.append(f"and {len(runs) - MAX_NAMED} more")

    return ", ".join(shown)


# --------------------------------------------------
def scan_cell(lengths, angles) -> List[int]:
    """
    Find frames whose unit cell is not a usable box

    A length or an angle that is not finite, a length that is zero or
    negative, or an angle outside 0 to 180 degrees cannot describe a box.

    The one case that is not a defect is a trajectory with no periodic box at
    all, which some writers record as zero lengths in every frame. That is why
    the all-zero file returns early instead of reporting every frame: the
    signal we are after is a box that disappears partway through a file whose
    other frames have one.
    """

    lengths = np.asarray(lengths, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    if lengths.ndim != 2 or angles.ndim != 2:
        return []

    no_box = (lengths == 0).all(axis=1)
    if no_box.all():
        return []

    bad = ~np.isfinite(lengths).all(axis=1)
    bad |= ~np.isfinite(angles).all(axis=1)
    bad |= (lengths <= 0).any(axis=1)
    bad |= (angles <= 0).any(axis=1)
    bad |= (angles >= 180).any(axis=1)

    return [int(i) for i in np.where(bad)[0]]


# --------------------------------------------------
def scan_coordinates(coords) -> Tuple[List[int], List[int], List[int]]:
    """
    Find all-zero frames and frames holding values that are not finite

    Both answers come out of one pass, because the coordinates are the only
    expensive thing this reads and there is no reason to read them twice. The
    two results never overlap: a frame of exact zeros is finite.
    """

    zero: List[int] = []
    nonfinite: List[int] = []
    huge: List[int] = []
    total = coords.shape[0]

    for start in range(0, total, CHUNK_FRAMES):
        block = np.asarray(coords[start : start + CHUNK_FRAMES])
        flat = block.reshape(block.shape[0], -1)

        finite = np.isfinite(flat).all(axis=1)
        for offset in np.where(~finite)[0]:
            nonfinite.append(start + int(offset))
        for offset in np.where(finite & ~flat.any(axis=1))[0]:
            zero.append(start + int(offset))

        big = finite & (np.abs(np.where(finite[:, None], flat, 0.0))
                        >= MAX_ABS_COORD).any(axis=1)
        for offset in np.where(big)[0]:
            huge.append(start + int(offset))

    return zero, nonfinite, huge


# --------------------------------------------------
def describe(name, frames, bad_cell, nonfinite, huge, zero) -> Optional[str]:
    """
    Turn frame indexes into the sentence that refuses the trajectory

    Shared by both readers so the two paths cannot phrase the same defect two
    different ways in the record.
    """

    problems: List[str] = []

    if bad_cell:
        problems.append(
            f"{len(bad_cell)} of {frames} frames record a unit cell that is "
            f"not a box (frames {format_runs(group_runs(bad_cell))})"
        )
    if nonfinite:
        problems.append(
            f"{len(nonfinite)} of {frames} frames hold coordinates that are "
            f"not finite numbers (frames {format_runs(group_runs(nonfinite))})"
        )
    if huge:
        problems.append(
            f"{len(huge)} of {frames} frames hold coordinates of "
            f"{MAX_ABS_COORD:.0e} angstroms or more, which is not a position "
            f"(frames {format_runs(group_runs(huge))})"
        )
    if zero:
        problems.append(
            f"{len(zero)} of {frames} frames hold 0.0 for every coordinate of "
            f"every atom (frames {format_runs(group_runs(zero))})"
        )

    if not problems:
        return None

    return f"{name}: " + "; ".join(problems)


# --------------------------------------------------
def screen_netcdf(trajectory: str, cell_only: bool) -> Verdict:
    """Screen a NetCDF trajectory, reading the cell without the coordinates"""

    name = os.path.basename(trajectory)

    with netcdf_file(trajectory, "r", mmap=True) as ncf:
        if "coordinates" not in ncf.variables:
            return Verdict(f"{name} has no coordinates", "")

        coords = ncf.variables["coordinates"]
        frames = int(coords.shape[0])

        if frames == 0:
            return Verdict(f"{name} holds no frames", "")

        bad_cell: List[int] = []
        if "cell_lengths" in ncf.variables and "cell_angles" in ncf.variables:
            bad_cell = scan_cell(
                ncf.variables["cell_lengths"][:],
                ncf.variables["cell_angles"][:],
            )

        zero: List[int] = []
        nonfinite: List[int] = []
        huge: List[int] = []
        if not cell_only:
            zero, nonfinite, huge = scan_coordinates(coords)

    how = "unit cell only" if cell_only else "cell and coordinates"

    return Verdict(
        describe(name, frames, bad_cell, nonfinite, huge, zero),
        f"{name}: screened {frames} frames ({how}), no damaged frames",
    )


# --------------------------------------------------
def screen_via_mdanalysis(trajectory: str) -> Verdict:
    """
    Screen an XTC, TRR or DCD frame by frame

    None of the three needs a topology to open, so nothing here depends on the
    structure or topology file being readable -- which matters, because this
    runs on trajectories we wrote as well as ones we were sent.

    An open failure is a refusal. See the module docstring for why that is the
    right way round: a screener that excuses itself when its reader breaks
    stops being a guard without anyone noticing.
    """

    name = os.path.basename(trajectory)

    # Imported here so the NetCDF path never pays for it: MDAnalysis takes
    # seconds to import, and this runs once per trajectory per job.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import MDAnalysis as mda

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            universe = mda.Universe(trajectory)
    except Exception as err:
        return Verdict(f"{name} cannot be opened for screening: {err}", "")

    zero: List[int] = []
    nonfinite: List[int] = []
    huge: List[int] = []
    lengths: List[List[float]] = []
    angles: List[List[float]] = []
    frames = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in universe.trajectory:
            frames += 1
            coords = np.asarray(step.positions, dtype=np.float64)

            if not np.isfinite(coords).all():
                nonfinite.append(step.frame)
            elif not coords.any():
                zero.append(step.frame)
            elif (np.abs(coords) >= MAX_ABS_COORD).any():
                huge.append(step.frame)

            box = step.dimensions
            if box is None:
                lengths.append([0.0, 0.0, 0.0])
                angles.append([0.0, 0.0, 0.0])
            else:
                lengths.append([float(v) for v in box[:3]])
                angles.append([float(v) for v in box[3:6]])

    if frames == 0:
        return Verdict(f"{name} holds no frames", "")

    bad_cell = scan_cell(lengths, angles)

    return Verdict(
        describe(name, frames, bad_cell, nonfinite, huge, zero),
        f"{name}: screened {frames} frames (cell and coordinates), "
        f"no damaged frames",
    )


# --------------------------------------------------
def screen(trajectory: str, cell_only: bool = False) -> Verdict:
    """
    Screen one trajectory

    The problem is set when the trajectory must not be converted. The note says
    what was done either way, so a pass-through is visible in the log rather
    than looking like a clean result.
    """

    name = os.path.basename(trajectory)
    lowered = trajectory.lower()

    if lowered.endswith(NETCDF_SUFFIXES):
        return screen_netcdf(trajectory, cell_only)

    if lowered.endswith(MDANALYSIS_SUFFIXES):
        return screen_via_mdanalysis(trajectory)

    return Verdict(None, f"{name}: NOT screened, no reader for this format")


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    try:
        verdict = screen(args.trajectory, args.cell_only)
    except Exception as err:
        # A trajectory in a format we do screen, that then fails mid-read, is a
        # refusal. Saying which file and why beats the missing-output-file
        # error this would otherwise surface as several minutes later.
        sys.exit(
            f"Cannot screen {os.path.basename(args.trajectory)}: {err}"
        )

    if verdict.problem:
        sys.exit(verdict.problem)

    print(verdict.note)


# --------------------------------------------------
if __name__ == "__main__":
    main()
