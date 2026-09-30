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

The third rule is a jump: an atom that moves MAX_JUMP or more between two
consecutive frames. DDD's damaged last frames move whole protein chains by one
constant offset, often well under MAX_ABS_COORD, so the size test alone passed
five of the twelve measured on 2026-09-30. A jump is not tested across a
boundary in time (a gap or a clock restart, where joined runs meet), because
between two runs the system can be anywhere. See `find_jumps`.

Kept deliberately in step with `check_amber.py` in MD-Repo/preflight-checks,
which is the contributor-facing tool: `scan_cell` and `scan_coordinates` are
the same two rules, and a submitter who runs that tool must not be told their
data is fine by one implementation and rejected by the other. Change one, change
both, and keep the two test corpora agreeing. `check_simulations.py` in the
same repo is the general successor (NetCDF, XTC, TRR and DCD, where
`check_amber.py` reads NetCDF only), and it is the one the jump rule
(`find_jumps`) should be added to; neither has it yet, so until then the
submitter-side tools pass the files it refuses. MDR-69, preflight-checks#2.
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

# An atom that moves this far between consecutive frames has not moved: the
# frame is damaged. DDD's damaged last frames shift whole protein chains by one
# constant vector of 1e4-1e7 A, and five of the twelve measured never reach
# MAX_ABS_COORD. Honest data moves far less: DDD at most 71.5 A per frame,
# AMBER runs 8-11 A, and the wrapped OpenMM run MDR00088497 272 A, which is its
# box diagonal -- an atom leaving one corner and coming back through the
# opposite one. So the limit is this, or twice the frame's own box diagonal if
# that is larger, which keeps a very large wrapped box from being refused.
#
# Why a jump and not a lower MAX_ABS_COORD: an unwrapped trajectory can drift
# thousands of angstroms from the origin over microseconds, honestly, while
# moving a few angstroms per frame.
MAX_JUMP = 5.0e3

# Two frames are consecutive in time when their spacing is within this share
# of the file's regular step. float32 times wander in the last digits.
STEP_TOLERANCE = 0.01

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
def box_diagonal(lengths, angles) -> Optional[float]:
    """
    The longest distance a wrap can move an atom in one box, or None

    Wrapping moves an atom by a sum of box vectors, at most one of each, so the
    longest move is the longest of a+b+c, a+b-c, a-b+c and -a+b+c. For a
    rectangular box that is the plain diagonal. No box -- zero, negative or
    non-finite lengths, or angles that make no cell -- gives None.
    """

    lengths = np.asarray(lengths, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if lengths.shape != (3,) or angles.shape != (3,):
        return None
    if not (np.isfinite(lengths).all() and np.isfinite(angles).all()):
        return None
    if (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any():
        return None

    alpha, beta, gamma = np.radians(angles)
    a_len, b_len, c_len = lengths
    a = np.array([a_len, 0.0, 0.0])
    b = np.array([b_len * np.cos(gamma), b_len * np.sin(gamma), 0.0])
    cx = c_len * np.cos(beta)
    cy = c_len * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c_len**2 - cx**2 - cy**2
    if not np.isfinite(cz_sq) or cz_sq <= 0:
        return None
    c = np.array([cx, cy, np.sqrt(cz_sq)])

    return float(max(np.linalg.norm(v) for v in (a + b + c, a + b - c, a - b + c, -a + b + c)))


# --------------------------------------------------
def jump_limit(diagonal: Optional[float]) -> float:
    """MAX_JUMP, or twice the box diagonal when there is a box that large"""

    if diagonal is None or not np.isfinite(diagonal) or diagonal <= 0:
        return MAX_JUMP

    return max(MAX_JUMP, 2.0 * float(diagonal))


# --------------------------------------------------
def regular_step(times) -> Optional[float]:
    """
    The file's usual spacing between frames, or None if time tells us nothing

    The median, not the most common value: float32 times differ in the last
    digits, so an exact count can split one real step into several. A file
    with no times, or whose times do not advance (ticket 2339 wrote zeros),
    gives None.
    """

    if times is None or len(times) < 2:
        return None

    step = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    if not np.isfinite(step) or step <= 0:
        return None

    return step


# --------------------------------------------------
def find_jumps(moves, times, diagonals) -> Tuple[List[int], List[int]]:
    """
    Frames whose atoms moved too far, and the time boundaries that excused one

    `moves[i]` is the largest one-atom move from frame i-1 to frame i (0 for
    frame 0). `diagonals[i]` is frame i's box diagonal, or None.

    A trajectory may be several runs joined, and between two runs the system
    can be anywhere, so a jump is not tested across a boundary in time: a step
    longer than the regular one, or one that goes back (a clock restart). Only
    a boundary into a block of two or more frames counts. Otherwise a frame
    damaged in its time as well as its coordinates -- a garbage time on a
    damaged last frame -- would excuse itself.

    Without usable times nothing can show a boundary, so every step is tested.
    A wrong refusal is loud and gets fixed; a missed defect is silent.
    """

    frames = len(moves)
    step = regular_step(times)

    irregular = [False] * frames
    if step is not None:
        spacing = np.diff(np.asarray(times, dtype=np.float64))
        for i in range(1, frames):
            irregular[i] = not abs(spacing[i - 1] - step) <= STEP_TOLERANCE * step

    boundaries = [
        i
        for i in range(1, frames)
        if irregular[i] and i + 1 < frames and not irregular[i + 1]
    ]
    excused = set(boundaries)

    jumps = [
        i
        for i in range(1, frames)
        if i not in excused
        and moves[i] >= jump_limit(None if diagonals is None else diagonals[i])
    ]

    return jumps, boundaries


# --------------------------------------------------
def jump_note(frames: int, times, boundaries: List[int]) -> str:
    """
    What the jump test did, for the note: silent only when it tested every
    step against a regular clock
    """

    if frames < 2:
        return ""
    if regular_step(times) is None:
        return "; no usable frame times, jump test applied to every step"
    if boundaries:
        return (
            f"; {len(boundaries) + 1} time segments (breaks at frames "
            f"{format_runs(group_runs(boundaries))}), jump test not applied "
            f"across them"
        )

    return ""


# --------------------------------------------------
def largest_moves(block, before) -> List[float]:
    """
    For each frame of a block, the largest one-atom move from the frame before

    `before` is the last frame of the previous block, or None at the start of
    the file, where the first frame's move is 0. A non-finite coordinate is
    reported by its own rule, so its move counts as 0 here.
    """

    moves: List[float] = []
    previous = before
    for frame in block:
        moves.append(0.0 if previous is None else largest_move(frame, previous))
        previous = frame

    return moves


# --------------------------------------------------
def largest_move(frame, before) -> float:
    """
    The largest one-atom move between two frames

    Squared distances, and one square root at the end: this runs on every
    frame of every trajectory, and a 400,000-atom frame is 400,000 of them.
    """

    with np.errstate(invalid="ignore", over="ignore"):
        # float32 is plenty: the question is 5,000 A against a few hundred.
        delta = np.subtract(frame, before, dtype=np.float32)
        squared = np.einsum("ij,ij->i", delta, delta)
    finite = squared[np.isfinite(squared)]

    return float(np.sqrt(finite.max())) if finite.size else 0.0


# --------------------------------------------------
def scan_coordinates(coords, moves=None) -> Tuple[List[int], List[int], List[int]]:
    """
    Find all-zero frames and frames holding values that are not finite

    Both answers come out of one pass, because the coordinates are the only
    expensive thing this reads and there is no reason to read them twice. The
    two results never overlap: a frame of exact zeros is finite.

    Given a list as `moves`, the same pass also appends each frame's largest
    one-atom move from the frame before, for `find_jumps`. The last frame of
    each block is carried into the next, so a jump at a block's first frame is
    still measured.
    """

    zero: List[int] = []
    nonfinite: List[int] = []
    huge: List[int] = []
    total = coords.shape[0]
    before = None

    for start in range(0, total, CHUNK_FRAMES):
        block = np.asarray(coords[start : start + CHUNK_FRAMES])
        flat = block.reshape(block.shape[0], -1)

        if moves is not None:
            moves.extend(largest_moves(block, before))
            before = block[-1]

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
def describe(name, frames, bad_cell, nonfinite, huge, zero, jumps=()) -> Optional[str]:
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
    if jumps:
        problems.append(
            f"{len(jumps)} of {frames} frames move an atom {MAX_JUMP:.0e} "
            f"angstroms or more (or twice the box diagonal) from the frame "
            f"before (frames {format_runs(group_runs(list(jumps)))})"
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
        jumps: List[int] = []
        boundaries: List[int] = []
        times = None
        if "time" in ncf.variables:
            times = np.array(ncf.variables["time"][:], dtype=np.float64)

        if not cell_only:
            moves: List[float] = []
            zero, nonfinite, huge = scan_coordinates(coords, moves)

            diagonals = None
            if "cell_lengths" in ncf.variables and "cell_angles" in ncf.variables:
                cell_lengths = np.array(ncf.variables["cell_lengths"][:])
                cell_angles = np.array(ncf.variables["cell_angles"][:])
                diagonals = [
                    box_diagonal(lengths, angles)
                    for lengths, angles in zip(cell_lengths, cell_angles)
                ]
            jumps, boundaries = find_jumps(moves, times, diagonals)

    how = "unit cell only" if cell_only else "cell and coordinates"
    note = f"{name}: screened {frames} frames ({how}), no damaged frames"
    if not cell_only:
        note += jump_note(frames, times, boundaries)

    return Verdict(
        describe(name, frames, bad_cell, nonfinite, huge, zero, jumps),
        note,
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
    moves: List[float] = []
    times: List[float] = []
    diagonals: List[Optional[float]] = []
    before = None
    frames = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in universe.trajectory:
            frames += 1
            coords = np.asarray(step.positions, dtype=np.float64)

            # DCD stores only a start and an interval, so its times always
            # read back regular and every step of one is tested.
            times.append(float(step.time))
            positions = step.positions
            moves.append(0.0 if before is None else largest_move(positions, before))
            before = positions.copy()
            diagonals.append(
                None
                if step.dimensions is None
                else box_diagonal(step.dimensions[:3], step.dimensions[3:6])
            )

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
    jumps, boundaries = find_jumps(moves, times, diagonals)

    return Verdict(
        describe(name, frames, bad_cell, nonfinite, huge, zero, jumps),
        f"{name}: screened {frames} frames (cell and coordinates), "
        f"no damaged frames" + jump_note(frames, times, boundaries),
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
