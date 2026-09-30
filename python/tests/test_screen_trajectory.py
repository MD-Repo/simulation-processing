"""Tests for screen_trajectory.py

The guard that runs before cpptraj, so the cases here are the ones that have
actually reached us rather than invented ones.

Ticket 2339 sent a block of frames written as exact zeros -- coordinates, box
lengths, box angles and time. Ticket 2371 sent frames of uninitialised memory:
coordinates running to the largest number a 32-bit float holds, some of them
not finite, with a unit cell of the same kind of garbage.

The case that decides the shape of this guard is `IVM-fxr-a.nc` of ticket
2371. Its two damaged frames carry a perfectly ordinary box -- 75.138 A and
109.471 degrees, the same as their neighbours -- and every damaged atom in
them is solvent. So a check that reads only the unit cell passes that file,
and the RMSD/RMSF ceiling passes it too because it measures the stripped
selection after conversion. The `full.xtc` cpptraj wrote from it holds a frame
that core-dumps the XTC reader. That is why `scan_coordinates` is not
optional, and why `test_valid_box_does_not_excuse_bad_coordinates` exists.

The fixtures here are the shared corpus named in the module docstring of both
this script and `check_amber.py` in MD-Repo/preflight-checks. The two carry
the same two rules and must agree: a submitter told their data is fine by one
and rejected by the other is worse off than with no tool at all.
"""

import os
import sys

import numpy as np
import pytest
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import screen_trajectory as s  # noqa: E402

ATOMS = 40
FRAMES = 6


# --------------------------------------------------
def write_nc(path, coords, lengths, angles, times="index"):
    """
    Write a minimal AMBER-convention NetCDF trajectory

    `times` is the frame times in ps; the default is the frame index. None
    leaves the time variable out, as contributor 289's MDR00048669 does.
    """

    out = netcdf_file(str(path), "w", version=2)
    out.Conventions = "AMBER"
    out.ConventionVersion = "1.0"
    out.createDimension("frame", None)
    out.createDimension("spatial", 3)
    out.createDimension("atom", coords.shape[1])
    out.createDimension("cell_spatial", 3)
    out.createDimension("cell_angular", 3)
    out.createDimension("label", 5)

    var = out.createVariable("spatial", "c", ("spatial",))
    var[:] = np.array(list("xyz"), dtype="c")
    var = out.createVariable("cell_spatial", "c", ("cell_spatial",))
    var[:] = np.array(list("abc"), dtype="c")
    var = out.createVariable("cell_angular", "c", ("cell_angular", "label"))
    var[:] = np.array([list("alpha"), list("beta "), list("gamma")], dtype="c")

    if times is not None:
        var = out.createVariable("time", "f", ("frame",))
        var.units = "picosecond"
        if isinstance(times, str):
            var[:] = np.arange(coords.shape[0], dtype=np.float32)
        else:
            var[:] = np.asarray(times, dtype=np.float32)

    var = out.createVariable(
        "coordinates", "f", ("frame", "atom", "spatial")
    )
    var.units = "angstrom"
    var[:] = coords

    var = out.createVariable("cell_lengths", "d", ("frame", "cell_spatial"))
    var.units = "angstrom"
    var[:] = lengths

    var = out.createVariable("cell_angles", "d", ("frame", "cell_angular"))
    var.units = "degree"
    var[:] = angles

    out.close()
    return str(path)


# --------------------------------------------------
def healthy():
    """Coordinates, lengths and angles of a trajectory with nothing wrong"""

    rng = np.random.default_rng(2339)
    coords = rng.uniform(-40, 40, (FRAMES, ATOMS, 3)).astype(np.float32)
    lengths = np.full((FRAMES, 3), 75.09, dtype=np.float64)
    angles = np.full((FRAMES, 3), 109.471219, dtype=np.float64)
    return coords, lengths, angles


# --------------------------------------------------
@pytest.fixture
def clean(tmp_path):
    coords, lengths, angles = healthy()
    return write_nc(tmp_path / "clean.nc", coords, lengths, angles)


# --------------------------------------------------
def test_a_healthy_trajectory_passes(clean):
    assert s.screen(clean, cell_only=False).problem is None


# --------------------------------------------------
def test_ticket_2339_shape_is_caught(tmp_path):
    """A block of frames written as exact zeros, box included"""

    coords, lengths, angles = healthy()
    coords[2:4] = 0.0
    lengths[2:4] = 0.0
    angles[2:4] = 0.0

    path = write_nc(tmp_path / "zeros.nc", coords, lengths, angles)
    problem = s.screen(path, cell_only=False).problem

    assert problem is not None
    assert "not a box" in problem
    assert "0.0 for every" in problem
    assert "2-3" in problem


# --------------------------------------------------
def test_ticket_2371_shape_is_caught(tmp_path):
    """Uninitialised memory in the coordinates and in the cell"""

    coords, lengths, angles = healthy()
    coords[4, :5] = np.nan
    coords[4, 5:10] = 3.3e38
    lengths[4] = [1.46e233, -1.04e-306, -1.60e-203]
    angles[4] = [1.67e214, -3.00e-79, 1.80e-5]

    path = write_nc(tmp_path / "garbage.nc", coords, lengths, angles)
    problem = s.screen(path, cell_only=False).problem

    assert problem is not None
    assert "not a box" in problem
    assert "not finite numbers" in problem
    assert "frames 4" in problem


# --------------------------------------------------
def test_valid_box_does_not_excuse_bad_coordinates(tmp_path):
    """
    IVM-fxr-a.nc: damaged frames under an ordinary box

    The box is untouched and every neighbouring frame is ordinary, so the
    cheap check has nothing to see. Reading the coordinates is the only way
    this file is ever refused.
    """

    coords, lengths, angles = healthy()
    coords[3, 7] = np.inf

    path = write_nc(tmp_path / "solventonly.nc", coords, lengths, angles)

    assert s.screen(path, cell_only=True).problem is None

    problem = s.screen(path, cell_only=False).problem
    assert problem is not None
    assert "not finite numbers" in problem


# --------------------------------------------------
def test_a_trajectory_with_no_periodic_box_is_not_a_defect(tmp_path):
    """Zero lengths in every frame means no box, not a broken box"""

    coords, lengths, angles = healthy()
    lengths[:] = 0.0
    angles[:] = 0.0

    path = write_nc(tmp_path / "nobox.nc", coords, lengths, angles)

    assert s.screen(path, cell_only=False).problem is None


# --------------------------------------------------
@pytest.mark.parametrize(
    "lengths_row,angles_row",
    [
        ([0.0, 75.0, 75.0], [109.5, 109.5, 109.5]),
        ([-75.0, 75.0, 75.0], [109.5, 109.5, 109.5]),
        ([np.nan, 75.0, 75.0], [109.5, 109.5, 109.5]),
        ([75.0, 75.0, 75.0], [0.0, 109.5, 109.5]),
        ([75.0, 75.0, 75.0], [180.0, 109.5, 109.5]),
        ([75.0, 75.0, 75.0], [109.5, np.inf, 109.5]),
        ([75.0, 75.0, 75.0], [6.2e116, 6.2e116, 6.2e116]),
    ],
)
def test_each_way_a_cell_can_be_wrong(tmp_path, lengths_row, angles_row):
    """
    Every one of these is a box no tool can use

    The last row is IVM-fxr-e.nc frame 5655, which a check written as "positive
    and finite" lets through: 6.2e116 is both. It is the angle that gives it
    away, which is why angles are tested as well as lengths.
    """

    coords, lengths, angles = healthy()
    lengths[1] = lengths_row
    angles[1] = angles_row

    path = write_nc(tmp_path / "cell.nc", coords, lengths, angles)
    problem = s.screen(path, cell_only=True).problem

    assert problem is not None
    assert "not a box" in problem
    assert "frames 1" in problem


# --------------------------------------------------
def test_a_format_we_have_no_reader_for_says_so(tmp_path):
    """Passing through is fine only because the note records that it happened"""

    path = tmp_path / "traj.dtr"
    path.write_bytes(b"a desmond trajectory, which we do not read")

    verdict = s.screen(str(path), cell_only=False)

    assert verdict.problem is None
    assert "NOT screened" in verdict.note


# --------------------------------------------------
def test_runs_are_grouped_for_the_message():
    assert s.group_runs([]) == []
    assert s.group_runs([4]) == [(4, 4)]
    assert s.group_runs([1, 2, 3, 9, 10, 40]) == [(1, 3), (9, 10), (40, 40)]


# --------------------------------------------------
def test_a_long_run_of_damage_does_not_fill_the_message():
    """
    2339 was 20 consecutive frames; a worse file could be thousands

    The message is stored in md_upload_instance_message, so it stays bounded.
    """

    runs = [(i * 2, i * 2) for i in range(50)]
    rendered = s.format_runs(runs)

    assert rendered.endswith(f"and {50 - s.MAX_NAMED} more")
    assert len(rendered) < 200


# --------------------------------------------------
def test_the_three_coordinate_answers_do_not_overlap():
    """One pass answers all three, and no frame appears in two of them"""

    coords = np.zeros((5, ATOMS, 3), dtype=np.float32)
    coords[1] = 1.5
    coords[2] = np.nan
    coords[4] = 1.5
    coords[4, 0, 0] = 2.1474836e7

    zero, nonfinite, huge = s.scan_coordinates(coords)

    assert zero == [0, 3]
    assert nonfinite == [2]
    assert huge == [4]
    assert not set(zero) & set(nonfinite) & set(huge)


# --------------------------------------------------
def test_the_xtc_saturation_value_is_caught(tmp_path):
    """
    XTC cannot store NaN, so the damage arrives as a finite number

    Writing a NaN through an XTC saturates its integer encoding and the value
    reads back as 21,474,836 angstroms. Every trajectory this pipeline writes
    is an XTC, and `.mdc` submissions are decompressed to one before the
    screen runs, so without a size test screening those formats would find
    nothing at all.
    """

    coords, lengths, angles = healthy()
    coords[3, 9] = 2.1474836e7

    path = write_nc(tmp_path / "saturated.nc", coords, lengths, angles)
    problem = s.screen(path, cell_only=False).problem

    assert problem is not None
    assert "not a position" in problem
    assert "frames 3" in problem


# --------------------------------------------------
def test_an_ordinary_large_system_is_not_too_large():
    """The limit is 0.1 mm; a big solvated box is a few hundred angstroms"""

    coords = np.full((3, ATOMS, 3), 950.0, dtype=np.float32)

    assert s.scan_coordinates(coords) == ([], [], [])


# --------------------------------------------------
def write_via_mdanalysis(path, coords, dimensions, times=None):
    """
    Write a trajectory in whatever format the extension names

    `times` sets each frame's time in ps. XTC and TRR store it as given; DCD
    stores only a start and an interval, so it reads back as regular whatever
    is passed.
    """

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mda.Universe.empty(coords.shape[1], trajectory=True)
    universe.load_new(coords, format=MemoryReader, dimensions=dimensions)
    with mda.Writer(str(path), coords.shape[1]) as writer:
        for step in universe.trajectory:
            if times is not None:
                step.time = times[step.frame]
            writer.write(universe.atoms)

    return str(path)


# --------------------------------------------------
def boxes(n, lengths=(50.0, 50.0, 50.0), angles=(90.0, 90.0, 90.0)):
    return np.tile(
        np.array(list(lengths) + list(angles), dtype=np.float32), (n, 1)
    )


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["xtc", "trr", "dcd"])
def test_a_healthy_trajectory_passes_in_every_format(tmp_path, ext):
    coords, _, _ = healthy()
    path = write_via_mdanalysis(tmp_path / f"clean.{ext}", coords, boxes(FRAMES))

    verdict = s.screen(path)

    assert verdict.problem is None
    assert "screened" in verdict.note


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["trr", "dcd"])
def test_non_finite_coordinates_are_caught_in_every_format(tmp_path, ext):
    """
    TRR and DCD store floats, so a NaN survives into them

    XTC is excluded on purpose: it cannot hold one, and what it does instead
    is the subject of test_the_xtc_saturation_value_is_caught.
    """

    coords, _, _ = healthy()
    coords[2, 3] = np.nan
    path = write_via_mdanalysis(tmp_path / f"nan.{ext}", coords, boxes(FRAMES))

    problem = s.screen(path).problem

    assert problem is not None
    assert "not finite numbers" in problem
    assert "frames 2" in problem


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["xtc", "trr", "dcd"])
def test_a_broken_box_is_caught_in_every_format(tmp_path, ext):
    coords, _, _ = healthy()
    dims = boxes(FRAMES)
    dims[4] = [50.0, 50.0, 50.0, 0.0, 90.0, 90.0]
    path = write_via_mdanalysis(tmp_path / f"box.{ext}", coords, dims)

    problem = s.screen(path).problem

    assert problem is not None
    assert "not a box" in problem


# --------------------------------------------------
def test_a_file_we_claim_to_read_but_cannot_open_is_refused(tmp_path):
    """
    The direction this must fail in, pinned by a test

    An earlier draft passed these through as "not screened". Then a leaked
    stub in another test module broke every MDAnalysis Universe in the
    process, and every damaged trajectory here passed while the suite stayed
    green. A guard that excuses itself when its reader breaks is not a guard.
    """

    path = tmp_path / "broken.xtc"
    path.write_bytes(b"this is not an xtc")

    verdict = s.screen(str(path))

    assert verdict.problem is not None
    assert "cannot be opened" in verdict.problem


# ==================================================
# Frame-to-frame jumps
#
# The proposed third coordinate rule (09-30, ddd host; kyclark-notes diary
# 2026-09-30, "DDD HOST"). A frame is refused when any atom moves MAX_JUMP
# (5,000 A) or more from the frame before -- or twice the frame's box
# diagonal, if that is larger -- unless the step is a boundary between two
# blocks of time. A one-frame block is never excused, and the absolute
# 1e6 A limit still covers every frame.
#
# Why: DDD's damaged last frames move whole protein chains by one constant
# offset of 1e4-1e7 A. Five of the twelve measured never reach 1e6 A, so
# MAX_ABS_COORD passes them. Honest data measured the same day moves far less:
# DDD at most 71.5 A per frame, contributor 289's AMBER runs 8-11 A, and the
# wrapped OpenMM run MDR00088497 up to 272 A, its box diagonal.
#
# These tests are written ahead of the code; until the rule exists they fail.
# ==================================================

CHAIN = 20  # the first CHAIN atoms play the protein chain the damage moves


# --------------------------------------------------
def steady(frames=FRAMES, atoms=ATOMS, box=50.0):
    """A trajectory whose atoms stay put apart from thermal noise"""

    rng = np.random.default_rng(88497)
    start = rng.uniform(5.0, box - 5.0, (atoms, 3))
    noise = rng.normal(0.0, 0.3, (frames, atoms, 3))
    coords = (start + noise).astype(np.float32)
    lengths = np.full((frames, 3), box, dtype=np.float64)
    angles = np.full((frames, 3), 90.0, dtype=np.float64)
    return coords, lengths, angles


# --------------------------------------------------
def direction():
    """A fixed direction for offsets, like the ones measured"""

    v = np.array([37144.0, 81529.0, 233850.0])
    return v / np.linalg.norm(v)


# --------------------------------------------------
def move_chain(coords, frame, distance):
    """Move the first CHAIN atoms of one frame by one constant vector"""

    coords[frame, :CHAIN] += (direction() * distance).astype(np.float32)


# --------------------------------------------------
def no_box(frames=FRAMES):
    """Dimensions for a trajectory with no periodic box, as a .mdc decodes"""

    return np.zeros((frames, 6), dtype=np.float32)


# --------------------------------------------------
# The rule, directly


# --------------------------------------------------
def test_the_jump_limit_has_a_floor_and_grows_with_the_box():
    """5,000 A, or twice the box diagonal when that is larger"""

    assert s.MAX_JUMP == 5.0e3
    assert s.jump_limit(None) == s.MAX_JUMP
    assert s.jump_limit(float("nan")) == s.MAX_JUMP
    assert s.jump_limit(0.0) == s.MAX_JUMP
    assert s.jump_limit(273.0) == s.MAX_JUMP
    assert s.jump_limit(5196.2) == pytest.approx(2 * 5196.2)


# --------------------------------------------------
def test_a_jump_on_a_regular_step_is_found():
    moves = [0.0, 4.0, 4.0, 4.0, 4.0, 250_425.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    jumps, boundaries = s.find_jumps(moves, times, None)

    assert jumps == [5]
    assert boundaries == []


# --------------------------------------------------
def test_a_jump_at_a_time_gap_is_excused():
    """Two joined runs: the second block starts at frame 3 and has 3 frames"""

    moves = [0.0, 4.0, 4.0, 20_000.0, 4.0, 4.0]
    times = [0.0, 10.0, 20.0, 5000.0, 5010.0, 5020.0]

    jumps, boundaries = s.find_jumps(moves, times, None)

    assert jumps == []
    assert boundaries == [3]


# --------------------------------------------------
def test_a_one_frame_block_at_the_end_is_not_excused():
    """
    A damaged last frame whose time is garbage too

    If a time gap excused any jump, a frame damaged in both its coordinates
    and its time would excuse itself. DDD's damaged last frames carry normal
    times, but nothing guarantees the next contributor's will.
    """

    moves = [0.0, 4.0, 4.0, 4.0, 4.0, 250_425.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 9.9e9]

    jumps, _ = s.find_jumps(moves, times, None)

    assert jumps == [5]


# --------------------------------------------------
def test_a_one_frame_block_in_the_middle_is_not_excused():
    """
    Frame 3 is damaged, time included, and frame 4 is back to normal

    Frame 3 is a block of one, so its jump stands. The step back into frame 4
    also looks like a boundary, and frames 4-5 are a real block, so that step
    is excused -- which is fine, because frame 3 already refuses the file.
    """

    moves = [0.0, 4.0, 4.0, 250_425.0, 250_425.0, 4.0]
    times = [0.0, 10.0, 20.0, -7.0e7, 40.0, 50.0]

    jumps, boundaries = s.find_jumps(moves, times, None)

    assert 3 in jumps
    assert 3 not in boundaries


# --------------------------------------------------
@pytest.mark.parametrize("times", [None, [0.0] * 6])
def test_with_no_usable_times_every_step_is_tested(times):
    """
    No time variable (MDR00048669), or every time zero (ticket 2339's frames)

    Nothing can show a gap, so nothing is excused. A wrong refusal is loud and
    gets fixed; a missed defect is silent.
    """

    moves = [0.0, 4.0, 4.0, 20_000.0, 4.0, 4.0]

    jumps, boundaries = s.find_jumps(moves, times, None)

    assert jumps == [3]
    assert boundaries == []


# --------------------------------------------------
def test_rounding_in_the_times_is_not_a_gap():
    """float32 times drift in the last digits; 1% of the step is allowed"""

    moves = [0.0] + [4.0] * 5
    times = [0.0, 10.0004, 19.9996, 30.0003, 40.0, 49.9998]

    _, boundaries = s.find_jumps(moves, times, None)

    assert boundaries == []


# --------------------------------------------------
def test_the_limit_follows_each_frames_box():
    moves = [0.0, 5196.0, 20_000.0]
    times = [0.0, 1.0, 2.0]
    diagonals = [5196.2, 5196.2, 5196.2]

    jumps, _ = s.find_jumps(moves, times, diagonals)

    assert jumps == [2]


# --------------------------------------------------
# The shapes that reached us, through screen()


# --------------------------------------------------
@pytest.mark.parametrize(
    "distance,source",
    [
        (15_425.0, "3fa3 Pro_lig30"),
        (20_603.0, "1m0o Pro_lig9"),
        (29_443.0, "4bzs Pro_lig25"),
        (178_431.0, "6of5 Pro_lig18"),
        (250_425.0, "6mlh Pro_lig33"),
    ],
)
def test_ddd_damaged_last_frames_the_old_screen_missed(tmp_path, distance, source):
    """
    The five of the twelve that stay under 1e6 A

    Median offsets as measured on 09-30. In each, the last frame moves the
    first chain(s) by one vector and leaves their shape intact.
    """

    coords, lengths, angles = steady()
    move_chain(coords, FRAMES - 1, distance)
    assert np.abs(coords).max() < s.MAX_ABS_COORD

    path = write_nc(tmp_path / "ddd.nc", coords, lengths, angles)
    problem = s.screen(path).problem

    assert problem is not None, source
    assert "from the frame before" in problem
    assert f"frames {FRAMES - 1}" in problem


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["xtc", "trr", "dcd"])
def test_ddd_damaged_last_frame_is_caught_in_every_format(tmp_path, ext):
    """A .mdc is decoded to XTC with no box before the screen sees it"""

    coords, _, _ = steady()
    move_chain(coords, FRAMES - 1, 250_425.0)
    path = write_via_mdanalysis(tmp_path / f"ddd.{ext}", coords, no_box())

    problem = s.screen(path).problem

    assert problem is not None
    assert "from the frame before" in problem
    assert f"frames {FRAMES - 1}" in problem


# --------------------------------------------------
def test_a_damaged_middle_frame_names_the_step_in_and_the_step_out(tmp_path):
    coords, lengths, angles = steady()
    move_chain(coords, 3, 20_000.0)

    path = write_nc(tmp_path / "middle.nc", coords, lengths, angles)
    problem = s.screen(path).problem

    assert problem is not None
    assert "frames 3-4" in problem


# --------------------------------------------------
def test_a_damaged_first_frame_is_named_by_the_step_after_it(tmp_path):
    """Frame 0 has no frame before it, so the jump shows at frame 1"""

    coords, lengths, angles = steady()
    move_chain(coords, 0, 20_000.0)

    path = write_nc(tmp_path / "first.nc", coords, lengths, angles)
    problem = s.screen(path).problem

    assert problem is not None
    assert "frames 1" in problem


# --------------------------------------------------
def test_a_jump_across_a_read_chunk_is_caught(tmp_path):
    """
    NetCDF is read CHUNK_FRAMES at a time

    The last frame of one chunk has to be carried into the next, or a jump at
    the first frame of a chunk is never compared with anything.
    """

    frames = s.CHUNK_FRAMES + 10
    coords, lengths, angles = steady(frames=frames)
    coords[s.CHUNK_FRAMES:, :CHAIN] += (direction() * 20_000.0).astype(np.float32)

    path = write_nc(tmp_path / "chunk.nc", coords, lengths, angles)
    problem = s.screen(path).problem

    assert problem is not None
    assert f"frames {s.CHUNK_FRAMES}" in problem


# --------------------------------------------------
# Honest data that must still pass


# --------------------------------------------------
def test_the_healthy_fixture_still_passes(clean):
    """
    healthy() places atoms at random each frame, so they move up to ~139 A
    between frames -- more than real data, and still far under the limit
    """

    assert s.screen(clean).problem is None


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc"])
def test_a_wrapped_trajectory_passes(tmp_path, ext):
    """
    MDR00088497: OpenMM, not unwrapped, 158 A box

    Every step some atom leaves one face and comes back through the opposite
    one; across a corner that is a whole box diagonal, 273 A.
    """

    box = 158.0
    coords, lengths, angles = steady(box=box)
    for frame in range(1, FRAMES, 2):
        coords[frame, 0] += box

    if ext == "nc":
        path = write_nc(tmp_path / "wrapped.nc", coords, lengths, angles)
    else:
        path = write_via_mdanalysis(
            tmp_path / "wrapped.xtc", coords, boxes(FRAMES, lengths=(box, box, box))
        )

    assert s.screen(path).problem is None


# --------------------------------------------------
def test_an_unwrapped_trajectory_far_from_the_origin_passes(tmp_path):
    """
    Unwrapped water can drift thousands of angstroms over microseconds

    Why the rule is a jump and not a lower absolute limit: these coordinates
    pass 20,000 A, and each step moves only 5 A.
    """

    coords, lengths, angles = steady()
    for frame in range(FRAMES):
        coords[frame] += np.float32(19_990.0 + 5.0 * frame)

    path = write_nc(tmp_path / "drift.nc", coords, lengths, angles)

    assert s.screen(path).problem is None


# --------------------------------------------------
def test_a_huge_box_raises_the_limit(tmp_path):
    """A wrap across a 3,000 A box is a 5,196 A move, over the 5,000 floor"""

    box = 3000.0
    coords, lengths, angles = steady(box=box)
    coords[3, 0] += box

    path = write_nc(tmp_path / "hugebox.nc", coords, lengths, angles)

    assert s.screen(path).problem is None


# --------------------------------------------------
def test_a_huge_box_does_not_excuse_what_it_cannot_explain(tmp_path):
    box = 3000.0
    coords, lengths, angles = steady(box=box)
    move_chain(coords, FRAMES - 1, 20_000.0)

    path = write_nc(tmp_path / "hugebox-bad.nc", coords, lengths, angles)
    problem = s.screen(path).problem

    assert problem is not None
    assert "from the frame before" in problem


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr"])
@pytest.mark.parametrize(
    "times",
    [
        [0.0, 10.0, 20.0, 5000.0, 5010.0, 5020.0],
        [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
    ],
    ids=["gap", "clock-restart"],
)
def test_joined_runs_pass_and_the_note_says_so(tmp_path, ext, times):
    """
    Two runs in one file: between them the system can be anywhere

    The second block is moved 20,000 A. The pass must be visible in the log,
    so the note names the time segments the jump test did not cross.
    """

    coords, lengths, angles = steady()
    coords[3:, :CHAIN] += (direction() * 20_000.0).astype(np.float32)

    if ext == "nc":
        path = write_nc(tmp_path / "joined.nc", coords, lengths, angles, times)
    else:
        path = write_via_mdanalysis(
            tmp_path / f"joined.{ext}", coords, boxes(FRAMES), times
        )

    verdict = s.screen(path)

    assert verdict.problem is None
    assert "time segments" in verdict.note


# --------------------------------------------------
def test_dcd_cannot_carry_a_gap_so_its_jumps_are_refused(tmp_path):
    """
    DCD stores a start and an interval, never per-frame times

    Whatever times the writer was given, every step reads back as regular.
    """

    coords, _, _ = steady()
    coords[3:, :CHAIN] += (direction() * 20_000.0).astype(np.float32)
    times = [0.0, 10.0, 20.0, 5000.0, 5010.0, 5020.0]
    path = write_via_mdanalysis(tmp_path / "joined.dcd", coords, boxes(FRAMES), times)

    problem = s.screen(path).problem

    assert problem is not None
    assert "frames 3" in problem


# --------------------------------------------------
@pytest.mark.parametrize(
    "times",
    [
        [0.0, 1.0, 2.0, 3.0, 4.0, 9.9e9],
        [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
    ],
    ids=["garbage-time", "zero-time"],
)
def test_a_damaged_last_frame_with_a_damaged_time_is_still_caught(tmp_path, times):
    coords, lengths, angles = steady()
    move_chain(coords, FRAMES - 1, 250_425.0)

    path = write_nc(tmp_path / "lasttime.nc", coords, lengths, angles, times)
    problem = s.screen(path).problem

    assert problem is not None
    assert f"frames {FRAMES - 1}" in problem


# --------------------------------------------------
def test_no_time_variable_tests_every_step_and_says_so(tmp_path):
    """
    Contributor 289's MDR00048669 has no time variable at all

    A clean file passes, with the note saying no time was available; a jump
    anywhere is refused, because nothing can show it is a gap.
    """

    coords, lengths, angles = steady()
    path = write_nc(tmp_path / "notime.nc", coords, lengths, angles, times=None)

    verdict = s.screen(path)
    assert verdict.problem is None
    assert "no usable frame times" in verdict.note

    coords[3:, :CHAIN] += (direction() * 20_000.0).astype(np.float32)
    path = write_nc(tmp_path / "notime-jump.nc", coords, lengths, angles, times=None)

    problem = s.screen(path).problem
    assert problem is not None
    assert "frames 3" in problem


# --------------------------------------------------
def test_mdr00072113_last_frame_is_caught_by_its_box(tmp_path):
    """
    Contributor 289, 215 frames: the last frame's box is 0,0,0 and 844 of 860
    atoms moved ~173 A. Public since before the screen existed. The cell rule
    catches it today; this pins it so the jump rule cannot be credited with
    it, or break it.
    """

    coords, lengths, angles = steady(box=52.3)
    coords[FRAMES - 1] += np.float32(100.0)
    lengths[FRAMES - 1] = 0.0

    path = write_nc(tmp_path / "72113.nc", coords, lengths, angles)
    problem = s.screen(path).problem

    assert problem is not None
    assert "not a box" in problem
    assert f"frames {FRAMES - 1}" in problem
