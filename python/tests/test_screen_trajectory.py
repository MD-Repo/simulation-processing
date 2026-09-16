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
def write_nc(path, coords, lengths, angles):
    """Write a minimal AMBER-convention NetCDF trajectory"""

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

    var = out.createVariable("time", "f", ("frame",))
    var.units = "picosecond"
    var[:] = np.arange(coords.shape[0], dtype=np.float32)

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
def write_via_mdanalysis(path, coords, dimensions):
    """Write a trajectory in whatever format the extension names"""

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mda.Universe.empty(coords.shape[1], trajectory=True)
    universe.load_new(coords, format=MemoryReader, dimensions=dimensions)
    with mda.Writer(str(path), coords.shape[1]) as writer:
        for _ in universe.trajectory:
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
