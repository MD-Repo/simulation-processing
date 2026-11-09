"""Tests for the trajectory rewrites: rescale_times.py, set_dcd_spacing.py,
cut_replicates.py, join_trajectories.py, split_at_resets.py, and the frame
walkers in trajframes.py they share.

Each file is built here (tests/trajbuild.py), small enough to hold raw
coordinates, with every coordinate carrying its frame number so a piece
can be told by what it holds. The promise under test is the Go code's:
only time fields are rewritten, frames are copied byte for byte, and a
piece holds exactly the frames it should. Ported from mdrepo-agent's
internal/traj tests when the rewrites moved here (mdrepo-agent#98).
"""

import io
import json
import os
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import trajframes as tf  # noqa: E402
import rescale_times  # noqa: E402
import set_dcd_spacing  # noqa: E402
import cut_replicates  # noqa: E402
import join_trajectories  # noqa: E402
import split_at_resets  # noqa: E402
import trajbuild as tb  # noqa: E402

HERE = os.path.dirname(rescale_times.__file__)


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def run_of(path):
    return tf.read_run(path)


def frame_coords(path):
    """The first coordinate of every frame, which is its frame number in
    the file it was built in."""
    run = tf.read_run(path)
    out = []
    with open(path, "rb") as fh:
        for i in range(len(run.frames)):
            a, _ = run.byte_range(i, i + 1)
            if run.format == "xtc":
                fh.seek(a + tf.XTC_HEADER + tf.XTC_BOX + 4)
                out.append(struct.unpack(">f", fh.read(4))[0])
            elif run.format == "trr":
                real = 8 if run.frames[i].wide else 4
                fh.seek(a + tf.TRR_FIXED_HEADER + 2 * real + 9 * real)
                out.append(
                    struct.unpack(">" + ("d" if real == 8 else "f"), fh.read(real))[0]
                )
            else:
                fh.seek(a + 4)
                out.append(struct.unpack(run.dcd.order + "f", fh.read(4))[0])
    return out


# --------------------------------------------------
def test_identify_by_the_first_bytes(tmp_path):
    assert tf.identify(write(tmp_path, "a.xtc", tb.xtc([0.0]))) == "xtc"
    assert tf.identify(write(tmp_path, "a.trr", tb.trr([0.0]))) == "trr"
    assert tf.identify(write(tmp_path, "a.dcd", tb.dcd(3, 1))) == "dcd"
    assert tf.identify(write(tmp_path, "b.dcd", tb.dcd(3, 1, little=False))) == "dcd"
    assert tf.identify(write(tmp_path, "a.txt", b"hello world")) is None
    d = tmp_path / "run_trj"
    d.mkdir()
    (d / "clickme.dtr").write_bytes(b"")
    assert tf.identify(str(d)) == "dtr"


def test_walkers_find_every_frame_and_its_stamp(tmp_path):
    x = run_of(write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0, 20.0])))
    assert x.times == [0.0, 10.0, 20.0] and x.skip == tf.XTC_HEADER
    assert [f.start for f in x.frames] == [0, 92, 184] and x.frames[-1].end == 276

    t = run_of(write(tmp_path, "a.trr", tb.trr([0.0, 5.0], double=True)))
    assert t.times == [0.0, 5.0] and all(f.wide for f in t.frames)
    assert t.skip == tf.TRR_FIXED_HEADER + 16

    d = run_of(write(tmp_path, "a.dcd", tb.dcd(4, 3)))
    assert d.times is None and len(d.frames) == 3 and d.dcd.natoms == 4
    assert len(d.head) == d.dcd.header_bytes


def test_a_truncated_xtc_is_damage(tmp_path):
    data = tb.xtc([0.0, 10.0])
    p = write(tmp_path, "a.xtc", data[:-7])
    with pytest.raises(tf.BadTrajectory, match="truncated"):
        tf.read_run(p)


def test_part_of_a_frame_header_is_damage(tmp_path):
    """One to fifteen bytes of a header after the last whole frame: a writer
    that stopped mid-frame, not a clean end. Only an empty read ends a
    file; otherwise a cut or rescale would publish the truncation."""
    data = tb.xtc([0.0, 10.0])
    p = write(tmp_path, "a.xtc", data + data[:7])
    with pytest.raises(tf.BadTrajectory, match="truncated at frame 2"):
        tf.read_run(p)
    data = tb.trr([0.0, 10.0])
    p = write(tmp_path, "a.trr", data + data[:10])
    with pytest.raises(tf.BadTrajectory, match="truncated at frame 2"):
        tf.read_run(p)


# --------------------------------------------------
def test_rescale_rewrites_only_the_xtc_time_fields(tmp_path):
    data = tb.xtc([0.0, 0.01, 0.02, 0.03])
    p = write(tmp_path, "a.xtc", data)

    report = rescale_times.rescale(p, 1000)

    assert report == {"format": "xtc", "frames": 4}
    after = open(p, "rb").read()
    assert tb.xtc_times(after) == pytest.approx([0, 10, 20, 30])
    # Everything but the four time fields is as it was.
    masked = bytearray(after)
    before = bytearray(data)
    for f in tf.walk_xtc(io.BytesIO(data), len(data), "a"):
        masked[f.time_offset : f.time_offset + 4] = b"\0\0\0\0"
        before[f.time_offset : f.time_offset + 4] = b"\0\0\0\0"
    assert masked == before


@pytest.mark.parametrize("double", [False, True])
def test_rescale_rewrites_trr_stamps_in_either_precision(tmp_path, double):
    p = write(tmp_path, "a.trr", tb.trr([1.0, 2.0, 3.0], double=double))
    assert rescale_times.rescale(p, 1000)["frames"] == 3
    assert tf.read_run(p).times == pytest.approx([1000, 2000, 3000])
    assert frame_coords(p) == [1.0, 2.0, 3.0]


def test_rescale_refuses_a_dcd_and_a_dtr(tmp_path):
    with pytest.raises(tf.Refused, match="DCD"):
        rescale_times.rescale(write(tmp_path, "a.dcd", tb.dcd(3, 2)), 1000)
    d = tmp_path / "run_trj"
    d.mkdir()
    (d / "clickme.dtr").write_bytes(b"")
    with pytest.raises(tf.Refused, match="DTR"):
        rescale_times.rescale(str(d), 1000)


def test_rescale_command_reports_a_refusal_with_exit_1(tmp_path):
    p = write(tmp_path, "a.dcd", tb.dcd(3, 2))
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "rescale_times.py"), "--trajectory", p, "--factor", "1000"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 1
    report = json.loads(r.stdout.strip().splitlines()[-1])
    assert report["format"] == "dcd" and "DCD" in report["refused"]
    assert "DCD" in r.stderr.strip().splitlines()[-1]

    p = write(tmp_path, "a.xtc", tb.xtc([0.0, 1.0]))
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "rescale_times.py"), "--trajectory", p, "--factor", "10"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {
        "format": "xtc",
        "frames": 2,
    }


def test_rescale_rewrites_an_amber_netcdf_time_variable(tmp_path):
    netcdf_file = pytest.importorskip("scipy.io").netcdf_file
    import numpy as np

    p = str(tmp_path / "a.nc")
    with netcdf_file(p, "w", version=1) as f:
        f.createDimension("frame", None)
        f.createDimension("atom", 2)
        f.createDimension("spatial", 3)
        c = f.createVariable("coordinates", "f", ("frame", "atom", "spatial"))
        t = f.createVariable("time", "f", ("frame",))
        t.units = b"picosecond"
        c[:] = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
        t[:] = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    assert tf.identify(p) == "netcdf"

    assert rescale_times.rescale(p, 1000) == {"format": "netcdf", "frames": 3}

    with netcdf_file(p, "r") as f:
        assert f.variables["time"][:].tolist() == pytest.approx([1, 2, 3])
        assert f.variables["coordinates"][:].flatten().tolist() == list(range(18))


def test_rescale_refuses_a_netcdf_time_in_nanoseconds(tmp_path):
    netcdf_file = pytest.importorskip("scipy.io").netcdf_file
    p = str(tmp_path / "a.nc")
    with netcdf_file(p, "w") as f:
        f.createDimension("frame", None)
        t = f.createVariable("time", "d", ("frame",))
        t.units = b"nanosecond"
        t[:] = [0.0, 1.0]
    with pytest.raises(tf.Refused, match="nanosecond"):
        rescale_times.rescale(p, 1000)


def test_rescale_rewrites_an_hdf5_time_dataset_compressed_or_not(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    p = str(tmp_path / "a.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("coordinates", data=np.zeros((3, 2, 3), np.float32))
        t = f.create_dataset("time", data=np.array([0.001, 0.002, 0.003], np.float32))
        t.attrs["units"] = "picoseconds"
    assert tf.identify(p) == "hdf5"
    assert rescale_times.rescale(p, 1000) == {"format": "hdf5", "frames": 3}
    with h5py.File(p, "r") as f:
        assert f["time"][:].tolist() == pytest.approx([1, 2, 3])

    # A time dataset behind a compression filter, as a NetCDF-4 writer
    # leaves it: h5py re-encodes its chunks and the coordinates' stay.
    q = str(tmp_path / "z.h5")
    with h5py.File(q, "w") as f:
        f.create_dataset(
            "coordinates",
            data=np.arange(18, dtype=np.float32).reshape(3, 2, 3),
            compression="gzip",
        )
        f.create_dataset(
            "time",
            data=np.array([1.0, 2.0, 3.0], np.float32),
            compression="gzip",
            shuffle=True,
        )
    assert rescale_times.rescale(q, 1000) == {"format": "hdf5", "frames": 3}
    with h5py.File(q, "r") as f:
        assert f["time"][:].tolist() == pytest.approx([1000, 2000, 3000])
        assert f["coordinates"][:].flatten().tolist() == list(range(18))
        assert f["time"].compression == "gzip", "the filter stays as it was"

    z = str(tmp_path / "int.h5")
    with h5py.File(z, "w") as f:
        f.create_dataset("coordinates", data=np.zeros((2, 2, 3), np.float32))
        f.create_dataset("time", data=np.array([0, 1], np.int64))
    with pytest.raises(tf.Refused, match="integers"):
        rescale_times.rescale(z, 1000)

    r = str(tmp_path / "h5md.h5")
    with h5py.File(r, "w") as f:
        g = f.create_group("particles/all/position")
        g.create_dataset("value", data=np.zeros((2, 2, 3), np.float32))
        g.create_dataset("time", data=np.array([0.0, 0.5], np.float64))
    assert rescale_times.rescale(r, 2) == {"format": "hdf5", "frames": 2}
    with h5py.File(r, "r") as f:
        assert f["particles/all/position/time"][:].tolist() == [0.0, 1.0]


# --------------------------------------------------
def test_set_dcd_spacing_rewrites_the_time_step(tmp_path):
    p = write(tmp_path, "a.dcd", tb.dcd(3, 2, delta=1.0, nsavc=1))
    report = set_dcd_spacing.set_spacing(p, 100)
    assert report["placeholder"] and report["was_ps"] == pytest.approx(tf.AKMA_PS)
    with open(p, "rb") as fh:
        h = tf.read_dcd_header(fh, "a.dcd")
    assert h.declared_ps_per_frame() == pytest.approx(100, rel=1e-6)
    assert frame_coords(p) == [1.0, 2.0]

    q = write(tmp_path, "b.dcd", tb.dcd(3, 2, delta=20.4548, nsavc=5000))
    report = set_dcd_spacing.set_spacing(q, 10)
    assert not report["placeholder"] and report["was_ps"] == pytest.approx(
        5000 * 20.4548 * tf.AKMA_PS
    )


# --------------------------------------------------
@pytest.mark.parametrize("fmt", ["xtc", "trr", "dcd"])
def test_cut_regroups_whole_frames(tmp_path, fmt):
    data = {"xtc": tb.xtc([float(i) for i in range(6)]), "trr": tb.trr([float(i) for i in range(6)]),
            "dcd": tb.dcd(3, 6)}[fmt]  # fmt: skip
    p = write(tmp_path, "all." + fmt, data)
    outs = [str(tmp_path / f"rep{i}.{fmt}") for i in range(3)]

    report = cut_replicates.cut(p, outs)

    assert report["frames"] == 6 and report["frames_each"] == 2
    for i, o in enumerate(outs):
        assert frame_coords(o) == [2 * i + 1, 2 * i + 2]
        run = tf.read_run(o)
        assert len(run.frames) == 2
        if fmt == "dcd":
            assert run.dcd.nset == 2, "the header states the frames that follow it"


def test_cut_refuses_frames_that_do_not_divide(tmp_path):
    p = write(tmp_path, "a.xtc", tb.xtc([0.0, 1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(tf.BadTrajectory, match="do not divide"):
        cut_replicates.cut(p, [str(tmp_path / "a"), str(tmp_path / "b")])


# --------------------------------------------------
def test_join_drops_the_frame_a_segment_repeats(tmp_path):
    a = write(tmp_path, "md.part0001.xtc", tb.xtc([0.0, 10.0, 20.0]))
    # The second segment begins with the frame the first ended on: same
    # time, same coordinates.
    b = write(
        tmp_path, "md.part0002.xtc", tb.xtc([20.0, 30.0, 40.0], coord=lambda f: f + 3)
    )
    out = str(tmp_path / "md.xtc")

    report = join_trajectories.join([a, b], out)

    assert report == {"format": "xtc", "frames": 5, "repeated": 1}
    assert frame_coords(out) == [1, 2, 3, 4, 5]
    assert tf.read_run(out).times == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_join_knows_a_repeated_dcd_frame_by_its_coordinates(tmp_path):
    a = write(tmp_path, "a.dcd", tb.dcd(3, 3))
    b = write(tmp_path, "b.dcd", tb.dcd(3, 2, coord=lambda f: f + 3))
    out = str(tmp_path / "ab.dcd")
    report = join_trajectories.join([a, b], out)
    assert report["frames"] == 4 and report["repeated"] == 1
    assert frame_coords(out) == [1, 2, 3, 4]
    assert tf.read_run(out).dcd.nset == 4


def test_join_refuses_segments_out_of_order_or_of_two_formats(tmp_path):
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0]))
    b = write(tmp_path, "b.xtc", tb.xtc([5.0, 15.0]))
    with pytest.raises(tf.BadTrajectory, match="not consecutive"):
        join_trajectories.join([a, b], str(tmp_path / "ab.xtc"))
    assert not (tmp_path / "ab.xtc").exists()
    c = write(tmp_path, "c.trr", tb.trr([20.0]))
    with pytest.raises(tf.BadTrajectory, match="one format"):
        join_trajectories.join([a, c], str(tmp_path / "ac.xtc"))


# --------------------------------------------------
def test_split_cuts_at_resets_and_drops_repeated_frames(tmp_path):
    # Two clones of 0-20 ps, the second repeating its 10 ps frame.
    times = [0.0, 10.0, 20.0, 0.0, 10.0, 10.0, 20.0]
    p = write(tmp_path, "clones.xtc", tb.xtc(times))

    report = split_at_resets.split(p, str(tmp_path / "out" / "clones_{k}of{n}.xtc"))

    assert report["split"] and report["frames"] == [3, 3] and report["dropped"] == 1
    assert report["resets"] == 1 and report["repeated"] == 1 and report["overlaps"] == 0
    assert [os.path.basename(x) for x in report["pieces"]] == [
        "clones_1of2.xtc",
        "clones_2of2.xtc",
    ]
    assert frame_coords(report["pieces"][0]) == [1, 2, 3]
    assert frame_coords(report["pieces"][1]) == [
        4,
        5,
        7,
    ], "the repeated frame 6 is left out"
    assert tf.read_run(report["pieces"][1]).times == [0.0, 10.0, 20.0]


def test_split_leaves_out_replayed_frames_after_an_overlap(tmp_path):
    # A run resumed from a checkpoint before the frame it last wrote: 30
    # back to 20, then 30 again, both already kept.
    times = [0.0, 10.0, 20.0, 30.0, 20.0, 30.0, 40.0]
    p = write(tmp_path, "resumed.xtc", tb.xtc(times))
    report = split_at_resets.split(p, str(tmp_path / "r_{k}of{n}.xtc"))
    assert (
        report["frames"] == [5] and report["dropped"] == 2 and report["overlaps"] == 1
    )
    assert frame_coords(report["pieces"][0]) == [1, 2, 3, 4, 7]


def test_split_with_nothing_to_split_writes_nothing(tmp_path):
    p = write(tmp_path, "one.xtc", tb.xtc([0.0, 10.0, 20.0]))
    report = split_at_resets.split(p, str(tmp_path / "one_{k}of{n}.xtc"))
    assert report["split"] is False and report["frame_count"] == 3
    assert "no time reset" in report["reason"]
    assert list(tmp_path.glob("one_*")) == []


def test_split_refuses_a_backward_step_to_a_time_never_kept_and_a_dcd(tmp_path):
    p = write(tmp_path, "odd.xtc", tb.xtc([0.0, 10.0, 20.0, 15.0, 30.0]))
    with pytest.raises(tf.BadTrajectory, match="backward time step"):
        split_at_resets.split(p, str(tmp_path / "odd_{k}of{n}.xtc"))
    d = write(tmp_path, "a.dcd", tb.dcd(3, 4))
    with pytest.raises(tf.BadTrajectory, match="stamps no frame times"):
        split_at_resets.split(d, str(tmp_path / "d_{k}of{n}.dcd"))


def test_split_command_reports_json(tmp_path):
    p = write(tmp_path, "c.xtc", tb.xtc([0.0, 1.0, 0.0, 1.0]))
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "split_at_resets.py"), "--trajectory", p,
         "--out-template", str(tmp_path / "c_{k}of{n}.xtc")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout.strip().splitlines()[-1])
    assert report["split"] and report["frames"] == [2, 2]
    assert all(os.path.exists(x) for x in report["pieces"])


# --------------------------------------------------
# What the first review of these commands found (simulation-processing#34).


def test_join_refuses_segments_of_different_atoms_or_precision(tmp_path):
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0], natoms=3))
    b = write(tmp_path, "b.xtc", tb.xtc([20.0, 30.0], natoms=5))
    with pytest.raises(tf.BadTrajectory, match="3 atoms a frame and b.xtc 5"):
        join_trajectories.join([a, b], str(tmp_path / "ab.xtc"))
    c = write(tmp_path, "c.trr", tb.trr([0.0, 10.0]))
    d = write(tmp_path, "d.trr", tb.trr([20.0, 30.0], double=True))
    with pytest.raises(tf.BadTrajectory, match="different precision"):
        join_trajectories.join([c, d], str(tmp_path / "cd.trr"))


def test_join_refuses_dcds_sampled_differently(tmp_path):
    """The joined DCD keeps the first segment's header, which times every
    frame by its save interval and time step: a segment saved at another
    spacing would be mistimed."""
    a = write(tmp_path, "a.dcd", tb.dcd(3, 2))
    b = write(tmp_path, "b.dcd", tb.dcd(3, 2, nsavc=5, coord=lambda f: f + 3))
    with pytest.raises(tf.BadTrajectory, match="every 1 steps .* every 5"):
        join_trajectories.join([a, b], str(tmp_path / "ab.dcd"))
    c = write(tmp_path, "c.dcd", tb.dcd(3, 2, delta=40.9096, coord=lambda f: f + 3))
    with pytest.raises(tf.BadTrajectory, match="40.9096"):
        join_trajectories.join([a, c], str(tmp_path / "ac.dcd"))
    assert not (tmp_path / "ab.dcd").exists() and not (tmp_path / "ac.dcd").exists()


def test_cut_refuses_one_file_named_for_two_pieces(tmp_path):
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0]))
    x = str(tmp_path / "x.xtc")
    with pytest.raises(ValueError, match="named for two pieces"):
        cut_replicates.cut(a, [x, x])
    with pytest.raises(ValueError, match="are one file, named for two pieces"):
        cut_replicates.cut(a, [x, str(tmp_path / "sub" / ".." / "x.xtc")])
    assert not os.path.exists(x)


def test_a_failed_split_removes_the_pieces_it_wrote(tmp_path):
    """The second piece cannot be written (a directory sits at its path):
    the first, already written, goes with it, so no partial set is left
    looking like a split."""
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0, 0.0, 10.0]))
    (tmp_path / "run_2.xtc").mkdir()
    with pytest.raises(OSError):
        split_at_resets.split(a, str(tmp_path / "run_{k}.xtc"))
    assert not (tmp_path / "run_1.xtc").exists()
    assert not (tmp_path / "run_1.xtc.part").exists()


def test_join_does_not_drop_a_frame_on_a_matching_placeholder_stamp(tmp_path):
    """Every frame stamped 0 ps, as a release of placeholders is: the
    segments' end frames differ in coordinates, so nothing is a repeat."""

    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 0.0, 0.0]))
    b = write(tmp_path, "b.xtc", tb.xtc([0.0, 0.0], coord=lambda f: f + 4))
    out = str(tmp_path / "ab.xtc")
    report = join_trajectories.join([a, b], out)
    assert report["frames"] == 5 and report["repeated"] == 0
    assert frame_coords(out) == [1, 2, 3, 4, 5]


def test_an_output_that_names_an_input_is_refused_and_the_input_kept(tmp_path):
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0]))
    b = write(tmp_path, "b.xtc", tb.xtc([20.0, 30.0], coord=lambda f: f + 3))
    with pytest.raises(tf.BadTrajectory, match="cannot be written over"):
        join_trajectories.join([a, b], a)
    assert frame_coords(a) == [1, 2], "the input is as it was"
    with pytest.raises(tf.BadTrajectory, match="cannot be written over"):
        cut_replicates.cut(a, [a, str(tmp_path / "c.xtc")])
    assert frame_coords(a) == [1, 2]


def test_a_failed_run_removes_only_what_it_created(tmp_path):
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0, 20.0]))  # 3 frames: no even cut
    pre = tmp_path / "pre.xtc"
    pre.write_bytes(b"not ours")
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "cut_replicates.py"), "--trajectory", a,
         "--out", str(pre), "--out", str(tmp_path / "x2.xtc")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 1
    assert pre.read_bytes() == b"not ours"
    assert not (tmp_path / "x2.xtc").exists()


def test_a_piece_is_written_whole_or_not_at_all(tmp_path):
    a = write(tmp_path, "a.xtc", tb.xtc([0.0, 10.0]))
    run = tf.read_run(a)
    dest = str(tmp_path / "out" / "p.xtc")
    with pytest.raises(tf.BadTrajectory, match="ended before"):
        tf.write_piece(dest, run, 2, [(a, 0, run.size + 100)])
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")


def test_set_dcd_spacing_refuses_a_header_with_no_save_interval(tmp_path):
    p = write(tmp_path, "a.dcd", tb.dcd(3, 2, delta=1.0, nsavc=0))
    with pytest.raises(tf.Refused, match="save interval is 0"):
        set_dcd_spacing.set_spacing(p, 100)
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "set_dcd_spacing.py"), "--trajectory", p, "--ps", "100"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 1
    assert "refused" in json.loads(r.stdout.strip().splitlines()[-1])
