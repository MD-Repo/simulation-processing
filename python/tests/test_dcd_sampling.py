"""Tests for the DCD frame-spacing recovery in cpptraj_gmx_traj_manipulation.py

Why this exists: cpptraj reports a DCD as holding only `coords`, so
trajectory_has_time_axis() returns None ("unknown"), and mdr-process treats
unknown as "trust the measurement". But conversion to XTC stamps cpptraj's
default 1 ps/frame onto the output, so the measurement is of a number cpptraj
invented. MDR00084214 is 100 ps/frame and 200 ns; the pipeline recorded
1 ps/frame and 2 ns, and 78 of the 80 DCD-sourced simulations in prod carry the
same fabricated 0.001 ns/frame.

dcd_sampling_ps() breaks that chain by reading DELTA and NSAVC out of the DCD
header, which cpptraj never exposes. The headers here are synthesised rather
than committed as fixtures: the parser reads the first 92 bytes and nothing
else, so a real multi-gigabyte trajectory would test exactly the same code.

The module is loaded with pytraj/parmed stubbed. Those are real dependencies of
the script, but they live in the simproc conda env that mdr-process runs it
under, while this suite runs in .venv -- which has pytest and no pytraj. The
functions under test touch neither.
"""

import importlib.util
import struct
import sys
import types

import pytest

# One AKMA time unit in ps -- the same constant the module uses, restated here
# so a typo in the module cannot silently agree with itself.
AKMA_TO_PS = 4.888821e-2


def _load_module():
    """Import the script with its conda-only dependencies stubbed out."""

    for name in ("pytraj", "parmed"):
        sys.modules.setdefault(name, types.ModuleType(name))
    spec = importlib.util.spec_from_file_location(
        "cpptraj_gmx_traj_manipulation", "cpptraj_gmx_traj_manipulation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cg = _load_module()


def make_dcd_header(
    nsavc=25000, delta=0.0818193182349205, charmm_version=24, endian="<", nset=2000
):
    """A 92-byte DCD header, the only part dcd_sampling_ps() reads.

    Offsets are from readdcd.h: 84 magic, "CORD", NSET, ISTART, NSAVC, then
    DELTA at 44 (float32 for CHARMM, float64 for X-PLOR) and the CHARMM
    version at 84. Defaults are MDR00084214's real values.
    """

    buf = bytearray(92)
    struct.pack_into(endian + "i", buf, 0, 84)
    buf[4:8] = b"CORD"
    struct.pack_into(endian + "i", buf, 8, nset)
    struct.pack_into(endian + "i", buf, 12, nsavc)  # ISTART
    struct.pack_into(endian + "i", buf, 16, nsavc)
    if charmm_version:
        struct.pack_into(endian + "f", buf, 44, delta)
    else:
        struct.pack_into(endian + "d", buf, 44, delta)
    struct.pack_into(endian + "i", buf, 84, charmm_version)
    return bytes(buf)


def write_dcd(tmp_path, name="t.dcd", **kwargs):
    path = tmp_path / name
    path.write_bytes(make_dcd_header(**kwargs))
    return str(path)


def test_recovers_the_real_spacing_of_mdr00084214(tmp_path):
    """The case that motivated all of this: 4 fs timestep, saved every 25000."""

    got = cg.dcd_sampling_ps(write_dcd(tmp_path))

    # MDAnalysis, reading the real 8.5 GB file, returns exactly this.
    assert got == pytest.approx(100.00000029814058, rel=1e-12)

    # And the value is self-consistent: 2000 frames at 100 ps is the 200 ns the
    # header's NSTEP x DELTA also gives.
    assert got * 25000 / 25000 == pytest.approx(100.0, rel=1e-6)


def test_charmm_and_xplor_store_delta_at_different_widths(tmp_path):
    """X-PLOR widens DELTA to a double over the offset CHARMM uses for a float.

    Reading the wrong width does not give a slightly wrong answer, it gives
    garbage, so the CHARMM-version flag has to be honoured.
    """

    charmm = cg.dcd_sampling_ps(write_dcd(tmp_path, "c.dcd", charmm_version=24))
    xplor = cg.dcd_sampling_ps(write_dcd(tmp_path, "x.dcd", charmm_version=0))
    assert charmm == pytest.approx(xplor, rel=1e-6)


def test_reads_big_endian_files(tmp_path):
    """DCDs get copied between machines; the leading 84 is the only tell."""

    little = cg.dcd_sampling_ps(write_dcd(tmp_path, "le.dcd", endian="<"))
    big = cg.dcd_sampling_ps(write_dcd(tmp_path, "be.dcd", endian=">"))
    assert big == pytest.approx(little, rel=1e-12)


def test_spacing_scales_with_nsavc(tmp_path):
    """Saving twice as often halves the spacing."""

    once = cg.dcd_sampling_ps(write_dcd(tmp_path, "a.dcd", nsavc=25000))
    twice = cg.dcd_sampling_ps(write_dcd(tmp_path, "b.dcd", nsavc=12500))
    assert twice == pytest.approx(once / 2, rel=1e-9)


@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"nsavc": 0}, "a frame saved every zero steps is not a spacing"),
        ({"nsavc": -5}, "negative NSAVC"),
        ({"delta": 0.0}, "a zero timestep would make the duration zero"),
        ({"delta": -1.0}, "negative timestep"),
    ],
)
def test_rejects_headers_that_would_poison_the_arithmetic(tmp_path, kwargs, why):
    """None falls back to the declaration; a bad number becomes a public record."""

    assert cg.dcd_sampling_ps(write_dcd(tmp_path, **kwargs)) is None, why


def test_rejects_files_that_are_not_dcds(tmp_path):
    bad_magic = bytearray(make_dcd_header())
    struct.pack_into("<i", bad_magic, 0, 12345)
    (tmp_path / "bad_magic.dcd").write_bytes(bytes(bad_magic))
    assert cg.dcd_sampling_ps(str(tmp_path / "bad_magic.dcd")) is None

    no_cord = bytearray(make_dcd_header())
    no_cord[4:8] = b"XXXX"
    (tmp_path / "no_cord.dcd").write_bytes(bytes(no_cord))
    assert cg.dcd_sampling_ps(str(tmp_path / "no_cord.dcd")) is None

    (tmp_path / "truncated.dcd").write_bytes(make_dcd_header()[:40])
    assert cg.dcd_sampling_ps(str(tmp_path / "truncated.dcd")) is None


def test_source_sampling_ps_only_claims_formats_it_can_read(tmp_path):
    """Recovery is a DCD-only affair, and a missing file is not an error."""

    dcd = write_dcd(tmp_path)
    assert cg.source_sampling_ps(dcd) is not None

    # An XTC keeps its own time axis through conversion; nothing to recover.
    xtc = tmp_path / "t.xtc"
    xtc.write_bytes(make_dcd_header())
    assert cg.source_sampling_ps(str(xtc)) is None

    assert cg.source_sampling_ps(str(tmp_path / "gone.dcd")) is None
    assert cg.source_sampling_ps(None) is None


def test_marker_line_is_what_mdr_process_parses(capsys):
    """The contract with parse_sampling_marker() in mdr-process/src/process.rs."""

    cg.report_source_sampling(100.00000029814058)
    line = capsys.readouterr().out.strip()
    assert line == "[mdrepo] source_sampling_ps=100.00000029814058"
    # Round-trips at full precision -- the Rust side parses this as f64.
    assert float(line.split("=")[1]) == 100.00000029814058

    cg.report_source_sampling(None)
    assert capsys.readouterr().out.strip() == "[mdrepo] source_sampling_ps=unknown"


def test_time_action_emits_nothing_when_nothing_was_recovered():
    """Guessing a spacing here would reproduce the bug with a different constant."""

    assert cg.time_action(None) == ""
    assert cg.time_action(100.0) == "time time0 0 dt 100.0\n"
