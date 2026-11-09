"""A trajectory with no periodic box must be written with a zero box.

MDR00105222 is a CA-only CHARMM PSF + DCD whose DCD header says it has no unit
cell (ICNTRL[10] == 0). Converted with the standalone cpptraj 6.4.4 that
replaced simproc on 2026-11-06, every one of its 20,000 frames came out of
full.xtc and minimal.xtc with the box [inf, inf, 0, 2.06, -inf, -inf, ...];
simproc's build of the same cpptraj version had written zeros. has_box()
judged the run box-less correctly, and the script then wrote nothing about the
box -- it only skipped `autoimage` -- so cpptraj filled the XTC box itself.
`box nobox` makes it write zeros.

The fixture inputs/nobox_dcd/ is that simulation's PSF and the first 3 frames
of its DCD (header patched to NSET = 3), 3,732 bytes, which reproduces it.
The tests that run cpptraj are skipped where cpptraj is not on PATH.
"""

import os
import re
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cpptraj_gmx_traj_manipulation as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "inputs", "nobox_dcd")
PSF = os.path.join(FIXTURE, "ca.psf")
DCD = os.path.join(FIXTURE, "ca3.dcd")
needs_cpptraj = pytest.mark.skipif(shutil.which("cpptraj") is None,
                                   reason="cpptraj not on PATH")


def boxes(xtc):
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    with XTCFile(xtc) as f:
        return [fr.box.copy() for fr in f]


# --- box_action ----------------------------------------------------------
def test_no_box_writes_box_nobox():
    assert c.box_action(False) == "box nobox\n"


def test_a_box_is_imaged():
    assert c.box_action(True) == "autoimage\n"


def test_the_old_name_is_gone():
    assert not hasattr(c, "autoimage_action")


# --- every cpptraj XTC writer settles the box -----------------------------
def test_every_xtc_written_by_cpptraj_has_a_box_line_first():
    """Each `trajout ... xtc` in the script is preceded, in its own input
    file, by a box_action() line. A new XTC writer added without one would
    bring the garbage box back for box-less runs."""
    src = open(c.__file__).read()
    writers = [m.start() for m in re.finditer(r'trajout \{[a-z_]+\} xtc', src)]
    assert len(writers) == 4  # stripped (minimal), Amber full, NAMD full, GROMACS wrap
    for pos in writers:
        block = src[src.rfind("open(", 0, pos):pos]
        assert "box_action(box_present)" in block, block[-300:]


def test_no_inline_autoimage_is_left():
    src = open(c.__file__).read()
    assert 'f.write("autoimage\\n")' not in src
    assert "Skipping autoimage" not in src


# --- process_stripped_trajectory's cpptraj inputs ---------------------------
@pytest.fixture
def no_cpptraj(monkeypatch):
    ran = []
    monkeypatch.setattr(c, "run_cpptraj",
                        lambda cppin, label, **kw: (ran.append(cppin), (0, ""))[1])
    monkeypatch.setattr(c, "fix_pdb_element_symbols", lambda path: None)
    return ran


@pytest.mark.parametrize("box_present,want,not_want", [
    (False, "box nobox", "autoimage"),
    (True, "autoimage", "box nobox"),
])
def test_stripped_inputs_settle_the_box(tmp_path, no_cpptraj, box_present,
                                        want, not_want):
    c.process_stripped_trajectory(PSF, DCD, str(tmp_path), ":WAT", "minimal",
                                  fit_mask="@CA", box_present=box_present)
    assert len(no_cpptraj) == 2
    for cppin in no_cpptraj:
        lines = open(cppin).read().splitlines()
        assert want in lines and not_want not in lines
        # Straight after trajin: before strip and rms, which change atoms
        # and coordinates the box line must still see as read.
        first_trajin = next(i for i, l in enumerate(lines) if l.startswith("trajin"))
        assert lines[first_trajin + 1] == want


def test_stripped_default_is_no_box(tmp_path, no_cpptraj):
    c.process_stripped_trajectory(PSF, DCD, str(tmp_path), ":WAT", "minimal",
                                  fit_mask="@CA")
    assert all("box nobox" in open(p).read() for p in no_cpptraj)


# --- real cpptraj on the fixture ---------------------------------------------
@needs_cpptraj
def test_fixture_has_no_box(tmp_path):
    frame = c.read_first_frame(PSF, DCD, str(tmp_path))
    assert not c.has_box(frame)


@needs_cpptraj
def test_fixture_still_shows_the_fault_without_box_nobox(tmp_path):
    """The control: plain trajin -> trajout. If this cpptraj build writes a
    clean zero box anyway, the fixture proves nothing here, so skip rather
    than fail."""
    xtc = tmp_path / "plain.xtc"
    cppin = tmp_path / "plain.in"
    cppin.write_text(f"parm {PSF}\ntrajin {DCD}\ntrajout {xtc} xtc\nrun\n")
    rv, _ = c.run_cpptraj(str(cppin), "plain", fatal=False)
    assert rv == 0
    got = boxes(str(xtc))
    if all(np.all(b == 0) for b in got):
        pytest.skip("this cpptraj build writes a zero box without box nobox")
    assert len(got) == 3


@needs_cpptraj
def test_minimal_xtc_of_a_box_less_run_has_a_zero_box(tmp_path):
    xtc, pdb, ref = c.process_stripped_trajectory(
        PSF, DCD, str(tmp_path), ":WAT", "minimal", fit_mask="@CA",
        box_present=False)
    assert xtc and os.path.isfile(xtc)
    got = boxes(xtc)
    assert len(got) == 3
    for b in got:
        assert np.all(np.isfinite(b)) and np.all(b == 0), b


@needs_cpptraj
def test_its_pdbs_carry_no_bogus_cryst1(tmp_path):
    _, pdb, ref = c.process_stripped_trajectory(
        PSF, DCD, str(tmp_path), ":WAT", "minimal", fit_mask="@CA",
        box_present=False)
    for path in (pdb, ref):
        for line in open(path):
            if line.startswith("CRYST1"):
                assert "inf" not in line.lower() and "nan" not in line.lower(), line
