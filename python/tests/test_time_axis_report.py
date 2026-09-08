"""Tests for the time-axis report cpptraj_gmx_traj_manipulation.py sends mdr-process

The wrapper reads cpptraj's trajin report to say whether the SOURCE trajectory
carried per-frame times. `false` is what makes mdr-process use the declared
sampling_frequency_ps instead of measuring the converted XTC, onto which
cpptraj stamps a default 1 ps/frame. A source format the parser does not
recognise must come back as "unknown" (None), never as "has time": a false
positive silently restores the fabricated spacing.

The module imports pytraj and parmed at the top, neither of which these tests
need; they are stubbed so the parser can be imported anywhere pytest runs.
"""

import sys
import types

import pytest

for name in ("pytraj", "parmed"):
    sys.modules.setdefault(name, types.ModuleType(name))

from cpptraj_gmx_traj_manipulation import trajectory_has_time_axis  # noqa: E402


NETCDF_WITH_TIME = (
    " 0: 'run.nc' is a NetCDF AMBER trajectory with coordinates, time, box, "
    "Parm top.prmtop (Truncated octahedron box) (reading 1 of 5614)"
)
NETCDF_WITHOUT_TIME = (
    " 0: 'run.nc' is a NetCDF AMBER trajectory with coordinates, box, "
    "Parm top.prmtop (Truncated octahedron box) (reading 1 of 5614)"
)
# What cpptraj prints for MDR00000376's 3zxw.exp02.md01.dry.dcd. No " with "
# clause at all.
CHARMM_DCD = (
    " 0: '3zxw.exp02.md01.dry.dcd' is a CHARMM DCD file (coords) Little Endian "
    "32 bit, Parm 3zxw.psf (Triclinic box) (reading 1 of 750)"
)
GROMACS_XTC = (
    " 0: 'full.xtc' is a GROMACS XTC file, Parm 3zxw.pdb (Triclinic box) "
    "(reading 1 of 750)"
)


# --------------------------------------------------
def test_netcdf_with_time_has_an_axis():
    assert trajectory_has_time_axis(f"noise\n{NETCDF_WITH_TIME}\nmore") is True


# --------------------------------------------------
def test_netcdf_without_time_has_none():
    assert trajectory_has_time_axis(NETCDF_WITHOUT_TIME) is False


# --------------------------------------------------
def test_charmm_dcd_has_no_time_axis():
    """A DCD stores no per-frame times; cpptraj's report has no contents list

    Before this was handled the DCD line matched nothing, the wrapper reported
    "unknown", and mdr-process measured the 1 ps/frame its own conversion had
    stamped onto the XTC.
    """

    assert trajectory_has_time_axis(CHARMM_DCD) is False


# --------------------------------------------------
def test_a_path_containing_time_does_not_count():
    line = NETCDF_WITHOUT_TIME.replace("top.prmtop", "time/top.prmtop")
    assert trajectory_has_time_axis(line) is False


# --------------------------------------------------
@pytest.mark.parametrize("output", ["", "nothing relevant here", GROMACS_XTC])
def test_an_unrecognised_report_is_unknown(output):
    """Unknown is not absence: an XTC carries its own times and is measured"""

    assert trajectory_has_time_axis(output) is None
