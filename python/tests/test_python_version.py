"""cpptraj_gmx_traj_manipulation.py runs only under the uv environment.

mdr-process 0.34.0 dropped the simproc conda env (Python 3.7.12). An older
mdr-process would still run the script under that interpreter; the script must
then stop with a message naming the fix, not a TypeError from a type hint.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(HERE, "cpptraj_gmx_traj_manipulation.py")
SIMPROC_PYTHON = os.path.expanduser("~/.local/share/mamba/envs/simproc/bin/python")


def test_the_script_starts_under_this_interpreter():
    out = subprocess.run(
        [sys.executable, SCRIPT, "--help"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "--top" in out.stdout


def test_the_version_check_comes_before_any_other_import():
    """Anything imported first could fail on an old interpreter instead"""
    lines = [l for l in open(SCRIPT) if l.strip() and not l.startswith("#")]
    assert lines[0].strip() == "import sys"
    assert lines[1].startswith("if sys.version_info < (3, 13):")


@pytest.mark.skipif(not os.path.isfile(SIMPROC_PYTHON), reason="no simproc env")
def test_an_old_interpreter_is_told_what_to_install():
    out = subprocess.run(
        [SIMPROC_PYTHON, SCRIPT, "--help"], capture_output=True, text=True
    )
    assert out.returncode != 0
    assert "needs Python 3.13+" in out.stderr
    assert "mdr-process 0.34.0" in out.stderr
    assert "Traceback" not in out.stderr
