"""
Tests for outputs.py: an output is published whole once its checks pass, or
left as it was.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import outputs  # noqa: E402


def test_the_staged_path_keeps_the_suffix():
    assert outputs.staged_path("/x/s.pdb") == "/x/s.part.pdb"
    assert outputs.staged_path("run_trj") == "run_trj.part"


def test_a_clean_exit_publishes_the_staged_file(tmp_path):
    out = tmp_path / "s.pdb"
    out.write_text("old")
    with outputs.staged(str(out)) as p:
        assert p == str(tmp_path / "s.part.pdb")
        open(p, "w").write("new")
    assert out.read_text() == "new"
    assert not os.path.exists(p)


def test_an_exception_removes_the_staged_file_and_keeps_the_output(tmp_path):
    out = tmp_path / "s.pdb"
    out.write_text("old")
    with pytest.raises(ValueError, match="the check"):
        with outputs.staged(str(out)) as p:
            open(p, "w").write("new")
            raise ValueError("the check failed")
    assert out.read_text() == "old"
    assert not os.path.exists(p)


def test_writing_nothing_publishes_nothing(tmp_path):
    out = tmp_path / "s.psf"
    with outputs.staged(str(out)):
        pass
    assert not out.exists()
