"""Tests for reduce_tpr.py

The real cut needs GROMACS: a 3-atom bonded molecule beside four single-atom
ones is built with grompp and cut to its first three atoms, and skips where
gmx is absent. The command's own parts, the index group, the dump reader and
the flow around gmx, are tested with a stand-in gmx on the PATH. Ported from
mdrepo-agent PR #87, which this command replaces.
"""

import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reduce_tpr as rt  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(rt.__file__), "reduce_tpr.py")


# --------------------------------------------------
def test_tpr_atoms_reads_gmx_dumps_count():
    assert (
        rt.tpr_atoms(
            "topology:\n   topology->name = foo\n   #atoms = 191\n   natoms = 191\n   ngrps  = 4\n"
        )
        == 191
    )
    assert rt.tpr_atoms("no count here") is None


def test_the_index_group_is_one_based_fifteen_to_a_line(tmp_path):
    p = tmp_path / "saved.ndx"
    rt.write_index(str(p), list(range(0, 17)))
    lines = p.read_text().splitlines()
    assert lines[0] == "[ saved ]"
    assert lines[1] == " ".join(str(i) for i in range(1, 16))
    assert lines[2] == "16 17"


# --------------------------------------------------
def fake_gmx(tmp_path, natoms=3, version="2024.2"):
    """A gmx on the PATH: convert-tpr copies -s to -o after reading the
    group name, dump reports natoms, --version reports a version."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gmx = bin_dir / "gmx"
    gmx.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  convert-tpr) read group; [ "$group" = saved ] || exit 3; cp "$3" "$7"; echo saved >> "$7" ;;\n'
        f"  dump) echo 'topology:'; echo '   natoms = {natoms}' ;;\n"
        f"  --version) echo 'GROMACS version:    {version}' ;;\n"
        "  *) exit 4 ;;\n"
        "esac\n"
    )
    gmx.chmod(gmx.stat().st_mode | stat.S_IEXEC)
    return str(gmx)


def test_reduce_runs_convert_tpr_with_the_selected_atoms(tmp_path, monkeypatch):
    gmx = fake_gmx(tmp_path, natoms=3)
    monkeypatch.setattr(rt, "select_indices", lambda tpr, sel: ([0, 1, 2], 7))
    tpr, out = tmp_path / "full.tpr", tmp_path / "saved.tpr"
    tpr.write_bytes(b"TPR")

    report = rt.reduce(str(tpr), "resname PRO", str(out), gmx)

    assert report == rt.Report(atoms=3, total=7, selected=3, gromacs="2024.2")
    assert out.read_bytes() == b"TPRsaved\n"


def test_reduce_refuses_a_count_that_differs_from_the_selection(tmp_path, monkeypatch):
    gmx = fake_gmx(tmp_path, natoms=5)
    monkeypatch.setattr(rt, "select_indices", lambda tpr, sel: ([0, 1, 2], 7))
    (tmp_path / "full.tpr").write_bytes(b"TPR")
    with pytest.raises(RuntimeError, match="holds 5 atoms and the selection 3"):
        rt.reduce(
            str(tmp_path / "full.tpr"), "resname PRO", str(tmp_path / "saved.tpr"), gmx
        )


def test_command_exits_2_without_gmx(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    p = subprocess.run(
        [sys.executable, SCRIPT, "--tpr", "x.tpr", "--selection", "protein", "--out", "y.tpr", "--gmx", "gmx-missing"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert p.returncode == 2
    assert "GROMACS is not available" in p.stderr


# --------------------------------------------------
TOP = """[ defaults ]
1 1 no 1.0 1.0
[ atomtypes ]
A 12.0 0.0 A 0.3 0.1
[ moleculetype ]
P 3
[ atoms ]
1 A 1 PRO C1 1 0.0 12.0
2 A 1 PRO C2 2 0.0 12.0
3 A 1 PRO C3 3 0.0 12.0
[ bonds ]
1 2 1 0.15 1000
2 3 1 0.15 1000
[ moleculetype ]
W 1
[ atoms ]
1 A 1 SOL O 1 0.0 12.0
[ system ]
t
[ molecules ]
P 1
W 4
"""
GRO = """test
    7
    1PRO     C1    1   1.000   1.000   1.000
    1PRO     C2    2   1.150   1.000   1.000
    1PRO     C3    3   1.300   1.000   1.000
    2SOL      O    4   2.000   2.000   2.000
    3SOL      O    5   2.300   2.000   2.000
    4SOL      O    6   2.000   2.300   2.000
    5SOL      O    7   2.300   2.300   2.000
   3.00000   3.00000   3.00000
"""
MDP = (
    "integrator = md\nnsteps = 0\ncutoff-scheme = Verlet\nrvdw = 0.9\nrcoulomb = 0.9\n"
)


def test_a_real_tpr_is_cut_to_the_saved_atoms(tmp_path):
    gmx = shutil.which(os.environ.get("GMX", "gmx"))
    if gmx is None:
        pytest.skip("no gmx on the PATH")
    pytest.importorskip("MDAnalysis")
    for name, body in (("topol.top", TOP), ("conf.gro", GRO), ("run.mdp", MDP)):
        (tmp_path / name).write_text(body)
    full = tmp_path / "full.tpr"
    p = subprocess.run(
        [gmx, "grompp", "-f", "run.mdp", "-c", "conf.gro", "-p", "topol.top", "-o", str(full), "-maxwarn", "10"],
        cwd=tmp_path, capture_output=True, text=True,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr

    out = tmp_path / "saved.tpr"
    report = rt.reduce(str(full), "resname PRO", str(out), gmx)

    assert (report.atoms, report.total, report.selected) == (3, 7, 3)
    assert report.gromacs
    import MDAnalysis as mda

    u = mda.Universe(str(out))
    assert len(u.atoms) == 3 and list(u.atoms.names) == ["C1", "C2", "C3"]
    assert len(u.bonds) == 2, "the molecule's bonds come with it"

    r = subprocess.run(
        [sys.executable, SCRIPT, "--tpr", str(full), "--selection", "resname SOL", "--out", str(tmp_path / "w.tpr"), "--gmx", gmx],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["atoms"] == 4
