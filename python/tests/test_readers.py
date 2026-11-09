"""Tests for count_atoms.py, frame_deviation.py and run_signature.py

The readers the agent ran inline until they moved here (mdrepo-agent#98).
The inputs are the ones test_write_structure.py builds: a CHARMM .psf of ten
carbons, a DCD whose every coordinate in frame f is f+1, a dipeptide in
water, and the Chamber .parm7 fixture.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("MDAnalysis")

import count_atoms as ca  # noqa: E402
import frame_deviation as fd  # noqa: E402
import run_signature as rs  # noqa: E402
import universe  # noqa: E402
from test_write_structure import (  # noqa: E402
    dipeptide_in_water,
    numbered_dcd,
    report_of,
    run,
    ten_atom_psf,
)

HERE = os.path.dirname(ca.__file__)
CHAMBER = os.path.join(HERE, "tests", "inputs", "chamber.parm7")


# --------------------------------------------------
def test_a_psf_is_counted_without_coordinates(tmp_path):
    psf = tmp_path / "run.psf"
    psf.write_text(ten_atom_psf())

    assert ca.count(str(psf)) == ca.Report(atoms=10, solvent=0)


def test_water_and_ions_are_counted_as_solvent(tmp_path):
    pdb = tmp_path / "s.pdb"
    pdb.write_text(dipeptide_in_water())

    assert ca.count(str(pdb)) == ca.Report(atoms=26, solvent=6)


def test_a_chamber_parm7_is_counted(tmp_path):
    assert universe.is_chamber(CHAMBER)
    assert ca.count(CHAMBER).atoms == 25


def test_a_top_is_gromacs_unless_it_is_amber(tmp_path):
    gmx, amber = tmp_path / "g.top", tmp_path / "a.top"
    gmx.write_text("[ defaults ]\n1 2 yes 0.5 0.8333\n")
    amber.write_text("%VERSION  VERSION_STAMP = V0001.000\n%FLAG TITLE\n")

    assert ca.is_gromacs_top(str(gmx))
    assert not ca.is_gromacs_top(str(amber))
    assert not ca.is_gromacs_top(str(tmp_path / "s.pdb"))


def test_itp_counts_infer_the_system_for_top_and_explicit_format(tmp_path, monkeypatch):
    top, itp = tmp_path / "system.top", tmp_path / "system.itp"
    top.write_text("[ defaults ]\n1 2 yes 0.5 0.8333\n")
    itp.write_text("[ moleculetype ]\nMOL 3\n")
    calls = []

    class Atoms:
        n_atoms = 4
        resnames = []

    class Universe:
        atoms = Atoms()

    def open_universe(path, **kwargs):
        calls.append((path, kwargs))
        return Universe()

    monkeypatch.setattr(ca, "open_universe", open_universe)

    assert ca.count(str(top)) == ca.Report(atoms=4, solvent=0)
    assert ca.count(str(itp), fmt="ITP") == ca.Report(atoms=4, solvent=0)
    assert calls == [
        (str(top), {"topology_format": "ITP", "infer_system": True}),
        (str(itp), {"topology_format": "ITP", "infer_system": True}),
    ]


def test_count_command_reports_json_and_refuses_an_unreadable_file(tmp_path):
    psf = tmp_path / "run.psf"
    psf.write_text(ten_atom_psf())
    script = os.path.join(HERE, "count_atoms.py")

    assert report_of(run(script, "--file", str(psf))) == {"atoms": 10, "solvent": 0}

    bad = tmp_path / "bad.pdb"
    bad.write_text("not a structure\n")
    p = run(script, "--file", str(bad))
    assert p.returncode == 1
    assert p.stderr.strip().splitlines()[-1].startswith("reading bad.pdb: ")


# --------------------------------------------------
def numbered_pdb(n: int, value: float) -> str:
    """n carbons, every coordinate value."""

    return "".join(
        "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
        % (i, i, value, value, value)
        for i in range(1, n + 1)
    ) + "END\n"


def test_a_structure_at_the_first_frame_deviates_by_nothing(tmp_path):
    pdb, dcd = tmp_path / "s.pdb", tmp_path / "run.dcd"
    pdb.write_text(numbered_pdb(10, 1.0))
    dcd.write_bytes(numbered_dcd(10, 3))

    report = fd.deviation(str(pdb), str(dcd))

    assert report.atoms == 10
    assert report.deviation == pytest.approx(0.0)


def test_a_structure_elsewhere_deviates_by_the_distance(tmp_path):
    pdb, dcd = tmp_path / "s.pdb", tmp_path / "run.dcd"
    pdb.write_text(numbered_pdb(10, 4.0))
    dcd.write_bytes(numbered_dcd(10, 3))

    script = os.path.join(HERE, "frame_deviation.py")
    report = report_of(run(script, "--structure", str(pdb), "--trajectory", str(dcd)))

    assert report["atoms"] == 10
    assert report["deviation"] == pytest.approx(3 * 3**0.5, abs=1e-3)


# --------------------------------------------------
def test_a_run_is_fingerprinted_by_its_ca_atoms(tmp_path):
    pdb, dcd = tmp_path / "s.pdb", tmp_path / "run.dcd"
    pdb.write_text(numbered_pdb(10, 1.0))
    dcd.write_bytes(numbered_dcd(10, 3))

    report = rs.signature(str(pdb), str(dcd), str(dcd))

    assert report.n == 10
    assert len(report.first) == 40 and len(report.last) == 40
    assert report == rs.signature(str(pdb), str(dcd), str(dcd)), "deterministic"


def test_a_run_fingerprint_excludes_calcium_ca_atoms(tmp_path):
    pdb, dcd = tmp_path / "s.pdb", tmp_path / "run.dcd"
    calcium = "".join(
        "ATOM  %5d  CA  CA A%4d    %8.3f%8.3f%8.3f  1.00  0.00          CA\n"
        % (i, i, 20.0 + i, 20.0, 20.0)
        for i in range(4, 7)
    )
    pdb.write_text(numbered_pdb(3, 1.0).replace("END\n", calcium + "END\n"))
    dcd.write_bytes(numbered_dcd(6, 3))

    report = rs.signature(str(pdb), str(dcd), str(dcd))

    assert report.n == 3


def test_a_run_with_too_few_ca_atoms_has_no_fingerprint(tmp_path):
    psf, dcd = tmp_path / "run.psf", tmp_path / "run.dcd"
    psf.write_text(ten_atom_psf())
    dcd.write_bytes(numbered_dcd(10, 2))

    script = os.path.join(HERE, "run_signature.py")
    report = report_of(run(script, "--topology", str(psf), "--first", str(dcd)))

    assert report == {"n": 0, "first": None, "last": None}
