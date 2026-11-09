"""Tests for generate_topology.py

The dipeptide is zenodo:5357152's case in miniature: its .cms listed every
heavy atom of the protein, then every hydrogen, and ParmEd, which takes a
residue to be a run of consecutive atoms, bonded 4,245 hydrogens to nothing.
The same atoms with each residue's together bond cleanly. Ported with the
checks from mdrepo-agent's internal/stage/topology_test.go when the step
moved here (mdrepo-agent#98).
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_topology as gt  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(gt.__file__), "generate_topology.py")

pytest.importorskip("parmed")


# --------------------------------------------------
def two_alanines(in_order: bool) -> str:
    """A dipeptide, each residue's atoms together or every heavy atom first."""

    def heavy(r):
        return [("N", r, 0), ("CA", r, 1.5), ("C", r, 3), ("O", r, 3.5), ("CB", r, 1.5)]

    def hyd(r):
        return [
            ("H", r, -0.5),
            ("HA", r, 1.5),
            ("HB1", r, 1),
            ("HB2", r, 2),
            ("HB3", r, 1.5),
        ]

    if in_order:
        atoms = heavy(1) + hyd(1) + heavy(2) + hyd(2)
    else:
        atoms = heavy(1) + heavy(2) + hyd(1) + hyd(2)
    lines = []
    for i, (name, res, x) in enumerate(atoms):
        lines.append(
            "ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
            % (i + 1, name, res, x + (res - 1) * 3.8, float(i % 3), 0.0, name[0])
        )
    return "\n".join(lines) + "\nEND\n"


# Each alanine: N 0, CA 1, C 2, O 3, CB 4, H 5, HA 6, HB1-3 7-9; the second
# at +10. No peptide bond.
ALANINE_BONDS = [
    [a + o, b + o]
    for o in (0, 10)
    for a, b in [[0, 1], [1, 2], [2, 3], [1, 4], [0, 5], [1, 6], [4, 7], [4, 8], [4, 9]]
]


def psf_bond_count(path: str) -> int:
    with open(path) as fh:
        for line in fh:
            if "!NBOND" in line:
                return int(line.split()[0])
    raise AssertionError("no bond section")


# --------------------------------------------------
def test_inferred_bonds_that_leave_lone_hydrogens_are_refused(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(False))
    psf.write_text("stale")

    report = gt.generate(str(pdb), str(psf))

    assert not report.ok()
    assert report.unbonded_h == 10, "all 10 hydrogens are bonded to nothing"
    assert "ALA 1 A H" in report.examples
    assert (
        psf.read_text() == "stale"
    ), "a refused topology is not written; what was at --out stays"
    assert not (tmp_path / "s.part.psf").exists()
    assert "10 hydrogens" in report.refusal()


def test_a_dipeptide_in_residue_order_bonds_every_hydrogen(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(True))
    psf.write_text("stale")  # replaced by the topology that passes

    report = gt.generate(str(pdb), str(psf))

    assert report.ok()
    assert report.atoms == 20
    assert report.bonds_from == "inferred"
    assert not report.renumbered
    assert psf_bond_count(psf) == 19, "9 in each alanine and the peptide bond"
    assert report.bonds == 19


def test_stated_bonds_are_the_topologys_bonds(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(True))

    report = gt.generate(str(pdb), str(psf), ALANINE_BONDS)

    assert report.ok()
    assert report.bonds_from == "stated"
    # No peptide bond stated, none in the topology: the inference is gone.
    assert psf_bond_count(psf) == 18


def test_stated_bonds_that_leave_a_hydrogen_alone_are_refused(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(True))

    report = gt.generate(str(pdb), str(psf), ALANINE_BONDS[:17])

    assert not report.ok()
    assert report.unbonded_h == 1
    assert not psf.exists()


def test_a_bond_outside_the_structure_is_an_error(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(True))

    with pytest.raises(ValueError, match="outside the 20"):
        gt.generate(str(pdb), str(psf), [[0, 20]])


# --------------------------------------------------
def test_wrap_safe_gives_each_block_past_the_wrap_its_own_chain(tmp_path):
    """GROMACS writes residue numbers modulo 10000. The 10000th residue and
    on get a fresh chain letter and a number from 1, so no (chain, number)
    pair repeats; the residues before keep chain A."""

    lines = []
    for r in range(1, 20002):
        lines.append(
            "ATOM  %5d  OW  SOL A%4d    %8.3f%8.3f%8.3f  1.00  0.00           O"
            % (r % 100000, r % 10000, r * 0.1, 0.0, 0.0)
        )
    src, dest = tmp_path / "w.pdb", tmp_path / "w.fixed.pdb"
    src.write_text("\n".join(lines) + "\nEND\n")

    gt.wrap_safe(str(src), str(dest))

    out = [l for l in dest.read_text().splitlines() if l.startswith("ATOM")]
    keys = [(l[21], l[22:26]) for l in out]
    assert len(set(keys)) == len(keys), "every (chain, number) pair is distinct"
    assert keys[0] == ("A", "   1")
    assert keys[9998] == ("A", "9999"), "the first 9999 residues keep their chain"
    assert keys[9999] == ("B", "   1"), "the 10000th starts the next block"
    assert keys[19997] == ("B", "9999")
    assert keys[19998] == ("C", "   1")
    assert src.read_text().count("SOL A") == 20001, "the PDB itself is untouched"


# --------------------------------------------------
def run(*argv):
    return subprocess.run(
        [sys.executable, SCRIPT, *argv], capture_output=True, text=True
    )


def test_command_reports_json_and_exits_0(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(True))

    p = run("--structure", str(pdb), "--out", str(psf))

    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout.strip().splitlines()[-1])
    assert report["atoms"] == 20 and report["bonds"] == 19
    assert psf.exists()


def test_command_refuses_wrong_bonds_with_the_reason_on_stderr(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    pdb.write_text(two_alanines(False))

    p = run("--structure", str(pdb), "--out", str(psf))

    assert p.returncode == 1
    report = json.loads(p.stdout.strip().splitlines()[-1])
    assert report["unbonded_h"] == 10
    assert "bonds are wrong" in p.stderr.strip().splitlines()[-1]
    assert not psf.exists()


def test_command_takes_stated_bonds_from_a_json_file(tmp_path):
    pdb, psf = tmp_path / "s.pdb", tmp_path / "s.psf"
    bonds = tmp_path / "bonds.json"
    pdb.write_text(two_alanines(True))
    bonds.write_text(json.dumps(ALANINE_BONDS))

    p = run("--structure", str(pdb), "--out", str(psf), "--bonds", str(bonds))

    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1])["bonds_from"] == "stated"
    assert psf_bond_count(psf) == 18


def test_an_unreadable_structure_exits_1_with_the_reason(tmp_path):
    p = run(
        "--structure", str(tmp_path / "missing.pdb"), "--out", str(tmp_path / "s.psf")
    )

    assert p.returncode == 1
    assert "missing.pdb" in p.stderr.strip().splitlines()[-1]
