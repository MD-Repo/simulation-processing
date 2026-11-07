"""Tests for first_frame_structure.py and subset_structure.py

The inputs are built here: a CHARMM .psf with no coordinates, a DCD whose
every coordinate in frame f is f+1, and a PDB of a dipeptide in water. Ported
from mdrepo-agent's internal/stage tests when the steps moved here
(mdrepo-agent#98).
"""

import json
import os
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("MDAnalysis")

import first_frame_structure as ffs  # noqa: E402
import psf_elements as pe  # noqa: E402
import subset_structure as ss  # noqa: E402

HERE = os.path.dirname(ffs.__file__)
FIRST_FRAME = os.path.join(HERE, "first_frame_structure.py")
SUBSET = os.path.join(HERE, "subset_structure.py")


# --------------------------------------------------
def numbered_dcd(n: int, frames: int) -> bytes:
    """A little-endian CHARMM DCD of n atoms whose every coordinate in frame
    f is f+1."""

    body = bytearray(84)
    body[0:4] = b"CORD"
    struct.pack_into("<i", body, 4, frames)
    struct.pack_into("<i", body, 4 + 4 * 2, 1)
    struct.pack_into("<f", body, 4 + 4 * 9, 20.4548)
    struct.pack_into("<i", body, 4 + 4 * 19, 24)
    out = bytearray()
    out += struct.pack("<i", 84) + body + struct.pack("<i", 84)
    out += struct.pack("<ii", 84, 1) + bytes(80) + struct.pack("<i", 84)
    out += struct.pack("<iii", 4, n, 4)
    for f in range(frames):
        for _ in range(3):
            out += struct.pack("<i", 4 * n)
            out += struct.pack("<%df" % n, *([float(f + 1)] * n))
            out += struct.pack("<i", 4 * n)
    return bytes(out)


def ten_atom_psf() -> str:
    """A CHARMM topology of ten alanine carbons and no coordinates."""

    lines = ["PSF", "", "       1 !NTITLE", " REMARKS ten", "", "      10 !NATOM"]
    for i in range(1, 11):
        lines.append(
            "%8d A        1        ALA      C%-3d     CT1    0.000000       12.0110           0"
            % (i, i)
        )
    lines += ["", "       0 !NBOND: bonds", "", "", "       0 !NTHETA: angles", "", ""]
    return "\n".join(lines) + "\n"


def dipeptide_in_water(elements: bool = True) -> str:
    """Two alanines and two waters, each residue's atoms together."""

    atoms = []
    for r in (1, 2):
        x0 = (r - 1) * 3.8
        for name, dx in (
            ("N", 0),
            ("CA", 1.5),
            ("C", 3),
            ("O", 3.5),
            ("CB", 1.5),
            ("H", -0.5),
            ("HA", 1.5),
            ("HB1", 1),
            ("HB2", 2),
            ("HB3", 1.5),
        ):
            atoms.append(("ATOM", name, "ALA", "A", r, x0 + dx, 0.0, 0.0))
    for r in (3, 4):
        for name, dx in (("OH2", 0.0), ("H1", 0.6), ("H2", -0.6)):
            atoms.append(("HETATM", name, "TIP3", "W", r, 20.0 + dx, r * 3.0, 0.0))
    lines = []
    for i, (rec, name, res, chain, resid, x, y, z) in enumerate(atoms, 1):
        el = ("%2s" % name[0]) if elements else "  "
        lines.append(
            "%-6s%5d  %-3s %-4s%s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %s  "
            % (rec, i, name, res, chain, resid, x, y, z, el)
        )
    return "\n".join(lines) + "\nEND\n"


def report_of(p: subprocess.CompletedProcess) -> dict:
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def run(script, *argv):
    return subprocess.run(
        [sys.executable, script, *argv], capture_output=True, text=True
    )


def element_columns(pdb):
    with open(pdb) as fh:
        return [l[76:78] for l in fh if l.startswith(("ATOM", "HETATM"))]


# --------------------------------------------------
def test_a_structure_is_written_from_the_topology_and_the_first_frame(tmp_path):
    psf, dcd, out = tmp_path / "run.psf", tmp_path / "run.dcd", tmp_path / "sys.pdb"
    psf.write_text(ten_atom_psf())
    dcd.write_bytes(numbered_dcd(10, 3))

    report = ffs.write(str(psf), str(dcd), str(out))

    assert report.atoms == 10
    assert report.elements == {"from": str(psf), "repartitioned": 0, "ions": 0}
    assert pe.pdb_atom_count(str(out)) == 10
    import MDAnalysis as mda

    u = mda.Universe(str(out))
    assert (u.atoms.positions == 1.0).all(), "the first frame's coordinates"
    assert element_columns(out) == [" C"] * 10, "elements from the .psf's masses"


def test_first_frame_command_reports_json(tmp_path):
    psf, dcd, out = tmp_path / "run.psf", tmp_path / "run.dcd", tmp_path / "sys.pdb"
    psf.write_text(ten_atom_psf())
    dcd.write_bytes(numbered_dcd(10, 3))

    report = report_of(
        run(
            FIRST_FRAME,
            "--topology",
            str(psf),
            "--trajectory",
            str(dcd),
            "--out",
            str(out),
        )
    )

    assert report["atoms"] == 10 and report["elements"]["from"] == str(psf)
    assert out.exists()


def test_a_chamber_parm7_is_read_through_parmed(tmp_path):
    """The first LEU residue, 25 atoms, of zenodo:6487373's
    bilayer_A176K.parm7, which MDAnalysis refuses."""

    top = os.path.join(HERE, "tests", "inputs", "chamber.parm7")
    dcd, out = tmp_path / "run.dcd", tmp_path / "sys.pdb"
    dcd.write_bytes(numbered_dcd(25, 2))

    report = ffs.write(top, str(dcd), str(out))

    assert report.atoms == 25 and report.chamber
    assert report.elements is None
    import MDAnalysis as mda

    u = mda.Universe(str(out))
    assert u.atoms.n_atoms == 25 and u.atoms.resnames[0] == "LEU"
    assert (u.atoms.positions == 1.0).all(), "the first frame's coordinates"


def test_an_amber_topology_is_not_taken_for_a_chamber_one(tmp_path):
    top = tmp_path / "plain.parm7"
    top.write_text("%VERSION  VERSION_STAMP = V0001.000\n%FLAG TITLE\n%FORMAT(20a4)\n")

    assert not ffs.is_chamber(str(top))
    assert not ffs.is_chamber(str(tmp_path / "missing.parm7"))


def test_first_frame_command_fails_on_a_mismatched_trajectory(tmp_path):
    psf, dcd, out = tmp_path / "run.psf", tmp_path / "run.dcd", tmp_path / "sys.pdb"
    psf.write_text(ten_atom_psf())
    dcd.write_bytes(numbered_dcd(7, 3))

    p = run(
        FIRST_FRAME, "--topology", str(psf), "--trajectory", str(dcd), "--out", str(out)
    )

    assert p.returncode == 1
    assert "run.psf" in p.stderr.strip().splitlines()[-1]
    assert not out.exists(), "nothing is left behind"


# --------------------------------------------------
def test_a_subset_keeps_the_selected_atoms_in_order(tmp_path):
    pdb, out = tmp_path / "sys.pdb", tmp_path / "sub.pdb"
    pdb.write_text(dipeptide_in_water())

    report = ss.subset(str(pdb), None, "protein", str(out))

    assert report.atoms == 20 and report.total == 26
    assert not report.released and report.mismatches == 0
    assert not report.from_topology
    assert report.elements is None
    assert pe.pdb_atom_count(str(out)) == 20
    import MDAnalysis as mda

    assert list(mda.Universe(str(out)).atoms.names[:3]) == ["N", "CA", "C"]


def test_a_subset_matched_to_the_released_topology_of_its_atoms(tmp_path):
    """A .psf of the saved atoms alone is the release's own account of them:
    the selection is compared with it name by name, and where the structure
    had no elements the .psf's masses give them."""

    pdb, psf, out = tmp_path / "sys.pdb", tmp_path / "prot.psf", tmp_path / "sub.pdb"
    pdb.write_text(dipeptide_in_water(elements=False))
    lines = ["PSF", "", "       1 !NTITLE", " REMARKS prot", "", "      20 !NATOM"]
    names = ["N", "CA", "C", "O", "CB", "H", "HA", "HB1", "HB2", "HB3"]
    masses = [14.007, 12.011, 12.011, 15.999, 12.011, 1.008, 1.008, 1.008, 1.008, 1.008]
    k = 0
    for r in (1, 2):
        for name, m in zip(names, masses):
            k += 1
            lines.append(
                "%8d A        %-8d ALA      %-8s CT1    0.000000     %10.4f           0"
                % (k, r, name, m)
            )
    lines += ["", "       0 !NBOND: bonds", "", ""]
    psf.write_text("\n".join(lines) + "\n")

    report = ss.subset(str(pdb), str(psf), "protein", str(out))

    assert report.released and report.mismatches == 0
    assert report.elements == {"from": str(psf), "repartitioned": 0, "ions": 0}
    assert element_columns(out)[:6] == [" N", " C", " C", " O", " C", " H"]


def test_a_subset_that_differs_from_the_released_topology_is_reported(tmp_path):
    pdb, psf, out = tmp_path / "sys.pdb", tmp_path / "prot.psf", tmp_path / "sub.pdb"
    pdb.write_text(dipeptide_in_water())
    lines = ["PSF", "", "       1 !NTITLE", " REMARKS prot", "", "      20 !NATOM"]
    for k in range(1, 21):
        lines.append(
            "%8d A        1        GLY      X%-7d CT1    0.000000       12.0110           0"
            % (k, k)
        )
    lines += ["", "       0 !NBOND: bonds", "", ""]
    psf.write_text("\n".join(lines) + "\n")

    report = ss.subset(str(pdb), str(psf), "protein", str(out))

    assert report.released and report.mismatches == 20
    assert report.examples[0].startswith("atom 1: selected N ALA1, released X1 GLY1")
    assert report.elements is None, "mismatched atoms are not given the .psf's elements"


def test_an_empty_selection_is_an_error(tmp_path):
    pdb, out = tmp_path / "sys.pdb", tmp_path / "sub.pdb"
    pdb.write_text(dipeptide_in_water())
    with pytest.raises(ValueError, match="holds no atoms"):
        ss.subset(str(pdb), None, "resname XXX", str(out))


def test_subset_command_reports_json_and_removes_its_output_on_failure(tmp_path):
    pdb, out = tmp_path / "sys.pdb", tmp_path / "sub.pdb"
    pdb.write_text(dipeptide_in_water())

    report = report_of(
        run(
            SUBSET,
            "--structure",
            str(pdb),
            "--selection",
            "resname TIP3",
            "--out",
            str(out),
        )
    )
    assert report["atoms"] == 6 and report["total"] == 26
    assert report["bonds"] is None or isinstance(report["bonds"], list)

    # A failed run removes the output it would have written, and leaves
    # alone one that was already there before it ran.
    fresh = tmp_path / "fresh.pdb"
    p = run(
        SUBSET,
        "--structure",
        str(pdb),
        "--selection",
        "resname XXX",
        "--out",
        str(fresh),
    )
    assert p.returncode == 1
    assert "holds no atoms" in p.stderr.strip().splitlines()[-1]
    assert not fresh.exists()
    p = run(
        SUBSET, "--structure", str(pdb), "--selection", "resname XXX", "--out", str(out)
    )
    assert p.returncode == 1
    assert (
        out.exists()
    ), "a file that was there before the run is not the run's to remove"


def test_a_tpr_structure_beside_a_top_is_still_its_own_topology(tmp_path, monkeypatch):
    """A .top reads only with its includes, so the .tpr stands in for it:
    the atoms' places are known and the report says which file served."""
    import MDAnalysis as mda

    calls = []
    real = mda.Universe

    class FakeUniverse:
        def __init__(self, *paths, **kw):
            calls.append(paths)
            self.u = real(str(tmp_path / "sys.pdb"))
            self.atoms = self.u.atoms

        def select_atoms(self, sel):
            return self.u.select_atoms(sel)

    (tmp_path / "sys.pdb").write_text(dipeptide_in_water())
    (tmp_path / "run.tpr").write_bytes(b"")
    (tmp_path / "topol.top").write_text("; includes")
    monkeypatch.setattr(mda, "Universe", FakeUniverse)

    r = ss.subset(
        str(tmp_path / "run.tpr"),
        str(tmp_path / "topol.top"),
        "all",
        str(tmp_path / "o.pdb"),
    )

    assert calls == [(str(tmp_path / "run.tpr"),)], "the .tpr is read, the .top is not"
    assert (
        r.from_topology and r.released and r.topology_used == str(tmp_path / "run.tpr")
    )
