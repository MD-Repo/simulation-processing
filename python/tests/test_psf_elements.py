"""Tests for psf_elements.py

Ported from mdrepo-agent's internal/stage/elements_test.go when the element
filling moved here (mdrepo-agent#98). The repartitioned masses are
zenodo:17477539's: methyl carbons of 5.963 beside hydrogens of 3.024.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psf_elements as pe  # noqa: E402


# --------------------------------------------------
def psf_of(tmp_path, atoms, bonds):
    """A .psf of the atoms, each "segid resid resname name type charge mass",
    and the bonds, 1-based."""

    lines = [
        "PSF EXT",
        "",
        "         1 !NTITLE",
        " REMARKS test",
        "",
        "%10d !NATOM" % len(atoms),
    ]
    for i, a in enumerate(atoms):
        f = a.split()
        lines.append(
            "%10d %-8s %-8s %-8s %-8s %-6s %14s%14s           0"
            % (i + 1, f[0], f[1], f[2], f[3], f[4], f[5], f[6])
        )
    lines += ["", "%10d !NBOND: bonds" % len(bonds)]
    row = ""
    for i, (a, b) in enumerate(bonds):
        row += "%10d%10d" % (a, b)
        if i % 4 == 3 or i == len(bonds) - 1:
            lines.append(row)
            row = ""
    lines += ["", "         0 !NTHETA: angles", ""]
    p = tmp_path / "sys.psf"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def pdb_of(tmp_path, names):
    """A PDB of the atoms with a blank element column, as MDAnalysis writes
    one from a .psf."""

    lines = ["CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1"]
    for i, n in enumerate(names):
        lines.append(
            "ATOM  %5d %-4s RES X   1       0.000   0.000   0.000  1.00  0.00      SEG "
            % (i + 1, n)
        )
    lines.append("END")
    p = tmp_path / "sys.pdb"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def element_columns(pdb):
    cols = []
    with open(pdb) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("ATOM  "):
                assert len(line) == 80, f"line is {len(line)} columns: {line!r}"
                cols.append(line[76:80])
    return cols


# --------------------------------------------------
def test_elements_come_from_masses_not_names(tmp_path):
    """HE21 is a hydrogen and CA a carbon; repartitioned mass is counted
    back; a lone ion with a whole charge is written with it; a massless
    site gets no element."""

    psf = psf_of(
        tmp_path,
        [
            "LIG 1 LIG C1 CG331 -0.27 5.963",  # a methyl carbon, lending 3 x 2.016
            "LIG 1 LIG H11 HGA3 0.09 3.024",
            "LIG 1 LIG H12 HGA3 0.09 3.024",
            "LIG 1 LIG H13 HGA3 0.09 3.024",
            "PRO 2 GLN NE2 NH2 -0.62 9.975",  # an amide nitrogen, lending 2 x 2.016
            "PRO 2 GLN HE21 H 0.32 3.024",
            "PRO 2 GLN HE22 H 0.30 3.024",
            "PRO 3 ALA CA CT1 0.07 12.011",  # no hydrogens bonded here
            "WAT 4 TIP3 OH2 OT -0.834 15.9994",
            "WAT 4 TIP3 H1 HT 0.417 1.008",  # unrepartitioned water, bonded H-H
            "WAT 4 TIP3 H2 HT 0.417 1.008",
            "ION 5 CAL CAL CAL 2.00 40.08",
            "ION 6 CLA CLA CLA -1.00 35.45",
            "LIG 7 LP LP1 LP 0.00 0.000",  # a massless site
        ],
        [(1, 2), (1, 3), (1, 4), (5, 6), (5, 7), (9, 10), (9, 11), (10, 11), (1, 14)],
    )
    pdb = pdb_of(
        tmp_path,
        [
            "C1",
            "H11",
            "H12",
            "H13",
            "NE2",
            "HE21",
            "HE22",
            "CA",
            "OH2",
            "H1",
            "H2",
            "CAL",
            "CLA",
            "LP1",
        ],
    )

    e = pe.fill_elements(pdb, psf)

    want = [
        " C  ",
        " H  ",
        " H  ",
        " H  ",
        " N  ",
        " H  ",
        " H  ",
        " C  ",
        " O  ",
        " H  ",
        " H  ",
        "CA2+",
        "CL1-",
        "    ",
    ]
    assert element_columns(pdb) == want
    assert e == pe.Elements(psf, repartitioned=2, ions=2)
    assert e.as_dict() == {"from": psf, "repartitioned": 2, "ions": 2}


def test_elements_follow_the_selected_atoms(tmp_path):
    """A subset's atoms are the topology's atoms it names, in order."""

    psf = psf_of(
        tmp_path,
        [
            "A 1 ALA N NH1 -0.47 14.007",
            "A 1 ALA C C 0.51 12.011",
            "A 1 ALA O O -0.51 15.999",
            "B 2 SOD SOD SOD 1.00 22.990",
        ],
        [(1, 2), (2, 3)],
    )
    pdb = pdb_of(tmp_path, ["O", "SOD"])

    pe.fill_elements(pdb, psf, [2, 3])

    assert element_columns(pdb) == [" O  ", "NA1+"]


def test_elements_are_not_guessed(tmp_path):
    """A mass that matches no element is not guessed at, and neither is a
    PDB that does not hold the topology's atoms."""

    psf = psf_of(tmp_path, ["CG 1 BEAD BB P5 0.0 72.0"], [])
    with pytest.raises(pe.ElementError, match="matches no element"):
        pe.fill_elements(pdb_of(tmp_path, ["BB"]), psf)

    psf = psf_of(
        tmp_path, ["A 1 ALA N NH1 -0.47 14.007", "A 1 ALA C C 0.51 12.011"], [(1, 2)]
    )
    with pytest.raises(pe.ElementError, match="holds 1 atoms and 2"):
        pe.fill_elements(pdb_of(tmp_path, ["N"]), psf)


def test_two_elements_within_tolerance_are_an_error(tmp_path):
    """Nothing in the table is within 0.1 of two elements, so a mass that
    lands between two is impossible by construction; a repartitioned mass
    that lands on none says so with the hydrogens' mass counted back."""

    psf = psf_of(
        tmp_path,
        ["LIG 1 LIG C1 CG331 -0.27 5.500", "LIG 1 LIG H11 HGA3 0.09 3.024"],
        [(1, 2)],
    )
    with pytest.raises(pe.ElementError, match=r"with its hydrogens'"):
        pe.fill_elements(pdb_of(tmp_path, ["C1", "H11"]), psf)


def test_pdb_lacks_elements(tmp_path):
    blank = pdb_of(tmp_path, ["N", "C"])
    assert pe.pdb_lacks_elements(blank)
    filled = tmp_path / "f.pdb"
    filled.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \nEND\n"
    )
    assert not pe.pdb_lacks_elements(str(filled))


def test_read_psf_reads_the_standard_layout_too(tmp_path):
    p = tmp_path / "s.psf"
    p.write_text(
        "PSF\n\n       1 !NTITLE\n REMARKS ten\n\n       2 !NATOM\n"
        "       1 A        1        ALA      N        NH1   -0.470000       14.0070           0\n"
        "       2 A        1        ALA      C        C      0.510000       12.0110           0\n"
        "\n       1 !NBOND: bonds\n       1       2\n\n       0 !NTHETA: angles\n\n"
    )
    t = pe.read_psf(str(p))
    assert t.name == ["N", "C"]
    assert t.mass == [14.007, 12.011]
    assert t.bonds == [(0, 1)]
    assert t.residue == ["A 1 ALA", "A 1 ALA"]


def test_read_psf_refuses_a_file_without_atoms(tmp_path):
    p = tmp_path / "s.psf"
    p.write_text("PSF\n\n       0 !NBOND\n")
    with pytest.raises(pe.ElementError, match="no !NATOM"):
        pe.read_psf(str(p))
