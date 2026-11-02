"""Tests for extract_bonds.py

Each case is one met on 2026-11-02 in a 40-simulation sample of the corpus
(work/psf-writer-2026-11-02 on Ken's laptop): BIOVIA's "!NBONDS" psf
header, full.pdb holding fewer atoms than the topology and in another
order, gmx trjconv's 1HD1-style hydrogen names, the bonds a plain vdW
guess misses (S-S, C-Br), the false S-H contact the heavy-only radii avoid,
MISATO's prmtop named .top, and a .top whose includes were not uploaded.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import extract_bonds as eb  # noqa: E402


# --------------------------------------------------
def pdb_text(atoms, box=None):
    """atoms: (name, resname, resid, element, x, y, z)"""

    lines = []
    if box:
        lines.append(
            f"CRYST1{box:9.3f}{box:9.3f}{box:9.3f}  90.00  90.00  90.00 P 1"
        )
    for k, (name, resname, resid, element, x, y, z) in enumerate(atoms, 1):
        field = f" {name:<3}" if len(name) < 4 else name
        lines.append(
            f"ATOM  {k:5d} {field} {resname:<4}A{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def psf_text(atoms, bonds, header="!NBOND: bonds"):
    """atoms: (segid, resid, resname, name); bonds 1-based"""

    lines = [
        "PSF EXT",
        "",
        "         1 !NTITLE",
        "* test",
        "",
        f"{len(atoms):10d} !NATOM",
    ]
    for k, (seg, resid, resname, name) in enumerate(atoms, 1):
        lines.append(
            f"{k:10d} {seg:<8} {resid:<8} {resname:<8} {name:<8} "
            f"X        0.000000       12.0110           0"
        )
    lines += ["", f"{len(bonds):10d} {header}"]
    flat = [str(i) for b in bonds for i in b]
    for k in range(0, len(flat), 8):
        lines.append(" ".join(f"{x:>10}" for x in flat[k : k + 8]))
    lines += ["", "         0 !NTHETA: angles", ""]
    return "\n".join(lines) + "\n"


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# A tiny system: ethanol-like residue LIG (C1-C2-O1 + H on O1),
# a water, and a sodium ion.
LIG = [
    ("C1", "LIG", 1, "C", 0.0, 0.0, 0.0),
    ("C2", "LIG", 1, "C", 1.52, 0.0, 0.0),
    ("O1", "LIG", 1, "O", 2.0, 1.35, 0.0),
    ("HO1", "LIG", 1, "H", 2.95, 1.35, 0.0),
]
NA = [("NA", "NA", 3, "NA", 6.0, 6.0, 6.0)]
WATER_PSF = [
    ("W", "2", "TIP3", "OH2"),
    ("W", "2", "TIP3", "H1"),
    ("W", "2", "TIP3", "H2"),
]


# --------------------------------------------------
def test_atom_key_turns_trjconv_hydrogen_names_back():
    assert eb.atom_key("1HD1") == "HD11"
    assert eb.atom_key("HD11") == "HD11"
    assert eb.atom_key(" CA ") == "CA"
    assert eb.atom_key("C1234") == "C123"


# --------------------------------------------------
def test_source_ranking_prefers_tpr_and_sniffs_prmtop(tmp_path):
    tpr = write(tmp_path, "md.tpr", "x")
    psf = write(tmp_path, "sys.psf", "x")
    misato = write(tmp_path, "production.top", "%VERSION  VERSION_STAMP\n")
    gmx = write(tmp_path, "topol.top", "[ defaults ]\n")
    other = write(tmp_path, "notes.txt", "x")
    ranked = eb.rank_sources([gmx, other, misato, psf, tpr])
    assert ranked == [
        ("tpr", tpr),
        ("psf", psf),
        ("prmtop", misato),
        ("top", gmx),
    ]


# --------------------------------------------------
@pytest.mark.parametrize("header", ["!NBOND: bonds", "!NBONDS"])
def test_psf_bonds_under_either_header(tmp_path, header):
    atoms = [("A", "1", "LIG", n) for n in ("C1", "C2", "O1", "HO1")]
    psf = write(
        tmp_path, "a.psf", psf_text(atoms, [(1, 2), (2, 3), (3, 4)], header)
    )
    top = eb.read_psf(psf)
    assert top.num_atoms == 4
    assert top.bonds == [(0, 1), (1, 2), (2, 3)]
    assert [r.name for r in top.residues] == ["LIG"]


# --------------------------------------------------
def test_psf_that_lists_too_few_bonds_is_refused(tmp_path):
    atoms = [("A", "1", "LIG", n) for n in ("C1", "C2")]
    text = psf_text(atoms, [(1, 2)]).replace(
        "         1 !NBOND", "         2 !NBOND"
    )
    with pytest.raises(ValueError, match="states 2 bonds"):
        eb.read_psf(write(tmp_path, "a.psf", text))


# --------------------------------------------------
def test_topology_bonds_map_onto_a_stripped_reordered_pdb(tmp_path):
    """The topology holds LIG, a water, then the ion; the pdb has no water
    and its LIG atoms in another order (as MISATO's do)."""

    top_atoms = (
        [("A", "1", "LIG", n) for n in ("C1", "C2", "O1", "HO1")]
        + WATER_PSF
        + [("I", "3", "NA", "NA")]
    )
    bonds = [(1, 2), (2, 3), (3, 4), (5, 6), (5, 7), (6, 7)]
    psf = write(tmp_path, "sys.psf", psf_text(top_atoms, bonds))
    reordered = [LIG[2], LIG[0], LIG[3], LIG[1]]  # O1, C1, HO1, C2
    pdb = write(tmp_path, "full.pdb", pdb_text(reordered + NA))

    result = eb.extract(pdb, [psf])
    assert result.source == "psf"
    assert result.num_atoms == 5
    # C1-C2, C2-O1, O1-HO1 in the pdb's order: O1=0, C1=1, HO1=2, C2=3
    assert result.bonds == [(0, 2), (0, 3), (1, 3)]
    assert result.dropped == {"TIP3": 1}
    assert (result.guess_missing, result.guess_extra) == (0, 0)
    assert result.warnings == []


# --------------------------------------------------
def test_a_source_that_does_not_map_falls_through_to_the_guess(tmp_path):
    other = [("A", "1", "ALA", n) for n in ("N", "CA", "C", "O")]
    psf = write(
        tmp_path, "other.psf", psf_text(other, [(1, 2), (2, 3), (3, 4)])
    )
    pdb = write(tmp_path, "full.pdb", pdb_text(LIG + NA))

    result = eb.extract(pdb, [psf])
    assert result.source == "guessed"
    assert result.bonds == [(0, 1), (1, 2), (2, 3)]
    assert any("other.psf not used" in w for w in result.warnings)


# --------------------------------------------------
def test_a_gromacs_top_with_missing_includes_falls_through(tmp_path):
    top = write(
        tmp_path,
        "topol.top",
        '#include "charmm36m.ff/forcefield.itp"\n[ system ]\nx\n'
        "[ molecules ]\nLIG 1\n",
    )
    pdb = write(tmp_path, "full.pdb", pdb_text(LIG))
    result = eb.extract(pdb, [top])
    assert result.source == "guessed"
    assert any("topol.top not used" in w for w in result.warnings)


# --------------------------------------------------
def test_guess_takes_disulfides_and_c_br_but_not_ions_or_h_h(tmp_path):
    atoms = [
        ("SG", "CYS", 1, "S", 0.0, 0.0, 0.0),
        ("SG", "CYS", 2, "S", 2.10, 0.0, 0.0),  # S-S 2.10: a bond
        ("C8", "LIG", 3, "C", 0.0, 5.0, 0.0),
        ("BR1", "LIG", 3, "BR", 1.98, 5.0, 0.0),  # C-Br 1.98: a bond
        ("S2", "LI2", 4, "S", 0.0, 10.0, 0.0),
        ("H5", "LI2", 4, "H", 1.70, 10.0, 0.0),  # S-H 1.70: a contact
        ("O", "WAT", 5, "O", 0.0, 15.0, 0.0),
        ("NA", "NA", 6, "NA", 2.30, 15.0, 0.0),  # Na-O 2.30: an ion
        ("H1", "WAT", 7, "H", 0.0, 20.0, 0.0),
        ("H2", "WAT", 7, "H", 0.75, 20.0, 0.0),  # H-H: never
    ]
    pdb = write(tmp_path, "full.pdb", pdb_text(atoms))
    result = eb.extract(pdb, [])
    assert result.source == "guessed"
    assert result.bonds == [(0, 1), (2, 3)]


# --------------------------------------------------
def test_guess_bonds_across_the_periodic_boundary(tmp_path):
    atoms = [
        ("C1", "LIG", 1, "C", 0.2, 5.0, 5.0),
        ("C2", "LIG", 1, "C", 18.7, 5.0, 5.0),
    ]  # 1.5 A apart through x
    pdb = write(tmp_path, "full.pdb", pdb_text(atoms, box=20.0))
    assert eb.extract(pdb, []).bonds == [(0, 1)]


# --------------------------------------------------
def test_element_falls_back_to_the_atom_name():
    assert eb.element_of("  ", "CL1") == "CL"
    assert eb.element_of("", "1HD1") == "H"
    assert eb.element_of(" N", "N") == "N"


# --------------------------------------------------
def test_main_writes_serials_and_summary(tmp_path, monkeypatch, capsys):
    pdb = write(tmp_path, "full.pdb", pdb_text(LIG))
    out, js = str(tmp_path / "bonds.tsv"), str(tmp_path / "bonds.json")
    monkeypatch.setattr(
        sys, "argv", ["extract_bonds.py", "-p", pdb, "-o", out, "-s", js]
    )
    eb.main()
    assert open(out).read() == "1\t2\n2\t3\n3\t4\n"
    summary = json.load(open(js))
    assert summary["source"] == "guessed"
    assert summary["num_bonds"] == 3
    assert "3 bonds for 4 atoms" in capsys.readouterr().out
