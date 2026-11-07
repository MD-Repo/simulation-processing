"""Tests for maestro.py and desmond_convert.py

tiny-out.cms is a two-component system, a ligand with a virtual site and two
TIP4P waters, laid out to catch the order mistakes msys does not make: a
full_system block to skip, a residue name that is null, a bond listed both
ways, and virtual sites that the ffio_sites template interleaves but msys
puts last. Ported from mdrepo-agent's internal/desmond tests when the
conversion moved here (mdrepo-agent#98).
"""

import copy
import io
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import maestro  # noqa: E402
import desmond_convert as dc  # noqa: E402
from maestro import Particle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TINY = os.path.join(HERE, "inputs", "desmond", "tiny-out.cms")
SCRIPT = os.path.join(os.path.dirname(dc.__file__), "desmond_convert.py")


def tiny() -> maestro.System:
    return maestro.load(TINY)


def particle(**kw) -> Particle:
    base = dict(
        component=0,
        chain="",
        segid="",
        resid=1,
        resname="",
        insertion="",
        name="",
        atomic_number=6,
        x=0.0,
        y=0.0,
        z=0.0,
        pseudo=False,
        chain_index=0,
        residue_index=0,
    )
    base.update(kw)
    return Particle(**base)


# --------------------------------------------------
def test_particles_are_in_msys_order():
    """Each component's atoms, then its virtual sites; the full_system block
    skipped. Not the ffio_sites template order (O H H M for TIP4P)."""

    s = tiny()
    assert [p.name for p in s.particles] == "C1 C2 V1 O H1 H2 O H1 H2 M M".split()
    for p in s.particles:
        assert p.pseudo == (p.name in ("V1", "M"))
        if p.pseudo:
            assert p.atomic_number == 0
        assert p.x != 99, "coordinates came from the full_system block"


def test_chains_residues_and_bonds_follow_msys():
    s = tiny()
    # Chains: (component, name, segment): the ligand's chain A and the
    # water's chain W. Residues: LIG 1 for C1 and V1, UNK 1 for C2, whose
    # name is <>, and two waters.
    assert (s.chains, s.residues) == (2, 4)
    assert s.particles[1].resname == "UNK", "a null residue name is UNK"
    p = s.particles[0]
    assert (p.resname, p.name, p.chain, p.segid) == (
        "LIG",
        "C1",
        "A",
        "LIG",
    ), "names are trimmed"
    # The ligand's bond is listed both ways; msys keeps one.
    assert len(s.bonds) == 5
    assert s.bonds[0] == (0, 1) and s.bonds[1] == (
        3,
        4,
    ), "particle indices, lower first"
    assert s.box[0][0] == 50 and s.box[2][2] == 50


def test_atoms_drop_the_virtual_sites():
    atoms, bonds, keep = tiny().atoms()
    assert len(atoms) == 8
    assert keep == [0, 1, 3, 4, 5, 6, 7, 8]
    assert len(bonds) == 5 and bonds[1] == (2, 3), "bonds renumbered to atoms"


def test_the_tokenizer_reads_as_msys_does():
    toks = maestro.tokenize('a#comment# b "x \\" y" <> "<>" c[3]{ } # to eol\nd')
    assert [t.text for t in toks] == [
        "a#comment#",
        "b",
        'x " y',
        "<>",
        "<>",
        "c",
        "[",
        "3",
        "]",
        "{",
        "}",
        "d",
    ]
    assert (
        not toks[3].quoted and toks[4].quoted
    ), "an unquoted <> is null, a quoted one a string"


def test_values_read_like_atoi_and_atof():
    assert maestro.Value("12abc").int() == 12
    assert maestro.Value("-3").int() == -3
    assert maestro.Value("x").int(7) == 0
    assert maestro.NULL.int(7) == 7
    assert maestro.Value("1.5e2xyz").float() == 150.0
    assert maestro.NULL.float(2.5) == 2.5


def test_an_alchemical_file_is_refused():
    def ct(stage):
        return (
            "f_m_ct {\n s_m_title\n i_fepio_stage\n :::\n x %d\n m_atom[1] {\n i_m_atomic_number\n"
            " :::\n 1 6\n :::\n }\n}\n" % stage
        )

    with pytest.raises(maestro.AlchemicalError):
        maestro.parse(ct(1) + ct(2))


def test_the_force_field_names_are_reported():
    assert tiny().forcefields == ["tiny ff", "tip4p"]


def test_a_bad_bond_is_an_error():
    text = (
        "f_m_ct {\n s_m_title\n :::\n x\n m_atom[1] {\n i_m_atomic_number\n :::\n 1 6\n :::\n }\n"
        " m_bond[1] {\n i_m_from\n i_m_to\n :::\n 1 1 2\n :::\n }\n}\n"
    )
    with pytest.raises(maestro.MaestroError, match="nonexistent"):
        maestro.parse(text)


# --------------------------------------------------
def test_the_pdb_has_the_atoms_in_order():
    s = tiny()
    atoms, _, _ = s.atoms()
    buf = io.StringIO()
    dc.write_pdb(buf, atoms, s.box)
    lines = buf.getvalue().strip().splitlines()
    assert lines[0].startswith("CRYST1   50.000   50.000   50.000  90.00  90.00  90.00")
    records = [l for l in lines if l.startswith("ATOM")]
    assert len(records) == 8
    # Columns: name 13-16, residue 18-21, chain 22, residue number 23-26,
    # x 31-38, segment 73-76, element 77-78.
    r = records[2]
    assert r[12:16] == " O  "
    assert r[17:21].strip() == "T4P" and r[21] == "W" and r[22:26].strip() == "1"
    assert r[30:38].strip() == "20.000"
    assert r[72:76].strip() == "WAT" and r[76:78].strip() == "O"


def test_spaces_inside_names_become_underscores():
    atoms = [
        particle(name="H1 1", resname="UNK", segid="L"),
        particle(name="H1 2", resname="UNK", segid="L"),
        particle(name="C1", resname="MY LIG"),
        particle(name="CA", resname="ALA", segid="PRO"),
    ]
    assert dc.unspace(atoms) == 3
    assert [a.name for a in atoms] == ["H1_1", "H1_2", "C1", "CA"]
    assert atoms[2].resname == "MY_LIG"


def test_blank_names_are_given_their_element_and_a_number():
    atoms = [
        particle(name="", resname="UNK", residue_index=0, atomic_number=8),
        particle(name="", resname="UNK", residue_index=0, atomic_number=6),
        particle(name="C1", resname="LIG", residue_index=1, atomic_number=6),
        particle(name="", resname="LIG", residue_index=1, atomic_number=6),
        particle(name="", resname="UNK", residue_index=0, atomic_number=6),
        particle(name="CA", resname="ALA", residue_index=2, atomic_number=6),
        particle(name="", resname="UNK", residue_index=3, atomic_number=0),
    ]
    assert dc.name_blanks(atoms) == 5
    assert " ".join(a.name for a in atoms) == "O1 C1 C1 C2 C2 CA X1"
    assert dc.name_blanks(atoms) == 0, "a second pass has nothing to name"
    buf = io.StringIO()
    dc.write_pdb(buf, atoms, maestro.ZERO_BOX)
    for l in buf.getvalue().splitlines():
        if l.startswith("ATOM"):
            assert l[12:16].strip(), f"a PDB record has no name: {l!r}"


def test_a_split_residue_is_brought_together():
    """Maestro lists the hydrogens it added after every heavy atom; each
    residue is brought together, and the bonds and trajectory indices follow
    its atoms."""

    atoms = [
        particle(name="N", residue_index=0),
        particle(name="CA", residue_index=0),
        particle(name="N", residue_index=1),
        particle(name="CA", residue_index=1),
        particle(name="H", residue_index=0),
        particle(name="H", residue_index=1),
        particle(name="O", residue_index=2),
    ]
    bonds = [(0, 1), (0, 4), (1, 2), (2, 5)]
    keep = [0, 1, 2, 3, 5, 6, 7]  # particle 4 a virtual site

    got, got_bonds, got_keep, n = dc.by_residue(atoms, bonds, keep)

    assert n == 2
    assert (
        " ".join(f"{a.name}{a.residue_index}" for a in got) == "N0 CA0 H0 N1 CA1 H1 O2"
    )
    assert got_keep == [0, 1, 5, 2, 3, 6, 7], "each atom keeps its own particle"
    assert got_bonds == [(0, 1), (0, 2), (1, 3), (3, 5)]

    in_order = [
        particle(residue_index=0),
        particle(residue_index=0),
        particle(residue_index=1),
    ]
    assert dc.by_residue(in_order, [], [0, 1, 2])[3] == 0


# --------------------------------------------------
pytest.importorskip("mdtraj")


def write_dtr(trj: str, order: str) -> None:
    """Frames of the tiny system with mdtraj, in the given particle order, at
    the Desmond layout's <run>_trj."""

    import numpy as np
    import mdtraj as md

    # msys order: C1 C2 V1 | O H1 H2 O H1 H2 M M. The template order
    # interleaves M.
    atoms = {
        "C1": (10, 10, 10), "C2": (11.5, 10, 10), "V1": (10.5, 10.5, 10),
        "O1": (20, 20, 20), "H11": (20.96, 20, 20), "H21": (19.76, 20.93, 20), "M1": (20.1, 20.1, 20),
        "O2": (30, 30, 30), "H12": (30.96, 30, 30), "H22": (29.76, 30.93, 30), "M2": (30.1, 30.1, 30),
    }  # fmt: skip
    names = {
        "msys": "C1 C2 V1 O1 H11 H21 O2 H12 H22 M1 M2",
        "template": "C1 C2 V1 O1 H11 H21 M1 O2 H12 H22 M2",
    }[order].split()
    f0 = np.array([atoms[n] for n in names], dtype=np.float32)
    shift = np.array([1.5, -2.0, 0.5], dtype=np.float32)
    xyz = np.stack([f0 + shift, f0])  # frame 1 is the .cms frame exactly
    tmp = trj + ".dtr"
    with md.formats.DTRTrajectoryFile(tmp, "w") as f:
        f.write(
            xyz,
            times=np.array([0.0, 10.0]),
            cell_lengths=np.full((2, 3), 50.0, np.float32),
            cell_angles=np.full((2, 3), 90.0, np.float32),
        )
    os.rename(tmp, trj)


def test_a_converted_run_maps_every_atom(tmp_path):
    import mdtraj as md

    trj = str(tmp_path / "tiny_trj")
    write_dtr(trj, "msys")
    pdb, xtc, bonds = (
        str(tmp_path / "s.pdb"),
        str(tmp_path / "s.xtc"),
        str(tmp_path / "b.json"),
    )

    r = dc.convert(TINY, trj, pdb, xtc, bonds)

    assert (r.atoms, r.particles, r.pseudo, r.frames, r.dt) == (8, 11, 3, 2, 10)
    assert r.pseudo_on == {"LIG": 1, "T4P": 2}, "the solute's one and the waters' two"
    assert r.match_frame == 1 and r.match_spread <= 1e-4, "the .cms is frame 1 exactly"
    assert r.matched
    assert r.bonds == 5 and r.forcefields == ["tiny ff", "tip4p"]
    assert r.renamed == 0 and r.named == 0 and r.regrouped == 0
    with open(bonds) as fh:
        assert json.load(fh) == [[0, 1], [2, 3], [2, 4], [5, 6], [5, 7]]

    # The .xtc holds the atoms, in nm, without the virtual sites.
    t = md.load(xtc, top=pdb)
    assert (t.n_frames, t.n_atoms) == (2, 8)
    assert round(float(t.xyz[1][2][0]), 3) == 2.0, "O1 x"
    assert round(float(t.xyz[1][7][1]), 3) == 3.093, "H22 y"


def test_a_misordered_trajectory_is_refused(tmp_path):
    """The same frames in the ffio_sites template order: the mapping this
    replaced. It must not convert quietly."""

    trj = str(tmp_path / "tiny_trj")
    write_dtr(trj, "template")
    with pytest.raises(ValueError, match="do not map"):
        dc.convert(TINY, trj, str(tmp_path / "s.pdb"), str(tmp_path / "s.xtc"))


def test_a_trajectory_of_another_system_is_refused(tmp_path):
    import numpy as np
    import mdtraj as md

    trj = str(tmp_path / "other_trj")
    with md.formats.DTRTrajectoryFile(trj + ".dtr", "w") as f:
        f.write(np.zeros((1, 7, 3), np.float32), times=np.array([0.0]),
                cell_lengths=np.full((1, 3), 50.0, np.float32), cell_angles=np.full((1, 3), 90.0, np.float32))  # fmt: skip
    os.rename(trj + ".dtr", trj)
    with pytest.raises(ValueError, match="not the same system"):
        dc.convert(TINY, trj, str(tmp_path / "s.pdb"), str(tmp_path / "s.xtc"))


def test_without_a_trajectory_only_the_pdb_is_written(tmp_path):
    pdb = str(tmp_path / "s.pdb")
    r = dc.convert(TINY, None, pdb, None)
    assert r.atoms == 8 and r.frames == 0 and r.match_frame == -1 and not r.matched
    assert os.path.exists(pdb)
    json.dumps(r._asdict())  # nothing in the report that JSON cannot carry


def test_command_reports_json_and_cleans_up_on_failure(tmp_path):
    trj = str(tmp_path / "tiny_trj")
    write_dtr(trj, "msys")
    pdb, xtc = str(tmp_path / "s.pdb"), str(tmp_path / "s.xtc")

    p = subprocess.run(
        [sys.executable, SCRIPT, "--cms", TINY, "--trajectory", trj, "--out-pdb", pdb, "--out-xtc", xtc],
        capture_output=True, text=True,
    )  # fmt: skip
    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout.strip().splitlines()[-1])
    assert report["atoms"] == 8 and report["frames"] == 2 and report["match_frame"] == 1

    write_dtr(str(tmp_path / "bad_trj"), "template")
    p = subprocess.run(
        [sys.executable, SCRIPT, "--cms", TINY, "--trajectory", str(tmp_path / "bad_trj"),
         "--out-pdb", str(tmp_path / "t.pdb"), "--out-xtc", str(tmp_path / "t.xtc")],
        capture_output=True, text=True,
    )  # fmt: skip
    assert p.returncode == 1
    assert "do not map" in p.stderr.strip().splitlines()[-1]
    assert not os.path.exists(tmp_path / "t.pdb") and not os.path.exists(
        tmp_path / "t.xtc"
    )


def test_a_negative_residue_number_stays_negative_in_the_pdb():
    """Maestro numbers N-terminal caps -1 and 0; folding -1 to 9999 would
    make it another residue."""
    assert dc.wrap_resid(-1) == -1 and dc.wrap_resid(0) == 0
    assert (
        dc.wrap_resid(9999) == 9999
        and dc.wrap_resid(10000) == 0
        and dc.wrap_resid(12345) == 2345
    )
    buf = io.StringIO()
    dc.write_pdb(buf, [particle(name="C", resname="ACE", resid=-1)], maestro.ZERO_BOX)
    assert buf.getvalue().splitlines()[0][22:26] == "  -1"


def test_frame_steps_run_on_across_chunks(tmp_path, monkeypatch):
    import numpy as np
    import mdtraj as md

    monkeypatch.setattr(dc, "CHUNK", 1)  # every frame its own write call
    trj = str(tmp_path / "tiny_trj")
    write_dtr(trj, "msys")
    pdb, xtc = str(tmp_path / "s.pdb"), str(tmp_path / "s.xtc")
    dc.convert(TINY, trj, pdb, xtc)
    with md.formats.XTCTrajectoryFile(xtc) as f:
        _, _, step, _ = f.read()
    assert step.tolist() == [0, 1]
