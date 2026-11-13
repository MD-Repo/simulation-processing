"""rebuild_minimal.py and write_minimal_commands (MISATO's lost ligand, #30)"""

import hashlib
import io
import json
import os
import shutil

import numpy as np
import pytest

import cpptraj_gmx_traj_manipulation as conv
import rebuild_minimal as rb

GMX = shutil.which("gmx") or (rb.GMX if os.path.exists(rb.GMX) else None)
needs_gmx = pytest.mark.skipif(GMX is None, reason="no gmx")


# --- write_minimal_commands ----------------------------------------------------
def commands(tmp_path, lipid=False, tpr=False):
    if tpr:
        (tmp_path / "full.tpr").write_text("")
    f = io.StringIO()
    conv.write_minimal_commands(f, str(tmp_path), lipid)
    return f.getvalue()


def test_minimal_commands_strip_ions_not_mol(tmp_path):
    text = commands(tmp_path)
    line = next(l for l in text.splitlines() if "& !r" in l)
    for ion in conv.KNOWN_IONS:
        assert ion in line.split("!r", 1)[1].split()
    assert "MOL" not in line
    assert (
        "-princ" in text
        and "-rotate 0 0 90" in text
        and "-fit rot+trans" in text
    )
    assert "minimal.gro" in text and "convert-tpr" not in text


def test_minimal_commands_strip_lipids_when_present(tmp_path):
    line = next(
        l for l in commands(tmp_path, lipid=True).splitlines() if "& !r" in l
    )
    for lipid in conv.KNOWN_LIPIDS:
        assert lipid in line
    line = next(l for l in commands(tmp_path).splitlines() if "& !r" in l)
    assert "POPC" not in line


def test_minimal_commands_use_a_tpr_when_there_is_one(tmp_path):
    assert "convert-tpr" in commands(tmp_path, tpr=True)


# --- residues ------------------------------------------------------------------
def pdb_text(residues):
    """ATOM lines for [(resname, resid, atoms)], cpptraj-style 4-letter names"""
    lines, i = [], 1
    for name, resid, n in residues:
        res = f"{name:>4}" if len(name) == 4 else f" {name:<3}"
        for k in range(n):
            lines.append(
                f"ATOM  {i:5}  C{k:<2}{res} A{resid:4}    "
                f"{i * 1.0:8.3f}{k * 1.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            )
            i += 1
    return "\n".join(lines) + "\nEND\n"


def write(tmp_path, name, residues):
    p = tmp_path / name
    p.write_text(pdb_text(residues))
    return str(p)


def test_residue_counts_read_4_letter_names(tmp_path):
    p = write(
        tmp_path,
        "a.pdb",
        [("ALA", 1, 3), ("MOL", 2, 2), ("TIP3", 3, 3), ("POPC", 4, 5)],
    )
    assert rb.residue_counts(p) == {"ALA": 1, "MOL": 1, "TIP3": 1, "POPC": 1}


def test_residue_counts_stop_at_the_first_model(tmp_path):
    p = tmp_path / "m.pdb"
    one = pdb_text([("ALA", 1, 2)]).replace("END\n", "")
    p.write_text("MODEL 1\n" + one + "ENDMDL\nMODEL 2\n" + one + "ENDMDL\n")
    assert rb.residue_counts(str(p)) == {"ALA": 1}


@pytest.mark.parametrize(
    "name", ["SOL", "sol", "TIP3", "NA", "Cl-", "POPC", "OHH", "T3H"]
)
def test_dropped_on_purpose(name):
    assert rb.dropped(name)


@pytest.mark.parametrize("name", ["MOL", "LIG", "ZN", "CA", "ALA", "HEM"])
def test_not_dropped(name):
    assert not rb.dropped(name)


def test_missing_from_minimal(tmp_path):
    full = write(
        tmp_path,
        "f.pdb",
        [("ALA", 1, 2), ("MOL", 2, 2), ("SOL", 3, 3), ("ZN", 4, 1)],
    )
    good = write(
        tmp_path, "g.pdb", [("ALA", 1, 2), ("MOL", 2, 2), ("ZN", 4, 1)]
    )
    bad = write(tmp_path, "b.pdb", [("ALA", 1, 2)])
    assert rb.missing_from_minimal(full, good) == []
    assert rb.missing_from_minimal(full, bad) == [
        "MOL (0 of 1)",
        "ZN (0 of 1)",
    ]


# --- arguments -----------------------------------------------------------------
def inputs(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "full.pdb").write_text("x")
    (d / "full.xtc").write_text("x")
    return str(d)


def test_args_refuse_writing_into_the_input(tmp_path):
    d = inputs(tmp_path)
    with pytest.raises(SystemExit):
        rb.get_args(["-i", d, "-o", d])


@pytest.mark.parametrize("dt", ["0", "-80"])
def test_args_refuse_a_bad_dt(tmp_path, dt):
    with pytest.raises(SystemExit):
        rb.get_args(
            ["-i", inputs(tmp_path), "-o", str(tmp_path / "o"), "--dt", dt]
        )


@pytest.mark.parametrize("gone", ["full.pdb", "full.xtc"])
def test_args_need_both_inputs(tmp_path, gone):
    d = inputs(tmp_path)
    os.remove(os.path.join(d, gone))
    with pytest.raises(SystemExit):
        rb.get_args(["-i", d, "-o", str(tmp_path / "o")])


def test_args_defaults(tmp_path):
    a = rb.get_args(["-i", inputs(tmp_path), "-o", str(tmp_path / "o")])
    assert (a.t0, a.dt, a.keep_temp) == (0.0, None, False)


# --- end to end with gmx --------------------------------------------------------
def system(tmp_path, frames=5, natoms_xtc=None):
    """Two ALA, a MOL ligand, two SOL waters and an NA: full.pdb + full.xtc
    with every frame at 0 ps (MISATO's XTCs)"""
    import MDAnalysis as mda

    d = tmp_path / "in"
    d.mkdir()
    atoms = (
        [
            ("N", "ALA", 1, "N"),
            ("CA", "ALA", 1, "C"),
            ("C", "ALA", 1, "C"),
            ("O", "ALA", 1, "O"),
        ]
        + [
            ("N", "ALA", 2, "N"),
            ("CA", "ALA", 2, "C"),
            ("C", "ALA", 2, "C"),
            ("O", "ALA", 2, "O"),
        ]
        + [
            ("C1", "MOL", 3, "C"),
            ("C2", "MOL", 3, "C"),
            ("O1", "MOL", 3, "O"),
        ]
        + [
            ("OW", "SOL", 4, "O"),
            ("HW1", "SOL", 4, "H"),
            ("HW2", "SOL", 4, "H"),
        ]
        + [
            ("OW", "SOL", 5, "O"),
            ("HW1", "SOL", 5, "H"),
            ("HW2", "SOL", 5, "H"),
        ]
        + [("NA", "NA", 6, "Na")]
    )
    rng = np.random.default_rng(7)
    base = rng.uniform(5, 15, size=(len(atoms), 3))
    lines = [
        "CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1"
    ]
    for i, ((name, res, rid, el), xyz) in enumerate(zip(atoms, base), 1):
        lines.append(
            f"ATOM  {i:5} {name:<4} {res:<3} A{rid:4}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2}"
        )
    (d / "full.pdb").write_text("\n".join(lines) + "\nEND\n")
    u = mda.Universe(str(d / "full.pdb"))
    n = natoms_xtc or len(atoms)
    with mda.coordinates.XTC.XTCWriter(str(d / "full.xtc"), n) as w:
        for k in range(frames):
            u.trajectory.ts.time = 0.0
            pos = base + rng.normal(0, 0.3, base.shape)
            if n != len(atoms):
                u2 = mda.Universe.empty(n, trajectory=True)
                u2.dimensions = [30, 30, 30, 90, 90, 90]
                u2.atoms.positions = pos[:n]
                w.write(u2.atoms)
            else:
                u.atoms.positions = pos
                w.write(u.atoms)
    return str(d)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


@needs_gmx
def test_rebuild_keeps_the_ligand_and_writes_times(tmp_path):
    d = system(tmp_path)
    before = {n: md5(os.path.join(d, n)) for n in os.listdir(d)}
    out = str(tmp_path / "out")
    assert rb.main(["-i", d, "-o", out, "--dt", "80", "-g", GMX]) == 0
    assert {
        n: md5(os.path.join(d, n)) for n in os.listdir(d)
    } == before  # inputs untouched
    # only the outputs; no temps, gmx backups or MDAnalysis offsets
    assert sorted(os.listdir(out)) == [
        "full.pdb",
        "full.xtc",
        "minimal.gro",
        "minimal.pdb",
        "minimal.xtc",
        "rebuild.json",
        "rebuild_minimal.sh",
        "sampled.xtc",
    ]

    res = rb.residue_counts(os.path.join(out, "minimal.pdb"))
    assert res == {"ALA": 2, "MOL": 1}
    report = json.load(open(os.path.join(out, "rebuild.json")))
    assert report["problems"] == []
    assert report["frames"] == {"full": 5, "minimal": 5, "sampled": 5}
    assert (report["full_atoms"], report["minimal_atoms"]) == (18, 11)

    t_full, x_full = rb.read_xtc(os.path.join(out, "full.xtc"))
    _, x_in = rb.read_xtc(os.path.join(d, "full.xtc"))
    assert np.allclose(t_full, [0, 80, 160, 240, 320])
    assert np.abs(x_full - x_in).max() < 1e-3
    t_min, _ = rb.read_xtc(os.path.join(out, "minimal.xtc"))
    assert np.allclose(t_min, t_full)
    for temp in ("minimal_temp.xtc", "min_princ.pdb", "index_minimal.ndx"):
        assert not os.path.exists(os.path.join(out, temp))


@needs_gmx
def test_rebuild_t0_and_keep_temp(tmp_path):
    d = system(tmp_path, frames=3)
    out = str(tmp_path / "out")
    assert (
        rb.main(
            [
                "-i",
                d,
                "-o",
                out,
                "--dt",
                "10",
                "--t0",
                "1000",
                "-g",
                GMX,
                "--keep-temp",
            ]
        )
        == 0
    )
    t, _ = rb.read_xtc(os.path.join(out, "full.xtc"))
    assert np.allclose(t, [1000, 1010, 1020])
    assert os.path.exists(os.path.join(out, "minimal_temp.xtc"))


@needs_gmx
def test_rebuild_without_dt_keeps_the_times(tmp_path):
    d = system(tmp_path, frames=3)
    out = str(tmp_path / "out")
    assert rb.main(["-i", d, "-o", out, "-g", GMX]) == 0
    assert md5(os.path.join(out, "full.xtc")) == md5(
        os.path.join(d, "full.xtc")
    )
    assert "time_axis" not in json.load(
        open(os.path.join(out, "rebuild.json"))
    )


@needs_gmx
def test_rebuild_fails_on_mismatched_inputs(tmp_path):
    d = system(
        tmp_path, frames=3, natoms_xtc=10
    )  # xtc has fewer atoms than the pdb
    out = str(tmp_path / "out")
    assert rb.main(["-i", d, "-o", out, "--dt", "80", "-g", GMX]) == 1
    assert json.load(open(os.path.join(out, "rebuild.json")))["error"]


# --- real data ------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("REBUILD_MINIMAL_REAL"),
    reason="set REBUILD_MINIMAL_REAL",
)
def test_real_misato_sim(tmp_path):
    """REBUILD_MINIMAL_REAL=<dir with a released full.pdb, full.xtc,
    minimal.pdb, minimal.xtc> (e.g. MDR00015163's): the rebuild is the old
    minimal plus the ligand, the protein within XTC rounding"""
    import MDAnalysis as mda

    src = os.environ["REBUILD_MINIMAL_REAL"]
    d = tmp_path / "in"
    d.mkdir()
    for n in ("full.pdb", "full.xtc"):
        shutil.copy(os.path.join(src, n), d / n)
    out = str(tmp_path / "out")
    assert (
        rb.main(["-i", str(d), "-o", out, "--dt", "80", "-g", GMX or rb.GMX])
        == 0
    )
    new = mda.Universe(
        os.path.join(out, "minimal.pdb"), os.path.join(out, "minimal.xtc")
    )
    old = mda.Universe(
        os.path.join(src, "minimal.pdb"), os.path.join(src, "minimal.xtc")
    )
    kept = new.select_atoms("not resname MOL")
    assert len(new.select_atoms("resname MOL")) > 0
    assert len(kept) == len(old.atoms)
    for i in range(len(old.trajectory)):
        new.trajectory[i]
        old.trajectory[i]
        assert np.abs(kept.positions - old.atoms.positions).max() <= 0.011, i
