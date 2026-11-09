"""Tests for check_outputs.py.

Every rule is tested both ways, with files built here: XTCs written through
MDAnalysis's XTCFile, PDB/PSF/mmCIF as text in the shapes our pipeline writes.
The cases that shaped the rules on real outputs (2026-11-09) are here too:
cpptraj's 4-letter residue names from column 17 (TIP3, 1511), .psf numbering
restarting per segment (info, not a fault), a box rotated by the RMS fit,
released sampled.xtc with every frame at 0 ps, and MDR00105222's box -- the
fault this exists for -- via the fixture in inputs/nobox_dcd run through real
cpptraj, with and without `box nobox`.
"""

import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_outputs as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "inputs", "nobox_dcd")
needs_cpptraj = pytest.mark.skipif(shutil.which("cpptraj") is None,
                                   reason="cpptraj not on PATH")

CUBE = np.eye(3) * 5.0
INF_BOX = np.array([[np.inf, np.inf, 0], [2.06, -np.inf, -np.inf], [0, 0, 0]])


# --- builders ------------------------------------------------------------------
ATOMS = [("N", "LYS", 15, "N"), ("CA", "LYS", 15, "C"), ("C", "LYS", 15, "C"),
         ("N", "GLY", 16, "N"), ("CA", "GLY", 16, "C"),
         ("OH2", "TIP3", 17, "O")]
XYZ = np.array([[1.0, 2.0, 3.0], [2.4, 2.1, 3.0], [3.0, 3.4, 3.1],
                [4.2, 3.6, 3.2], [5.0, 4.8, 3.4], [9.0, 9.0, 9.0]])


def write_xtc(path, n=3, natoms=len(ATOMS), box=CUBE, times=None, xyz=None,
              boxes=None):
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    times = list(range(0, 10 * n, 10)) if times is None else times
    base = (XYZ / 10.0 if xyz is None else xyz / 10.0)[:natoms]
    with XTCFile(str(path), "w") as fh:
        for i, t in enumerate(times):
            b = boxes[i] if boxes is not None else box
            fh.write(base.astype(np.float32), np.asarray(b, dtype=np.float32),
                     i, float(t), 1000.0)
    return str(path)


def pdb_line(i, name, resname, resid, element, xyz, chain="A"):
    # cpptraj: a 4-letter name in columns 13-16 starts at 13, a shorter at 14;
    # a 4-letter residue name starts at column 17.
    nm = name if len(name) == 4 else f" {name:<3}"
    rn = f"{resname:>4}" if len(resname) == 4 else f" {resname:<3}"
    return (f"ATOM  {i:5d} {nm}{rn} {chain}{resid:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          "
            f"{element:>2}\n")


def write_pdb(path, atoms=ATOMS, xyz=XYZ, cryst1=None, models=1):
    with open(path, "w") as fh:
        if cryst1:
            fh.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"
                     % tuple(cryst1))
        for m in range(models):
            if models > 1:
                fh.write(f"MODEL     {m + 1:4d}\n")
            for i, (a, x) in enumerate(zip(atoms, xyz), start=1):
                fh.write(pdb_line(i, *a, x + m))
            if models > 1:
                fh.write("ENDMDL\n")
        fh.write("END\n")
    return str(path)


def write_psf(path, atoms=ATOMS):
    with open(path, "w") as fh:
        fh.write("PSF EXT\n\n       1 !NTITLE\n REMARKS test\n\n")
        fh.write(f"{len(atoms):10d} !NATOM\n")
        for i, (name, resname, resid, el) in enumerate(atoms, start=1):
            fh.write(f"{i:10d} PROA     {resid:<8d} {resname:<8} {name:<8} "
                     f"{name[:4]:<6}  0.000000  12.0110  0\n")
        fh.write("\n         0 !NBOND: bonds\n")
    return str(path)


def write_mmcif(path, atoms=ATOMS, xyz=XYZ):
    rows = []
    for i, ((name, resname, resid, el), x) in enumerate(zip(atoms, xyz), start=1):
        rows.append(f"ATOM {i} {el} {name} {resname} A {resid} "
                    f"{x[0]:.3f} {x[1]:.3f} {x[2]:.3f} 1")
    with open(path, "w") as fh:
        fh.write("data_test\nloop_\n_atom_site.group_PDB\n_atom_site.id\n"
                 "_atom_site.type_symbol\n_atom_site.auth_atom_id\n"
                 "_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
                 "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n"
                 "_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
                 "_atom_site.pdbx_PDB_model_num\n" + "\n".join(rows) + "\n")
    return str(path)


def full_set(d, box=CUBE, cryst1=(50, 50, 50, 90, 90, 90), **xtc_kw):
    d.mkdir(exist_ok=True)
    write_psf(d / "full.psf")
    write_pdb(d / "full.pdb", cryst1=cryst1)
    write_mmcif(d / "full.mmcif")
    write_xtc(d / "full.xtc", box=box, **xtc_kw)
    write_psf(d / "minimal.psf", ATOMS[:5])
    write_pdb(d / "minimal.pdb", ATOMS[:5], XYZ[:5])
    write_xtc(d / "minimal.xtc", natoms=5, box=box, **xtc_kw)
    return str(d)


def levels(findings, check=None):
    return [(f.level, f.file) for f in findings
            if f.level != "info" and (check is None or f.check == check)]


# --- box geometry ------------------------------------------------------------------
@pytest.mark.parametrize("box,kind", [
    (np.zeros((3, 3)), "zero"),
    (np.full((3, 3), 1e-8), "zero"),
    (CUBE, "cell"),
    (INF_BOX, "non-finite"),
    (np.full((3, 3), np.nan), "non-finite"),
    (np.diag([-7083.2, 0, 0]), "bad"),                       # 105222 rep_10
    (np.diag([1992603.5, 5, 5]), "bad"),                    # absurdly long
    (np.diag([5, 5, 0.01]), "bad"),                         # one vector tiny
    (np.array([[5, 0, 0], [5, 0.001, 0], [0, 0, 5]]), "bad"),  # flat
    (np.array([[5, 0, 0], [0, 5, 0], [2.5, 2.5, 3.54]]), "cell"),  # triclinic
])
def test_box_kind(box, kind):
    assert C.box_kind(np.asarray(box, dtype=float)) == kind


def test_a_rotated_cell_is_the_same_cell():
    th = math.radians(37)
    rot = np.array([[math.cos(th), -math.sin(th), 0],
                    [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    box = np.array([[5, 0, 0], [0, 6, 0], [1, 1, 7.0]])
    assert C.box_kind(box @ rot.T) == "cell"
    assert C.same_cell(box, box @ rot.T)
    assert not C.same_cell(box, box * 1.01)


def test_cryst1_round_trip():
    box = C.box_from_cryst1((50, 60, 70, 80, 85, 95))
    lens, angs = C.lengths_angles(box)
    assert np.allclose(lens, [5, 6, 7], atol=1e-9)
    assert np.allclose(angs, [80, 85, 95], atol=1e-9)


# --- check 1: box --------------------------------------------------------------------
def test_a_consistent_box_passes(tmp_path):
    f = C.run_checks([full_set(tmp_path / "s")], box_present=True)
    assert levels(f) == []


def test_no_box_passes_with_no_box(tmp_path):
    d = full_set(tmp_path / "s", box=np.zeros((3, 3)), cryst1=None)
    assert levels(C.run_checks([d], box_present=False)) == []


def test_the_105222_box_fails(tmp_path):
    d = full_set(tmp_path / "s", box=INF_BOX, cryst1=None)
    f = C.run_checks([d])
    assert ("fail", "full.xtc") in levels(f, "box")
    assert ("fail", "minimal.xtc") in levels(f, "box")
    msg = next(x.message for x in f if x.file == "full.xtc" and x.check == "box")
    assert "3 of 3 frames have a non-finite box" in msg


def test_some_frames_with_a_box_and_some_without_fails(tmp_path):
    p = write_xtc(tmp_path / "full.xtc", boxes=[CUBE, np.zeros((3, 3)), CUBE])
    f = C.check_box({"full.xtc": C.read_xtc(p)}, {}, None)
    assert levels(f) == [("fail", "full.xtc")]
    assert "1 frames have no box and 2 have one" in f[0].message


@pytest.mark.parametrize("present,box", [(True, np.zeros((3, 3))), (False, CUBE)])
def test_the_box_must_match_the_pipelines_decision(tmp_path, present, box):
    p = write_xtc(tmp_path / "full.xtc", box=box)
    f = C.check_box({"full.xtc": C.read_xtc(p)}, {}, present)
    assert levels(f) == [("fail", "full.xtc")]


def test_sampled_may_have_no_box_when_the_run_has_one(tmp_path):
    p = write_xtc(tmp_path / "sampled.xtc", box=np.zeros((3, 3)))
    assert levels(C.check_box({"sampled.xtc": C.read_xtc(p)}, {}, True)) == []


def test_cryst1_must_match_the_first_frame(tmp_path):
    d = full_set(tmp_path / "s", cryst1=(52, 50, 50, 90, 90, 90))
    assert ("fail", "full.pdb") in levels(C.run_checks([d]), "box")


def test_missing_cryst1_with_a_box_warns(tmp_path):
    d = full_set(tmp_path / "s", cryst1=None)
    assert levels(C.run_checks([d]), "box") == [("warn", "full.pdb")]


def test_a_cryst1_without_a_box_fails_but_the_no_box_cryst1_passes(tmp_path):
    d = full_set(tmp_path / "a", box=np.zeros((3, 3)),
                 cryst1=(50, 50, 50, 90, 90, 90))
    assert ("fail", "full.pdb") in levels(C.run_checks([d]), "box")
    d = full_set(tmp_path / "b", box=np.zeros((3, 3)), cryst1=(1, 1, 1, 90, 90, 90))
    assert levels(C.run_checks([d]), "box") == []


# --- check 2: time ---------------------------------------------------------------------
def frames(tmp_path, name, times, natoms=3):
    return C.read_xtc(write_xtc(tmp_path / name, n=len(times), natoms=natoms,
                                times=times))


def test_times_in_order_pass(tmp_path):
    f = C.check_time({"full.xtc": frames(tmp_path, "full.xtc", [0, 10, 20])}, None)
    assert levels(f) == []


def test_time_going_backwards_fails(tmp_path):
    f = C.check_time({"full.xtc": frames(tmp_path, "full.xtc", [0, 10, 5, 20])}, None)
    assert levels(f) == [("fail", "full.xtc")]
    assert "10.0 -> 5.0" in f[0].message


def test_a_repeated_frame_time_warns(tmp_path):
    f = C.check_time({"full.xtc": frames(tmp_path, "full.xtc", [0, 10, 10, 20])}, None)
    assert levels(f) == [("warn", "full.xtc")]
    assert "repeat" in f[0].message


def test_no_time_axis_warns_once(tmp_path):
    f = C.check_time({"sampled.xtc": frames(tmp_path, "sampled.xtc", [0] * 5)}, None)
    assert levels(f) == [("warn", "sampled.xtc")]
    assert "no time axis: all 5 frames are at 0 ps" in f[0].message


def test_skipped_time_is_only_reported(tmp_path):
    f = C.check_time({"full.xtc": frames(tmp_path, "full.xtc", [0, 10, 20, 100, 110])},
                     None)
    assert levels(f) == []
    assert any(x.level == "info" and "uneven spacing" in x.message for x in f)


def test_full_and_minimal_must_have_the_same_frames(tmp_path):
    f = C.check_time({"full.xtc": frames(tmp_path, "full.xtc", [0, 10, 20]),
                      "minimal.xtc": frames(tmp_path, "minimal.xtc", [0, 10])}, None)
    assert levels(f) == [("fail", "minimal.xtc")]
    f = C.check_time({"full.xtc": frames(tmp_path, "f2.xtc", [0, 10, 20]),
                      "minimal.xtc": frames(tmp_path, "m2.xtc", [0, 10, 30])}, None)
    assert levels(f) == [("fail", "minimal.xtc")]


def test_source_times_match(tmp_path):
    full = frames(tmp_path, "full.xtc", [0, 10, 20, 100])
    f = C.check_time({"full.xtc": full}, np.array([1000, 1010, 1020, 1100.0]), "s.xtc")
    assert levels(f) == []


def test_a_grid_stamped_over_skipped_source_time_fails(tmp_path):
    """The conversion's `time time0 0 dt` would turn 0,10,20,100 into 0..30"""
    full = frames(tmp_path, "full.xtc", [0, 10, 20, 30])
    f = C.check_time({"full.xtc": full}, np.array([0, 10, 20, 100.0]), "s.xtc")
    assert levels(f) == [("fail", "full.xtc")]
    assert "source 100 ps" in f[0].message


def test_a_different_frame_count_from_the_source_warns(tmp_path):
    full = frames(tmp_path, "full.xtc", [0, 10, 20])
    f = C.check_time({"full.xtc": full}, np.arange(0, 130.0), "s.xtc")
    assert levels(f) == [("warn", "full.xtc")]


def test_source_times_read_from_xtc_and_netcdf_not_dcd(tmp_path):
    p = write_xtc(tmp_path / "src.xtc", times=[5, 15, 25])
    assert list(C.read_source_times(p)) == [5, 15, 25]
    from scipy.io import netcdf_file

    nc = tmp_path / "src.nc"
    with netcdf_file(str(nc), "w") as fh:
        fh.createDimension("frame", 3)
        v = fh.createVariable("time", "f", ("frame",))
        v[:] = [2.0, 4.0, 8.0]
    assert list(C.read_source_times(str(nc))) == [2, 4, 8]
    assert C.read_source_times(os.path.join(FIXTURE, "ca3.dcd")) is None


# --- check 3: atoms ----------------------------------------------------------------------
def test_readers_agree_on_our_shapes(tmp_path):
    pdb = C.read_pdb(write_pdb(tmp_path / "x.pdb"))
    psf = C.read_psf(write_psf(tmp_path / "x.psf"))
    cif = C.read_mmcif(write_mmcif(tmp_path / "x.mmcif"))
    assert [a.resname for a in pdb.atoms][-1] == "TIP3"     # column 17
    assert [(a.name, a.resname, a.resid) for a in pdb.atoms] == \
        [(a.name, a.resname, a.resid) for a in psf.atoms] == \
        [(a.name, a.resname, a.resid) for a in cif.atoms]
    assert np.allclose(pdb.xyz, cif.xyz)


def test_pdb_reads_the_first_model_only(tmp_path):
    assert len(C.read_pdb(write_pdb(tmp_path / "m.pdb", models=3)).atoms) == len(ATOMS)


def test_files_that_agree_pass(tmp_path):
    assert levels(C.run_checks([full_set(tmp_path / "s")]), "atoms") == []


def test_atom_counts_must_agree(tmp_path):
    d = full_set(tmp_path / "s")
    write_pdb(os.path.join(d, "full.pdb"), ATOMS[:5], XYZ[:5],
              cryst1=(50, 50, 50, 90, 90, 90))
    f = C.run_checks([d])
    assert ("fail", "full") in levels(f, "atoms")


@pytest.mark.parametrize("i,field,value,expect", [
    (1, 0, "CB", "atom name"),
    (3, 1, "ALA", "residue name"),
    (0, 3, "XP", "not element symbols"),
])
def test_one_wrong_atom_fails(tmp_path, i, field, value, expect):
    atoms = [list(a) for a in ATOMS]
    atoms[i][field] = value
    d = full_set(tmp_path / "s")
    write_mmcif(os.path.join(d, "full.mmcif"), [tuple(a) for a in atoms])
    f = [x for x in C.run_checks([d]) if x.level == "fail"]
    assert f and f[0].file == "full.mmcif" and expect in f[0].message


def test_long_names_compare_as_far_as_the_narrow_format_holds(tmp_path):
    """A .psf name of 6 characters against the .pdb's 4"""
    atoms = list(ATOMS)
    atoms[5] = ("OH2", "TIP3XX", 17, "O")
    d = full_set(tmp_path / "s")
    write_psf(os.path.join(d, "full.psf"), atoms)
    assert levels(C.run_checks([d]), "atoms") == []


def test_psf_numbering_per_segment_is_info(tmp_path):
    atoms = [a if a[2] != 17 else (a[0], a[1], 1, a[3]) for a in ATOMS]
    d = full_set(tmp_path / "s")
    write_psf(os.path.join(d, "full.psf"), atoms)
    f = C.run_checks([d])
    assert levels(f, "atoms") == []
    assert any(x.level == "info" and "numbered differently" in x.message for x in f)


def test_residues_split_differently_fails(tmp_path):
    """Same names, but the .pdb puts GLY's N into LYS"""
    atoms = list(ATOMS)
    atoms[3] = ("N", "LYS", 15, "N")
    atoms[4] = ("CA", "LYS", 16, "C")
    d = full_set(tmp_path / "s")
    write_pdb(os.path.join(d, "full.pdb"), [(n, "LYS", r, e) for n, _, r, e in atoms],
              cryst1=(50, 50, 50, 90, 90, 90))
    f = [x for x in C.run_checks([d]) if x.level == "fail" and x.check == "atoms"]
    assert f and f[0].file == "full.pdb"


def test_structure_coordinates_must_equal_frame_0(tmp_path):
    d = full_set(tmp_path / "s")
    moved = XYZ.copy()
    moved[2] += 0.5
    write_pdb(os.path.join(d, "full.pdb"), xyz=moved, cryst1=(50, 50, 50, 90, 90, 90))
    f = [x for x in C.run_checks([d]) if x.level == "fail"]
    assert f[0].file == "full.pdb" and "atom 3" in f[0].message


def test_xtc_precision_is_not_a_difference(tmp_path):
    d = full_set(tmp_path / "s")
    nudged = XYZ + 0.004                       # under one XTC step (0.01 A)
    write_pdb(os.path.join(d, "full.pdb"), xyz=nudged, cryst1=(50, 50, 50, 90, 90, 90))
    assert levels(C.run_checks([d]), "atoms") == []


def test_gro_files_are_ignored(tmp_path):
    d = full_set(tmp_path / "s")
    with open(os.path.join(d, "full.gro"), "w") as fh:
        fh.write("garbage\n 1\n")
    assert "full.gro" not in C.find_files([d])
    assert levels(C.run_checks([d])) == []


def test_files_split_across_dirs_first_dir_wins(tmp_path):
    """An MDR-108 build: new/ (psf, mmcif) beside orig/ (pdb, sampled.xtc)"""
    orig, new = tmp_path / "orig", tmp_path / "new"
    orig.mkdir(), new.mkdir()
    write_pdb(orig / "minimal.pdb", ATOMS[:5], XYZ[:5])
    write_xtc(orig / "sampled.xtc", natoms=5)
    write_psf(new / "minimal.psf", ATOMS[:5])
    write_mmcif(new / "minimal.mmcif", ATOMS[:5], XYZ[:5])
    write_pdb(new / "minimal.pdb", ATOMS[:4], XYZ[:4])     # shadowed? no: new first
    files = C.find_files([str(new), str(orig)])
    assert files["minimal.pdb"].startswith(str(new))
    assert ("fail", "minimal") in levels(C.run_checks([str(new), str(orig)]), "atoms")
    assert levels(C.run_checks([str(orig), str(new)]), "atoms") == []


def test_an_unreadable_file_is_a_failure_not_a_crash(tmp_path):
    d = full_set(tmp_path / "s")
    with open(os.path.join(d, "full.mmcif"), "w") as fh:
        fh.write("not a cif")
    f = C.run_checks([d])
    assert ("fail", "full.mmcif") in levels(f, "atoms")


# --- command line ----------------------------------------------------------------------
def test_main_exit_codes_and_json(tmp_path, capsys):
    good = full_set(tmp_path / "good")
    out = tmp_path / "f.json"
    assert C.main([good, "--box", "--json", str(out)]) == 0
    assert all(x["level"] == "info" for x in json.loads(out.read_text()))
    bad = full_set(tmp_path / "bad", box=INF_BOX, cryst1=None)
    assert C.main([bad, "-q"]) == 1
    printed = capsys.readouterr().out
    assert "FAIL box" in printed and "INFO" not in printed.split("FAIL", 1)[1]


def test_an_empty_dir_checks_nothing_and_passes(tmp_path):
    assert C.main([str(tmp_path)]) == 0


# --- real cpptraj on 105222's fixture -------------------------------------------------
def cpptraj_minimal(tmp_path, box_line):
    import cpptraj_gmx_traj_manipulation as conv

    psf, dcd = os.path.join(FIXTURE, "ca.psf"), os.path.join(FIXTURE, "ca3.dcd")
    out = tmp_path / ("fixed" if box_line else "plain")
    out.mkdir()
    cppin = out / "c.in"
    lines = [f"parm {psf}", f"trajin {dcd}"] + ([box_line] if box_line else []) + \
        [f"trajout {out}/minimal.xtc xtc", f"trajout {out}/minimal.pdb pdb include_ep onlyframes 1",
         "run"]
    cppin.write_text("\n".join(lines) + "\n")
    rv, _ = conv.run_cpptraj(str(cppin), "test", fatal=False)
    assert rv == 0
    shutil.copy(psf, out / "minimal.psf")
    return str(out)


@needs_cpptraj
def test_the_fixed_conversion_passes(tmp_path):
    d = cpptraj_minimal(tmp_path, "box nobox")
    f = C.run_checks([d], box_present=False)
    assert [x for x in f if x.level == "fail"] == []


@needs_cpptraj
def test_the_unfixed_conversion_fails_the_box_check(tmp_path):
    d = cpptraj_minimal(tmp_path, None)
    boxes = C.read_xtc(os.path.join(d, "minimal.xtc")).boxes
    if all(C.box_kind(b) == "zero" for b in boxes):
        pytest.skip("this cpptraj build writes a zero box without box nobox")
    f = C.run_checks([d], box_present=False)
    assert ("fail", "minimal.xtc") in levels(f, "box")


# --- real outputs (opt-in) --------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("CHECK_OUTPUTS_DIRS"),
                    reason="set CHECK_OUTPUTS_DIRS to processed folders, ':'-separated")
def test_real_processed_folders_pass():
    for d in os.environ["CHECK_OUTPUTS_DIRS"].split(":"):
        fails = [x for x in C.run_checks([d]) if x.level == "fail"]
        assert fails == [], (d, fails)
