"""read_first_frame and its parsers: cpptraj in place of pytraj.

pytraj was the only thing that tied cpptraj_gmx_traj_manipulation.py to the
simproc conda env. These pin the replacement's behaviour, including the cases
where pytraj failed silently (an unreadable frame read as "no box").

The tests marked `needs_cpptraj` run a real cpptraj (MDREPO_CPPTRAJ, or the
one on PATH) over tiny systems written here; they skip when there is none.
Checked against pytraj on 19 real prod sims (AMBER prmtop with nc/dcd, PSF
with xtc/dcd, PDB with xtc, unreadable .mdc and a stripped xtc): the same box
decision, boxes within 1e-6, the same atom and residue names.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import cpptraj_gmx_traj_manipulation as c  # noqa: E402

CPPTRAJ = os.environ.get("MDREPO_CPPTRAJ") or shutil.which("cpptraj")
needs_cpptraj = pytest.mark.skipif(not CPPTRAJ, reason="no cpptraj")


@pytest.fixture
def cpptraj_on_path(monkeypatch):
    """Put the cpptraj under test first on PATH, as mdr-process does"""
    monkeypatch.setenv(
        "PATH",
        os.path.dirname(os.path.abspath(CPPTRAJ))
        + os.pathsep
        + os.environ["PATH"],
    )


# --------------------------------------------------
# restart_box
def restart(natoms, box_line=None, title="Cpptraj Generated Restart"):
    lines = [title, f"{natoms:6d}  0.0000000E+00"]
    coords = ["%12.7f" % (0.1 * i) for i in range(3 * natoms)]
    for i in range(0, len(coords), 6):
        lines.append("".join(coords[i : i + 6]))
    if box_line is not None:
        lines.append(box_line)
    return lines


def test_restart_box_reads_the_six_values():
    box = "  50.7280595  50.7280595  50.7280595 109.4712190 109.4712190 109.4712190"
    assert c.restart_box(restart(862, box)) == [
        50.7280595,
        50.7280595,
        50.7280595,
        109.471219,
        109.471219,
        109.471219,
    ]


def test_restart_box_none_when_there_is_no_box_line():
    assert c.restart_box(restart(862)) is None


@pytest.mark.parametrize("natoms", [1, 2, 3, 4, 5])
def test_restart_box_counts_coordinate_lines_for_odd_atom_counts(natoms):
    """Two atoms to a line; one atom is half a line, still one line"""
    box = "  10.0000000  20.0000000  30.0000000  90.0000000  90.0000000  90.0000000"
    assert c.restart_box(restart(natoms, box))[:3] == [10.0, 20.0, 30.0]
    assert c.restart_box(restart(natoms)) is None


def test_restart_box_reads_by_column_when_values_touch():
    """1,000 A or more fills all 12 characters (F12.7): no space to split on"""
    box = "1234.56789011234.5678901  50.0000000  90.0000000  90.0000000  90.0000000"
    assert c.restart_box(restart(1, box))[:3] == [
        1234.5678901,
        1234.5678901,
        50.0,
    ]


def test_restart_box_pads_a_lengths_only_line_with_right_angles():
    box = "  10.0000000  20.0000000  30.0000000"
    assert c.restart_box(restart(1, box)) == [
        10.0,
        20.0,
        30.0,
        90.0,
        90.0,
        90.0,
    ]


def test_restart_box_ignores_blank_trailing_lines():
    assert c.restart_box(restart(2) + ["", "   "]) is None


def test_restart_box_rejects_a_short_box_line():
    with pytest.raises(ValueError, match="short box line"):
        c.restart_box(restart(1, "  10.0000000"))


def test_restart_box_rejects_a_file_with_no_atom_count():
    with pytest.raises(ValueError, match="no atom count"):
        c.restart_box(["title only"])


def test_a_zero_box_from_a_restart_is_not_usable():
    """A run without a cell can still write zero lengths"""
    box = "   0.0000000   0.0000000   0.0000000  90.0000000  90.0000000  90.0000000"
    frame = c.FirstFrame(c.restart_box(restart(1, box)), set(), set())
    assert not c.has_box(frame)


# --------------------------------------------------
# info_names: real cpptraj 6.4.4 output (prod MDR00072133 and MDR00001511)
ATOMINFO = """\
#Atom Name  #Res Name  #Mol Type   Charge     Mass GBradius El     rVDW     eVDW
    1 N        1 MET      1 N3     0.1592   7.9620   1.5500  N   1.8240   0.1700
    2 H1       1 MET      1 H      0.1984   3.0240   1.3000  H   0.6000   0.0157
    5 CA       1 MET      1 CX     0.0221   7.0480   1.7000  C   1.9080   0.1094
  861 Na+     59 Na+      4 Na+    1.0000  22.9900   1.5000 NA   1.4670   0.0296
"""

RESINFO = """\
#Res  Name First  Last Natom #Orig  #Mol C I
    1 AC5      1    19    19     1     1
    2 VAL     20    35    16     2     1
   60 Na+    862   862     1    60     5
"""


def test_info_names_reads_atom_names(tmp_path):
    path = tmp_path / "atoms.dat"
    path.write_text(ATOMINFO)
    assert c.info_names(str(path)) == {"N", "H1", "CA", "Na+"}


def test_info_names_reads_residue_names(tmp_path):
    path = tmp_path / "res.dat"
    path.write_text(RESINFO)
    assert c.info_names(str(path)) == {"AC5", "VAL", "Na+"}


def test_info_names_of_a_header_only_table_is_empty(tmp_path):
    path = tmp_path / "res.dat"
    path.write_text("#Res  Name First  Last Natom #Orig  #Mol C I\n\n")
    assert c.info_names(str(path)) == set()


def test_backbone_and_lipids_from_info_names(tmp_path):
    """What the callers do with the names"""
    atoms = tmp_path / "atoms.dat"
    atoms.write_text(ATOMINFO)
    names = c.info_names(str(atoms))
    assert c.backbone_fit_mask(names) != "@*"
    res = tmp_path / "res.dat"
    res.write_text(RESINFO)
    assert not c.info_names(str(res)) & set(c.KNOWN_LIPIDS)


# --------------------------------------------------
# read_first_frame against a real cpptraj
def pdb(path, natoms=3, cryst1=None, resname="ALA", names=("N", "CA", "C")):
    lines = []
    if cryst1:
        lines.append(
            "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1" % cryst1
        )
    for i in range(natoms):
        name = names[i % len(names)]
        lines.append(
            # name 13-16, altLoc 17, resName 18-21 (4 wide: lipids), chain 22
            "ATOM  %5d %-4s %-4sA%4d    %8.3f%8.3f%8.3f  1.00  0.00           %s"
            % (
                i + 1,
                name if len(name) == 4 else " " + name,
                resname,
                1 + i // 3,
                1.0 + i,
                2.0,
                3.0,
                name[0],
            )
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def write_traj(tmp_path, top, fmt, ext, box=None):
    """Use cpptraj itself to write a one-frame trajectory of `top`"""
    out = tmp_path / f"traj.{ext}"
    cppin = tmp_path / f"make_{ext}.in"
    cmds = [f"parm {top}", f"trajin {top}"]
    if box:
        cmds.append("box x %g y %g z %g alpha 90 beta 90 gamma 90" % box)
    cmds += [f"trajout {out} {fmt}", "run"]
    cppin.write_text("\n".join(cmds) + "\n")
    rc = os.system(f"cpptraj -i {cppin} > {tmp_path}/make_{ext}.log 2>&1")
    assert rc == 0 and out.is_file(), (
        tmp_path / f"make_{ext}.log"
    ).read_text()
    return str(out)


@needs_cpptraj
@pytest.mark.parametrize(
    "fmt,ext", [("xtc", "xtc"), ("dcd", "dcd"), ("netcdf", "nc")]
)
def test_box_of_a_trajectory_with_a_cell(tmp_path, cpptraj_on_path, fmt, ext):
    top = pdb(tmp_path / "top.pdb", 6, names=("N", "CA", "C"))
    traj = write_traj(tmp_path, top, fmt, ext, box=(40.5, 41.25, 42.0))
    frame = c.read_first_frame(top, traj, str(tmp_path))
    assert frame.frame_error is None
    assert c.has_box(frame)
    assert frame.box[:3] == pytest.approx([40.5, 41.25, 42.0], abs=1e-4)
    assert frame.box[3:] == pytest.approx([90.0, 90.0, 90.0])
    assert frame.atom_names == {"N", "CA", "C"}
    assert frame.residue_names == {"ALA"}


@needs_cpptraj
def test_no_box_from_a_pdb_without_cryst1(tmp_path, cpptraj_on_path):
    top = pdb(tmp_path / "top.pdb", 3)
    frame = c.read_first_frame(top, top, str(tmp_path))
    assert frame.frame_error is None
    assert not c.has_box(frame)


@needs_cpptraj
def test_box_from_a_pdb_with_cryst1(tmp_path, cpptraj_on_path):
    top = pdb(
        tmp_path / "top.pdb", 3, cryst1=(30.0, 31.0, 32.0, 90.0, 90.0, 90.0)
    )
    frame = c.read_first_frame(top, top, str(tmp_path))
    assert frame.box[:3] == pytest.approx([30.0, 31.0, 32.0])


def psf(path, atoms):
    """A minimal CHARMM PSF: (segid, resid, resname, name, type, mass) per atom"""
    lines = [
        "PSF",
        "",
        "       1 !NTITLE",
        " REMARKS test",
        "",
        "%8d !NATOM" % len(atoms),
    ]
    for i, (seg, resid, resname, name, typ, mass) in enumerate(atoms, 1):
        lines.append(
            "%8d %-4s %-4d %-4s %-4s %-4s %14.6f%14.4f%12d"
            % (i, seg, resid, resname, name, typ, 0.0, mass, 0)
        )
    lines += ["", "       0 !NBOND: bonds", ""]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@needs_cpptraj
def test_lipid_names_keep_four_letters_from_a_psf(tmp_path, cpptraj_on_path):
    """KNOWN_LIPIDS are 4 letters; a PSF keeps them (a PDB keeps only 3, in
    cpptraj and pytraj alike), so a membrane is found"""
    top = psf(
        tmp_path / "memb.psf",
        [
            ("MEMB", 1, "POPC", "P", "PL", 30.97),
            ("MEMB", 1, "POPC", "O11", "OSLP", 15.999),
            ("MEMB", 1, "POPC", "C12", "CTL2", 12.011),
        ],
    )
    frames = pdb(tmp_path / "memb.pdb", 3, names=("P", "O11", "C12"))
    frame = c.read_first_frame(top, frames, str(tmp_path))
    assert frame.frame_error is None
    assert frame.residue_names == {"POPC"}
    assert frame.residue_names & set(c.KNOWN_LIPIDS)
    assert frame.atom_names == {"P", "O11", "C12"}
    assert c.backbone_fit_mask(frame.atom_names) == "@*"


@needs_cpptraj
def test_backbone_names_from_a_psf(tmp_path, cpptraj_on_path):
    top = psf(
        tmp_path / "prot.psf",
        [
            ("PROA", 1, "ALA", "N", "NH1", 14.007),
            ("PROA", 1, "ALA", "CA", "CT1", 12.011),
            ("PROA", 1, "ALA", "C", "C", 12.011),
        ],
    )
    frames = pdb(tmp_path / "prot.pdb", 3)
    frame = c.read_first_frame(top, frames, str(tmp_path))
    assert c.backbone_fit_mask(frame.atom_names) != "@*"
    assert not frame.residue_names & set(c.KNOWN_LIPIDS)


@needs_cpptraj
def test_an_unreadable_frame_is_no_box_not_an_error(
    tmp_path, cpptraj_on_path
):
    """pytraj printed "Global index 0 is out of range" and went on with a zero
    box; keep that outcome, but say so (prod .mdc files, MDR00095051)"""
    top = pdb(tmp_path / "top.pdb", 3)
    junk = tmp_path / "rep4.mdc"
    junk.write_bytes(b"\x00\x01not a trajectory\n" * 20)
    conf = tmp_path / "conf.pdb"
    frame = c.read_first_frame(
        top, str(junk), str(tmp_path), conf_pdb=str(conf)
    )
    assert frame.box is None and not c.has_box(frame)
    assert frame.frame_error and "exit" in frame.frame_error
    assert frame.atom_names == {"N", "CA", "C"}
    assert not conf.exists()


@needs_cpptraj
def test_an_atom_count_mismatch_is_no_box_not_an_error(
    tmp_path, cpptraj_on_path
):
    """A stripped trajectory with the whole-system topology (MDR00004438)"""
    small = pdb(tmp_path / "small.pdb", 3)
    traj = write_traj(tmp_path, small, "xtc", "xtc", box=(20.0, 20.0, 20.0))
    big = pdb(tmp_path / "big.pdb", 6)
    frame = c.read_first_frame(big, traj, str(tmp_path))
    assert frame.box is None
    assert "atoms" in frame.frame_error.lower() or "exit" in frame.frame_error


@needs_cpptraj
def test_an_unreadable_topology_raises(tmp_path, cpptraj_on_path):
    bad = tmp_path / "broken.psf"
    bad.write_text("this is not a psf\n")
    with pytest.raises(RuntimeError, match="could not read topology"):
        c.read_first_frame(str(bad), str(bad), str(tmp_path))


@needs_cpptraj
def test_a_missing_topology_raises(tmp_path, cpptraj_on_path):
    with pytest.raises(RuntimeError, match="could not read topology"):
        c.read_first_frame(
            str(tmp_path / "nope.prmtop"),
            str(tmp_path / "nope.nc"),
            str(tmp_path),
        )


@needs_cpptraj
def test_conf_pdb_is_the_first_frame(tmp_path, cpptraj_on_path):
    top = pdb(tmp_path / "top.pdb", 6)
    traj = write_traj(tmp_path, top, "xtc", "xtc", box=(40.0, 40.0, 40.0))
    conf = tmp_path / "conf.pdb"
    c.read_first_frame(top, traj, str(tmp_path), conf_pdb=str(conf))
    atoms = [
        l
        for l in conf.read_text().splitlines()
        if l.startswith(("ATOM", "HETATM"))
    ]
    assert len(atoms) == 6
    assert any(l.startswith("CRYST1") for l in conf.read_text().splitlines())


@needs_cpptraj
def test_stale_outputs_from_a_previous_run_are_not_reused(
    tmp_path, cpptraj_on_path
):
    """A second read over a bad frame must not pick up the first read's box"""
    top = pdb(tmp_path / "top.pdb", 6)
    good = write_traj(tmp_path, top, "xtc", "xtc", box=(40.0, 40.0, 40.0))
    assert c.has_box(c.read_first_frame(top, good, str(tmp_path)))
    junk = tmp_path / "junk.mdc"
    junk.write_bytes(b"junk\n")
    assert c.read_first_frame(top, str(junk), str(tmp_path)).box is None


@needs_cpptraj
def test_the_read_prints_nothing_when_it_works(
    tmp_path, cpptraj_on_path, capsys
):
    """pytraj's read was silent; cpptraj's chatter must not reach the log"""
    top = pdb(tmp_path / "top.pdb", 3)
    c.read_first_frame(top, top, str(tmp_path))
    assert capsys.readouterr().out == ""


def test_no_pytraj_left():
    """The point of the change: the script must import without pytraj"""
    source = open(c.__file__).read()
    assert "import pytraj" not in source
    for call in ("pt.iterload(", "pt.load(", "pt.write_traj(filename"):
        assert call not in source
    assert "pytraj" not in sys.modules or sys.modules["pytraj"] is None
