"""The converted files describe the same atoms (review CONV-2).

cpptraj drops extra points (Amber TIP4P-Ew/OPC water sites, CHARMM lone
pairs) from PDB output unless told `include_ep`, while XTC output keeps
them. So a TIP4P-Ew system's full.pdb held 809 atoms against 1068 in
full.xtc, parmed could not build full.gro ("cannot reshape array") and the
run failed; a PSF with lone pairs published a full.pdb that MDAnalysis
refuses to pair with its full.xtc. Nothing compared the counts.
"""

import os
import re
import struct
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _name in ("pytraj", "parmed"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

import cpptraj_gmx_traj_manipulation as c  # noqa: E402


# --------------------------------------------------
def write_pdb(path, n):
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(
                f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    {0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}"
                "  1.00  0.00           C\n"
            )
        fh.write("END\n")


def write_gro(path, n):
    with open(path, "w") as fh:
        fh.write(f"title\n{n:>5}\n")
        for i in range(n):
            fh.write(f"{i + 1:>5}ALA     CA{i + 1:>5}{0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}\n")
        fh.write("   4.00000   4.00000   4.00000\n")


def write_xtc(path, n):
    """Only the frame header's first two words matter: magic 1995, natoms"""
    with open(path, "wb") as fh:
        fh.write(struct.pack(">iiif", 1995, n, 0, 0.0))


# --------------------------------------------------
def test_the_three_counts_are_read(tmp_path):
    write_pdb(tmp_path / "x.pdb", 7)
    write_gro(tmp_path / "x.gro", 7)
    write_xtc(tmp_path / "x.xtc", 7)
    assert c.pdb_atom_count(tmp_path / "x.pdb") == 7
    assert c.gro_atom_count(tmp_path / "x.gro") == 7
    assert c.xtc_atom_count(tmp_path / "x.xtc") == 7


def test_agreeing_files_pass_and_a_missing_one_is_not_checked(tmp_path):
    for prefix, n in (("full", 1068), ("minimal", 30)):
        write_pdb(tmp_path / f"{prefix}.pdb", n)
        write_gro(tmp_path / f"{prefix}.gro", n)
        write_xtc(tmp_path / f"{prefix}.xtc", n)
    # minimal_lipid never written: nothing to compare
    assert c.check_atom_counts(str(tmp_path)) == []


def test_a_pdb_without_the_extra_points_is_a_mismatch(tmp_path):
    """The TIP4P-Ew case: 809 in the PDB, 1068 in the XTC and GRO"""
    write_pdb(tmp_path / "full.pdb", 809)
    write_gro(tmp_path / "full.gro", 1068)
    write_xtc(tmp_path / "full.xtc", 1068)
    problems = c.check_atom_counts(str(tmp_path))
    assert problems == ["full.pdb has 809 atoms, full.gro 1068, full.xtc 1068"]


def test_a_file_that_cannot_be_read_is_reported(tmp_path):
    write_pdb(tmp_path / "full.pdb", 3)
    write_gro(tmp_path / "full.gro", 3)
    (tmp_path / "full.xtc").write_bytes(b"\x00\x00\x00\x01\x00\x00\x00\x03")
    (problem,) = c.check_atom_counts(str(tmp_path))
    assert problem.startswith("full: ") and "1995" in problem


def test_every_cpptraj_pdb_output_includes_extra_points():
    """A `trajout ... pdb` without include_ep drops them again"""
    source = open(c.__file__).read()
    outputs = re.findall(r"trajout \{[^}]+\} pdb[^\n]*", source)
    assert len(outputs) >= 8
    missing = [line for line in outputs if "include_ep" not in line]
    assert missing == []
