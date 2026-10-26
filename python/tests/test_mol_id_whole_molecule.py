"""A ligand split across the periodic boundary is read whole (review F6).

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_mol_id_whole_molecule.py -q
"""

import pathlib

import numpy as np
import MDAnalysis as mda

import mol_id

INPUTS = pathlib.Path(__file__).parent / "inputs/ligands"
VPX = INPUTS / "vorapaxar_namd_frame.pdb"


def _box(path):
    for line in open(path):
        if line.startswith("CRYST1"):
            return [float(line[6:15]), float(line[15:24]), float(line[24:33])]
    raise AssertionError("no CRYST1")


def _wrapped_copy(tmp_path):
    """The fixture with the second half of the ligand's atoms moved one
    box vector along x, as an XTC stores a molecule straddling the cell"""
    a = _box(VPX)[0]
    out = tmp_path / "wrapped.pdb"
    lines = []
    atoms = 0
    for line in open(VPX):
        if line.startswith(("ATOM", "HETATM")):
            atoms += 1
            if atoms > 34:
                x = float(line[30:38]) + a
                line = line[:30] + f"{x:8.3f}" + line[38:]
        lines.append(line)
    out.write_text("".join(lines))
    return out


def test_a_ligand_split_across_the_box_reads_as_the_unwrapped_one(tmp_path, monkeypatch):
    (whole,) = mol_id.structure_to_smiles(str(VPX))
    (split,) = mol_id.structure_to_smiles(str(_wrapped_copy(tmp_path)))
    assert split["inchikey"] == whole["inchikey"]
    assert split["smiles"] == whole["smiles"]
    assert split["formula"] == whole["formula"]

    # The same file without the step is the two fragments F6 describes
    monkeypatch.setattr(mol_id, "_whole_positions", lambda sel: None)
    (broken,) = mol_id.structure_to_smiles(str(_wrapped_copy(tmp_path)))
    assert broken["inchikey"] != whole["inchikey"]


def test_the_selection_keeps_its_own_positions():
    u = mda.Universe(str(VPX))
    sel = u.select_atoms("resname VPX")
    before = sel.positions.copy()
    mol_id._read_coords(sel)
    assert np.array_equal(sel.positions, before)


def test_a_molecule_longer_than_half_the_box_is_walked_along_its_bonds(tmp_path):
    """A chain of 6 atoms 1.5 A apart (7.5 A) in a 10 A cell, its last two
    atoms wrapped. Placed each nearest the atom it was reached from, the
    chain is straight again; placed nearest the FIRST atom, as a single
    minimum-image shift would, atoms past 5 A would fold back"""
    pdb = tmp_path / "chain.pdb"
    xs = [0.5, 2.0, 3.5, 5.0, 6.5 - 10.0, 8.0 - 10.0]
    text = "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1\n" + "".join(
        f"HETATM{i + 1:5d}  C{i + 1:<2} LIG A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        for i, x in enumerate(xs)
    ) + "END\n"
    pdb.write_text(text)
    u = mda.Universe(str(pdb))
    got = mol_id._whole_positions(u.atoms)
    assert got is not None
    np.testing.assert_allclose(got[:, 0], [0.5, 2.0, 3.5, 5.0, 6.5, 8.0], atol=1e-3)

    # Already whole: nothing to do
    u.atoms.positions = got
    assert mol_id._whole_positions(u.atoms) is None


def test_no_box_means_no_change():
    u = mda.Universe(str(VPX))
    u.dimensions = None
    assert mol_id._whole_positions(u.select_atoms("resname VPX")) is None
