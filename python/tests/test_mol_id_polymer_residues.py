"""A residue joined into a chain is part of the chain, whatever its name.

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_mol_id_polymer_residues.py -q
"""

import pathlib
import warnings

import MDAnalysis as mda
import numpy as np

import mol_id

INPUTS = pathlib.Path(__file__).parent / "inputs/ligands"
GLA = INPUTS / "gla_chain_namd_frame.pdb"


def _universe(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mda.Universe(str(path))


def _pdb(tmp_path, atoms) -> str:
    """A PDB of (resname, resid, name, element, x, y, z)."""
    lines = [
        f"ATOM  {i + 1:5d} {name:<4} {rn:<4}A{rid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {el:>2}"
        for i, (rn, rid, name, el, x, y, z) in enumerate(atoms)
    ]
    p = tmp_path / "t.pdb"
    p.write_text("\n".join(lines) + "\nEND\n")
    return str(p)


def test_a_modified_residue_in_the_chain_is_not_a_ligand():
    """Two gamma-carboxyglutamates (CGU) of a Gla domain, peptide-bonded to a
    leucine and to each other, beside vorapaxar. Only vorapaxar is a ligand;
    taken for one, CGU was read as an amino-aldehyde and published."""

    u = _universe(GLA)

    assert set(mol_id.find_ligand_resnames(u)) == {"VPX"}
    got = mol_id.structure_to_smiles(str(GLA))
    assert [g["resname"] for g in got] == ["VPX"]


def test_a_free_copy_of_a_chain_residue_is_still_a_ligand():
    """The same CGU once in the chain and once on its own, 30 A away: the
    name is still a ligand, and the free copy is the one read."""

    u = _universe(GLA)
    free = u.select_atoms("resname CGU and resid 6")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = mda.Merge(u.atoms, free)
    moved = merged.atoms[len(u.atoms) :]
    moved.positions = moved.positions + np.array([30.0, 0.0, 0.0])
    moved.residues.resids = 99

    assert "CGU" in mol_id.find_ligand_resnames(merged)
    linked = mol_id._polymer_residues(merged, {"CGU"})
    assert moved.residues[0].resindex not in linked
    assert len(linked) == 2


def test_a_ligand_bonded_to_a_side_chain_stays_a_ligand(tmp_path):
    """A covalent inhibitor acylating a lysine: its carbonyl carbon, named C,
    sits 1.34 A from the lysine's NZ. That is not a backbone bond."""

    path = _pdb(
        tmp_path,
        [
            ("LYS", 1, "N", "N", 0.000, 0.000, 0.000),
            ("LYS", 1, "CA", "C", 1.460, 0.000, 0.000),
            ("LYS", 1, "C", "C", 2.000, 1.420, 0.000),
            ("LYS", 1, "NZ", "N", 1.460, -4.000, 0.000),
            ("LIG", 2, "C", "C", 1.460, -5.340, 0.000),
            ("LIG", 2, "O", "O", 2.500, -5.900, 0.000),
            ("LIG", 2, "C1", "C", 0.200, -6.100, 0.000),
        ],
    )

    assert set(mol_id.find_ligand_resnames(_universe(path))) == {"LIG"}


def test_an_acetyl_cap_is_part_of_its_chain(tmp_path):
    """An ACE cap's carbonyl carbon bonded to the next residue's backbone N."""

    path = _pdb(
        tmp_path,
        [
            ("ACE", 1, "CH3", "C", -1.520, 0.000, 0.000),
            ("ACE", 1, "C", "C", 0.000, 0.000, 0.000),
            ("ACE", 1, "O", "O", 0.615, 1.065, 0.000),
            ("ALA", 2, "N", "N", 0.665, -1.152, 0.000),
            ("ALA", 2, "CA", "C", 2.125, -1.152, 0.000),
            ("ALA", 2, "C", "C", 2.665, 0.268, 0.000),
        ],
    )

    assert mol_id.find_ligand_resnames(_universe(path)) == {}


def test_retinal_on_a_lysine_is_reported_as_retinal():
    """Rhodopsin's chromophore: retinal joined to a lysine's NZ as a Schiff
    base, the whole one chain residue (LYR). The lysine is the chain's; the
    retinal is reported, as released from it -- C=NZ read back as C=O."""

    got = mol_id.structure_to_smiles(str(INPUTS / "retinal_lysine_peptide.pdb"))

    assert [g["resname"] for g in got] == ["LYR"]
    assert got[0]["formula"] == "C20H28O"
    assert got[0]["cut_from"] == "LYR NZ"
    # retinal's skeleton; the stereo block follows the frame
    assert got[0]["inchikey"].startswith("NCYCYZXNIZJOKI-")


def test_a_palmitoyl_on_a_cysteine_is_reported_as_palmitic_acid():
    """A palmitoyl thioester on a cysteine's SG, one chain residue (CYP): the
    thioester read back as the acid."""

    got = mol_id.structure_to_smiles(
        str(INPUTS / "palmitoyl_cysteine_peptide.pdb")
    )

    assert [g["resname"] for g in got] == ["CYP"]
    assert got[0]["smiles"] == "CCCCCCCCCCCCCCCC(=O)O"
    assert got[0]["cut_from"] == "CYP SG"
