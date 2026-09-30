"""Bond orders from the explicit hydrogens, not from a simulated frame's angles.

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_mol_id_bond_orders.py -q
"""

import pathlib

from openbabel import openbabel as ob

import mol_id

INPUTS = pathlib.Path(__file__).parent / "inputs/ligands"


def _ob_from_pdb(text: str):
    conv = ob.OBConversion()
    conv.SetInFormat("pdb")
    mol = ob.OBMol()
    conv.ReadString(mol, text)
    mol.PerceiveBondOrders()
    return mol


def _pdb(atoms) -> str:
    """HETATM lines for (name, element, x, y, z)."""
    return "".join(
        f"HETATM{i + 1:5d} {name:<4} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {el:>2}\n"
        for i, (name, el, x, y, z) in enumerate(atoms)
    ) + "END\n"


def test_a_double_bond_bent_by_the_frame_is_still_double():
    """Vorapaxar from the first frame of a NAMD run. Its vinyl carbons are
    1.36 A apart with three neighbours each, but the frame bends their angles
    past what OpenBabel types as sp2: read by geometry, the bond is single and
    each carbon gains an implicit hydrogen, C29H35FN2O4. The file's 33
    hydrogens leave only the double bond."""

    got = mol_id.structure_to_smiles(
        str(INPUTS / "vorapaxar_namd_frame.pdb"), resname="VPX"
    )
    got = got[0] if isinstance(got, list) else got

    assert got["formula"] == "C29H33FN2O4"
    assert got["charge"] == 0
    assert "/C=C/" in got["smiles"]


def test_a_united_atom_residue_keeps_the_geometric_reading():
    """HO-CH2-CH2-OH with its carbons' hydrogens folded into them, as a
    united-atom force field writes it. The valences alone could only be met
    by a charged C=C, which the frame holds at single-bond length: refused."""

    mol = _ob_from_pdb(
        _pdb(
            [
                ("O1", "O", 0.000, 0.000, 0.000),
                ("C1", "C", 1.430, 0.000, 0.000),
                ("C2", "C", 1.940, 1.440, 0.000),
                ("O2", "O", 3.370, 1.440, 0.000),
                ("H1", "H", -0.320, 0.900, 0.000),
                ("H2", "H", 3.690, 0.540, 0.000),
            ]
        )
    )

    assert mol_id._bond_orders_from_hydrogens(mol) is None


def test_a_residue_without_hydrogens_is_left_to_openbabel():
    mol = _ob_from_pdb(
        _pdb(
            [
                ("C1", "C", 0.000, 0.000, 0.000),
                ("O1", "O", 1.230, 0.000, 0.000),
            ]
        )
    )

    assert mol_id._bond_orders_from_hydrogens(mol) is None


def test_a_residue_cut_from_a_chain_keeps_the_geometric_reading():
    """An acetyl cap cut from its peptide, as a ligand selection cuts it: the
    carbonyl carbon has lost its bond to the next residue's nitrogen. The
    valences alone can only close it as an acylium, C#[O+], which is refused
    -- the same reading that made a pyroglutamate a +2 ion."""

    mol = _ob_from_pdb(
        _pdb(
            [
                ("CH3", "C", 0.000, 0.000, 0.000),
                ("C", "C", 1.520, 0.000, 0.000),
                ("O", "O", 2.135, 1.065, 0.000),
                ("H1", "H", -0.363, -1.028, 0.000),
                ("H2", "H", -0.363, 0.514, 0.890),
                ("H3", "H", -0.363, 0.514, -0.890),
            ]
        )
    )

    assert mol_id._bond_orders_from_hydrogens(mol) is None


def test_bond_orders_the_search_cannot_settle_are_left_to_openbabel():
    """Phytate with its twelve charges unwritten, as cpptraj writes a PDB: the
    search for bond orders has every phosphate oxygen to choose among, and
    without a bound runs for hours. It gives up within the iteration budget."""

    got = mol_id.structure_to_smiles(
        str(INPUTS / "phytate_embedded.pdb"), resname="IHP"
    )
    got = got[0] if isinstance(got, list) else got

    assert got["formula"] == "C6H18O24P6"


def test_charmm_ions_are_background():
    for resname in ("SOD", "POT", "CLA", "CAL", "CES", "LIT", "RUB", "BAR", "CAD"):
        assert mol_id._is_skipped_residue(resname), resname
