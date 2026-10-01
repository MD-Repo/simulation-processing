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


def test_a_thioacid_is_not_read_charge_separated():
    """Thioacetic acid, every hydrogen in place, its sulfur listed before its
    oxygen as a residue lists a cysteine's SG before an acyl's O. Given that
    order the valence search puts the double bond on sulfur,
    C(=[SH+])[O-] -- a charge-separated form, refused -- and the acid read
    from the geometry stands. The C-S is set to the 1.74 A of the palmitoyl
    thioester this was found on, which a length check cannot refuse without
    refusing strained double bonds too."""

    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms

    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)S"))
    AllChem.EmbedMolecule(m, randomSeed=3)
    heavy = sorted(
        (a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1),
        key=lambda i: m.GetAtomWithIdx(i).GetSymbol() == "O",
    )
    hydrogens = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1]
    m = Chem.RenumberAtoms(m, heavy + hydrogens)
    # The C-S of the palmitoyl thioester this was found on, in its frame.
    rdMolTransforms.SetBondLength(m.GetConformer(), 1, 2, 1.74)
    mol = _ob_from_pdb(Chem.MolToPDBBlock(m, flavor=2 | 8))

    assert mol_id._bond_orders_from_hydrogens(mol) is None
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    assert conv.WriteString(mol).split()[0] == "CC(=O)S"


def test_charmm_ions_are_background():
    for resname in ("SOD", "POT", "CLA", "CAL", "CES", "LIT", "RUB", "BAR", "CAD"):
        assert mol_id._is_skipped_residue(resname), resname


def _united_atom(smiles: str):
    """`smiles` embedded with every hydrogen, then with its carbons' hydrogens
    folded into them, as a united-atom force field writes it: an OBMol with
    the real geometry and only the polar hydrogens."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    pos = m.GetConformer().GetPositions()
    atoms = [
        (f"{a.GetSymbol()}{a.GetIdx() + 1}", a.GetSymbol(), *pos[a.GetIdx()])
        for a in m.GetAtoms()
        if not (
            a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() == 6
        )
    ]
    return _ob_from_pdb(_pdb(atoms))


def test_a_united_atom_aromatic_ring_keeps_the_geometric_reading():
    """Hydroquinone with its ring CH united. The valences can only be met with
    a triple bond and cumulated double bonds in the ring,
    OC1=C=C=C(O)C#C1, at aromatic length, under a single bond's: refused by
    their own lengths."""

    assert (
        mol_id._bond_orders_from_hydrogens(_united_atom("Oc1ccc(O)cc1"))
        is None
    )


def test_a_united_atom_vinyl_keeps_the_geometric_reading():
    """HO-CH=CH-OH with its CH united: the valences call for OC#CO, a triple
    bond at double-bond length."""

    assert mol_id._bond_orders_from_hydrogens(_united_atom("O/C=C/O")) is None


def test_a_real_nitrile_and_alkyne_are_still_settled():
    """The triple bonds a ligand does carry are far shorter than the ones a
    united-atom residue forces, and stand."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    for smiles in ("N#Cc1ccccc1", "C#Cc1ccccc1"):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, randomSeed=7)
        AllChem.MMFFOptimizeMolecule(m)
        pos = m.GetConformer().GetPositions()
        mol = _ob_from_pdb(
            _pdb(
                [
                    (
                        f"{a.GetSymbol()}{a.GetIdx() + 1}",
                        a.GetSymbol(),
                        *pos[a.GetIdx()],
                    )
                    for a in m.GetAtoms()
                ]
            )
        )
        got = mol_id._bond_orders_from_hydrogens(mol)
        assert got is not None, smiles
        conv = ob.OBConversion()
        conv.SetOutFormat("can")
        assert "#" in conv.WriteString(got), smiles
