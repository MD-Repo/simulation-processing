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
    without a bound runs for hours. It gives up within the iteration budget,
    at each charge it tries; -12 is past _MAX_SEARCHED_CHARGE. OpenBabel's
    reading, a hydrogen on every phosphate oxygen, is the neutral form a
    ligand is recorded in."""

    got = mol_id.structure_to_smiles(
        str(INPUTS / "phytate_embedded.pdb"), resname="IHP"
    )
    got = got[0] if isinstance(got, list) else got

    assert got["formula"] == "C6H18O24P6"


def test_a_thioacid_is_not_read_charge_separated():
    """Thioacetic acid, every hydrogen in place, its sulfur listed before its
    oxygen as a residue lists a cysteine's SG before an acyl's O. Given that
    order the valence search puts the double bond on sulfur,
    C(=[SH+])[O-] -- a charge-separated form, refused -- and the second pass,
    with the sulfur last, reads the acid. The C-S is set to the 1.74 A of the
    palmitoyl thioester this was found on, which a length check cannot refuse
    without refusing strained double bonds too."""

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

    got = mol_id._bond_orders_from_hydrogens(mol)

    assert got is not None
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    assert conv.WriteString(got).split()[0] == "CC(=O)S"


def test_a_thiazole_listed_sulfur_first_is_read_on_a_second_pass():
    """Thiazole-4-carboxamide with its sulfur listed first. In that order
    every charge the search tries gives a form the checks refuse; with the
    sulfur listed last it gives the thiazole. The atoms come back in the
    order they went in."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("NC(=O)c1cscn1"))
    AllChem.EmbedMolecule(m, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(m)
    heavy = sorted(
        (a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1),
        key=lambda i: m.GetAtomWithIdx(i).GetAtomicNum() != 16,
    )
    hydrogens = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 1]
    m = Chem.RenumberAtoms(m, heavy + hydrogens)
    mol = _ob_from_pdb(Chem.MolToPDBBlock(m, flavor=2 | 8))

    got = mol_id._bond_orders_from_hydrogens(mol)

    assert got is not None
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    assert conv.WriteString(got).split()[0] == "NC(=O)c1cscn1"
    assert [a.GetAtomicNum() for a in ob.OBMolAtomIter(got)] == [
        a.GetAtomicNum() for a in ob.OBMolAtomIter(mol)
    ]


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


def _all_atom(smiles: str, tmp_path, united: bool = False) -> dict:
    """mol_id's reading of `smiles` embedded with every hydrogen and written
    as cpptraj writes a ligand: one residue, no CONECT, no formal charges.
    With `united`, the hydrogens on carbon are left out."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    pos = m.GetConformer().GetPositions()
    atoms = [
        (f"{a.GetSymbol()}{a.GetIdx() + 1}", a.GetSymbol().upper(), *pos[a.GetIdx()])
        for a in m.GetAtoms()
        if not (
            united
            and a.GetAtomicNum() == 1
            and a.GetNeighbors()[0].GetAtomicNum() == 6
        )
    ]
    path = tmp_path / "ligand.pdb"
    path.write_text(_pdb(atoms))
    return mol_id.structure_to_smiles(str(path), resname="LIG")[0]


def _key(smiles: str) -> str:
    return mol_id._inchikey_from_smiles(smiles)


AMP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)({0})({0}))[C@@H](O)[C@H]1O"
ATP = (
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)({0})OP(=O)({0})OP(=O)({0})({0}))"
    "[C@@H](O)[C@H]1O"
)


def test_a_ligand_of_charge_two_or_more_has_its_charge_tried(tmp_path):
    """Review F2. No file states a ligand's charge, and tried at 0 and +-1
    alone AMP(2-) came back with its adenine charge-separated,
    [N+]1=CN=C2C1=[N+]=CN=C2N, ATP(4-) with a dihydro-adenine and spermine(4+)
    with four neutral four-valent nitrogens. Each is now found at its own
    charge, and recorded neutral."""

    for charged, neutral, charge in (
        (AMP.format("[O-]"), AMP.format("O"), -2),
        (ATP.format("[O-]"), ATP.format("O"), -4),
        ("[O-]C(=O)CC(O)(CC(=O)[O-])C(=O)[O-]", "OC(=O)CC(O)(CC(=O)O)C(=O)O", -3),
        ("[NH3+]CCC[NH2+]CCCC[NH2+]CCC[NH3+]", "NCCCNCCCCNCCCN", 4),
        ("C[NH+]1CC[NH+](CC1)Cc1ccccc1", "CN1CCN(CC1)Cc1ccccc1", 2),
    ):
        got = _all_atom(charged, tmp_path)
        assert got["simulated_charge"] == charge, charged
        assert _key(got["simulated_smiles"]) == _key(charged), charged
        assert got["charge"] == 0, charged
        assert got["inchikey"] == _key(neutral), charged
        assert "+" not in got["smiles"] and "-" not in got["smiles"], charged


def test_the_fewest_charged_atoms_win_then_the_lowest_charge():
    """Travis's tie-break (2026-11-12). The charges run 0, -1, +1, -2, ...,
    +-6; an assignment with fewer charged atoms beats one with more, and
    between two with as many the one of lower total charge is kept."""

    assert mol_id._charges_to_try(0) == [
        0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6
    ]  # fmt: skip
    assert mol_id._charges_to_try(-8)[-1] == -8


def test_a_neutral_ligand_and_a_zwitterion_are_left_as_they_are(tmp_path):
    """The search stops at 0 for a neutral molecule, and a charge no proton
    settles stays: a nitro group, a quaternary ammonium."""

    for smiles in ("[O-][N+](=O)c1ccccc1", "CC(=O)OCC[N+](C)(C)C", "c1ccncc1"):
        got = _all_atom(smiles, tmp_path)
        assert got["inchikey"] == _key(smiles), smiles
        assert "simulated_smiles" not in got, smiles

    glycine = _all_atom("[NH3+]CC(=O)[O-]", tmp_path)
    assert glycine["inchikey"] == _key("NCC(=O)O")
    assert glycine["simulated_charge"] == 0


def test_a_tetrazolate_is_not_read_as_a_cation(tmp_path):
    """Losartan's tetrazolate came back C1=[N+]=NN=N1, charge +1 for -1: the
    search at -1 puts the charge on carbon, refused, and at +1 found a ring
    nitrogen with two double bonds. That is refused too, and the ligand is
    recorded as the neutral tetrazole."""

    losartan = "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)-c1ccccc1-c1nn[n-]n1"
    got = _all_atom(losartan, tmp_path)

    assert "[N+]" not in got["smiles"]
    assert got["charge"] == 0
    assert got["formula"] == "C22H23ClN6O"


def test_a_united_atom_heteroaromatic_ring_gets_no_triple_bonds(tmp_path):
    """Review F3. With its ring CH united and its NH explicit, 2-pyridone's
    valences are met by [H]n1c#cc#cc1=O, which RDKit calls aromatic, and the
    check of bond lengths passed over aromatic bonds. It reads the Kekule
    form now, and no ring this small holds a triple bond at any length."""

    for smiles in (
        "O=c1cccc[nH]1",
        "O=c1cc[nH]cc1",
        "O=c1cc[nH]c(=O)[nH]1",
        "c1c[nH]cn1",
        "c1cc[nH]c1",
    ):
        got = _all_atom(smiles, tmp_path, united=True)
        assert "#" not in got["smiles"], (smiles, got["smiles"])
        assert got["num_heavy_atoms"] == _all_atom(smiles, tmp_path)["num_heavy_atoms"]


def test_an_all_atom_aromatic_ring_stretched_by_the_frame_is_still_read():
    """What the Kekule check must not cost: vorapaxar's NAMD frame holds one
    aromatic C-C at 1.50 A, past the margin for a double bond. A ring RDKit
    calls aromatic is exempt from that margin and from no other."""

    got = mol_id.structure_to_smiles(
        str(INPUTS / "vorapaxar_namd_frame.pdb"), resname="VPX"
    )[0]

    assert got["formula"] == "C29H33FN2O4"


def test_a_nitro_group_is_not_read_two_electrons_over():
    """With the charge range widened, a nitro group the search cannot settle
    at 0 was settled at -2 as N([O-])[O-] and recorded, neutral, as a
    dihydroxylamine: MDR00000544's ligand, which OpenBabel had read right,
    and MDR00020883's aminothiazole, which the sulfur-last pass had. Found
    by running the old and new mol_id over the corpus, 2026-11-12."""

    from rdkit import Chem

    m = Chem.MolFromSmiles("c1ccccc1N([O-])[O-]")
    nitrogen = next(a for a in m.GetAtoms() if a.GetAtomicNum() == 7)
    assert mol_id._implausible(nitrogen)

    for smiles in ("c1ccccc1[N+](=O)[O-]", "CC(=O)N[O-]", "C[N+](C)(C)[O-]"):
        m = Chem.MolFromSmiles(smiles)
        assert not any(mol_id._implausible(a) for a in m.GetAtoms()), smiles
