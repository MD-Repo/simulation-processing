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


def _modified_peptide(tmp_path, middle, resname, join, group, double=False):
    """Ala-X-Ala with `group` (SMILES, its first atom bonded to `join`) on
    the middle residue's side chain, the whole residue named `resname`,
    embedded with every hydrogen and written without CONECT records, as the
    pipeline's minimal.pdb is."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rw = Chem.RWMol(Chem.MolFromSequence(f"A{middle}A"))
    at = None
    for a in rw.GetAtoms():
        info = a.GetPDBResidueInfo()
        if info.GetResidueNumber() == 2:
            info.SetResidueName(resname)
            if info.GetName().strip() == join:
                at = a.GetIdx()
    grp = Chem.MolFromSmiles(group)
    first = rw.GetNumAtoms()
    for k, a in enumerate(grp.GetAtoms()):
        i = rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
        rw.GetAtomWithIdx(i).SetMonomerInfo(
            Chem.AtomPDBResidueInfo(
                f" {a.GetSymbol()}X{k}"[:4].ljust(4),
                residueName=resname,
                residueNumber=2,
                chainId="A",
                isHeteroAtom=True,
            )
        )
    for b in grp.GetBonds():
        rw.AddBond(
            b.GetBeginAtomIdx() + first,
            b.GetEndAtomIdx() + first,
            b.GetBondType(),
        )
    rw.AddBond(
        at, first, Chem.BondType.DOUBLE if double else Chem.BondType.SINGLE
    )
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m, addCoords=False, addResidueInfo=True)
    # Each residue's atoms together, or MDAnalysis splits the residue.
    order = sorted(
        range(m.GetNumAtoms()),
        key=lambda i: (
            m.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueNumber(),
            i,
        ),
    )
    m = Chem.RenumberAtoms(m, order)
    assert AllChem.EmbedMolecule(m, randomSeed=11) == 0
    AllChem.MMFFOptimizeMolecule(m)
    p = tmp_path / f"{resname}.pdb"
    p.write_text(Chem.MolToPDBBlock(m, flavor=2))
    return str(p)


def test_conect_records_for_the_ligand_alone_still_find_the_peptide_bonds(
    tmp_path,
):
    """The Gla fixture with CONECT records for vorapaxar only, as a file
    written by a tool that bonds HETATM residues alone. Stated bonds used to
    turn the distance check off, so the CGU were never linked; and asked of
    CGU alone, the answer differed from asked of every candidate."""

    lines = GLA.read_text().splitlines()
    u = _universe(GLA)
    vpx = u.select_atoms("resname VPX")
    heavy = np.array([not n.startswith("H") for n in vpx.names])
    d = np.linalg.norm(vpx.positions[:, None] - vpx.positions[None], axis=-1)
    cut = np.where(heavy[:, None] & heavy[None], 1.9, 1.25)
    conect = [
        f"CONECT{vpx[i].id:5d}{vpx[j].id:5d}"
        for i, j in zip(*np.nonzero(np.triu((d < cut) & (d > 0))))
    ]
    end = next(k for k, l in enumerate(lines) if l.startswith("END"))
    path = tmp_path / "gla_conect.pdb"
    path.write_text("\n".join(lines[:end] + conect + lines[end:]) + "\n")
    u = _universe(path)
    assert len(u.atoms.bonds) == len(conect)

    assert set(mol_id.find_ligand_resnames(u)) == {"VPX"}
    cgu = mol_id._polymer_residues(u, {"CGU"})
    assert len(cgu) == 2
    assert cgu <= mol_id._polymer_residues(u, {"CGU", "VPX"})


def test_a_ligand_that_is_a_chain_of_nonstandard_residues_stays(tmp_path):
    """Three nonstandard residues peptide-bonded to each other and to no
    standard residue, as cyclosporin's are: a ligand, not a chain's
    residues. It used to vanish, leaving "No ligand-like residue found"."""

    path = _pdb(
        tmp_path,
        [
            ("MLE", 1, "N", "N", 0.000, 0.000, 0.000),
            ("MLE", 1, "CA", "C", 1.460, 0.000, 0.000),
            ("MLE", 1, "C", "C", 2.000, 1.420, 0.000),
            ("MVA", 2, "N", "N", 3.330, 1.420, 0.000),
            ("MVA", 2, "CA", "C", 3.870, 2.840, 0.000),
            ("MVA", 2, "C", "C", 5.330, 2.840, 0.000),
            ("SAR", 3, "N", "N", 5.870, 4.260, 0.000),
            ("SAR", 3, "CA", "C", 7.330, 4.260, 0.000),
            ("SAR", 3, "C", "C", 7.870, 5.680, 0.000),
        ],
    )

    u = _universe(path)
    assert mol_id._polymer_residues(u, {"MLE", "MVA", "SAR"}) == set()
    assert set(mol_id.find_ligand_resnames(u)) == {"MLE", "MVA", "SAR"}


def test_a_thioether_is_not_released_as_an_alcohol(tmp_path):
    """Farnesyl on a cysteine's SG by a thioether. Hydrolysis does not break
    that bond; read as released, it would be farnesol, which no one put
    there. Not recorded as a ligand, and said."""

    path = _modified_peptide(
        tmp_path, "C", "CYF", "SG", "CC=C(C)CCC=C(C)CCC=C(C)C"
    )
    notes = []

    assert mol_id.find_ligand_resnames(_universe(path), notes) == {}
    assert len(notes) == 1
    assert "15 heavy atoms covalently bound at CYF SG" in notes[0]
    assert "not one that hydrolysis would break" in notes[0]


def test_biotin_on_a_lysine_is_released_by_its_amide(tmp_path):
    """Biocytin: biotin's carboxyl as an amide on a lysine's NZ. Hydrolysed,
    that is biotin."""

    path = _modified_peptide(
        tmp_path,
        "K",
        "BTK",
        "NZ",
        "C(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12",
    )
    notes = []
    got = mol_id.structure_to_smiles(path, notes=notes)

    assert [g["resname"] for g in got] == ["BTK"]
    assert got[0]["formula"] == "C10H16N2O3S"
    assert got[0]["cut_from"] == "BTK NZ"
    assert any("C10H16N2O3S was found covalently bound" in n for n in notes)


def test_a_small_modification_genetically_encoded_is_the_residue_own(
    tmp_path,
):
    """Pyrrolysine's methylpyrroline-carbonyl, 8 heavy atoms on NZ: an amino
    acid of the genetic code, not a ligand on a lysine. Passed over, and
    said."""

    path = _modified_peptide(tmp_path, "K", "PYL", "NZ", "C(=O)C1N=CCC1C")
    notes = []

    assert mol_id.find_ligand_resnames(_universe(path), notes) == {}
    assert notes == [
        "Residue PYL 2 is joined into a polymer chain, so it was not taken "
        "for a ligand."
    ]


def test_notes_reach_stderr_for_mdr_process():
    """mdr-process reads each note from a "[mdrepo] note=" line on stderr."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "mol_id.py", "smiles-from-structure", str(GLA)],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(mol_id.__file__).parent,
    )

    assert out.returncode == 0, out.stderr
    assert (
        f"{mol_id.NOTE_MARKER}Residue CGU 6 is joined into a polymer chain"
        in out.stderr
    )
