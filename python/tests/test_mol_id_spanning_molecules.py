"""A ligand made of several residues is one molecule (review F5).

mol_id took a ligand to be one residue. A peptide written with nonstandard
residue names, a glycan, or a lipid written the Amber Lipid21 way as a head
and two tails was cut at its residue boundaries, each piece had its open
valences filled, and the pieces were stored as the simulation's ligands.

Decided 2026-11-12 (Travis): a peptide is a ligand by sequence, X for a
residue with no one-letter code; glycans and multi-residue lipids are refused
for now; one copy of a ligand is enough.

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_mol_id_spanning_molecules.py -q
"""

import json
import subprocess
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import mol_id

BACKBONE = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)")


def _embedded(smiles: str):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=7) != 0:
        AllChem.EmbedMolecule(m, randomSeed=7, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    return m


def _pdb_lines(m, residue_of: dict, names: dict, first_serial: int = 1) -> str:
    """ATOM lines, no CONECT, for atoms grouped by residue in `residue_of`
    ({atom index: (resname, resid)}), written residue by residue."""
    pos = m.GetConformer().GetPositions()
    order = sorted(range(m.GetNumAtoms()), key=lambda i: (residue_of[i][1], i))
    out = []
    for serial, i in enumerate(order, first_serial):
        a = m.GetAtomWithIdx(i)
        resname, resid = residue_of[i]
        name = names.get(i, f"{a.GetSymbol()}{i + 1}")
        x, y, z = pos[i]
        out.append(
            f"HETATM{serial:5d} {name:<4} {resname:>3} L{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
            f"{a.GetSymbol().upper():>2}\n"
        )
    return "".join(out)


def _peptide(smiles: str, resnames: list) -> str:
    """PDB text for the peptide `smiles`, its residues named `resnames` in
    order along the backbone, each with atoms N, CA, C and O."""
    m = _embedded(smiles)
    matches = m.GetSubstructMatches(BACKBONE)
    assert len(matches) == len(resnames), (len(matches), resnames)
    residue_of, names = {}, {}
    backbone = {i for match in matches for i in match}
    for k, (n, ca, c, o) in enumerate(matches):
        res = (resnames[k], k + 1)
        names.update({n: "N", ca: "CA", c: "C", o: "O"})
        for i in (n, ca, c, o):
            residue_of[i] = res
        # The side chain: out from CA, and whatever hangs on N (an N-methyl)
        stack = [ca, n]
        while stack:
            i = stack.pop()
            for nb in m.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j not in residue_of and j not in backbone:
                    residue_of[j] = res
                    stack.append(j)
    # The C-terminal OXT and its hydrogen
    for a in m.GetAtoms():
        if a.GetIdx() not in residue_of:
            residue_of[a.GetIdx()] = residue_of[a.GetNeighbors()[0].GetIdx()]
    for a in m.GetAtoms():
        if a.GetIdx() not in residue_of:
            residue_of[a.GetIdx()] = residue_of[a.GetNeighbors()[0].GetIdx()]
    return _pdb_lines(m, residue_of, names)


def _split(smiles: str, parts: list) -> str:
    """PDB text for `smiles` with its atoms shared out among residues:
    `parts` is [(resname, SMARTS of that residue's heavy atoms)]."""
    m = _embedded(smiles)
    residue_of = {}
    for resid, (resname, smarts) in enumerate(parts, 1):
        (match,) = m.GetSubstructMatches(Chem.MolFromSmarts(smarts))[:1]
        for i in match:
            residue_of.setdefault(i, (resname, resid))
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 1:
            residue_of[a.GetIdx()] = residue_of[a.GetNeighbors()[0].GetIdx()]
    assert len(residue_of) == m.GetNumAtoms()
    return _pdb_lines(m, residue_of, {})


# H-D-Ala-Sar-D-Ala-MeLeu-OH: cyclosporin's residues, the review's example
TETRAPEPTIDE = "C[C@@H](N)C(=O)N(C)CC(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)O"
TETRAPEPTIDE_NAMES = ["DAL", "SAR", "DAL", "MLE"]


def _read(tmp_path, text: str):
    path = tmp_path / "ligand.pdb"
    path.write_text(text + "END\n")
    notes, peptides = [], []
    found = mol_id.structure_to_smiles(str(path), notes=notes, peptides=peptides)
    return found, peptides, notes


def test_a_peptide_of_nonstandard_residues_is_one_ligand_by_sequence(tmp_path):
    """It came back as three ligands: DAL C[C@@H](C=O)N, an aminoaldehyde,
    MLE and SAR likewise, the second DAL dropped."""

    found, peptides, notes = _read(
        tmp_path, _peptide(TETRAPEPTIDE, TETRAPEPTIDE_NAMES)
    )

    assert found == []
    assert peptides == [
        {
            "sequence": "XXXX",
            "sequence_type": "protein",
            "resnames": ["DAL", "SAR", "DAL", "MLE"],
            "resids": [1, 2, 3, 4],
            "cyclic": False,
            "resname": "DAL",
        }
    ]
    assert len(notes) == 1 and "one peptide of 4 residues" in notes[0]


def test_a_cyclic_peptide_is_one_ligand_too(tmp_path):
    """Head to tail, as cyclosporin is: every residue has a residue before
    it, and the sequence starts at the first in the file."""

    cyclic = "C[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)NC(=O)CN(C)C1=O"
    found, peptides, _ = _read(
        tmp_path, _peptide(cyclic, ["DAL", "MLE", "DAL", "SAR"])
    )

    assert found == []
    assert len(peptides) == 1
    assert peptides[0]["cyclic"] is True
    assert peptides[0]["sequence"] == "XXXX"
    assert sorted(peptides[0]["resids"]) == [1, 2, 3, 4]
    assert peptides[0]["resids"][0] == 1


def test_caps_are_named_and_left_out_of_the_sequence(tmp_path):
    text = _peptide(TETRAPEPTIDE, TETRAPEPTIDE_NAMES)
    # An acetyl on the N-terminus, 1.33 A from it, as residue ACE 0
    m = _embedded("CC(=O)N")
    pos = m.GetConformer().GetPositions()
    c, n = pos[1], pos[3]
    first_n = next(
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in text.splitlines()
        if line[12:16].strip() == "N" and line[22:26].strip() == "1"
    )
    shift = [first_n[k] - n[k] for k in range(3)]
    cap = "".join(
        f"HETATM{900 + i:5d} {name:<4} ACE L   0    "
        f"{pos[i][0] + shift[0]:8.3f}{pos[i][1] + shift[1]:8.3f}"
        f"{pos[i][2] + shift[2]:8.3f}  1.00  0.00          {el:>2}\n"
        for i, (name, el) in enumerate([("CH3", "C"), ("C", "C"), ("O", "O")])
    )
    assert abs(sum((c[k] - n[k]) ** 2 for k in range(3)) ** 0.5 - 1.36) < 0.1

    _, peptides, _ = _read(tmp_path, cap + text)

    assert len(peptides) == 1
    assert peptides[0]["resnames"] == ["ACE", "DAL", "SAR", "DAL", "MLE"]
    assert peptides[0]["sequence"] == "XXXX"


def test_a_lipid_written_as_head_and_tails_is_refused(tmp_path):
    """POPC the Lipid21 way, PA + PC + OL, came back as a palmitic aldehyde,
    an oleic aldehyde and a glycerophosphocholine. Not recorded, and said."""

    popc = (
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)"
        "OC(=O)CCCCCCC/C=C\\CCCCCCCC"
    )
    text = _split(
        popc,
        [
            ("PA", "[CH3]CCCCCCCCCCCCCCC(=O)"),
            ("OL", "[CH3]CCCCCCC/C=C\\CCCCCCCC(=O)"),
            ("PC", "OC[CH](COP(=O)([O-])OCC[N+](C)(C)C)O"),
        ],
    )

    found, peptides, notes = _read(tmp_path, text)

    assert found == [] and peptides == []
    assert len(notes) == 1
    assert "PA 1, OL 2 and PC 3" in notes[0]
    assert "not recorded" in notes[0]


def test_a_glycan_is_refused_and_a_ligand_beside_it_is_still_read(tmp_path):
    """Two sugars joined by a glycosidic bond are one molecule of two
    residues: refused. Benzamidine, bonded to nothing, is read as before."""

    chitobiose = (
        "CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)"
        "[C@H]2NC(C)=O)[C@@H](CO)O[C@H]1O"
    )
    sugar = "CC(=O)N[CH]1[CH]([OH])[CH]([OH0])[CH](CO)O[CH]1[OH]"
    other = "[CH]1(O[CH](CO)[CH](O)[CH](O)[CH]1NC(C)=O)"
    glycan = _split(chitobiose, [("NAG", sugar), ("NDG", other)])

    m = _embedded("NC(=N)c1ccccc1")
    pos = m.GetConformer().GetPositions()
    benzamidine = "".join(
        f"HETATM{800 + i:5d} {a.GetSymbol() + str(i + 1):<4} BEN L   9    "
        f"{pos[i][0] + 30:8.3f}{pos[i][1]:8.3f}{pos[i][2]:8.3f}"
        f"  1.00  0.00          {a.GetSymbol().upper():>2}\n"
        for i, a in enumerate(m.GetAtoms())
    )

    found, peptides, notes = _read(tmp_path, glycan + benzamidine)

    assert [f["resname"] for f in found] == ["BEN"]
    assert found[0]["inchikey"] == mol_id._inchikey_from_smiles("NC(=N)c1ccccc1")
    assert peptides == []
    assert any("NAG 1 and NDG 2" in n and "glycan" in n for n in notes)


def test_two_ligands_in_contact_are_still_two(tmp_path):
    """A hydrogen bond is not a bond: two residues 2.8 A apart, O to O."""

    m = _embedded("CO")
    pos = m.GetConformer().GetPositions()
    away = (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])
    # The second methanol is the first turned about the point 1.4 A past its
    # oxygen, on the C-O line: oxygen facing oxygen, 2.8 A apart.
    mirrored = 2 * (pos[1] + 1.4 * away) - pos

    def methanol(resname, resid, at):
        return "".join(
            f"HETATM{resid * 100 + i:5d} {a.GetSymbol() + str(i + 1):<4} "
            f"{resname} L{resid:4d}    {at[i][0]:8.3f}{at[i][1]:8.3f}"
            f"{at[i][2]:8.3f}  1.00  0.00          {a.GetSymbol().upper():>2}\n"
            for i, a in enumerate(m.GetAtoms())
        )

    found, peptides, notes = _read(
        tmp_path, methanol("MOH", 1, pos) + methanol("MEO", 2, mirrored)
    )

    assert sorted(f["resname"] for f in found) == ["MEO", "MOH"]
    assert peptides == [] and notes == []


def test_the_command_line_writes_a_peptide_beside_the_small_molecules(tmp_path):
    """`both --outfile` is what mdr-process reads: {structure, name} for a
    small molecule, {peptide} for a peptide ligand."""

    path = tmp_path / "ligand.pdb"
    path.write_text(_peptide(TETRAPEPTIDE, TETRAPEPTIDE_NAMES) + "END\n")
    out = tmp_path / "inferred_ligands.json"
    script = mol_id.__file__

    done = subprocess.run(
        [sys.executable, script, "both", str(path), "--outfile", str(out)],
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
    assert [list(e) for e in json.loads(out.read_text())] == [["peptide"]]
    assert "[mdrepo] note=Residues DAL 1, SAR 2, DAL 3 and MLE 4" in done.stderr
