"""Chain IDs and elements of the PDBs the GROMACS path writes.

`restore_chain_ids`, which this replaces, mapped residue NUMBER to chain ID
from the submitter's PDB (last writer wins) and rewrote every residue with
that number. GROMACS already writes correct chain IDs from the TPR, one
letter per molecule with each chain's own numbering, and pdb2gmx restarts
every chain at 1, so a 2-chain system (A 1-12, B 1-8) came out as
B 1-8 / A 9-12 / B 1-8 and the splitter cut chain A in two (review
CONV-1, SEQ-1). The transfer is now atom for atom or not at all.
"""

import importlib.util
import os
import sys
import types


def _load_module():
    """Import the script with its conda-only dependencies stubbed out, only
    while its body runs: a stub left in sys.modules breaks MDAnalysis for
    every later test in the process (see test_dcd_sampling.py)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    planted = [
        name
        for name in ("pytraj", "parmed")
        if sys.modules.setdefault(name, types.ModuleType(name)).__spec__ is None
    ]
    try:
        spec = importlib.util.spec_from_file_location(
            "cpptraj_gmx_traj_manipulation",
            os.path.join(here, "cpptraj_gmx_traj_manipulation.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in planted:
            sys.modules.pop(name, None)


c = _load_module()


# --------------------------------------------------
def atom(serial, name, resname, chain, resnum, element, x=0.0):
    """One PDB record in GROMACS's layout (4-char residue names in 18-21)"""
    return (
        f"ATOM  {serial:>5} {name:<4} {resname:<4}{chain:1}{resnum:>4}    "
        f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00          {element:>2}\n"
    )


def residues(chains, start_serial=1):
    """ATOM lines for `chains`: a list of (chain ID, [(resname, resnum)]),
    three backbone atoms per residue"""
    lines = []
    s = start_serial
    for chain, res in chains:
        for resname, resnum in res:
            for name in ("N", "CA", "C"):
                lines.append(atom(s, name, resname, chain, resnum, name[0]))
                s += 1
        lines.append("TER\n")
    return lines


def write(path, lines):
    with open(path, "w") as fh:
        fh.writelines(["CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1\n"])
        fh.writelines(lines)
        fh.write("END\n")


def chain_ids(path):
    """(chain, resnum) of each residue in file order"""
    out = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], int(line[22:26]))
            if not out or out[-1] != key:
                out.append(key)
    return out


TWO_CHAINS = [
    ("A", [("MET", 1), ("ALA", 2), ("LYS", 3)]),
    ("B", [("GLY", 1), ("SER", 2)]),
]


# --------------------------------------------------
def test_gromacs_chain_ids_survive_restarted_numbering(tmp_path):
    """CONV-1's shape: both chains numbered from 1. GROMACS's letters are
    right and the submitter's agree; nothing may be relabelled by number"""

    ref = tmp_path / "submitted.pdb"
    full = tmp_path / "full.pdb"
    write(ref, residues(TWO_CHAINS))
    write(full, residues(TWO_CHAINS))

    assert c.transfer_chain_ids(str(ref), str(full), str(tmp_path / "out.pdb"))
    assert chain_ids(tmp_path / "out.pdb") == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 1),
        ("B", 2),
    ]


def test_chains_merged_into_one_moleculetype_get_their_letters_back(tmp_path):
    """pdb2gmx -merge all puts both chains in one moleculetype, which
    GROMACS letters as one molecule. The submitter's PDB, the same atoms
    in the same order, says where B starts"""

    ref = tmp_path / "submitted.pdb"
    full = tmp_path / "full.pdb"
    write(ref, residues(TWO_CHAINS))
    write(full, residues([("A", TWO_CHAINS[0][1]), ("A", TWO_CHAINS[1][1])]))

    out = tmp_path / "out.pdb"
    assert c.transfer_chain_ids(str(ref), str(full), str(out))
    assert chain_ids(out) == [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2)]


def test_gromacs_renumbering_of_ions_does_not_block_the_transfer(tmp_path):
    """GROMACS renumbers single-residue molecules after the last chain
    residue: the submitter's chloride 999 is residue 4 in full.pdb, and
    the merged protein has no letters at all (seen with GROMACS 2025.4)"""

    ref = tmp_path / "submitted.pdb"
    full = tmp_path / "full.pdb"
    write(ref, residues(TWO_CHAINS) + [atom(50, "CL1", "LIG", "X", 999, "Cl")])
    write(
        full,
        residues([(" ", TWO_CHAINS[0][1]), (" ", TWO_CHAINS[1][1])])
        + [atom(50, "CL1", "LIG", " ", 4, "Cl")],
    )
    out = tmp_path / "out.pdb"
    assert c.transfer_chain_ids(str(ref), str(full), str(out))
    assert chain_ids(out) == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 1),
        ("B", 2),
        ("X", 4),
    ]


def test_a_blank_reference_chain_keeps_the_gromacs_letter(tmp_path):
    ref = tmp_path / "submitted.pdb"
    full = tmp_path / "full.pdb"
    write(ref, residues([(" ", TWO_CHAINS[0][1]), ("B", TWO_CHAINS[1][1])]))
    write(full, residues(TWO_CHAINS))

    out = tmp_path / "out.pdb"
    assert c.transfer_chain_ids(str(ref), str(full), str(out))
    assert chain_ids(out) == [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2)]


def test_a_reference_of_other_atoms_is_not_applied(tmp_path):
    """The submitter's PDB is often the one BEFORE pdb2gmx: no hydrogens,
    no water, so fewer atoms. Then GROMACS's own IDs stand and nothing is
    written"""

    ref = tmp_path / "submitted.pdb"
    full = tmp_path / "full.pdb"
    write(ref, residues(TWO_CHAINS))
    water = [("X", [("SOL", 10), ("SOL", 11)])]
    write(full, residues(TWO_CHAINS + water))
    out = tmp_path / "out.pdb"
    assert not c.transfer_chain_ids(str(ref), str(full), str(out))
    assert not out.exists()

    # The same count, but not the same atoms
    other = [
        ("A", [("MET", 1), ("ALA", 2), ("LYS", 3)]),
        ("B", [("TRP", 1), ("SER", 2)]),
    ]
    write(ref, residues(other))
    assert not c.transfer_chain_ids(str(ref), str(full.with_name("x.pdb")), str(out))
    write(full, residues(TWO_CHAINS))
    assert not c.transfer_chain_ids(str(ref), str(full), str(out))
    assert not out.exists()


def test_minimal_takes_full_pdb_columns_in_order(tmp_path):
    """minimal.pdb is a subset of full.pdb's atoms in the same order. The
    IDs come across by position, including for a 4-character residue name
    GROMACS read one column late (POPC -> OPC)"""

    full = tmp_path / "full.pdb"
    minimal = tmp_path / "minimal.pdb"
    lines = residues(TWO_CHAINS)
    lines += [atom(50, "N", "POPC", "C", 7, "N"), atom(51, "C1", "POPC", "C", 7, "C")]
    write(full, lines)
    # Every CA, the lipid (as gmx reads it), chains blank, elements lost
    sub = [
        atom(1, "CA", "MET", " ", 1, "C"),
        atom(2, "CA", "LYS", " ", 3, "C"),
        atom(3, "CA", "SER", " ", 2, "C"),
        atom(4, "N", "OPC", " ", 7, "N"),
    ]
    write(minimal, sub)

    assert c.copy_pdb_columns(str(full), str(minimal), element=False)
    assert chain_ids(minimal) == [("A", 1), ("A", 3), ("B", 2), ("C", 7)]


def elements(path):
    return [
        line[76:78].strip()
        for line in open(path)
        if line.startswith(("ATOM", "HETATM"))
    ]


def test_minimal_takes_the_elements_trjconv_wrote_from_the_tpr(tmp_path):
    """editconf re-derives elements from atom names: CL1 -> C, BR1 -> B,
    SE1 -> S, FE1 -> F (review F1). full.pdb, from the TPR's atomic
    numbers, has them right; minimal.pdb takes them atom for atom"""

    full = tmp_path / "full.pdb"
    minimal = tmp_path / "minimal.pdb"
    ligand = [
        ("C1", "C"),
        ("CL1", "Cl"),
        ("BR1", "Br"),
        ("SE1", "Se"),
        ("FE1", "Fe"),
        ("ZN1", "Zn"),
    ]
    write(
        full,
        residues(TWO_CHAINS)
        + [
            atom(60 + i, name, "LIG", "A", 9, el) for i, (name, el) in enumerate(ligand)
        ],
    )
    broken = {"C": "C", "Cl": "C", "Br": "B", "Se": "S", "Fe": "F", "Zn": ""}
    write(
        minimal,
        [atom(1, "CA", "MET", "A", 1, "C")]
        + [
            atom(2 + i, name, "LIG", "A", 9, broken[el])
            for i, (name, el) in enumerate(ligand)
        ],
    )

    assert c.copy_pdb_columns(str(full), str(minimal))
    assert elements(minimal) == ["C", "C", "Cl", "Br", "Se", "Fe", "Zn"]
    assert chain_ids(minimal) == [("A", 1), ("A", 9)]


def test_a_blank_source_element_leaves_the_target_element(tmp_path):
    full = tmp_path / "full.pdb"
    minimal = tmp_path / "minimal.pdb"
    write(full, [atom(1, "CA", "MET", "A", 1, "")])
    write(minimal, [atom(1, "CA", "MET", " ", 1, "C")])
    assert c.copy_pdb_columns(str(full), str(minimal))
    assert elements(minimal) == ["C"]
    assert chain_ids(minimal) == [("A", 1)]


def test_an_atom_missing_from_full_pdb_leaves_minimal_untouched(tmp_path):
    full = tmp_path / "full.pdb"
    minimal = tmp_path / "minimal.pdb"
    write(full, residues(TWO_CHAINS))
    write(
        minimal, [atom(1, "CA", "MET", " ", 1, "C"), atom(2, "CA", "TRP", " ", 9, "C")]
    )
    before = minimal.read_text()
    assert not c.copy_pdb_columns(str(full), str(minimal))
    assert minimal.read_text() == before


def test_label_gromacs_pdbs_keeps_the_gromacs_full_pdb(tmp_path):
    ref = tmp_path / "submitted.pdb"
    write(ref, residues(TWO_CHAINS))
    write(
        tmp_path / "full.pdb",
        residues([("A", TWO_CHAINS[0][1]), ("A", TWO_CHAINS[1][1])]),
    )
    write(
        tmp_path / "minimal.pdb",
        [atom(1, "CA", "MET", "A", 1, "C"), atom(2, "CA", "SER", "A", 2, "C")],
    )

    c.label_gromacs_pdbs(str(tmp_path), str(ref), lipid_present=False)

    assert chain_ids(tmp_path / "full.pdb") == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 1),
        ("B", 2),
    ]
    assert chain_ids(tmp_path / "full.gmx.pdb") == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("A", 1),
        ("A", 2),
    ]
    assert chain_ids(tmp_path / "minimal.pdb") == [("A", 1), ("B", 2)]


def test_label_gromacs_pdbs_without_a_pdb_coordinate_file(tmp_path):
    """A .gro coordinate file has no chain IDs: GROMACS's stand, and
    minimal.pdb still takes them from full.pdb"""

    write(tmp_path / "full.pdb", residues(TWO_CHAINS))
    write(
        tmp_path / "minimal.pdb",
        [atom(1, "CA", "MET", " ", 1, "C"), atom(2, "CA", "SER", " ", 2, "C")],
    )
    c.label_gromacs_pdbs(str(tmp_path), str(tmp_path / "conf.gro"), lipid_present=False)
    assert not (tmp_path / "full.gmx.pdb").exists()
    assert chain_ids(tmp_path / "minimal.pdb") == [("A", 1), ("B", 2)]
