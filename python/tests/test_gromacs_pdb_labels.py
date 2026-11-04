"""Chain IDs and elements of the PDBs the GROMACS path writes.

minimal.pdb takes full.pdb's chain IDs and elements atom for atom. Which
chain IDs full.pdb carries is decided by mdr-process (`assign_chain_ids`,
MDR-106) for every path, not here.
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


def test_label_gromacs_pdbs_leaves_full_pdb_and_copies_to_minimal(tmp_path):
    """full.pdb keeps the IDs GROMACS wrote (mdr-process decides them
    after); minimal.pdb takes full.pdb's"""

    write(tmp_path / "full.pdb", residues(TWO_CHAINS))
    write(
        tmp_path / "minimal.pdb",
        [atom(1, "CA", "MET", " ", 1, "C"), atom(2, "CA", "SER", " ", 2, "C")],
    )
    c.label_gromacs_pdbs(str(tmp_path), lipid_present=False)
    assert chain_ids(tmp_path / "minimal.pdb") == [("A", 1), ("B", 2)]


# --------------------------------------------------
def placed(serial, name, element, x, y=0.0, z=0.0, resnum=9):
    """One LIG atom at (x, y, z)"""
    line = atom(serial, name, "LIG", "A", resnum, element)
    return line[:30] + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}" + line[54:]


def test_halogens_written_as_c_and_b_are_set_to_cl_and_br(tmp_path):
    """DDD's Pro_lig.pdb writes CL1 as C and BR1 as B; on the no-TPR path
    full.pdb copies that column. A halogen is bonded to one heavy atom and
    no hydrogen"""

    full = tmp_path / "full.pdb"
    write(
        full,
        [
            placed(1, "C1", "C", 0.0),
            placed(2, "CL1", "C", 1.75),
            placed(3, "C2", "C", -1.4),
            placed(4, "BR1", "B", -3.3),
            placed(5, "CL", "C", 20.0, resnum=10),  # a lone ion
        ],
    )

    assert c.fix_halogen_elements(str(full)) == 3
    assert elements(full) == ["C", "Cl", "C", "Br", "Cl"]


def test_atoms_merely_named_like_halogens_are_left(tmp_path):
    """The same corpus has a CF3 carbon named CL1 (MDR00092758) and
    sulfurs named SE1 (MDR00090909); a methyl named CL2 has hydrogens"""

    full = tmp_path / "full.pdb"
    write(
        full,
        [
            placed(1, "CL1", "C", 0.0),
            placed(2, "F1", "F", 1.35),
            placed(3, "F2", "F", -0.45, 1.27),
            placed(4, "F3", "F", -0.45, -0.64, 1.1),
            placed(5, "C2", "C", -0.5, -0.7, -1.3),
            placed(6, "SE1", "S", 5.0),
            placed(7, "C3", "C", 6.8),
            placed(8, "C4", "C", 3.2),
            placed(9, "CL2", "C", 10.0),
            placed(10, "C5", "C", 11.5),
            placed(11, "H1", "H", 9.0),
        ],
    )
    before = full.read_text()

    assert c.fix_halogen_elements(str(full)) == 0
    assert full.read_text() == before


def test_label_gromacs_pdbs_gives_minimal_the_corrected_halogens(tmp_path):
    write(
        tmp_path / "full.pdb",
        [placed(1, "C1", "C", 0.0), placed(2, "CL1", "C", 1.75)],
    )
    write(tmp_path / "minimal.pdb", [placed(1, "CL1", "C", 1.75)])

    c.label_gromacs_pdbs(str(tmp_path), lipid_present=False)

    assert elements(tmp_path / "full.pdb") == ["C", "Cl"]
    assert elements(tmp_path / "minimal.pdb") == ["Cl"]
