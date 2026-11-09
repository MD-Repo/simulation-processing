#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Write a .psf topology of exactly the atoms a PDB holds, in its order,
         for a submission that ships a structure and no topology

MDRepo takes a topology file with every simulation. A release that ships only
a PDB and its trajectories has none, so one is written from the PDB with
ParmEd: the same atoms in the same order, so it lines up with the trajectories
frame by frame, with bonds inferred from residue templates and distance, or
the bonds the caller states when the release's own format states them (a
Desmond .cms lists its bonds; the PDB written from it does not).

This is a topology of the submitted structure, deposited beside it as
`topology_file_name`. It is not the standard .psf mdr-process writes of
full.pdb for the ML representations (extract_bonds.py); that one is built
after the fact from whatever topology the submission has, and this is what
gives such a submission a topology to begin with.

Until 2026-11 the agent (MD-Repo/mdrepo-agent, internal/stage/topology.go)
ran this inline. It moved here with the rest of the file-producing steps
(mdrepo-agent#98) so that a person's submission and an agent's follow the
same rule.

Two things the PDB can do to ParmEd, both met on real records:

  * GROMACS writes a residue number modulo 10000, so a solvated box's waters
    wrap and ParmEd folds a wrapped residue into an earlier one of the same
    number and refuses the file ("Coordinate mismatch"). Past the first 10000
    residues each further block gets a chain letter of its own, which keeps
    every (chain, number) pair distinct; residues before the wrap, the
    protein among them, keep their chains. zenodo:11780149's 79366-atom
    trjconv PDB is the case.

  * ParmEd checks a model's coordinates against its atoms only at ENDMDL,
    keyed by atom name and residue, so atoms that share a key are counted
    once and the file is refused ("Coordinate mismatch"): an oligomer whose
    subunits all carry one chain letter and the same numbers (zenodo:7986576's
    MthK tetramer, written by trjconv inside MODEL and ENDMDL), or a residue
    with two atoms named alike. A file of one model is read without the
    wrapper, and then the atoms are read in order, as many as it lists.

  Either rewritten copy is written to a temporary directory, never beside
  the PDB, whose directory is the deposit's; the PDB itself is not changed.

  * ParmEd takes a residue to be a run of consecutive atoms, and bonds an
    atom no template names by distance. Wrong names or a residue listed in
    two places break that without an error: zenodo:5357152's .cms listed
    every hydrogen after the protein's heavy atoms, and the PSF bonded 4,245
    hydrogens to nothing. A hydrogen bonded to no heavy atom, or a C, N, O, P
    or S atom bonded to nothing inside a residue of several atoms, says the
    bonds are wrong (a monatomic ion or a lone water oxygen is neither), and
    the topology is refused: no .psf is written and the run exits 1. The
    report still goes to stdout so the caller can say what was wrong.

Output: the .psf at --out, and one JSON object on stdout:

  {"atoms": N, "bonds": N, "unbonded_h": N, "isolated": N,
   "examples": [...], "renumbered": bool, "bonds_from": "inferred"|"stated"}

Exit 0 when the topology was written and its bonds pass; 1 with the reason
as the last line of stderr when they do not or the PDB cannot be read; 2
when parmed is not installed.
"""

import argparse
import json
import os
import sys
import tempfile
from typing import NamedTuple, Optional

import outputs

# The alphabet of chain letters the renumbering may hand out.
CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"

# How many residues one chain letter holds: the PDB field is four digits.
RESIDUES_PER_CHAIN = 9999

# Elements whose atoms cannot stand alone inside a residue of several atoms.
BONDED_HEAVY = (6, 7, 8, 15, 16)


class Args(NamedTuple):
    structure: str
    out: str
    bonds: Optional[str]


class Report(NamedTuple):
    atoms: int
    bonds: int
    unbonded_h: int
    isolated: int
    examples: list[str]
    renumbered: bool
    bonds_from: str

    def ok(self) -> bool:
        return self.unbonded_h == 0 and self.isolated == 0

    def refusal(self) -> str:
        return (
            f"the generated topology leaves {self.unbonded_h} hydrogens bonded "
            f"to no heavy atom and {self.isolated} atoms bonded to nothing "
            f"inside a residue ({'; '.join(self.examples)}), so its bonds are wrong"
        )


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Write a .psf of exactly the atoms a PDB holds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--structure", required=True, metavar="PDB", help="Structure file"
    )
    parser.add_argument(
        "-o", "--out", required=True, metavar="PSF", help="Topology to write"
    )
    parser.add_argument(
        "-b",
        "--bonds",
        metavar="JSON",
        help=(
            "Bonds the topology must carry, as a JSON list of [i, j] pairs of "
            "0-based atom indices in the PDB's order; without it the bonds are "
            "inferred from the PDB"
        ),
    )
    a = parser.parse_args()
    return Args(a.structure, a.out, a.bonds)


# --------------------------------------------------
def wrap_safe(path: str, dest: str) -> None:
    """Write a copy of the PDB whose residues past the first 9999 take a
    chain letter of their own, so no (chain, number) pair repeats."""

    with open(path) as fh:
        lines = fh.read().splitlines(True)
    used = {
        line[21]
        for line in lines
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21 and line[21] != " "
    }
    letters = [c for c in CHAIN_LETTERS if c not in used]
    out, last, n = [], None, 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) > 26:
            key = (line[17:20], line[21], line[22:27])
            if key != last:
                n += 1
                last = key
            if n > RESIDUES_PER_CHAIN:
                block = (n - 1) // RESIDUES_PER_CHAIN
                if block - 1 >= len(letters):
                    raise ValueError("too many residues to renumber past the wrap")
                line = (
                    line[:21]
                    + letters[block - 1]
                    + "%4d" % (((n - 1) % RESIDUES_PER_CHAIN) + 1)
                    + line[26:]
                )
        out.append(line)
    with open(dest, "w") as fh:
        fh.writelines(out)


# --------------------------------------------------
def single_model(path: str, dest: str) -> bool:
    """Write a copy of the PDB without its MODEL and ENDMDL lines, when it
    holds at most one model. False, and nothing written, when it holds
    several: those are frames, and they have to agree."""

    with open(path) as fh:
        lines = fh.read().splitlines(True)
    if sum(1 for line in lines if line.startswith("MODEL")) > 1:
        return False
    with open(dest, "w") as fh:
        fh.writelines(l for l in lines if not l.startswith(("MODEL", "ENDMDL")))
    return True


# --------------------------------------------------
def load_structure(path: str):
    """Load the PDB with ParmEd, rewriting one it refuses with "Coordinate
    mismatch": first with residues renumbered past the wrap, then without a
    single model's MODEL and ENDMDL. Returns the structure and whether it was
    renumbered."""

    import parmed as pmd

    try:
        return pmd.load_file(path), False
    except pmd.exceptions.PDBError as e:
        if "Coordinate mismatch" not in str(e):
            raise
        err = e
    # The rewritten copies go in a directory of their own: the PDB's is the
    # deposit's, and every file in it is part of the deposit.
    with tempfile.TemporaryDirectory(prefix="psf-") as tmp:
        wrapped = os.path.join(tmp, "renumbered.pdb")
        wrap_safe(path, wrapped)
        try:
            return pmd.load_file(wrapped), True
        except pmd.exceptions.PDBError as e:
            if "Coordinate mismatch" not in str(e):
                raise
            err = e
        one = os.path.join(tmp, "single.pdb")
        if single_model(path, one):
            try:
                return pmd.load_file(one), False
            except pmd.exceptions.PDBError as e:
                if "Coordinate mismatch" not in str(e):
                    raise
                err = e
    raise err


# --------------------------------------------------
def state_bonds(st, pairs: list) -> None:
    """Replace the structure's bonds with the stated pairs."""

    import parmed as pmd

    n = len(st.atoms)
    if any(not isinstance(i, int) or i < 0 or i >= n for pair in pairs for i in pair):
        raise ValueError(f"a bond names an atom outside the {n} in the structure")
    for b in list(st.bonds):
        b.delete()
    del st.bonds[:]
    for i, j in pairs:
        st.bonds.append(pmd.Bond(st.atoms[i], st.atoms[j]))


# --------------------------------------------------
def check_bonds(st) -> tuple[int, int, list[str]]:
    """Count the hydrogens bonded to no heavy atom and the heavy atoms bonded
    to nothing inside a residue of several atoms, naming a few."""

    unbonded_h, isolated, examples = 0, 0, []
    for a in st.atoms:
        if a.atomic_number == 1 and not any(
            b.atomic_number > 1 for b in a.bond_partners
        ):
            unbonded_h += 1
        elif (
            a.atomic_number in BONDED_HEAVY
            and not a.bond_partners
            and len(a.residue.atoms) > 1
        ):
            isolated += 1
        else:
            continue
        if len(examples) < 5:
            chain = f" {a.residue.chain}" if a.residue.chain else ""
            examples.append(f"{a.residue.name} {a.residue.number}{chain} {a.name}")
    return unbonded_h, isolated, examples


# --------------------------------------------------
def generate(structure: str, out: str, bonds: Optional[list] = None) -> Report:
    """Write the .psf of the structure at out and report on it. A topology
    whose bonds are wrong is not written, and a file already at out is left
    as it was."""

    st, renumbered = load_structure(structure)
    if bonds is not None:
        state_bonds(st, bonds)
    unbonded_h, isolated, examples = check_bonds(st)
    report = Report(
        atoms=len(st.atoms),
        bonds=len(st.bonds),
        unbonded_h=unbonded_h,
        isolated=isolated,
        examples=examples,
        renumbered=renumbered,
        bonds_from="stated" if bonds is not None else "inferred",
    )
    if report.ok():
        # Written beside out and moved onto it whole, so a save that fails
        # midway leaves a topology already at out as it was.
        with outputs.staged(out) as psf:
            st.save(psf, format="psf", overwrite=True)
    return report


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        import parmed  # noqa: F401
    except ImportError:
        sys.stderr.write(f"parmed is not installed in {sys.executable}\n")
        sys.exit(2)

    bonds = None
    if args.bonds:
        with open(args.bonds) as fh:
            bonds = json.load(fh)
    try:
        report = generate(args.structure, args.out, bonds)
    except Exception as e:  # the reason is the last line of stderr
        sys.stderr.write(f"{args.structure}: {e}\n")
        sys.exit(1)

    print(json.dumps(report._asdict()))
    if not report.ok():
        sys.stderr.write(report.refusal() + "\n")
        sys.exit(1)


# --------------------------------------------------
if __name__ == "__main__":
    main()
