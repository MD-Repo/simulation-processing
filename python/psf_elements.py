"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Fill the element column of a PDB written from a CHARMM .psf, from the
         topology's masses and bonds rather than from the atoms' names

A CHARMM .psf names atoms and their types and carries no elements, so
MDAnalysis writes the PDB's element column blank, and whatever reads the PDB
next guesses each element from the atom's name. Names are not elements:
glutamine's HE21 is a hydrogen that reads as helium, and a protein's CA is
carbon, not calcium. zenodo:17477539's structure, written from its .psf and
first frame, reached mdr-process's ligand inference as vorapaxar with 33
helium atoms.

The topology's masses and bonds are the source instead. Under hydrogen mass
repartitioning a heavy atom lends mass to its hydrogens -- the same release's
methyl carbons weigh 5.963 beside three hydrogens of 3.024 -- so the mass each
bonded hydrogen holds over its own is counted back before the nearest standard
atomic weight is taken. An atom whose mass matches no element, or two, stops
the write: the element is not guessed.

A single-atom residue, bonded to nothing, whose charge in the topology is a
whole nonzero number is an ion, and that charge is written as its formal
charge; without it a reader takes Ca2+ for neutral calcium and completes its
valence with hydrogens.

Used by subset_structure.py and first_frame_structure.py, the two commands
that write a PDB from a .psf. Ported from mdrepo-agent's
internal/stage/elements.go when those steps moved here (mdrepo-agent#98).
"""

import os
from typing import NamedTuple, Optional

# Standard atomic weights of the elements biomolecular simulations hold.
STANDARD_WEIGHTS = {
    "H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078,
    "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "As": 74.922,
    "Se": 78.971, "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Mo": 95.95,
    "Ru": 101.07, "Rh": 102.91, "Ag": 107.87, "Cd": 112.41, "I": 126.90,
    "Cs": 132.91, "Ba": 137.33, "W": 183.84, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Tl": 204.38, "Pb": 207.2,
}  # fmt: skip

# Force fields round weights differently (16.00, 15.9994, 15.999); the closest
# two elements above, Ni and Co, are 0.24 apart.
WEIGHT_TOLERANCE = 0.1

# Below HYDROGEN_MASS an atom is a hydrogen, repartitioned up to four times its
# weight; below VIRTUAL_MASS it is a massless site or a Drude particle, which
# has no element.
HYDROGEN_MASS = 4.6
VIRTUAL_MASS = 0.9


class PSFAtoms(NamedTuple):
    """What a .psf states of each atom that settles its element: its residue
    (segid, resid and resname as one string), name, charge and mass, and the
    bonds, 0-based."""

    residue: list[str]
    name: list[str]
    charge: list[float]
    mass: list[float]
    bonds: list[tuple[int, int]]


class Elements(NamedTuple):
    """Where a written PDB's element column came from."""

    source: str
    repartitioned: int
    ions: int

    def as_dict(self) -> dict:
        return {
            "from": self.source,
            "repartitioned": self.repartitioned,
            "ions": self.ions,
        }


class ElementError(ValueError):
    """An element that cannot be settled, or a PDB that is not the topology's
    atoms."""


# --------------------------------------------------
def is_psf(path: str) -> bool:
    return path.lower().endswith(".psf")


# --------------------------------------------------
def read_psf(path: str) -> PSFAtoms:
    """Read the atom and bond sections of a CHARMM .psf. Fields are read by
    whitespace, which serves the standard, EXT and XPLOR layouts alike."""

    name = os.path.basename(path)

    def count(line: str, section: str) -> Optional[int]:
        f = line.split()
        if len(f) < 2 or not f[1].startswith(section):
            return None
        try:
            return int(f[0])
        except ValueError:
            return None

    residue, names, charge, mass, bonds = [], [], [], [], []
    with open(path) as fh:
        lines = iter(fh)
        natom = None
        for line in lines:
            n = count(line, "!NATOM")
            if n is None:
                continue
            natom = n
            for i in range(n):
                try:
                    atom = next(lines)
                except StopIteration:
                    raise ElementError(f"{name} ends inside its atoms")
                f = atom.split()
                if len(f) < 8:
                    raise ElementError(
                        f"{name} atom {i + 1}: {atom.rstrip()!r} is not an atom line"
                    )
                try:
                    q, m = float(f[6]), float(f[7])
                except ValueError:
                    raise ElementError(
                        f"{name} atom {i + 1}: no charge and mass in {atom.rstrip()!r}"
                    )
                residue.append(f"{f[1]} {f[2]} {f[3]}")
                names.append(f[4])
                charge.append(q)
                mass.append(m)
            break
        if natom is None:
            raise ElementError(f"{name} has no !NATOM section")
        for line in lines:
            n = count(line, "!NBOND")
            if n is None:
                continue
            ids: list[int] = []
            for line in lines:
                if len(ids) >= 2 * n:
                    break
                for w in line.split():
                    try:
                        i = int(w)
                    except ValueError:
                        i = 0
                    if i < 1 or i > natom:
                        raise ElementError(f"{name}: {w!r} in its bonds is no atom")
                    ids.append(i - 1)
            if len(ids) != 2 * n:
                raise ElementError(
                    f"{name} states {n} bonds and lists {len(ids)} atoms of them"
                )
            bonds = [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]
            break
    return PSFAtoms(residue, names, charge, mass, bonds)


# --------------------------------------------------
def settle_elements(t: PSFAtoms) -> tuple[list[str], list[int], int, int]:
    """Give each atom its element, blank for a virtual site, and each ion its
    formal charge. Returns the elements, the formal charges, and how many
    heavy atoms had repartitioned mass counted back and how many ions were
    charged."""

    n = len(t.mass)
    lent = [0.0] * n
    bonded = [0] * n
    for i, j in t.bonds:
        for a, h in ((i, j), (j, i)):
            bonded[a] += 1
            if t.mass[a] >= HYDROGEN_MASS and t.mass[h] < HYDROGEN_MASS:
                lent[a] += t.mass[h]
                if t.mass[h] >= VIRTUAL_MASS:
                    lent[a] -= STANDARD_WEIGHTS["H"]

    elements = [""] * n
    repartitioned = 0
    for i, m in enumerate(t.mass):
        if m < VIRTUAL_MASS:
            continue
        if m < HYDROGEN_MASS:
            elements[i] = "H"
            continue
        mass = m + lent[i]
        near = sorted(
            el
            for el, w in STANDARD_WEIGHTS.items()
            if abs(w - mass) <= WEIGHT_TOLERANCE
        )
        if len(near) != 1:
            counted = f" ({mass:.3f} with its hydrogens')" if lent[i] != 0 else ""
            match = " and ".join(near) if near else "no element"
            raise ElementError(
                f"atom {i + 1} ({t.name[i]}, {t.residue[i]}) has mass {m:.3f}{counted}, "
                f"which matches {match}"
            )
        elements[i] = near[0]
        if abs(lent[i]) > 0.01:
            repartitioned += 1

    size: dict[str, int] = {}
    for r in t.residue:
        size[r] = size.get(r, 0) + 1
    formal = [0] * n
    ions = 0
    for i, q in enumerate(t.charge):
        if size[t.residue[i]] != 1 or bonded[i] != 0:
            continue
        r = round(q)
        if r != 0 and abs(q - r) < 0.01:
            formal[i] = int(r)
            ions += 1
    return elements, formal, repartitioned, ions


# --------------------------------------------------
def fill_elements(pdb: str, psf: str, atoms: Optional[list[int]] = None) -> Elements:
    """Write the elements and formal charges the .psf settles into the PDB.
    The PDB's atoms are the topology's atoms `atoms`, in order, or all of them
    when atoms is None."""

    t = read_psf(psf)
    elements, formal, repartitioned, ions = settle_elements(t)
    with open(pdb) as fh:
        lines = fh.read().splitlines(True)
    k = 0
    for i, line in enumerate(lines):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        at = k
        if atoms is not None:
            if k >= len(atoms):
                raise ElementError(
                    f"{os.path.basename(pdb)} holds more atoms than were selected"
                )
            at = atoms[k]
        if at >= len(elements):
            raise ElementError(
                f"{os.path.basename(pdb)} holds more atoms than {os.path.basename(psf)}"
            )
        body = line.rstrip("\r\n")
        nl = line[len(body) :]
        body = body.ljust(76)
        charge = "  "
        if formal[at]:
            q = formal[at]
            charge = f"{abs(q)}{'+' if q > 0 else '-'}"
        lines[i] = f"{body[:76]}{elements[at].upper():>2}{charge:>2}{nl}"
        k += 1
    want = len(atoms) if atoms is not None else len(elements)
    if k != want:
        raise ElementError(
            f"{os.path.basename(pdb)} holds {k} atoms and {want} were written to it"
        )
    tmp = pdb + ".elements"
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    os.replace(tmp, pdb)
    return Elements(psf, repartitioned, ions)


# --------------------------------------------------
def pdb_atom_count(path: str) -> int:
    """The ATOM and HETATM records of the first model: what the commands
    that write a PDB check against the atoms they meant to write."""

    n = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                n += 1
            elif line.startswith("ENDMDL"):
                break
    return n


# --------------------------------------------------
def pdb_lacks_elements(path: str) -> bool:
    """Whether any atom of the PDB has a blank element column."""

    with open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM  ", "HETATM")) and (
                len(line.rstrip("\r\n")) < 78 or not line[76:78].strip()
            ):
                return True
    return False
