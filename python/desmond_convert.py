#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Convert a Desmond run (an -out.cms and its _trj) to a PDB and an
         XTC that MDAnalysis reads, dropping the virtual sites from both

MDRepo reads neither a .cms nor a DTR, so a run is converted whole: the atoms
of its -out.cms, read as msys reads them (maestro.py), to a PDB, and the
frames of its _trj to an XTC with mdtraj, with the virtual sites dropped
from both. Everything after that -- the topology, the frame checks, the time
axis -- runs on the converted files as on any release's.

Three things the .cms can do to a PDB and the PSF made from it, each met on a
real record and fixed in the converted copies:

  * A .cms may list a residue's atoms apart: Maestro appends the hydrogens
    it adds after every heavy atom of the chain. msys groups residues by
    key, so the run is fine, but a PDB reader starts a residue at each
    change of residue, and the PSF built from that PDB had every residue
    twice, heavy atoms in one and hydrogens in the other, the hydrogens
    bonded to nothing. zenodo:5357152, 2026-09-29: all 530 protein residues
    split, 4,245 unbonded hydrogens. Each residue's atoms are brought
    together, and the trajectory indices follow them so the XTC follows the
    PDB (--regroup, the default).

  * A .cms may name atoms "H1 1", and msys keeps the space; so does a
    fixed-column PDB. But the PSF generated from that PDB is read by
    splitting on whitespace, and every field after the name shifts.
    zenodo:5357152, 2026-09-28: "Could not convert partial charge [H1]".
    Whitespace inside a name becomes an underscore, one character for one.

  * A ligand Maestro never named has a blank name in its .cms, the PSF
    carries the name and type fields empty, and a reader that splits on
    whitespace finds seven fields where it wants eight. zenodo:4064775,
    2026-11-05: 44 atoms of an UNK ligand, "Array is too short". A blank
    name becomes the element's symbol and a number within its residue (C1,
    C2, N1), skipping names the residue has.

The mapping of frames onto atoms is checked, because a wrong order converts
quietly into nonsense. The .cms is one frame of the run -- the last, or for
a merged trajectory the end of an earlier segment -- so some frame should
match it to within one shift of the box origin; failing that, the bond
lengths in the first frame should be bond lengths. Either establishes the
mapping; neither refuses the run.

Moved here from mdrepo-agent's internal/desmond (convert.go, pdb.go,
frames.py) with the rest of the file-producing steps (mdrepo-agent#98),
taking in agent PR #101's naming of blank atoms.

Output: the PDB at --out-pdb; with --trajectory, the XTC at --out-xtc; with
--out-bonds, the .cms bonds between the PDB's atoms as a JSON list of
0-based [i, j] pairs in the PDB's order, which generate_topology.py takes
as --bonds. Each is written beside its path and moved onto it only once the
run has converted, so a refusal leaves whatever was at the path as it was.
One JSON object on stdout:

  {"atoms": N, "particles": N, "pseudo": N, "pseudo_on": {resname: N},
   "renamed": N, "named": N, "regrouped": N, "bonds": N,
   "forcefields": [...], "frames": N, "first_time": ps, "dt": ps,
   "match_frame": i, "match_spread": nm, "matched": bool, "bond_p99": nm | null}

"matched" says the .cms coordinates are frame match_frame to within the
tolerance; otherwise the mapping rested on the bond lengths.

Exit 0 when converted; 1 with the reason as the last line of stderr when the
.cms cannot be read, the trajectory is not the same system, or the atoms do
not map; 2 when mdtraj is not installed.
"""

import argparse
import contextlib
import json
import math
import os
import sys
from typing import NamedTuple, Optional

import maestro
import outputs
from maestro import Particle

# Element symbols by atomic number.
ELEMENTS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr "
    "Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra "
    "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()

# Tolerances for the mapping checks. The .cms stores six decimals of
# angstroms and the .dtr single precision, so a right mapping matches a frame
# to well under 0.001 nm. Bonds are under 0.2 nm; a wrong mapping puts bonded
# atoms nanometres apart.
MAX_MATCH_SPREAD = 0.001
MAX_BOND_P99 = 0.25

# Frames read from the .dtr at a time.
CHUNK = 100


class Args(NamedTuple):
    cms: str
    trajectory: Optional[str]
    out_pdb: str
    out_xtc: Optional[str]
    out_bonds: Optional[str]


class Frames(NamedTuple):
    frames: int
    first_time: float
    dt: float
    match_frame: int
    match_spread: float
    bond_p99: Optional[float]


class Result(NamedTuple):
    atoms: int
    particles: int
    pseudo: int
    pseudo_on: dict[str, int]
    renamed: int
    named: int
    regrouped: int
    bonds: int
    forcefields: list[str]
    frames: int
    first_time: float
    dt: float
    match_frame: int
    match_spread: float
    matched: bool
    bond_p99: Optional[float]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Convert a Desmond run to a PDB and an XTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--cms", required=True, metavar="CMS", help="The run's -out.cms"
    )
    parser.add_argument(
        "-f",
        "--trajectory",
        metavar="DIR",
        help="The run's _trj directory (holding clickme.dtr); without it only the PDB is written",
    )
    parser.add_argument(
        "-p", "--out-pdb", required=True, metavar="PDB", help="Structure to write"
    )
    parser.add_argument(
        "-x",
        "--out-xtc",
        metavar="XTC",
        help="Trajectory to write; required with --trajectory",
    )
    parser.add_argument(
        "-b",
        "--out-bonds",
        metavar="JSON",
        help="Write the .cms bonds between the PDB's atoms here, for generate_topology.py --bonds",
    )
    a = parser.parse_args()
    if a.trajectory and not a.out_xtc:
        # Exit 1, not argparse's 2: a bad value is this run's refusal, and the
        # contract keeps 2 for an environment that lacks something.
        sys.stderr.write("--trajectory needs --out-xtc" + "\n")
        sys.exit(1)
    return Args(a.cms, a.trajectory, a.out_pdb, a.out_xtc, a.out_bonds)


# --------------------------------------------------
def element(anum: int) -> str:
    return ELEMENTS[anum] if 0 < anum < len(ELEMENTS) else "X"


# --------------------------------------------------
def by_residue(atoms, bonds, keep):
    """Put each residue's atoms together, residues in the order they first
    appear and atoms in their order within each, renumbering the bonds and
    keep to match. Returns the new lists and how many residues were brought
    together; a file already in residue order comes back as it is."""

    # residue_index counts residues as they first appear, so a stable sort
    # on it keeps the residues in file order.
    order = sorted(range(len(atoms)), key=lambda i: atoms[i].residue_index)
    # A residue split in the file reappears after another has begun.
    seen, split = set(), set()
    for i, a in enumerate(atoms):
        if (
            i > 0
            and a.residue_index != atoms[i - 1].residue_index
            and a.residue_index in seen
        ):
            split.add(a.residue_index)
        seen.add(a.residue_index)
    if not split:
        return atoms, bonds, keep, 0
    pos = [0] * len(atoms)
    for n, o in enumerate(order):
        pos[o] = n
    new_atoms = [atoms[o] for o in order]
    new_keep = [keep[o] for o in order]
    new_bonds = []
    for a, b in bonds:
        x, y = pos[a], pos[b]
        new_bonds.append((x, y) if x <= y else (y, x))
    return new_atoms, new_bonds, new_keep, len(split)


def unspace(atoms: list[Particle]) -> int:
    """Replace whitespace inside atom, residue and segment names with
    underscores; how many atoms changed."""

    def fix(s: str) -> tuple[str, bool]:
        if " " not in s and "\t" not in s:
            return s, False
        return s.replace(" ", "_").replace("\t", "_"), True

    n = 0
    for a in atoms:
        a.name, c1 = fix(a.name)
        a.resname, c2 = fix(a.resname)
        a.segid, c3 = fix(a.segid)
        if c1 or c2 or c3:
            n += 1
    return n


def name_blanks(atoms: list[Particle]) -> int:
    """Give an atom the .cms names nothing the symbol of its element and a
    number, C1, C2, N1, counting each element in its residue and skipping a
    name the residue already has; how many atoms were named."""

    used: dict[int, set[str]] = {}
    for a in atoms:
        if a.name:
            used.setdefault(a.residue_index, set()).add(a.name)
    nxt: dict[tuple[int, int], int] = {}
    n = 0
    for a in atoms:
        if a.name:
            continue
        el = element(a.atomic_number)
        names = used.setdefault(a.residue_index, set())
        key = (a.residue_index, a.atomic_number)
        while True:
            nxt[key] = nxt.get(key, 0) + 1
            name = f"{el}{nxt[key]}"
            if name not in names:
                a.name = name
                names.add(name)
                break
        n += 1
    return n


# --------------------------------------------------
def cell_parameters(v: maestro.Box):
    """Cell vectors to lengths and angles."""

    def norm(x):
        return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])

    def angle(x, y):
        d = x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
        return math.degrees(math.acos(d / (norm(x) * norm(y))))

    return (
        norm(v[0]),
        norm(v[1]),
        norm(v[2]),
        angle(v[1], v[2]),
        angle(v[0], v[2]),
        angle(v[0], v[1]),
    )


def wrap_resid(resid: int) -> int:
    """A residue number as the four-column field holds it: wrapped past 9999
    as mdtraj wraps it, with a negative number (Maestro's caps) kept
    negative rather than folded to a large positive one."""
    return resid if -999 <= resid <= 9999 else int(math.fmod(resid, 10000))


def write_pdb(fh, atoms: list[Particle], box: maestro.Box) -> None:
    """Write atoms as a PDB, the system's structure and topology for
    MDAnalysis: names, residues, chains and segments as msys reads them, and
    the .cms coordinates. Serial and residue numbers wrap past the columns'
    width, as mdtraj writes them; the chain is the first character of the
    chain name and the segment its first four."""

    if box != maestro.ZERO_BOX:
        a, b, c, alpha, beta, gamma = cell_parameters(box)
        fh.write(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
        )
    for i, p in enumerate(atoms):
        name = p.name if len(p.name) >= 4 else " " + p.name
        chain = p.chain[:1] if p.chain else " "
        ins = p.insertion[:1] if p.insertion else " "
        fh.write(
            "ATOM  %5d %-4.4s %-4.4s%1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4.4s%2s\n"
            % ((i + 1) % 100000, name, p.resname, chain, wrap_resid(p.resid), ins,
               p.x, p.y, p.z, 1.0, 0.0, p.segid, element(p.atomic_number))
        )  # fmt: skip
    fh.write("END\n")


# --------------------------------------------------
def copy_frames(dtr: str, out: str, particles: int, keep, ref, bonds) -> Frames:
    """Copy the .dtr to an .xtc keeping the atoms `keep`, and measure the
    mapping against the .cms coordinates `ref` (angstroms) and the bonds."""

    import numpy as np
    import mdtraj as md

    keep = np.asarray(keep, dtype=np.int64)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3) / 10.0
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)

    frames, times = 0, []
    best_spread, best_frame = float("inf"), -1
    bond_p99 = None
    with md.formats.DTRTrajectoryFile(dtr) as f, md.formats.XTCTrajectoryFile(
        out, "w"
    ) as x:
        while True:
            xyz, t, lengths, angles = f.read(n_frames=CHUNK)
            if len(xyz) == 0:
                break
            if xyz.shape[1] != particles:
                raise ValueError(
                    f"the trajectory has {xyz.shape[1]} particles per frame and the .cms "
                    f"{particles}: they are not the same system"
                )
            atoms = xyz[:, keep, :] / 10.0  # angstroms to nm
            lengths = lengths / 10.0
            box = md.utils.lengths_and_angles_to_box_vectors(
                lengths[:, 0],
                lengths[:, 1],
                lengths[:, 2],
                angles[:, 0],
                angles[:, 1],
                angles[:, 2],
            )
            # mdtraj numbers the frames of each write call from 0 unless
            # told otherwise; the step has to run on across chunks.
            x.write(
                atoms.astype(np.float32),
                time=t.astype(np.float32),
                step=np.arange(frames, frames + len(atoms), dtype=np.int32),
                box=np.stack(box, axis=1).astype(np.float32),
            )
            for i in range(len(atoms)):
                # The .cms is one frame of the run. The frame it is differs
                # from it by one shift, the box origin.
                v = ref - atoms[i]
                v -= lengths[i] * np.round(v / lengths[i])
                spread = float(v.std(axis=0).max())
                if spread < best_spread:
                    best_spread, best_frame = spread, frames + i
            if bond_p99 is None and len(bonds):
                # Bond lengths in the first frame, minimum image: a wrong
                # mapping pulls bonded atoms far apart.
                d = atoms[0][bonds[:, 0]] - atoms[0][bonds[:, 1]]
                d -= lengths[0] * np.round(d / lengths[0])
                bond_p99 = float(np.percentile(np.linalg.norm(d, axis=1), 99))
            frames += len(atoms)
            times.extend(float(v) for v in t)

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    return Frames(
        frames, times[0] if times else 0.0, dt, best_frame, best_spread, bond_p99
    )


# --------------------------------------------------
def convert(cms: str, trajectory: Optional[str], out_pdb: str, out_xtc: Optional[str],
            out_bonds: Optional[str] = None, regroup: bool = True) -> Result:  # fmt: skip
    """Convert the run and report on it."""

    system = maestro.load(cms)
    atoms, bonds, keep = system.atoms()
    if not atoms:
        raise ValueError(f"{os.path.basename(cms)} holds no atoms")
    regrouped = 0
    if regroup:
        atoms, bonds, keep, regrouped = by_residue(atoms, bonds, keep)
    renamed = unspace(atoms)
    named = name_blanks(atoms)
    pseudo_on: dict[str, int] = {}
    for p in system.particles:
        if p.pseudo:
            pseudo_on[p.resname] = pseudo_on.get(p.resname, 0) + 1

    # Every output is staged beside its path and published together once the
    # run has converted: a mapping that fails leaves a PDB or bonds file
    # already at its path as it was.
    frames = Frames(0, 0.0, 0.0, -1, float("inf"), None)
    with contextlib.ExitStack() as stack:
        with open(stack.enter_context(outputs.staged(out_pdb)), "w") as fh:
            write_pdb(fh, atoms, system.box)
        if out_bonds:
            with open(stack.enter_context(outputs.staged(out_bonds)), "w") as fh:
                json.dump([list(b) for b in bonds], fh)
        if trajectory:
            dtr = trajectory
            if os.path.isdir(dtr):
                dtr = os.path.join(dtr, "clickme.dtr")
            ref = [(p.x, p.y, p.z) for p in atoms]
            xtc = stack.enter_context(outputs.staged(out_xtc))
            frames = copy_frames(dtr, xtc, len(system.particles), keep, ref, bonds)
            # Either check establishes the mapping: the .cms matching a frame,
            # or, for a .cms from outside the saved frames, sane bond lengths.
            if frames.match_spread > MAX_MATCH_SPREAD and (
                frames.bond_p99 is None or frames.bond_p99 > MAX_BOND_P99
            ):
                raise ValueError(
                    "atoms do not map: the .cms matches no frame (best spread "
                    f"{frames.match_spread:.3f} nm) and bonds run to {frames.bond_p99 or 0:.2f} nm"
                )

    return Result(
        atoms=len(atoms),
        particles=len(system.particles),
        pseudo=len(system.particles) - len(atoms),
        pseudo_on=pseudo_on,
        renamed=renamed,
        named=named,
        regrouped=regrouped,
        bonds=len(bonds),
        forcefields=system.forcefields,
        frames=frames.frames,
        first_time=frames.first_time,
        dt=frames.dt,
        match_frame=frames.match_frame,
        match_spread=frames.match_spread if frames.frames else 0.0,
        matched=frames.frames > 0 and frames.match_spread <= MAX_MATCH_SPREAD,
        bond_p99=frames.bond_p99,
    )


# --------------------------------------------------
def main() -> None:
    args = get_args()
    if args.trajectory:
        try:
            import mdtraj  # noqa: F401
        except ImportError:
            sys.stderr.write(f"mdtraj is not installed in {sys.executable}\n")
            sys.exit(2)

    try:
        result = convert(
            args.cms, args.trajectory, args.out_pdb, args.out_xtc, args.out_bonds
        )
    except Exception as e:  # convert() staged its outputs: none is left behind
        reason = " ".join(str(e).split())
        sys.stderr.write(f"converting {os.path.basename(args.cms)}: {reason}\n")
        sys.exit(1)
    print(json.dumps(result._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
