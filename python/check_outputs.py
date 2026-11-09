#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-11-09
Purpose: Check that a simulation's processed files agree with each other

WHY THIS EXISTS. The checks we had looked at each file alone -- atom counts,
coordinate ranges, a .gro box line being present -- and at storage (size,
checksum). Nothing read what was inside a trajectory beyond that. So a box-less
run whose XTC frames carried an uninitialized box ([inf, -inf, ...], or
-7083.2 nm, or -6.9e14 nm) passed everything, for years: cpptraj writes
whatever memory holds when told nothing about the box, and an old-vs-new byte
comparison could not tell that from noise. Found 2026-11-09 on MDR00105222
(simulation-processing #44).

Three checks, each a function returning Findings:

  box    Every frame's box is finite and either zero in every frame or a real
         cell in every frame (three vectors 0.1-10,000 nm enclosing a volume);
         it matches the pipeline's own box decision when given; full.pdb's
         CRYST1 agrees with full.xtc's first frame.
         Lengths and angles are compared, not the matrix: the RMS fit rotates
         the box with the coordinates.

  time   Frame times are finite and never go backwards (fail); two frames at
         the same time -- a restart joined with its boundary frame twice --
         warn, as does a file whose frames all share one time (released
         sampled.xtc from 2025 imports: every frame at 0 ps); uneven spacing
         is reported only, since some simulations skip
         time. full.xtc and minimal.xtc have the same frames. Given the source
         trajectory, its stored times must match full.xtc's: the conversion
         can stamp an even grid over a source that skipped time.

  atoms  Every file of a set (full: psf, pdb, mmcif, xtc; minimal: the same
         plus sampled.xtc) has the same atoms in the same order as the .psf,
         the topology, with the same names and residue names, split into the
         same residues (numbering may differ: .psf restarts it per segment,
         cpptraj's .pdb does not; reported as info); elements are real
         element symbols; .pdb and .mmcif coordinates equal the trajectory's
         first frame within the formats' precision. .gro files are not
         checked (to be deprecated; full.gro is the submitter's structure).

Read-only. A missing file is skipped, so the same checks run on a fresh
`process` output (no .psf/.mmcif yet), on an MDR-108 build (orig/ + new/, no
full.xtc), or on a release fetched from IRODS.

usage: check_outputs.py DIR [DIR ...] [--box | --no-box]
                        [--source-traj FILE] [--json OUT]
Exit 1 if any check fails.
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# XTC stores nm to 3 decimals by default (0.01 A steps), PDB A to 3, GRO nm to
# 3. The worst honest disagreement is about 0.01 A; 0.02 A leaves room.
COORD_TOL_A = 0.02
BOX_LEN_RTOL = 2e-3
BOX_ANGLE_TOL_DEG = 0.2
ZERO = 1e-6
SOURCE_TIME_EXTS = {".xtc", ".trr", ".nc", ".ncdf", ".netcdf"}

ELEMENTS = set("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu
Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs
Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl
Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh
Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D
""".split())
ELEMENTS_UPPER = {e.upper() for e in ELEMENTS}


class Finding(NamedTuple):
    """One result: level is fail, warn or info"""
    check: str
    level: str
    file: str
    message: str


class Atom(NamedTuple):
    name: str
    resname: str
    resid: int
    element: str


class Structure(NamedTuple):
    """A structure or topology file's atoms; xyz in angstroms, or None"""
    path: str
    atoms: List[Atom]
    xyz: Optional[np.ndarray]
    resid_wrap: Optional[int]
    resname_width: Optional[int]
    name_width: Optional[int]


class Frames(NamedTuple):
    """What the checks need from a trajectory, read once"""
    path: str
    natoms: int
    times: np.ndarray
    boxes: np.ndarray            # (n, 3, 3) nm
    first_xyz: Optional[np.ndarray]   # angstroms


# --------------------------------------------------
def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Check that a simulation's processed files agree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dirs", metavar="DIR", nargs="+",
                        help="Folder(s) holding the files; the first with a "
                        "given file name wins")
    box = parser.add_mutually_exclusive_group()
    box.add_argument("--box", dest="box_present", action="store_true",
                     default=None, help="The pipeline found a periodic box")
    box.add_argument("--no-box", dest="box_present", action="store_false",
                     help="The pipeline found no periodic box")
    parser.add_argument("--source-traj", metavar="FILE",
                        help="The source trajectory full.xtc came from")
    parser.add_argument("--json", metavar="OUT", help="Write findings as JSON")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Print fails and warns only")
    return parser.parse_args(argv)


# --------------------------------------------------
def find_files(dirs: Sequence[str]) -> Dict[str, str]:
    """{file name: path} for the names the checks know, first dir wins"""

    names = [f"{s}.{e}" for s in ("full", "minimal")
             for e in ("xtc", "pdb", "psf", "mmcif")] + ["sampled.xtc"]
    found: Dict[str, str] = {}
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if n not in found and os.path.isfile(p):
                found[n] = p
    return found


# --- readers ---------------------------------------------------------------
def read_xtc(path: str) -> Frames:
    """Times, boxes and the first frame of an XTC"""

    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    times, boxes, first, natoms = [], [], None, 0
    with XTCFile(path) as fh:
        for fr in fh:
            if first is None:
                first = np.array(fr.x, dtype=float) * 10.0
                natoms = len(fr.x)
            times.append(float(fr.time))
            boxes.append(np.array(fr.box, dtype=float))
    return Frames(path, natoms, np.array(times),
                  np.array(boxes).reshape(-1, 3, 3), first)


def read_pdb(path: str) -> Structure:
    """ATOM/HETATM records, the first model only"""

    atoms, xyz = [], []
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec in ("ATOM  ", "HETATM"):
                atoms.append(Atom(line[12:16].strip(), line[16:21].strip(),
                                  _int(line[22:26]), line[76:78].strip()))
                xyz.append((float(line[30:38]), float(line[38:46]),
                            float(line[46:54])))
            elif rec.startswith("ENDMDL"):
                break
    return Structure(path, atoms, np.array(xyz, dtype=float).reshape(-1, 3),
                     10000, 4, 4)


def read_pdb_cryst1(path: str) -> Optional[Tuple[float, ...]]:
    """CRYST1 a b c alpha beta gamma, or None if there is none"""

    with open(path) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                return tuple(float(line[i:j]) for i, j in
                             ((6, 15), (15, 24), (24, 33), (33, 40),
                              (40, 47), (47, 54)))
            if line.startswith(("ATOM  ", "HETATM")):
                return None
    return None


def read_psf(path: str) -> Structure:
    """The !NATOM section: id segid resid resname name type charge mass"""

    atoms = []
    with open(path) as fh:
        for line in fh:
            if "!NATOM" in line:
                n = int(line.split()[0])
                for _ in range(n):
                    f = next(fh).split()
                    atoms.append(Atom(f[4], f[3], _int(f[2]), ""))
                break
    return Structure(path, atoms, None, None, None, None)


def read_mmcif(path: str) -> Structure:
    """_atom_site, first model"""

    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    d = MMCIF2Dict(path)

    def col(*keys):
        for k in keys:
            if f"_atom_site.{k}" in d:
                return d[f"_atom_site.{k}"]
        return None

    names = col("auth_atom_id", "label_atom_id")
    comps = col("auth_comp_id", "label_comp_id")
    seqs = col("auth_seq_id", "label_seq_id")
    elems = col("type_symbol") or [""] * len(names)
    models = col("pdbx_PDB_model_num")
    xs, ys, zs = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    atoms, xyz = [], []
    first_model = models[0] if models else None
    for i in range(len(names)):
        if models and models[i] != first_model:
            break
        atoms.append(Atom(names[i].strip('"'), comps[i], _int(seqs[i]),
                          "" if elems[i] in "?." else elems[i]))
        xyz.append((float(xs[i]), float(ys[i]), float(zs[i])))
    return Structure(path, atoms, np.array(xyz, dtype=float).reshape(-1, 3),
                     None, None, None)


def read_source_times(path: str) -> Optional[np.ndarray]:
    """Stored frame times of a source that has them; None for formats that
    only carry a header spacing (DCD) or that cannot be read here"""

    ext = os.path.splitext(path)[1].lower()
    if ext not in SOURCE_TIME_EXTS:
        return None
    if ext in (".xtc", ".trr"):
        from MDAnalysis.lib.formats.libmdaxdr import TRRFile, XTCFile

        cls = XTCFile if ext == ".xtc" else TRRFile
        with cls(path) as fh:
            return np.array([float(fr.time) for fr in fh])
    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=False) as nc:
        if "time" not in nc.variables:
            return None
        return np.array(nc.variables["time"][:], dtype=float)


def _int(s) -> int:
    try:
        return int(str(s).strip())
    except ValueError:
        return 0


# --- box geometry ------------------------------------------------------------
def box_kind(box: np.ndarray) -> str:
    """zero, cell, bad or non-finite, for one 3x3 box in nm"""

    if not np.all(np.isfinite(box)):
        return "non-finite"
    if np.all(np.abs(box) < ZERO):
        return "zero"
    lens = np.linalg.norm(box, axis=1)
    if lens.min() < 0.1 or lens.max() > 1e4:
        return "bad"
    if abs(np.linalg.det(box)) < 0.1 * float(np.prod(lens)):
        return "bad"
    return "cell"


def lengths_angles(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vector lengths (nm) and alpha, beta, gamma (degrees)"""

    a, b, c = box
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))

    def ang(u, v, lu, lv):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / (lu * lv)))))

    return (np.array([la, lb, lc]),
            np.array([ang(b, c, lb, lc), ang(a, c, la, lc), ang(a, b, la, lb)]))


def box_from_cryst1(c: Sequence[float]) -> np.ndarray:
    """3x3 box (nm) from CRYST1 a b c (A) alpha beta gamma (degrees)"""

    a, b, cc = (x / 10.0 for x in c[:3])
    al, be, ga = (math.radians(x) for x in c[3:6])
    v1 = np.array([a, 0.0, 0.0])
    v2 = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = cc * math.cos(be)
    cy = cc * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    v3 = np.array([cx, cy, math.sqrt(max(cc * cc - cx * cx - cy * cy, 0.0))])
    return np.array([v1, v2, v3])


def same_cell(b1: np.ndarray, b2: np.ndarray) -> bool:
    l1, a1 = lengths_angles(b1)
    l2, a2 = lengths_angles(b2)
    return bool(np.all(np.abs(l1 - l2) <= BOX_LEN_RTOL * np.maximum(l1, l2))
                and np.all(np.abs(a1 - a2) <= BOX_ANGLE_TOL_DEG))


# --- check 1: box --------------------------------------------------------------
def check_box(trajs: Dict[str, Frames], files: Dict[str, str],
              box_present: Optional[bool]) -> List[Finding]:
    out = []
    kinds_by_file = {}
    for name, fr in trajs.items():
        kinds = [box_kind(b) for b in fr.boxes]
        counts = {k: kinds.count(k) for k in set(kinds)}
        kinds_by_file[name] = counts
        for k in ("non-finite", "bad"):
            if k in counts:
                i = kinds.index(k)
                out.append(Finding("box", "fail", name,
                                   f"{counts[k]} of {len(kinds)} frames have a "
                                   f"{k} box (first at frame {i}: "
                                   f"{_fmt(fr.boxes[i])})"))
        if "zero" in counts and "cell" in counts:
            out.append(Finding("box", "fail", name,
                               f"{counts['zero']} frames have no box and "
                               f"{counts['cell']} have one"))
        if box_present is not None and set(counts) <= {"zero", "cell"}:
            if box_present and "zero" in counts and name != "sampled.xtc":
                out.append(Finding("box", "fail", name,
                                   "the run has a box but these frames have none"))
            if not box_present and "cell" in counts:
                out.append(Finding("box", "fail", name,
                                   "the run has no box but these frames have one"))

    fx = trajs.get("full.xtc")
    if fx is not None and len(fx.boxes) and box_kind(fx.boxes[0]) == "cell":
        first = fx.boxes[0]
        if "full.pdb" in files:
            c = read_pdb_cryst1(files["full.pdb"])
            if c is None:
                out.append(Finding("box", "warn", "full.pdb",
                                   "no CRYST1 although full.xtc has a box"))
            elif not same_cell(box_from_cryst1(c), first):
                out.append(Finding("box", "fail", "full.pdb",
                                   f"CRYST1 {c} differs from full.xtc frame 0"))
    elif fx is not None and "full.pdb" in files:
        c = read_pdb_cryst1(files["full.pdb"])
        if c is not None and c[:3] != (1.0, 1.0, 1.0) and any(c[:3]):
            if box_kind(box_from_cryst1(c)) != "zero":
                out.append(Finding("box", "fail", "full.pdb",
                                   f"CRYST1 {c} although full.xtc has no box"))
    if not out:
        out.append(Finding("box", "info", "",
                           "; ".join(f"{n}: {c}" for n, c in kinds_by_file.items())
                           or "no trajectory to check"))
    return out


# --- check 2: time ---------------------------------------------------------------
def check_time(trajs: Dict[str, Frames],
               source_times: Optional[np.ndarray],
               source_name: str = "") -> List[Finding]:
    out = []
    for name, fr in trajs.items():
        t = fr.times
        if not np.all(np.isfinite(t)):
            out.append(Finding("time", "fail", name,
                               f"{int(np.sum(~np.isfinite(t)))} frame times are "
                               "not finite"))
            continue
        if len(t) > 1 and np.all(t == t[0]):
            out.append(Finding("time", "warn", name,
                               f"no time axis: all {len(t)} frames are at "
                               f"{t[0]:g} ps"))
            continue
        d = np.diff(t)
        back = np.flatnonzero(d < 0)
        same = np.flatnonzero(d == 0)
        if len(back):
            i = int(back[0])
            out.append(Finding("time", "fail", name,
                               f"time goes backwards {len(back)} times (first "
                               f"at frame {i + 1}: {t[i]} -> {t[i + 1]} ps)"))
        if len(same):
            i = int(same[0])
            out.append(Finding("time", "warn", name,
                               f"{len(same)} frames repeat the previous frame's "
                               f"time (first at frame {i + 1}, {t[i]} ps)"))
        pos = d[d > 0]
        if len(pos):
            step = float(np.median(pos))
            gaps = int(np.sum(np.abs(pos - step) > 1e-3 * max(step, 1e-9)))
            if gaps:
                out.append(Finding("time", "info", name,
                                   f"uneven spacing: {gaps} of {len(pos)} steps "
                                   f"differ from the median {step:g} ps "
                                   f"(largest {float(pos.max()):g} ps)"))

    full, mini = trajs.get("full.xtc"), trajs.get("minimal.xtc")
    if full is not None and mini is not None:
        if len(full.times) != len(mini.times):
            out.append(Finding("time", "fail", "minimal.xtc",
                               f"{len(mini.times)} frames; full.xtc has "
                               f"{len(full.times)}"))
        elif not np.allclose(full.times, mini.times, rtol=0, atol=1e-3):
            out.append(Finding("time", "fail", "minimal.xtc",
                               "frame times differ from full.xtc's"))

    if full is not None and source_times is not None:
        if len(source_times) != len(full.times):
            out.append(Finding("time", "warn", "full.xtc",
                               f"{len(full.times)} frames; the source "
                               f"{source_name} has {len(source_times)}"))
        else:
            src = source_times - source_times[0]
            got = full.times - full.times[0]
            tol = 1e-3 * max(float(np.median(np.diff(src))) if len(src) > 1 else 1.0, 1e-3)
            bad = np.flatnonzero(np.abs(src - got) > tol)
            if len(bad):
                i = int(bad[0])
                out.append(Finding("time", "fail", "full.xtc",
                                   f"{len(bad)} frame times differ from the "
                                   f"source's (first at frame {i}: source "
                                   f"{src[i]:g} ps from its start, full.xtc "
                                   f"{got[i]:g} ps)"))
    if not out:
        out.append(Finding("time", "info", "", "frame times in order"))
    return out


# --- check 3: atoms ---------------------------------------------------------------
def same_name(a: str, b: str, width: Optional[int]) -> bool:
    """Equal, or equal as far as the narrower format can hold"""
    if a == b:
        return True
    if width and (len(a) > width or len(b) > width):
        return a[:width] == b[:width]
    return False


def residue_starts(atoms: List[Atom]) -> List[int]:
    """1-based index of each residue's first atom: a new residue starts
    where the residue number or name changes"""
    starts, prev = [], None
    for i, a in enumerate(atoms):
        key = (a.resid, a.resname)
        if key != prev:
            starts.append(i + 1)
            prev = key
    return starts


def check_atom_set(label: str, structs: List[Structure],
                   trajs: List[Frames]) -> List[Finding]:
    out = []
    counts = {os.path.basename(s.path): len(s.atoms) for s in structs}
    counts.update({os.path.basename(f.path): f.natoms for f in trajs})
    if len(set(counts.values())) > 1:
        out.append(Finding("atoms", "fail", label,
                           "atom counts differ: " +
                           ", ".join(f"{k} {v}" for k, v in counts.items())))
        return out

    ref = structs[0] if structs else None
    for s in structs:
        name = os.path.basename(s.path)
        bad_el = sorted({a.element for a in s.atoms
                         if a.element and a.element.upper() not in ELEMENTS_UPPER})
        if bad_el:
            out.append(Finding("atoms", "fail", name,
                               f"not element symbols: {', '.join(bad_el)}"))
        if ref is None or s is ref:
            continue
        # Narrowest format in the whole set: the .mmcif takes its names from
        # the .pdb, so a long .psf name reaches it already cut to 4.
        nw = min([x.name_width for x in structs if x.name_width] + [8])
        rw = min([x.resname_width for x in structs if x.resname_width] + [8])
        wrap = min([w for w in (s.resid_wrap, ref.resid_wrap) if w] or [0])
        renumbered = None
        for i, (a, b) in enumerate(zip(ref.atoms, s.atoms)):
            what = None
            if not same_name(a.name, b.name, nw):
                what = f"atom name {a.name!r} vs {b.name!r}"
            elif not same_name(a.resname, b.resname, rw):
                what = f"residue name {a.resname!r} vs {b.resname!r}"
            elif a.element and b.element and a.element.upper() != b.element.upper():
                what = f"element {a.element!r} vs {b.element!r}"
            elif (a.resid % wrap if wrap else a.resid) != \
                    (b.resid % wrap if wrap else b.resid):
                if renumbered is None:
                    renumbered = (i, a.resid, b.resid)
            if what:
                out.append(Finding("atoms", "fail", name,
                                   f"atom {i + 1} differs from "
                                   f"{os.path.basename(ref.path)}: {what}"))
                break
        else:
            rb, sb = residue_starts(ref.atoms), residue_starts(s.atoms)
            if rb != sb:
                i = next(k for k, (x, y) in enumerate(zip(rb + [None], sb + [None]))
                         if x != y)
                out.append(Finding("atoms", "fail", name,
                                   f"residues split differently from "
                                   f"{os.path.basename(ref.path)} (residue "
                                   f"{i + 1}: starts at atom "
                                   f"{(rb + [None])[i]} vs {(sb + [None])[i]})"))
            elif renumbered is not None:
                i, x, y = renumbered
                out.append(Finding("atoms", "info", name,
                                   f"residues numbered differently from "
                                   f"{os.path.basename(ref.path)} (from atom "
                                   f"{i + 1}: residue {x} vs {y})"))

    first = next((f for f in trajs if f.first_xyz is not None), None)
    if first is not None:
        for s in structs:
            if s.xyz is None or len(s.xyz) != len(first.first_xyz):
                continue
            dev = np.abs(s.xyz - first.first_xyz).max(axis=1)
            worst = int(np.argmax(dev))
            if dev[worst] > COORD_TOL_A:
                out.append(Finding("atoms", "fail", os.path.basename(s.path),
                                   f"coordinates differ from "
                                   f"{os.path.basename(first.path)} frame 0 "
                                   f"by up to {dev[worst]:.3f} A (atom "
                                   f"{worst + 1}); {int(np.sum(dev > COORD_TOL_A))}"
                                   f" atoms beyond {COORD_TOL_A} A"))
    return out


def check_atoms(files: Dict[str, str], trajs: Dict[str, Frames]) -> List[Finding]:
    out = []
    # .psf first: it is the topology the others are checked against.
    readers = {"psf": read_psf, "pdb": read_pdb, "mmcif": read_mmcif}
    for label in ("full", "minimal"):
        structs = []
        for ext, reader in readers.items():
            p = files.get(f"{label}.{ext}")
            if p:
                try:
                    structs.append(reader(p))
                except Exception as e:  # unreadable is itself a finding
                    out.append(Finding("atoms", "fail", f"{label}.{ext}",
                                       f"cannot be read: {e}"))
        tr = [trajs[n] for n in (f"{label}.xtc",) + (("sampled.xtc",) if label == "minimal" else ())
              if n in trajs]
        if len(structs) + len(tr) >= 2:
            out.extend(check_atom_set(label, structs, tr))
    if not out:
        out.append(Finding("atoms", "info", "", "files agree"))
    return out


def _fmt(box: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.4g}" for v in np.asarray(box).ravel()) + "]"


# --------------------------------------------------
def run_checks(dirs: Sequence[str], box_present: Optional[bool] = None,
               source_traj: Optional[str] = None) -> List[Finding]:
    """All three checks over the files found in dirs"""

    files = find_files(dirs)
    trajs, out = {}, []
    for n in ("full.xtc", "minimal.xtc", "sampled.xtc"):
        if n in files:
            try:
                trajs[n] = read_xtc(files[n])
            except Exception as e:
                out.append(Finding("read", "fail", n, f"cannot be read: {e}"))
    out.extend(check_box(trajs, files, box_present))
    src = None
    if source_traj:
        try:
            src = read_source_times(source_traj)
        except Exception as e:
            out.append(Finding("time", "warn", source_traj,
                               f"source times not read: {e}"))
    out.extend(check_time(trajs, src, os.path.basename(source_traj or "")))
    out.extend(check_atoms(files, trajs))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = get_args(argv)
    findings = run_checks(args.dirs, args.box_present, args.source_traj)
    for f in findings:
        if args.quiet and f.level == "info":
            continue
        print(f"{f.level.upper():4} {f.check:5} {f.file or '-'}: {f.message}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([f._asdict() for f in findings], fh, indent=2)
    fails = sum(f.level == "fail" for f in findings)
    warns = sum(f.level == "warn" for f in findings)
    print(f"{fails} fail, {warns} warn")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
