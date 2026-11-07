#!/usr/bin/env python3
"""
Author : Nick Eddy <eddy.nicholas@gmail.com>
Date   : 2026-11-06
Purpose: Cut a GROMACS run input (.tpr) of the whole system down to the atoms
         a selection picks, for a release whose trajectories hold only part
         of the system and that ships no topology of that part

A GROMACS release often saves only part of its system in the trajectories
(compressed-x-grps = Protein, or the solvent stripped afterwards) and ships
the whole system's .tpr and nothing else as a topology. MDRepo needs a
topology of exactly the saved atoms, and a .psf generated from a PDB of
them is refused beside GROMACS files. The release's own .tpr, cut to those
atoms, is the topology the release did not ship: it keeps the force field's
parameters and the run's own settings, and it is what `gmx convert-tpr`
exists to make, given an index group of the atoms to keep.

The atoms are the ones the MDAnalysis selection picks from the .tpr, so a
structure written from the same .tpr with the same selection
(subset_structure.py) holds the same atoms in the same order. The reduced
.tpr is read back with `gmx dump` to count its atoms, and the report says
which GROMACS did the cutting, since the deposited description names it.
This is the most common topology problem met so far (mdrepo-agent#70, #91):
zenodo:3778678's buildit.tpr cut to its 191 saved atoms passed
`mdr-process validate` where the generated .psf had not.

Replaces mdrepo-agent PR #87, which did this in the agent; it moved here
with the rest of the file-producing steps (mdrepo-agent#98).

Output: the reduced .tpr at --out, and one JSON object on stdout:

  {"atoms": N, "total": N, "selected": N, "gromacs": "2024.2" | ""}

Exit 0 when written; 1 with the reason as the last line of stderr when the
.tpr cannot be read, the selection holds no atoms, or gmx fails or writes
a .tpr of another count; 2 when gmx or MDAnalysis is not available.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from typing import NamedTuple, Optional

# The index group written for the saved atoms, and the name fed to gmx.
GROUP = "saved"


class Args(NamedTuple):
    tpr: str
    selection: str
    out: str
    gmx: str


class Report(NamedTuple):
    atoms: int
    total: int
    selected: int
    gromacs: str


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Cut a whole-system .tpr to the atoms a selection picks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--tpr", required=True, metavar="TPR", help="The whole system's run input"
    )
    parser.add_argument(
        "-l",
        "--selection",
        required=True,
        metavar="SEL",
        help="MDAnalysis selection of the saved atoms",
    )
    parser.add_argument(
        "-o", "--out", required=True, metavar="TPR", help="Reduced run input to write"
    )
    parser.add_argument(
        "-g", "--gmx", default=os.environ.get("GMX", "gmx"), help="GROMACS executable"
    )
    a = parser.parse_args()
    return Args(a.tpr, a.selection, a.out, a.gmx)


# --------------------------------------------------
def select_indices(tpr: str, selection: str) -> tuple[list[int], int]:
    """The 0-based indices the selection picks from the .tpr, and the atom
    count of the whole system."""

    warnings.filterwarnings("ignore")
    import MDAnalysis as mda

    u = mda.Universe(tpr)
    ag = u.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(
            f"the selection {selection!r} holds no atoms of {os.path.basename(tpr)}"
        )
    return [int(i) for i in ag.indices], len(u.atoms)


def write_index(path: str, indices: list[int]) -> None:
    """A GROMACS .ndx of one group, 1-based, fifteen to a line."""

    with open(path, "w") as fh:
        fh.write(f"[ {GROUP} ]\n")
        for i in range(0, len(indices), 15):
            fh.write(" ".join(str(a + 1) for a in indices[i : i + 15]) + "\n")


_NATOMS = re.compile(r"^\s*natoms\s*=\s*(\d+)", re.M)
_VERSION = re.compile(r"^GROMACS version:\s*(\S+)", re.M)


def tpr_atoms(dump: str) -> Optional[int]:
    """The atom count in gmx dump's text."""
    m = _NATOMS.search(dump)
    return int(m.group(1)) if m else None


def gmx_version(gmx: str) -> str:
    try:
        out = subprocess.run(
            [gmx, "--version"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    m = _VERSION.search(out)
    return m.group(1) if m else ""


def last_line(s: str) -> str:
    lines = [l for l in s.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


# --------------------------------------------------
def reduce(tpr: str, selection: str, out: str, gmx: str = "gmx") -> Report:
    """Write the .tpr cut to the selection at out and report on it."""

    indices, total = select_indices(tpr, selection)
    with tempfile.TemporaryDirectory(prefix="reduce-tpr-") as scratch:
        ndx = os.path.join(scratch, "saved.ndx")
        write_index(ndx, indices)
        p = subprocess.run(
            [gmx, "convert-tpr", "-s", os.path.abspath(tpr), "-n", ndx, "-o", os.path.abspath(out)],
            input=GROUP + "\n", capture_output=True, text=True, cwd=scratch, timeout=600,
        )  # fmt: skip
        if p.returncode != 0:
            raise RuntimeError(
                f"gmx convert-tpr of {os.path.basename(tpr)}: {last_line(p.stderr)}"
            )
        dump = subprocess.run(
            [gmx, "dump", "-s", os.path.abspath(out)],
            capture_output=True,
            text=True,
            cwd=scratch,
            timeout=600,
        )
        if dump.returncode != 0:
            raise RuntimeError(
                f"gmx dump of the reduced {os.path.basename(tpr)}: {last_line(dump.stderr)}"
            )
    n = tpr_atoms(dump.stdout)
    if n is None:
        raise RuntimeError(f"the reduced {os.path.basename(tpr)} states no atom count")
    if n != len(indices):
        raise RuntimeError(
            f"{os.path.basename(tpr)} reduced to the selection holds {n} atoms and the selection {len(indices)}"
        )
    return Report(n, total, len(indices), gmx_version(gmx))


# --------------------------------------------------
def main() -> None:
    args = get_args()
    if shutil.which(args.gmx) is None:
        sys.stderr.write(f"GROMACS is not available: no {args.gmx} on the PATH\n")
        sys.exit(2)
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        sys.stderr.write(f"MDAnalysis is not installed in {sys.executable}\n")
        sys.exit(2)
    # Only what this run created is removed on failure, not a file that was
    # already there.
    existed = os.path.exists(args.out)
    try:
        report = reduce(args.tpr, args.selection, args.out, args.gmx)
    except Exception as e:
        if not existed and os.path.exists(args.out):
            os.remove(args.out)
        sys.stderr.write(
            f"reducing {os.path.basename(args.tpr)}: {' '.join(str(e).split())}\n"
        )
        sys.exit(1)
    print(json.dumps(report._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
