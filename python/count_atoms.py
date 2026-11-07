#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-11-07
Purpose: Count the atoms a structure or topology file holds, and how many of
         them are solvent

The agent measures every released structure and topology before it decides
what to deposit: whether a structure holds the trajectories' atoms, whether
a topology is of the whole system or of the saved part. The file is read
with MDAnalysis, the reader MDRepo's own processing uses, and a
topology-only file -- a .tpr, .psf, .prmtop, .top -- loads as a Universe
without coordinates. Solvent is water and ions, by residue name.

A .top is AMBER's when it carries %VERSION or %FLAG, and GROMACS's
otherwise, which MDAnalysis reads as its ITP format. A CHARMM .cor is read
with --format CRD, MDAnalysis's name for it. A CHARMM-GUI Chamber .parm7 is
read through ParmEd (universe.py).

Moved here from mdrepo-agent's internal/stage/released.go (mdrepo-agent#98):
everything that reads or writes a structure, topology or trajectory is a
command here.

Output: one JSON object on stdout:

  {"atoms": N, "solvent": N}

Exit 0 when read; 1 with the reason as the last line of stderr when the file
cannot be read; 2 when MDAnalysis is not installed.
"""

import argparse
import json
import os
import sys
from typing import NamedTuple, Optional

from universe import open_universe

# Residue names of water and ions across the common force fields.
SOLVENT = {
    "SOL", "HOH", "WAT", "H2O", "TIP3", "TIP4", "TIP5", "TIP3P", "TIP4P", "T3P",
    "T4P", "SPC", "SPCE", "OPC", "OPC3", "NA", "CL", "K", "MG", "CA", "ZN", "LI",
    "CS", "RB", "F", "BR", "I", "NA+", "CL-", "K+", "MG2+", "CA2+", "ZN2+", "SOD",
    "CLA", "POT", "CAL", "MG2", "IOD", "CES",
}  # fmt: skip


class Args(NamedTuple):
    file: str
    format: Optional[str]


class Report(NamedTuple):
    atoms: int
    solvent: int


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Count the atoms in a structure or topology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--file", required=True, metavar="FILE", help="Structure or topology"
    )
    parser.add_argument(
        "--format", metavar="FMT", help="MDAnalysis format, e.g. CRD for a .cor"
    )
    a = parser.parse_args()
    return Args(a.file, a.format or None)


# --------------------------------------------------
def is_gromacs_top(path: str) -> bool:
    """Whether a .top is GROMACS's rather than AMBER's."""

    if not path.lower().endswith(".top"):
        return False
    with open(path, errors="replace") as fh:
        head = fh.read(4096)
    return "%VERSION" not in head and "%FLAG" not in head


# --------------------------------------------------
def count(path: str, fmt: Optional[str] = None) -> Report:
    """The atoms in the file and how many are water or ions."""

    if fmt is None and is_gromacs_top(path):
        fmt = "ITP"
    if fmt == "ITP":
        u = open_universe(path, topology_format="ITP")
    elif fmt:
        u = open_universe(path, topology_format=fmt, format=fmt)
    else:
        u = open_universe(path)
    solvent = 0
    try:
        solvent = sum(1 for r in u.atoms.resnames if str(r).upper() in SOLVENT)
    except Exception:  # a topology with no residue names counts none
        pass
    return Report(int(u.atoms.n_atoms), int(solvent))


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        sys.stderr.write(f"MDAnalysis is not installed in {sys.executable}\n")
        sys.exit(2)

    try:
        report = count(args.file, args.format)
    except Exception as e:
        reason = " ".join(str(e).split())
        sys.stderr.write(f"reading {os.path.basename(args.file)}: {reason}\n")
        sys.exit(1)
    print(json.dumps(report._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
