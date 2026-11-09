#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-11-07
Purpose: Fingerprint a run by its first and last frames, so that the same run
         released twice is found

A release can hold one run at more than one selection or stride: the whole
system and the protein alone, every frame and every tenth. Deposited
separately they would be two simulations of one run. The fingerprint is the
distances of the CA atoms (P for nucleic acids) from the first of them,
rounded to 0.01 A, at the first frame of the first trajectory and the last
frame of the last. Two runs whose fingerprints agree on the atoms they share
are the same run.

Moved here from mdrepo-agent's internal/assembly/verify.go (mdrepo-agent#98):
everything that reads or writes a structure, topology or trajectory is a
command here.

Output: one JSON object on stdout:

  {"n": N, "first": SHA1, "last": SHA1}     n is 0, with no digests, when
                                            the run has under 3 CA or P atoms

Exit 0 when read; 1 with the reason as the last line of stderr when the files
cannot be read together; 2 when MDAnalysis is not installed.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import NamedTuple, Optional

from universe import open_universe


class Args(NamedTuple):
    topology: str
    first: str
    last: str


class Report(NamedTuple):
    n: int
    first: Optional[str]
    last: Optional[str]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Fingerprint a run by its first and last frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t", "--topology", required=True, metavar="FILE", help="Topology"
    )
    parser.add_argument(
        "--first", required=True, metavar="FILE", help="First trajectory"
    )
    parser.add_argument(
        "--last", metavar="FILE", help="Last trajectory (default: the first)"
    )
    a = parser.parse_args()
    return Args(a.topology, a.first, a.last or a.first)


# --------------------------------------------------
def pick(u):
    """The CA atoms, or the P atoms, when there are at least three."""

    for sel in ("protein and name CA", "name P"):
        ag = u.select_atoms(sel)
        if len(ag) >= 3:
            return ag
    return None


# --------------------------------------------------
def digest(ag) -> str:
    """The SHA1 of the atoms' distances from the first, to 0.01 A."""

    import numpy as np

    p = ag.positions.astype(np.float64)
    d = np.round(np.linalg.norm(p - p[0], axis=1), 2)
    return hashlib.sha1(d.tobytes()).hexdigest()


# --------------------------------------------------
def signature(topology: str, first: str, last: str) -> Report:
    """The run's fingerprint at its first and last frames."""

    u = open_universe(topology, first)
    ag = pick(u)
    if ag is None:
        return Report(0, None, None)
    u.trajectory[0]
    f = digest(ag)
    if last != first:
        u = open_universe(topology, last)
        ag = pick(u)
    u.trajectory[-1]
    return Report(len(ag), f, digest(ag))


# --------------------------------------------------
def main() -> None:
    args = get_args()
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        sys.stderr.write(f"MDAnalysis is not installed in {sys.executable}\n")
        sys.exit(2)

    try:
        report = signature(args.topology, args.first, args.last)
    except Exception as e:
        reason = " ".join(str(e).split())
        sys.stderr.write(f"reading {os.path.basename(args.topology)}: {reason}\n")
        sys.exit(1)
    print(json.dumps(report._asdict()))


# --------------------------------------------------
if __name__ == "__main__":
    main()
