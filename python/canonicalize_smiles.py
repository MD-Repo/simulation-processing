#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-06-22
Purpose: Canonicalize SMILES strings

A pure filter: takes N SMILES as command-line arguments and prints N canonical
SMILES, one per line, in the same order. It never reads or writes any file --
the caller (mdr-process) holds the metadata in memory and substitutes the
canonical forms itself, so the on-disk TOML is left exactly as the submitter
sent it. Exits non-zero naming the first SMILES OpenBabel cannot parse.
"""

import sys
from typing import Optional

from openbabel import pybel

# OpenBabel accepts non-standard protonation such as [N+H3] (which strict RDKit
# parsing rejects as invalid) and canonicalizes it to [NH3+]. Quiet its logger.
pybel.ob.obErrorLog.SetOutputLevel(0)


# --------------------------------------------------
def canonicalize(smiles: str) -> Optional[str]:
    """Canonical SMILES via OpenBabel, or None if the SMILES is invalid."""
    try:
        mol = pybel.readstring("smi", smiles)
    except Exception:
        return None
    canonical = mol.write("can").strip()
    return canonical.split()[0] if canonical else None


# --------------------------------------------------
def main() -> None:
    canonical = []
    for smiles in sys.argv[1:]:
        result = canonicalize(smiles)
        if result is None:
            sys.exit(f"Invalid SMILES '{smiles}'")
        canonical.append(result)

    if canonical:
        print("\n".join(canonical))


# --------------------------------------------------
if __name__ == "__main__":
    main()
