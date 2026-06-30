#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-06-22
Purpose: Canonicalize a SMILES string
"""

import argparse
import sys
import toml
from typing import NamedTuple, Optional

from openbabel import pybel

# OpenBabel accepts non-standard protonation such as [N+H3] (which strict RDKit
# parsing rejects as invalid) and canonicalizes it to [NH3+]. Quiet its logger.
pybel.ob.obErrorLog.SetOutputLevel(0)


class Args(NamedTuple):
    file: str


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Canonicalize SMILES strings in TOML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("file", metavar="FILE", help="Input TOML string")

    args = parser.parse_args()

    return Args(file=args.file)


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
    args = get_args()

    data = toml.load(args.file)
    num_changed = 0
    errors = []

    for ligand in data.get("ligands", []):
        if orig_smiles := ligand.get("smiles"):
            if new_smiles := canonicalize(orig_smiles):
                if new_smiles != orig_smiles:
                    ligand["smiles"] = new_smiles
                    num_changed += 1
            else:
                errors.append(f"Invalid SMILES '{orig_smiles}'")
        else:
            errors.append("Missing SMILES")

    if errors:
        sys.exit("Errors: {}".format(", ".join(errors)))

    print(f"Changed {num_changed} SMILES")
    if num_changed > 0:
        with open(args.file, "wt") as fh:
            toml.dump(data, fh)


# --------------------------------------------------
if __name__ == "__main__":
    main()
