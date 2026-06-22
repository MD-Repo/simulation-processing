#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-06-22
Purpose: Canonicalize a SMILES string
"""

import argparse
import sys
from typing import NamedTuple

from rdkit import Chem
from rdkit.Chem import MolToSmiles


class Args(NamedTuple):
    smiles: str


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Canonicalize a SMILES string",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("smiles", metavar="SMILES", help="Input SMILES string")

    args = parser.parse_args()

    return Args(smiles=args.smiles)


# --------------------------------------------------
def main() -> None:
    args = get_args()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"Error: invalid SMILES: {args.smiles!r}", file=sys.stderr)
        sys.exit(1)

    print(MolToSmiles(mol))


# --------------------------------------------------
if __name__ == "__main__":
    main()
