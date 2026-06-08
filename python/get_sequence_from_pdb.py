#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-02-04
Purpose: Get sequence from PDB
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
from typing import NamedTuple


class Args(NamedTuple):
    """Command-line arguments"""

    pdb_path: str
    out_file: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Get sequence from PDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("pdb_path", metavar="FILE", help="Path to PDB file")

    parser.add_argument(
        "-o",
        "--out-file",
        help="Output filename",
        metavar="FILE",
        default="sequence.fa",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pdb_path):
        parser.error(f"Invalid PDB path '{args.pdb_path}'")

    return Args(pdb_path=args.pdb_path, out_file=args.out_file)


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", args.pdb_path)
    seqs = defaultdict(list)  # chain_id -> list of single-letter codes

    for model in structure:
        # Use the first model only for sequence (typical for PDB)
        for chain in model:
            chain_id = chain.id if chain.id != " " else "_"

            for res in chain:
                # Only standard residues (residue id starting with blank ' ')
                # Skip HETATM, water, ligands, etc.
                if res.id[0] != " ":
                    continue

                # skip lipids and membrane components
                resname = res.resname.strip().upper()

                if aa := VARIANT.get(resname, CANON.get(resname, "")):
                    seqs[chain_id].append(aa)

    if not seqs:
        sys.exit(f"Failed to find sequence in structure '{args.pdb_path}'")

    out_fh = open(args.out_file, "wt")
    for seq_num, residues in enumerate(seqs.values(), start=1):
        # Skip all Xs
        if set(list(residues)) != set("X"):
            print(">{}\n{}".format(seq_num, "".join(residues)), file=out_fh)

    print(f"Done, see '{args.out_file}'")


# --------------------------------------------------
# Base mapping for canonical residues (3-letter -> 1-letter)
CANON = {
    key.upper(): val
    for key, val in IUPACData.protein_letters_3to1_extended.items()
}

# Biopython's extended map includes SEC ('SEC': 'U') and PYL ('PYL': 'O') in some versions,
# but we'll explicitly handle a broad set below.
# Hand-tuned mapping for common variants and PTMs
# Feel free to extend this list as you encounter new codes.
VARIANT = {
    # Protonation/tautomer states
    "ASH": "D",
    "GLH": "E",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",  # histidine tautomers (generic to H)
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",  # AMBER-style histidines
    # Cysteine family
    "CYX": "C",  # disulfide-linked
    "CYM": "C",  # deprotonated thiolate
    "CYN": "C",  # protonation/alt naming
    "CME": "C",  # S-methylcysteine -> C
    "CSO": "C",
    "CSS": "C",
    "CSX": "C",  # oxidized/bridged variants -> C
    "OCS": "C",  # cysteinyl-serine mixed disulfide -> C
    # Phosphorylated residues
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
    # Methionine & analogs
    "MSE": "M",
    "FME": "M",  # selenomethionine, N-formyl-Met -> M
    # Lysine carboxylation/methylation
    "KCX": "K",  # carboxylated lysine
    "MLZ": "K",
    "MLY": "K",
    "M3L": "K",
    "KPI": "K",  # methylated lysines -> K
    # Glutamate variants
    "CGU": "E",  # gamma-carboxyglutamate -> E (sequence-level)
    "PCA": "E",  # pyroglutamate (from Glu/Gln cyclization) -> E
    # Histidine variants
    "HIC": "H",
    "MHO": "M",  # 4-methylhistidine; methionine sulfoxide -> M
    # Ser/Thr variants
    "OMT": "T",
    "SME": "M",  # O-methylthreonine; S-methylmethionine -> M
    # Tyrosine variants
    "TYI": "Y",
    "TYS": "Y",  # iodinated; sulfated (PTR already handled as phospho)
    # Termini caps (omit from sequence)
    "ACE": "",
    "NME": "",
    "NH2": "",
    "FOR": "",  # acetyl, N-methyl, amidation, formyl
    # Special chromophores/crosslinks mapped to backbone residue
    "LYR": "K",  # Lys-retinal protonated Schiff base -> treat as Lys at sequence level
    # Rare canonical names sometimes seen
    "SEC": "U",  # selenocysteine
    "PYL": "O",  # pyrrolysine
    # (generic covalent adduct; keep as unknown at sequence level)
    "COV": "X",
    # (histidine-like/protonated histidine surrogate)
    "HRG": "H",
    # (α-aminoisobutyric acid; approximated as alanine)
    "AIB": "A",
}

LIPID_AND_MEMBRANE = {
    "POP",
    "POPC",
    "POPE",
    "POPG",
    "CHL",
    "CHOL",
}


# --------------------------------------------------
if __name__ == "__main__":
    main()
