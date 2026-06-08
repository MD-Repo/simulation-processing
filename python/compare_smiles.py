#!/usr/bin/env python3
"""
compare_smiles.py — Compare two SMILES at multiple levels of strictness:

  1) Exact:        canonical SMILES identical
  2) Connectivity: same heavy-atom skeleton (ignoring charge, H, stereo)
  3) Stereo:       same connectivity + same stereochemistry
  4) Full InChI:   same InChI (connectivity + stereo + charge + protonation)

The key insight: InChI separates molecular identity into layers.
The connectivity layer (/c) captures the heavy-atom bond graph
independent of protonation state, charge, and stereochemistry.
This is what lets you recognize that NMN+ and neutral NMN are
"the same molecule in different protonation states."

Usage:
    python compare_smiles.py SMILES_1 SMILES_2

Requires: pip install openbabel-wheel
"""

import argparse
import json
import sys
from openbabel import openbabel as ob

ob.obErrorLog.SetOutputLevel(ob.obError)


def parse_smiles(smi: str) -> ob.OBMol:
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    mol = ob.OBMol()
    if not conv.ReadString(mol, smi):
        raise ValueError(f"Failed to parse SMILES: {smi}")
    return mol


def to_canonical(mol: ob.OBMol) -> str:
    conv = ob.OBConversion()
    conv.SetOutFormat("can")
    return conv.WriteString(mol).strip().split("\t")[0]


def to_inchi(mol: ob.OBMol) -> str:
    conv = ob.OBConversion()
    conv.SetOutFormat("inchi")
    return conv.WriteString(mol).strip()


def inchi_layers(inchi: str) -> dict:
    """
    Parse an InChI string into its component layers.

    Key layers:
      formula  — molecular formula (includes all H)
      c        — connectivity of heavy atoms (the bond graph)
      h        — hydrogen layer (which heavy atoms carry H)
      q        — charge layer
      p        — proton balance
      t        — stereo (tetrahedral)
      b        — stereo (double bond E/Z)
      m        — stereo (mirror image)
      s        — stereo type
    """
    parts = inchi.split("/")
    layers = {"raw": inchi}
    if len(parts) > 1:
        layers["formula"] = parts[1]
    for p in parts[2:]:
        if p and p[0].isalpha():
            key = p[0]
            layers[key] = p
        elif p.startswith("+") or p.startswith("-"):
            # charge/proton layers like p+1
            layers["p"] = p
    return layers


def compare(smi1: str, smi2: str) -> dict:
    mol1 = parse_smiles(smi1)
    mol2 = parse_smiles(smi2)

    can1 = to_canonical(mol1)
    can2 = to_canonical(mol2)

    inchi1 = to_inchi(mol1)
    inchi2 = to_inchi(mol2)

    layers1 = inchi_layers(inchi1)
    layers2 = inchi_layers(inchi2)

    # Level 1: exact canonical SMILES
    exact = can1 == can2

    # Level 2: same connectivity (heavy-atom bond graph)
    conn1 = layers1.get("c", "")
    conn2 = layers2.get("c", "")
    same_connectivity = conn1 == conn2 and conn1 != ""

    # Level 3: same connectivity + stereochemistry
    stereo_layers = ["c", "t", "b", "m"]
    same_stereo = same_connectivity and all(
        layers1.get(k, "") == layers2.get(k, "") for k in stereo_layers
    )

    # Level 4: full InChI match
    same_inchi = inchi1 == inchi2

    # Diagnose what differs
    differences = []
    if not same_connectivity:
        differences.append("connectivity (different molecules)")
    else:
        if layers1.get("formula") != layers2.get("formula"):
            differences.append(
                f"formula ({layers1.get('formula')} vs {layers2.get('formula')})"
            )
        if layers1.get("h") != layers2.get("h"):
            differences.append("hydrogen attachment")
        if layers1.get("q", "") != layers2.get("q", ""):
            differences.append("charge")
        if layers1.get("p", "") != layers2.get("p", ""):
            differences.append("protonation")
        for k, label in [("t", "tetrahedral stereo"), ("b", "E/Z stereo")]:
            if layers1.get(k, "") != layers2.get(k, ""):
                differences.append(label)

    return {
        "smi1_canonical": can1,
        "smi2_canonical": can2,
        "formula1": mol1.GetFormula().rstrip("+-"),
        "formula2": mol2.GetFormula().rstrip("+-"),
        "charge1": mol1.GetTotalCharge(),
        "charge2": mol2.GetTotalCharge(),
        "exact_match": exact,
        "same_connectivity": same_connectivity,
        "same_connectivity_and_stereo": same_stereo,
        "same_inchi": same_inchi,
        "differences": differences,
        "inchi1": inchi1,
        "inchi2": inchi2,
        "connectivity_layer": conn1 if same_connectivity else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two SMILES strings at multiple levels of strictness."
    )
    parser.add_argument("smiles1", metavar="SMILES_1", help="First SMILES string")
    parser.add_argument("smiles2", metavar="SMILES_2", help="Second SMILES string")
    parser.add_argument(
        "-o", "--outfile", metavar="FILE", help="Write JSON output to FILE instead of stdout"
    )
    args = parser.parse_args()

    r = compare(args.smiles1, args.smiles2)

    output = json.dumps(r, indent=2)

    if args.outfile:
        with open(args.outfile, "w") as fh:
            fh.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
