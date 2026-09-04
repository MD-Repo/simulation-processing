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
from typing import List, Tuple

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


def split_fragments(smi: str) -> List[str]:
    """The dot-separated components of a SMILES, in input order."""

    return [f for f in smi.split(".") if f]


def largest_fragment(smi: str) -> Tuple[str, List[str]]:
    """(largest fragment, everything else) ranked by heavy-atom count.

    WHY THIS EXISTS. The "inferred" side of every comparison in this
    pipeline comes from OpenBabel perceiving bonds from simulated 3D
    coordinates (mol_id.py's ConnectTheDots/PerceiveBondOrders). That
    routinely detaches a piece of an intact ligand -- a phosphonate, an
    exocyclic amine, or a bare H2 -- and emits it as a separate
    dot-component. The declared side, being a lookup from a reference
    table, is always a single component.

    So a multi-fragment inferred SMILES is usually one molecule that was
    mis-perceived, not a genuinely different one, and comparing the whole
    multi-component string against a single-component declaration reports
    a connectivity mismatch for a molecule that is actually right. Three
    bundles blocked in the 2026-09 prod wave this way (1yq7 .OP(=O)=O,
    3in3 .N, 4dhm .[HH]); 4dhm's main component was character-identical
    to the declaration.

    Ties keep the first fragment, so the result is deterministic. An
    unparseable fragment counts as zero heavy atoms rather than raising --
    the caller is already in a diagnostic path and a hard failure here
    would lose the comparison it can still make.
    """

    fragments = split_fragments(smi)
    if len(fragments) < 2:
        return smi, []

    def heavy(frag: str) -> int:
        try:
            return parse_smiles(frag).NumHvyAtoms()
        except ValueError:
            return 0

    ranked = sorted(fragments, key=heavy, reverse=True)
    return ranked[0], ranked[1:]


def connectivity_of(smi: str) -> str:
    """The InChI /c layer for a SMILES, or "" if InChI is unavailable."""

    try:
        return inchi_layers(to_inchi(parse_smiles(smi))).get("c", "")
    except (ValueError, RuntimeError):
        return ""


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

    # An EMPTY /c layer means InChI generation failed, not that the two
    # molecules differ. The strict predicate above cannot tell those apart
    # (conn1 != "" makes a missing layer look like a mismatch), so report
    # availability separately and let callers avoid blocking on it.
    inchi_available = bool(conn1) and bool(conn2)

    # Level 2b: same connectivity after discarding detached fragments. See
    # largest_fragment() for why the inferred side is so often split. This
    # is ADDITIVE -- same_connectivity above keeps its exact meaning,
    # because mdr-process's resolve_ligands() keys off it.
    formula1 = mol1.GetFormula().rstrip("+-")
    formula2 = mol2.GetFormula().rstrip("+-")

    main1, dropped1 = largest_fragment(can1)
    main2, dropped2 = largest_fragment(can2)
    if same_connectivity:
        same_connectivity_largest_fragment = True
    elif not inchi_available:
        same_connectivity_largest_fragment = False
    else:
        main_conn1 = conn1 if not dropped1 else connectivity_of(main1)
        main_conn2 = conn2 if not dropped2 else connectivity_of(main2)
        same_connectivity_largest_fragment = (
            main_conn1 == main_conn2 and main_conn1 != ""
        )

    # Every atom present, but distributed over more than one component:
    # OpenBabel broke a bond that is really there rather than inventing an
    # extra molecule. 1yq7 is the case that forced this -- one of a
    # bisphosphonate's two P-C bonds was cut, so the LARGEST fragment is
    # genuinely missing a phosphonate and largest-fragment matching cannot
    # rescue it, yet the declared molecule is unarguably the one in the
    # structure file: C7H11NO7P2 on both sides.
    #
    # This is the line BLOCK is supposed to draw. ligand_check's bar is
    # "the declared ligand is not the molecule in the structure file"; if
    # the atom inventory is identical and only the bond graph is split,
    # that statement is false and BLOCK is the wrong verdict. Where the
    # inventory really does differ -- 3in3, C20H17N5O vs C20H21N5O, four
    # hydrogens adrift because an imine was perceived as a saturated ring
    # plus free ammonia -- this stays False and the BLOCK stands.
    same_gross_formula = formula1 == formula2
    multi_fragment = len(split_fragments(can1)) > 1 or len(split_fragments(can2)) > 1
    fragment_artifact = (
        multi_fragment
        and inchi_available
        and (same_connectivity_largest_fragment or same_gross_formula)
    )

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
        if not inchi_available:
            differences.append("inchi unavailable (no connectivity layer)")
        elif fragment_artifact:
            dropped = ", ".join(dropped1 + dropped2) or "none"
            kind = (
                "detached fragment"
                if same_connectivity_largest_fragment
                else "split bond graph, all atoms present"
            )
            differences.append(f"{kind} ({dropped})")
        else:
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
        "formula1": formula1,
        "formula2": formula2,
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
        # The quantity the connectivity verdict is actually computed from.
        # Callers used to report formula1/formula2 next to a connectivity
        # verdict, which prints identical strings for two isomers -- or for
        # a molecule with a bare H2 hanging off it.
        "connectivity1": conn1,
        "connectivity2": conn2,
        "inchi_available": inchi_available,
        "same_connectivity_largest_fragment": same_connectivity_largest_fragment,
        "same_gross_formula": same_gross_formula,
        "fragment_artifact": fragment_artifact,
        "num_fragments1": len(split_fragments(can1)),
        "num_fragments2": len(split_fragments(can2)),
        "largest_fragment1": main1,
        "largest_fragment2": main2,
        "discarded_fragments1": dropped1,
        "discarded_fragments2": dropped2,
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
