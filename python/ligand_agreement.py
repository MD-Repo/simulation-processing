#!/usr/bin/env python3
"""
ligand_agreement.py — when a submitter declares BOTH `smiles` and `inchi` for
one ligand, decide whether the two describe the same molecule.

WHY THIS IS NOT ligand_check.py, AND WHY ITS BAR IS STRICTER. ligand_check
compares what the submitter declared against what OpenBabel perceived from
simulated coordinates. Those are two different sources describing one molecule,
so they disagree on protonation and stereo perception even when nothing is
wrong, and the bar there is connectivity alone -- measured, a stricter rule
flags 3,599 of 6,659 bundles to find 26 real problems.

Here both sides are the SUBMITTER'S OWN, inside one file. A difference is not
two instruments disagreeing, it is one person contradicting themselves, and
there is no measurement noise to forgive. So a hydrogen-layer difference STOPS
here and merely FLAGS there. That is deliberate. Do not "simplify" this into a
call to compare_smiles.compare() and a read of `same_connectivity` -- that is
the connectivity bar wearing this function's name, and it would accept a
declared pair that disagrees about protonation.

THE VERSION CAVEAT, which is inherent and cannot be engineered away. We compute
an InChI from the declared SMILES with our own toolkit, and compare it against
an InChI string the submitter generated with theirs. The formula, /c and /h
layers are stable across InChI releases for ordinary organics; the stereo layers
are exactly where releases have changed. That asymmetry is why a stereo
difference is only decisive when BOTH sides carry stereo -- see row 3.
"""

from typing import Tuple

import compare_smiles

# Verdicts.
ACCEPT = "accept"
STOP = "stop"
UNVERIFIABLE = "unverifiable"

# The layers that describe stereochemistry. /t and /b are the assignments; /m
# and /s qualify them and never appear alone.
STEREO_LAYERS = ("t", "b", "m", "s")


def _stereo(layers: dict) -> dict:
    return {k: v for k, v in layers.items() if k in STEREO_LAYERS}


def agree(smiles: str, inchi: str) -> Tuple[str, str]:
    """Verdict for one ligand's declared pair: (ACCEPT|STOP|UNVERIFIABLE, why).

    The rows are evaluated in order and the order is load-bearing -- see the
    UNVERIFIABLE row for the case that made it so.
    """

    try:
        mol = compare_smiles.parse_smiles(smiles)
    except (ValueError, RuntimeError):
        return STOP, f"smiles does not parse: {smiles}"

    from_smiles = compare_smiles.to_inchi(mol)
    if not from_smiles:
        return UNVERIFIABLE, "no InChI could be computed from the declared smiles"

    ours = compare_smiles.inchi_layers(from_smiles)
    theirs = compare_smiles.inchi_layers(inchi)

    our_formula = ours.get("formula", "")
    their_formula = theirs.get("formula", "")
    if not their_formula:
        return UNVERIFIABLE, "the declared inchi has no formula layer"

    # A DIFFERENT FORMULA IS DECIDABLE EVEN WITHOUT A /c LAYER, and this test
    # has to come first for that reason. A monatomic species has no
    # connectivity layer by construction rather than by failure, so the
    # empty-/c row below would otherwise answer "we could not check" for
    # [Na+] declared against zinc's InChI -- where the formulas plainly differ
    # and the answer is knowable. Same trap as the one recorded against
    # ligand_check.check() for the inferred side.
    if our_formula != their_formula:
        return STOP, f"formula differs: {our_formula} from smiles, {their_formula} declared"

    our_c = ours.get("c", "")
    their_c = theirs.get("c", "")
    if not our_c or not their_c:
        # Formulas already agree, so there is nothing left that this pair can
        # settle. Not a mismatch -- an absent layer is a failed comparison, the
        # same doctrine compare_smiles and ligand_check already follow.
        return UNVERIFIABLE, "no connectivity layer on one side; formulas agree"

    if our_c != their_c:
        return STOP, "connectivity differs"

    if ours.get("h", "") != theirs.get("h", ""):
        return STOP, "hydrogen layer differs: same skeleton, different protons"

    our_stereo = _stereo(ours)
    their_stereo = _stereo(theirs)

    if our_stereo and their_stereo:
        if our_stereo != their_stereo:
            return STOP, "stereochemistry conflicts"
        return ACCEPT, "identical, including stereochemistry"

    if our_stereo or their_stereo:
        # One side is simply less specific, which is not a contradiction. Keep
        # the side that says more -- the corpus makes this the common case, not
        # the corner: every one of the 6,664 declared ligands in collection 5
        # carries no stereochemistry at all.
        richer = "inchi" if their_stereo else "smiles"
        return ACCEPT, f"same skeleton; only {richer} carries stereochemistry"

    return ACCEPT, "identical; neither side carries stereochemistry"
