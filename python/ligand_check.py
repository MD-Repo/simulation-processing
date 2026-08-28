#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-28
Purpose: The DDD prod wave's ligand verdict, shared by preflight_ligands.py
         (which vets the corpus before anything is imported) and
         bulk_process_local.py (which re-checks inline during the run).

         WHY THIS IS NOT mdr-process's OWN CHECK: process.rs's
         resolve_ligands() accepts a ligand when ANY of exact_match,
         same_connectivity, same_connectivity_and_stereo or same_inchi
         holds. same_connectivity ignores charge and protonation, which is
         almost certainly how MDR-55 passed -- a neutral 3,6-dihydropyridine
         standing in for a declared +1 pyridinium. The CheckedLigand verdict
         it computes is never persisted, so we redo the comparison here
         against the same compare_smiles.py it shells out to.

         WHY THE BAR IS CONNECTIVITY AND NOT exact_match: measured over the
         883 bundles Phase B had finished on 2026-08-28, exact_match passes
         only 14.8%. The declared SMILES is not the submitter describing
         what they simulated -- it is our own fix_ligand_smiles.py fill from
         PDBbind's reference table -- while the inferred SMILES is OpenBabel
         perceiving bonds and protonation from simulated 3D coordinates.
         Those two disagree on protonation state and stereo perception even
         when nothing is wrong, so exact_match measures the sourcing
         mismatch rather than a data defect.

         A DIFFERENT CONNECTIVITY, though, means the declared ligand is not
         the molecule in the structure file. That is a real error, it is
         7.8% of the corpus, and it is what BLOCK is for. Everything short
         of it is recorded as FLAG: imported, but visible in the record for
         a contributor report.
"""

import os
from typing import List, Optional, Sequence, Tuple

import compare_smiles

BLOCK = "block"
FLAG = "flag"
PASS = "pass"

# Worst-first, so a multi-ligand bundle takes its worst verdict.
SEVERITY = {BLOCK: 2, FLAG: 1, PASS: 0}

# MDR-40: OpenBabel trusts the PDB element column, and 11gs/Pro_lig.pdb has
# atoms named CL1/CL2 recorded as element C. Only two-letter symbols are
# checked -- flagging every one-letter disagreement would be noise.
TWO_LETTER_ELEMENTS = ("CL", "BR", "FE", "ZN", "MG", "MN", "NA", "CA",
                       "CU", "SE", "NI", "CO")


# --------------------------------------------------
def worst(verdicts: Sequence[str]) -> str:
    """The most severe verdict in a sequence, PASS if empty."""

    return max(verdicts, key=lambda v: SEVERITY[v], default=PASS)


# --------------------------------------------------
def element_column_disagreements(pdb_path: str) -> List[str]:
    """Ligand atoms whose PDB element column contradicts the element implied
    by the atom name. OpenBabel trusts that column silently, so a
    disagreement means the inference's own INPUT may be wrong -- worth
    flagging regardless of what the SMILES comparison then says (MDR-40).
    """

    bad = []
    try:
        with open(pdb_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("HETATM"):
                    continue
                name = line[12:16].strip().upper()
                element = line[76:78].strip().upper()
                if not name or not element:
                    continue
                implied = next(
                    (s for s in TWO_LETTER_ELEMENTS if name.startswith(s)),
                    None,
                )
                if implied and element != implied:
                    bad.append(f"{name}:{element}!={implied}")
    except OSError as e:
        return [f"unreadable: {e}"]

    return sorted(set(bad))


# --------------------------------------------------
def best_comparison(declared: str, candidates: Sequence[str]) -> Optional[dict]:
    """Compare one declared SMILES against every inferred candidate and
    return the closest result. A bundle can infer several ligands and only
    one of them is the right partner for this declared entry, so the best
    match is the meaningful one -- not the first.
    """

    best = None
    best_rank = -1

    for candidate in candidates:
        try:
            result = compare_smiles.compare(declared, candidate)
        except (ValueError, RuntimeError):
            continue

        if result["exact_match"]:
            rank = 4
        elif result["same_inchi"]:
            rank = 3
        elif result["same_connectivity_and_stereo"]:
            rank = 2
        elif result["same_connectivity"]:
            rank = 1
        else:
            rank = 0

        if rank > best_rank:
            best, best_rank = result, rank

    return best


# --------------------------------------------------
def check(
    declared: Sequence[str],
    candidates: Sequence[str],
    structure_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Verdict for one bundle: (BLOCK|FLAG|PASS, detail).

    BLOCK when a declared ligand's connectivity does not match anything
    inferred -- the declared molecule is not the one in the structure file.
    Also BLOCK when there is nothing to compare, since then nothing was
    ever actually checked (resolve_ligands' else-branch adopts whatever it
    inferred, unverified).
    """

    if not declared:
        return BLOCK, "no declared ligand smiles; inference adopted unchecked"
    if not candidates:
        return BLOCK, "no inferred structure smiles to compare against"

    verdicts, notes = [], []

    for num, smiles in enumerate(declared):
        result = best_comparison(smiles, candidates)
        if result is None:
            verdicts.append(BLOCK)
            notes.append(f"ligand[{num}] unparseable")
            continue

        if result["exact_match"]:
            verdicts.append(PASS)
            continue

        if not result["same_connectivity"]:
            verdicts.append(BLOCK)
            notes.append(
                f"ligand[{num}] connectivity "
                f"({result['formula1']} vs {result['formula2']})"
            )
            continue

        verdicts.append(FLAG)
        notes.append(
            f"ligand[{num}] " + (", ".join(result["differences"]) or "not exact")
        )

    if structure_path and os.path.isfile(structure_path):
        bad = element_column_disagreements(structure_path)
        if bad:
            verdicts.append(FLAG)
            notes.append("element column: " + " ".join(bad[:5]))

    detail = " ".join(" ".join(notes).split())[:300]
    return worst(verdicts), detail
