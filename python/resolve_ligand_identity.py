#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-09-09
Purpose: Fill in whichever ligand notation the submitter did not supply

`mdr-meta` accepts a ligand declared by `smiles`, by `inchi`, or by both --
`ligand_declares_identity` requires only that one be present. Import needs
more than that: `md_ligand.smiles` is NOT NULL, so before this existed an
InChI-only ligand passed `mdr-meta check` and then failed at import, and
`inchikey` and `identity_software` were columns nothing ever wrote.

A pure filter, like `canonicalize_smiles.py`: ligands in on stdin, results out
on stdout, in the same order. It never reads or writes a file, so the TOML the
submitter sent is left exactly as they wrote it and neither declared value is
ever rewritten in it.

WHAT THE CALLER MUST NOT DO WITH THE RESULT. `md_ligand.declared_identity`
records which notation the submitter actually supplied, and it is only
recoverable BEFORE this runs -- afterwards every ligand carries both and the
distinction is gone forever. Read it off the submitter's own document, not off
these results. The Rust caller has a test pinning exactly that.

WHY NOT A PEP-723 INLINE DEPENDENCY BLOCK. `mdr-process` launches this with
`uv run`, and an inline block makes uv build an isolated environment from that
header and ignore the project's. `pyproject.toml` already declares both
`rdkit` and `openbabel-wheel`, so with no block uv uses the project and both
are present. A block here would have to re-list them, and getting that wrong
fails only in production -- which nearly shipped once already with `mol_id.py`.

FAILURE POLICY: REFUSE ONLY WHAT CANNOT BE STORED. RDKit checks valences where
OpenBabel does not, and it rejects molecules that are perfectly real as
submitted -- of the 121,240 `md_ligand` rows backfilled on 2026-09-09 it
refused 32, about half of them valence cases like a neutral four-bonded
nitrogen. Making that fatal would turn a missing column into a failed
submission, so a ligand whose InChI cannot be computed is stored with a null
`inchi` and `inchikey` rather than blocked. The two things that DO stop a
ligand are a SMILES OpenBabel cannot parse (unchanged from
`canonicalize_smiles.py`) and an InChI-only ligand whose InChI RDKit will not
read, because that one cannot produce the NOT NULL column at all.
"""

import argparse
import json
import sys
from typing import Dict, List, NamedTuple, Optional

from rdkit import Chem, RDLogger

import compare_smiles
import ligand_agreement

# RDKit narrates its refusals on stderr, which would interleave with the
# messages this script writes for the submitter.
RDLogger.DisableLog("rdApp.*")


class Args(NamedTuple):
    """Command-line arguments"""

    infile: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Fill in whichever ligand notation was not supplied",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--infile",
        help='JSON list of {"name", "smiles", "inchi"}, "-" for stdin',
        metavar="FILE",
        default="-",
    )

    args = parser.parse_args()

    return Args(infile=args.infile)


# --------------------------------------------------
def inchikey_of(inchi: str) -> Optional[str]:
    """The InChIKey for an InChI string, or None if the InChI is not readable.

    `Chem.InchiToInchiKey` is a hash with no validation behind it -- asked for
    the key of `InChI=1S/NOPE` it returns `DIAJDFQKHMMVGV-UHFFFAOYSA-N` rather
    than failing. So the string is put through `MolFromInchi` first, which does
    refuse it. Without that gate a malformed InChI acquires a well-formed key
    and every consumer downstream reads it as real.
    """

    if Chem.MolFromInchi(inchi) is None:
        return None

    return Chem.InchiToInchiKey(inchi) or None


# --------------------------------------------------
def from_smiles(smiles: str, name: str) -> Dict[str, Optional[str]]:
    """Canonicalize the SMILES and derive the InChI and key from it"""

    try:
        mol = compare_smiles.parse_smiles(smiles)
    except (ValueError, RuntimeError):
        sys.exit(f'Ligand "{name}": invalid SMILES {smiles!r}')

    canonical = compare_smiles.to_canonical(mol)

    # inchi_of already prefers RDKit 1.07.3 and falls back to OpenBabel 1.04,
    # and reports which one answered -- that report IS identity_software, and
    # is why this does not call to_inchi() and guess afterwards.
    inchi, software = compare_smiles.inchi_of(mol)

    # ONLY AN RDKit InChI IS STORED, and this is the one place the rule differs
    # from `compare_smiles`. That module falls back to OpenBabel 1.04 on
    # purpose, and is right to: a COMPARISON puts both sides through the same
    # library, so the version cancels out, and refusing the 29.5% of inferred
    # structures RDKit will not accept would be an outage rather than an
    # upgrade. Its own docstring says a comparison must not mix the two.
    #
    # A STORED value has no other side to cancel against. It is matched later
    # against whatever an outside database computed, and InChI is canonical
    # only within a version -- the key spends a character encoding which. A
    # column holding some 1.04 and some 1.07.3 strings is one an exact-match
    # consumer cannot use, which is the whole purpose of storing it.
    #
    # Measured 2026-09-09 against the production backfill: with this rule the
    # resolver reproduces it exactly, including leaving the same rows null.
    # Allowing the fallback instead recovered 5 rows in 100 and made the
    # column bi-toolkit, so a reprocess would have disagreed with the backfill
    # on precisely the molecules already known to be difficult.
    if software != compare_smiles.RDKIT:
        inchi = ""

    # The three land together or not at all, and the key is the gate. An InChI
    # RDKit cannot read back is one we cannot key, and storing an unkeyable
    # InChI is worse than storing none: it looks like an identifier and cannot
    # be matched against anything.
    key = inchikey_of(inchi) if inchi else None
    if not key:
        # See FAILURE POLICY: storable without an InChI, so store it.
        return {
            "smiles": canonical,
            "inchi": None,
            "inchikey": None,
            "identity_software": None,
        }

    return {
        "smiles": canonical,
        "inchi": inchi,
        "inchikey": key,
        "identity_software": software,
    }


# --------------------------------------------------
def from_inchi(inchi: str, name: str) -> Dict[str, Optional[str]]:
    """Derive the SMILES from the InChI, keeping the InChI as submitted

    The derived SMILES is NOT the string the submitter would have written.
    Obeticholic acid's round-trips to a different atom ordering and different
    ring-closure numbers while its InChIKey matches exactly, which is the
    correct outcome: SMILES has no canonical form, InChI does. What makes this
    honest rather than a silent substitution is `declared_identity`, which
    records that this row's SMILES came from us and not from them.
    """

    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        sys.exit(
            f'Ligand "{name}": RDKit cannot read the declared inchi, and it '
            f"is the only structure given, so no SMILES can be derived from "
            f"it -- {inchi}"
        )

    smiles = Chem.MolToSmiles(mol)
    if not smiles:
        sys.exit(f'Ligand "{name}": no SMILES could be derived from the inchi')

    return {
        "smiles": smiles,
        "inchi": inchi,
        "inchikey": Chem.MolToInchiKey(mol) or None,
        "identity_software": compare_smiles.RDKIT,
    }


# --------------------------------------------------
def from_both(smiles: str, inchi: str, name: str) -> Dict[str, Optional[str]]:
    """Keep both as submitted, having checked they describe one molecule

    Nothing is derived here, so `identity_software` stays null: it records the
    toolkit behind a value WE produced, and both of these are the submitter's.

    The agreement bar is deliberately stricter than the one used to compare a
    declared ligand against the structure -- there the two sides are different
    sources describing one molecule and protonation differences are forgiven;
    here both are the submitter's own, inside one file, so a hydrogen-layer
    difference is a person contradicting themselves. `ligand_agreement` owns
    that rule and the reasoning behind it.
    """

    try:
        mol = compare_smiles.parse_smiles(smiles)
    except (ValueError, RuntimeError):
        sys.exit(f'Ligand "{name}": invalid SMILES {smiles!r}')

    canonical = compare_smiles.to_canonical(mol)

    verdict, why = ligand_agreement.agree(smiles, inchi)
    if verdict == ligand_agreement.STOP:
        sys.exit(
            f'Ligand "{name}": the declared smiles and inchi describe '
            f"different molecules -- {why}"
        )

    return {
        "smiles": canonical,
        "inchi": inchi,
        "inchikey": inchikey_of(inchi),
        "identity_software": None,
    }


# --------------------------------------------------
def resolve(ligand: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Fill in whichever notation is missing for one ligand"""

    name = ligand.get("name") or "<unnamed>"
    smiles = (ligand.get("smiles") or "").strip() or None
    inchi = (ligand.get("inchi") or "").strip() or None

    if smiles and inchi:
        return from_both(smiles, inchi, name)
    if smiles:
        return from_smiles(smiles, name)
    if inchi:
        return from_inchi(inchi, name)

    # `ligand_declares_identity` in libmdrepo rejects this before a bundle can
    # reach here. Named rather than assumed, so a change there surfaces as a
    # message instead of a KeyError further down.
    sys.exit(f'Ligand "{name}": declares neither smiles nor inchi')


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    text = sys.stdin.read() if args.infile == "-" else open(args.infile).read()

    try:
        ligands: List[Dict[str, Optional[str]]] = json.loads(text)
    except json.JSONDecodeError as err:
        sys.exit(f"Input is not JSON: {err}")

    if not isinstance(ligands, list):
        sys.exit("Input must be a JSON list of ligands")

    json.dump([resolve(ligand) for ligand in ligands], sys.stdout)
    sys.stdout.write("\n")


# --------------------------------------------------
if __name__ == "__main__":
    main()
