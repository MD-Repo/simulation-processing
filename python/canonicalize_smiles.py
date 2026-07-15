#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-06-22
Purpose: Canonicalize a SMILES string
"""

import argparse
import re
import shutil
import sys
import toml
from pathlib import Path
from typing import NamedTuple, Optional

from openbabel import pybel

# OpenBabel accepts non-standard protonation such as [N+H3] (which strict RDKit
# parsing rejects as invalid) and canonicalizes it to [NH3+]. Quiet its logger.
pybel.ob.obErrorLog.SetOutputLevel(0)

# Matches a `smiles = "..."` line, capturing everything before/after the
# quoted value so it can be rewritten in place.
SMILES_LINE = re.compile(
    r'(?P<prefix>^\s*smiles\s*=\s*)"(?P<value>(?:[^"\\]|\\.)*)"(?P<suffix>\s*(?:#.*)?)$',
    re.MULTILINE,
)


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
def main() -> None:
    args = get_args()

    try:
        data = toml.load(args.file)
    except FileNotFoundError:
        sys.exit(f'Error: File not found: "{args.file}"')
    except toml.TomlDecodeError as err:
        data = repair_backslash_smiles(args.file, err)

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
            fh.write(literalize_backslash_smiles(toml.dumps(data)))


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
def repair_backslash_smiles(file: str, original_err: toml.TomlDecodeError) -> dict:
    """
    Retry a TOML file that failed to parse because a "smiles" value in a
    [[ligands]] table contains a literal backslash (valid SMILES stereo-bond
    syntax, but an invalid escape in a TOML basic string). Any offending
    double-quoted "smiles" values are rewritten as single-quoted (literal)
    strings, which TOML does not escape-process. The original file is backed
    up before the repaired version is written. Exits on failure.
    """
    text = Path(file).read_text()

    def fix(match: re.Match) -> str:
        value = match.group("value")
        if "\\" not in value or "'" in value:
            return match.group(0)
        return f"{match.group('prefix')}'{value}'{match.group('suffix')}"

    repaired = SMILES_LINE.sub(fix, text)

    if repaired == text:
        sys.exit(f'Error: Invalid TOML in "{file}": {original_err}')

    try:
        data = toml.loads(repaired)
    except toml.TomlDecodeError as err:
        sys.exit(
            f'Error: Invalid TOML in "{file}": {original_err} '
            f"(repair attempt also failed: {err})"
        )

    backup = f"{file}.bak"
    shutil.copy2(file, backup)
    with open(file, "wt") as fh:
        fh.write(repaired)
    print(
        f'Repaired backslash-quoted "smiles" values in "{file}" '
        f'(original backed up to "{backup}")',
        file=sys.stderr,
    )

    return data


# --------------------------------------------------
def literalize_backslash_smiles(text: str) -> str:
    """
    Rewrite double-quoted "smiles" values that contain a backslash as
    single-quoted (literal) strings, so the backslash is stored the same
    way it appears in SMILES (single, unescaped) rather than as TOML's
    doubled basic-string escape (e.g. "\\\\c1ccccc1").
    """

    def fix(match: re.Match) -> str:
        value = match.group("value")
        try:
            literal_value = value.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            return match.group(0)
        if "\\" not in literal_value or "'" in literal_value:
            return match.group(0)
        return f"{match.group('prefix')}'{literal_value}'{match.group('suffix')}"

    return SMILES_LINE.sub(fix, text)


# --------------------------------------------------
if __name__ == "__main__":
    main()
