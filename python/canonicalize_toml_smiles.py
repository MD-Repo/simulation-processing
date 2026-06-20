#!/usr/bin/env python3
"""
canonicalize_toml_smiles.py — Rewrite ligand SMILES in an mdrepo metadata toml
to OpenBabel canonical form before processing.

Some upstream SMILES strings use non-standard protonation notation
(e.g. [N+H3], [N+H2], [N+H]) which is rejected by mdr-process's SMILES
validator. OpenBabel parses these correctly and emits standard canonical SMILES
(e.g. [NH3+], [NH2+], [NH+]) that pass validation. This script is called
automatically by mdr-process before validation.

Usage:
    python canonicalize_toml_smiles.py mdrepo-metadata.toml [...]
    uv run canonicalize_toml_smiles.py mdrepo-metadata.toml
"""

import re
import sys

from openbabel import openbabel as ob

ob.obErrorLog.SetOutputLevel(ob.obError)


def _make_converter() -> ob.OBConversion:
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    conv.SetOutFormat("can")
    return conv


def canonicalize(smiles: str, conv: ob.OBConversion) -> str:
    mol = ob.OBMol()
    if conv.ReadString(mol, smiles):
        return conv.WriteString(mol).strip().split("\t")[0]
    return smiles


def fix_toml(path: str) -> int:
    conv = _make_converter()
    with open(path) as f:
        content = f.read()

    changes = 0

    def replace(m: re.Match) -> str:
        nonlocal changes
        original = m.group(2)
        canonical = canonicalize(original, conv)
        if canonical != original:
            changes += 1
        return m.group(1) + canonical + m.group(3)

    updated = re.sub(
        r'^(smiles = ")([^"]+)(")',
        replace,
        content,
        flags=re.MULTILINE,
    )

    if changes:
        with open(path, "w") as f:
            f.write(updated)

    return changes


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <toml> [...]", file=sys.stderr)
        sys.exit(1)

    for path in sys.argv[1:]:
        n = fix_toml(path)
        if n:
            print(f"{path}: canonicalized {n} SMILES string{'s' if n != 1 else ''}")


if __name__ == "__main__":
    main()
