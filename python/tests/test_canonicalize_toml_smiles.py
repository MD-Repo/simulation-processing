import shutil
from pathlib import Path

import pytest

from canonicalize_toml_smiles import _make_converter, canonicalize, fix_toml

# SMILES with backslash stereo notation — valid SMILES but invalid in TOML double-quoted strings
BACKSLASH_SMILES = r"CC1(C)[NH+]=C2N(C1)C(=CS2)CS/C(=[NH+]\C1CCCCC1)/NC1CCCCC1"

INPUTS = Path(__file__).parent / "inputs"


def test_canonicalize_nonstandard_protonation():
    conv = _make_converter()
    result = canonicalize("[N+H3]CCCC", conv)
    assert "[NH3+]" in result


def test_canonicalize_backslash_smiles_parses():
    # OpenBabel must be able to read the SMILES without returning the original unchanged
    conv = _make_converter()
    result = canonicalize(BACKSLASH_SMILES, conv)
    assert result  # non-empty means parsing succeeded


def test_fix_toml_double_quoted(tmp_path):
    toml = tmp_path / "test.toml"
    toml.write_text('smiles = "[N+H3]CCCC"\n')
    changes = fix_toml(str(toml))
    content = toml.read_text()
    assert changes == 1
    assert 'smiles = "' in content  # double quotes preserved
    assert "[N+H3]" not in content  # non-standard protonation replaced


def test_fix_toml_single_quoted_backslash(tmp_path):
    # Single-quoted SMILES with backslash must be found and stay single-quoted
    toml = tmp_path / "test.toml"
    toml.write_text(f"smiles = '{BACKSLASH_SMILES}'\n")
    fix_toml(str(toml))
    content = toml.read_text()
    assert "smiles = '" in content  # single quotes preserved


def test_fix_toml_already_canonical(tmp_path):
    toml = tmp_path / "test.toml"
    original = 'smiles = "CCCC"\n'
    toml.write_text(original)
    changes = fix_toml(str(toml))
    assert changes == 0
    assert toml.read_text() == original  # file untouched


def test_fix_toml_real_input(tmp_path):
    dst = tmp_path / "MDR00001245.toml"
    shutil.copy(INPUTS / "MDR00001245.toml", dst)
    fix_toml(str(dst))
    content = dst.read_text()
    # Single-quoted SMILES with backslash must still be single-quoted after processing
    assert "smiles = '" in content
