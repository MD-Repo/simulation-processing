import subprocess
import sys


def run(smiles: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "canonicalize_toml_smiles.py", smiles],
        capture_output=True,
        text=True,
    )


def test_simple_smiles():
    result = run("CCO")
    assert result.returncode == 0
    assert result.stdout.strip() == "CCO"


def test_canonical_form_reordered():
    # RDKit canonicalizes atom ordering
    result = run("OCC")
    assert result.returncode == 0
    assert result.stdout.strip() == "CCO"


def test_ring_canonicalization():
    # Different ring traversal orders should produce the same canonical SMILES
    result1 = run("C1CCCCC1")
    result2 = run("C1CCCC(C1)")
    assert result1.returncode == 0
    assert result1.stdout.strip() == result2.stdout.strip()


def test_backslash_stereo_smiles():
    smiles = r"CC1(C)[NH+]=C2N(C1)C(=CS2)CS/C(=[NH+]\C1CCCCC1)/NC1CCCCC1"
    result = run(smiles)
    assert result.returncode == 0
    assert result.stdout.strip()  # non-empty canonical SMILES


def test_invalid_smiles_exits_nonzero():
    result = run("not-a-smiles")
    assert result.returncode != 0
    assert "Error" in result.stderr
