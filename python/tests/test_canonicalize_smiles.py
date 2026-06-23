import subprocess
import sys
import toml
from pathlib import Path


def make_toml(tmp_path: Path, smiles_list: list[str], name: str = "test.toml") -> Path:
    toml_file = tmp_path / name
    toml_file.write_text(toml.dumps({"ligands": [{"smiles": s} for s in smiles_list]}))
    return toml_file


def run(toml_file: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "canonicalize_smiles.py", str(toml_file)],
        capture_output=True,
        text=True,
    )


def test_simple_smiles(tmp_path):
    toml_file = make_toml(tmp_path, ["CCO"])
    result = run(toml_file)
    assert result.returncode == 0
    assert "Changed 0 SMILES" in result.stdout


def test_canonical_form_reordered(tmp_path):
    toml_file = make_toml(tmp_path, ["OCC"])
    result = run(toml_file)
    assert result.returncode == 0
    assert "Changed 1 SMILES" in result.stdout
    data = toml.loads(toml_file.read_text())
    assert data["ligands"][0]["smiles"] == "CCO"


def test_ring_canonicalization(tmp_path):
    toml1 = make_toml(tmp_path, ["C1CCCCC1"], "ring1.toml")
    toml2 = make_toml(tmp_path, ["C1CCCC(C1)"], "ring2.toml")
    result1 = run(toml1)
    result2 = run(toml2)
    assert result1.returncode == 0
    assert result2.returncode == 0
    smiles1 = toml.loads(toml1.read_text())["ligands"][0]["smiles"]
    smiles2 = toml.loads(toml2.read_text())["ligands"][0]["smiles"]
    assert smiles1 == smiles2


def test_backslash_stereo_smiles(tmp_path):
    smiles = r"CC1(C)[NH+]=C2N(C1)C(=CS2)CS/C(=[NH+]\C1CCCCC1)/NC1CCCCC1"
    toml_file = make_toml(tmp_path, [smiles])
    result = run(toml_file)
    assert result.returncode == 0
    data = toml.loads(toml_file.read_text())
    assert data["ligands"][0]["smiles"]  # non-empty canonical SMILES


def test_invalid_smiles_exits_nonzero(tmp_path):
    toml_file = make_toml(tmp_path, ["not-a-smiles"])
    result = run(toml_file)
    assert result.returncode != 0
    assert "Errors" in result.stderr
