"""Tests for --replace-mismatched-name (2026-10-27).

The 958 DDD bundles whose TOML ligand name disagrees with the PDBbind table:
902 of them have coordinates that confirm the TABLE's molecule
(preflight_ligands.py pass/flag), so the submitter's name is about another
molecule -- his exporter named a different HET group from the same entry.
Ken, 2026-10-27: publish the table's name there. fix_ligand_smiles.py used to
fill only a BLANK name; this option also replaces a disagreeing one, keeping
the submitter's in a comment, and only alongside --allow-name-mismatch.
"""

import sys
from pathlib import Path

import pytest
import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bulk_process_local as B  # noqa: E402
import fix_ligand_smiles as F  # noqa: E402

TOML = """\
pdb_id = "2fl6"

[[contributors]]
name = "Maodong Li"

[[ligands]]
name = "ADENOSINE-5'-DIPHOSPHATE"

[[solutes]]
name = "WAT"
"""

FILLED = "name added by test; the submitter left it blank"
REPLACED = "name replaced by test; the submitter's name was"


def ligand(text: str) -> dict:
    return toml.loads(text)["ligands"][0]


def test_default_leaves_a_disagreeing_name_alone():
    new, _ = F.insert_smiles(TOML, "CCO", "src", name="ETHANOL",
                             name_source=FILLED)
    assert ligand(new) == {"name": "ADENOSINE-5'-DIPHOSPHATE",
                           "smiles": "CCO"}
    assert REPLACED not in new


def test_replace_name_writes_the_tables_name_and_keeps_the_old_one():
    new, _ = F.insert_smiles(TOML, "CCO", "src", name="ETHANOL",
                             name_source=FILLED, replace_name=True,
                             replaced_source=REPLACED)
    assert ligand(new) == {"name": "ETHANOL", "smiles": "CCO"}
    assert f"# {REPLACED}: ADENOSINE-5'-DIPHOSPHATE" in new
    doc = toml.loads(new)
    # [[contributors]] and [[solutes]] also have `name`; neither moves.
    assert doc["contributors"][0]["name"] == "Maodong Li"
    assert doc["solutes"][0]["name"] == "WAT"


def test_replace_name_ignores_a_name_that_differs_only_in_case():
    new, _ = F.insert_smiles(TOML, "CCO", "src",
                             name="adenosine-5'-diphosphate",
                             name_source=FILLED, replace_name=True,
                             replaced_source=REPLACED)
    assert ligand(new)["name"] == "ADENOSINE-5'-DIPHOSPHATE"
    assert REPLACED not in new


def test_a_blank_name_is_still_filled_with_its_own_note():
    text = TOML.replace("name = \"ADENOSINE-5'-DIPHOSPHATE\"", 'name = ""')
    for replace_name in (False, True):
        new, _ = F.insert_smiles(text, "CCO", "src", name="ETHANOL",
                                 name_source=FILLED,
                                 replace_name=replace_name,
                                 replaced_source=REPLACED)
        assert ligand(new)["name"] == "ETHANOL"
        assert FILLED in new and REPLACED not in new


def test_an_old_name_with_quotes_and_newlines_stays_inside_its_comment():
    text = TOML.replace("name = \"ADENOSINE-5'-DIPHOSPHATE\"",
                        'name = "3\'-3\\"-x\\nnext"')
    new, _ = F.insert_smiles(text, "CCO", "src", name="ETHANOL",
                             name_source=FILLED, replace_name=True,
                             replaced_source=REPLACED)
    assert ligand(new) == {"name": "ETHANOL", "smiles": "CCO"}
    note = [l for l in new.splitlines() if REPLACED in l]
    assert note == [f"# {REPLACED}: 3'-3\"-x next"]


def test_cli_refuses_replace_without_allow(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "fix_ligand_smiles.py", "x", "--replace-mismatched-name"])
    with pytest.raises(SystemExit) as exc:
        F.get_args()
    assert exc.value.code == 2


def test_cli_end_to_end_replaces_and_keeps_a_backup(tmp_path, monkeypatch):
    bundle = tmp_path / "2fl6"
    bundle.mkdir()
    (bundle / "mdrepo-metadata.toml").write_text(TOML)
    table = tmp_path / "table.tsv"
    table.write_text("pdb_id\tligand_name\tcanonical_smiles\n"
                     "2fl6\tETHANOL\tCCO\n")
    monkeypatch.setattr(sys, "argv", [
        "fix_ligand_smiles.py", str(bundle), "--table", str(table),
        "--allow-name-mismatch", "--replace-mismatched-name"])
    F.main()
    text = (bundle / "mdrepo-metadata.toml").read_text()
    assert ligand(text) == {"name": "ETHANOL", "smiles": "CCO"}
    assert "the submitter's name, which names a different molecule, was: " \
        "ADENOSINE-5'-DIPHOSPHATE" in text
    assert (bundle / "mdrepo-metadata.toml.orig").read_text() == TOML


def test_cli_end_to_end_without_replace_keeps_the_old_name(
        tmp_path, monkeypatch):
    bundle = tmp_path / "2fl6"
    bundle.mkdir()
    (bundle / "mdrepo-metadata.toml").write_text(TOML)
    table = tmp_path / "table.tsv"
    table.write_text("pdb_id\tligand_name\tcanonical_smiles\n"
                     "2fl6\tETHANOL\tCCO\n")
    monkeypatch.setattr(sys, "argv", [
        "fix_ligand_smiles.py", str(bundle), "--table", str(table),
        "--allow-name-mismatch"])
    F.main()
    text = (bundle / "mdrepo-metadata.toml").read_text()
    assert ligand(text) == {"name": "ADENOSINE-5'-DIPHOSPHATE",
                            "smiles": "CCO"}


@pytest.mark.parametrize("replace", [False, True])
def test_bulk_passes_the_flag_only_when_asked(tmp_path, monkeypatch, replace):
    seen = []

    class Proc:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(B.subprocess, "run",
                        lambda cmd, **k: (seen.append(cmd), Proc())[1])
    args = B.Args(*([None] * len(B.Args._fields)))._replace(
        fix_smiles="fix_ligand_smiles.py", smiles_table="table.tsv",
        allow_name_mismatch=True, replace_mismatched_name=replace)
    B.fix_smiles("/some/bundle", args)
    assert ("--replace-mismatched-name" in seen[0]) is replace


def test_bulk_cli_refuses_replace_without_allow(tmp_path, monkeypatch):
    survey = tmp_path / "s.tsv"
    survey.write_text("bundle\tclassification\tdetail\n")
    monkeypatch.setattr(sys, "argv", [
        "bulk_process_local.py", "-s", str(survey),
        "--replace-mismatched-name"])
    with pytest.raises(SystemExit) as exc:
        B.get_args()
    assert exc.value.code == 2
