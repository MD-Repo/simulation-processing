"""Tests for the --allow-name-mismatch route through the two DDD drivers.

Item 1 of the 2026-09-15 holdout plan writes the reference table's SMILES
even where its ligand name disagrees with the TOML's, and accepts it only
where the bundle's own coordinates confirm it. The two halves must stay
coupled: preflight_ligands.py judges with the override on, and
bulk_process_local.py may use the override only on bundles that preflight
passed or flagged. A real run checks ligands only AFTER import, so without
the gate a block goes straight into prod -- which is how 1ph0, 1uvu, 2imd,
2uyq and 3d6o got there on 2026-09-22.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bulk_process_local as B  # noqa: E402
import preflight_ligands as P  # noqa: E402


def write_tsv(path: Path, header: str, rows) -> str:
    path.write_text(header + "\n" + "".join("\t".join(r) + "\n" for r in rows))
    return str(path)


def bulk_args(tmp_path: Path, **kw) -> B.Args:
    survey = write_tsv(
        tmp_path / "survey.tsv", "bundle\tclassification\tdetail",
        [("aaaa", "name_mismatch", ""), ("bbbb", "name_mismatch", ""),
         ("cccc", "name_mismatch", ""), ("dddd", "name_mismatch", ""),
         ("eeee", "no_smiles", "")],
    )
    base = B.Args(*([None] * len(B.Args._fields)))._replace(
        survey_tsv=survey, go_classes=("name_mismatch",),
        record_file=str(tmp_path / "processed.tsv"),
        fix_smiles="fix_ligand_smiles.py", smiles_table="table.tsv",
        allow_name_mismatch=False, preflight_tsv=None)
    return base._replace(**kw)


def test_without_preflight_the_survey_alone_decides(tmp_path):
    assert B.eligible_bundles(bulk_args(tmp_path)) == [
        "aaaa", "bbbb", "cccc", "dddd"]


def test_preflight_keeps_only_pass_and_flag(tmp_path):
    verdicts = write_tsv(
        tmp_path / "ligands.tsv", "timestamp\tresult\tbundle\tdetail",
        [("t", "pass", "aaaa", ""), ("t", "flag", "bbbb", "stereo"),
         ("t", "block", "cccc", "connectivity differs"),
         # dddd is absent: never vetted, so never eligible
         ("t", "flag", "eeee", "")],  # wrong survey class, still out
    )
    args = bulk_args(tmp_path, preflight_tsv=verdicts)
    assert B.eligible_bundles(args) == ["aaaa", "bbbb"]


def test_preflight_later_verdict_wins(tmp_path):
    verdicts = write_tsv(
        tmp_path / "ligands.tsv", "timestamp\tresult\tbundle\tdetail",
        [("t1", "error", "aaaa", "timed out"), ("t2", "flag", "aaaa", ""),
         ("t1", "pass", "bbbb", ""), ("t2", "block", "bbbb", "")],
    )
    args = bulk_args(tmp_path, preflight_tsv=verdicts)
    assert B.eligible_bundles(args) == ["aaaa"]


@pytest.mark.parametrize("allow", [False, True])
def test_bulk_fix_smiles_passes_the_flag_only_when_asked(
        tmp_path, monkeypatch, allow):
    seen = []

    class Proc:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(B.subprocess, "run",
                        lambda cmd, **k: (seen.append(cmd), Proc())[1])
    B.fix_smiles("/some/bundle",
                 bulk_args(tmp_path, allow_name_mismatch=allow))
    assert ("--allow-name-mismatch" in seen[0]) is allow


@pytest.mark.parametrize("allow", [False, True])
def test_preflight_fill_smiles_passes_the_flag_only_when_asked(
        monkeypatch, allow):
    seen = []
    monkeypatch.setattr(P.subprocess, "run",
                        lambda cmd, **k: seen.append(cmd))
    args = P.Args(*([None] * len(P.Args._fields)))._replace(
        fix_smiles="fix_ligand_smiles.py", smiles_table="table.tsv",
        allow_name_mismatch=allow)
    P.fill_smiles("/some/bundle", args)
    assert ("--allow-name-mismatch" in seen[0]) is allow


def test_cli_refuses_the_override_without_a_preflight(tmp_path, monkeypatch):
    survey = write_tsv(tmp_path / "s.tsv", "bundle\tclassification\tdetail",
                       [])
    monkeypatch.setattr(sys, "argv", [
        "bulk_process_local.py", "-s", survey, "--allow-name-mismatch"])
    with pytest.raises(SystemExit) as exc:
        B.get_args()
    assert exc.value.code == 2


def test_cli_refuses_a_missing_preflight_file(tmp_path, monkeypatch):
    survey = write_tsv(tmp_path / "s.tsv", "bundle\tclassification\tdetail",
                       [])
    monkeypatch.setattr(sys, "argv", [
        "bulk_process_local.py", "-s", survey, "--allow-name-mismatch",
        "--preflight", str(tmp_path / "nope.tsv")])
    with pytest.raises(SystemExit):
        B.get_args()


def test_cli_accepts_the_override_with_a_preflight(tmp_path, monkeypatch):
    survey = write_tsv(tmp_path / "s.tsv", "bundle\tclassification\tdetail",
                       [])
    verdicts = write_tsv(tmp_path / "v.tsv",
                         "timestamp\tresult\tbundle\tdetail", [])
    monkeypatch.setattr(sys, "argv", [
        "bulk_process_local.py", "-s", survey, "--allow-name-mismatch",
        "--preflight", verdicts, "--work-dir", str(tmp_path / "w")])
    args = B.get_args()
    assert args.allow_name_mismatch is True
    assert args.preflight_tsv == verdicts
    assert args.dry_run is True
