"""Tests for export_public_metadata.py's index.json

index.json is what someone holding only the tarball reads first: the date,
the count, the ID list and, since 2026-11-10, which of those an AI agent
imported (Ken). The agent list is read from the records in the same
tarball, so the two cannot disagree, and a record without the flag stops
the export rather than publishing an empty list that only looks like
"none". No database and no network: records are written to a temp dir in
the shape mdr-export writes them.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import export_public_metadata as E  # noqa: E402


def record(sim_dir: Path, sim_id: int, **fields) -> None:
    """One MDRnnnnnnnn.json shaped like mdr-export's Summary"""
    body = {
        "id": sim_id,
        "is_deprecated": False,
        "is_placeholder": False,
        "creation_date": "2026-11-10T00:00:00Z",
        "is_agent_imported": False,
        "meta": {"description": "x"},
    }
    body.update(fields)
    (sim_dir / f"MDR{sim_id:08}.json").write_text(json.dumps(body))


@pytest.fixture
def dirs(tmp_path):
    sim_dir = tmp_path / "build" / "simulations"
    sim_dir.mkdir(parents=True)
    return tmp_path / "build", sim_dir


def read_index(build_dir: Path) -> dict:
    return json.loads((build_dir / "index.json").read_text())


# --------------------------------------------------
def test_agent_imports_are_listed(dirs):
    build_dir, sim_dir = dirs
    record(sim_dir, 2)
    record(sim_dir, 530, is_agent_imported=True)
    record(sim_dir, 10, is_agent_imported=True)
    record(sim_dir, 7)
    assert E.write_index(str(build_dir), str(sim_dir), "prod") == 4
    index = read_index(build_dir)
    assert index["simulations"] == [
        "MDR00000002",
        "MDR00000007",
        "MDR00000010",
        "MDR00000530",
    ]
    assert index["agent_imported"] == ["MDR00000010", "MDR00000530"]
    assert index["agent_imported_count"] == 2


def test_none_imported_gives_an_empty_list_not_a_missing_key(dirs):
    # Today's prod: 0 agent imports. The key is still there, so a consumer
    # can tell "none" from "this index predates the field"
    build_dir, sim_dir = dirs
    record(sim_dir, 2)
    record(sim_dir, 5)
    E.write_index(str(build_dir), str(sim_dir), "prod")
    index = read_index(build_dir)
    assert index["agent_imported"] == []
    assert index["agent_imported_count"] == 0


def test_agent_list_is_a_sorted_subset_of_simulations(dirs):
    build_dir, sim_dir = dirs
    for i in (900, 3, 47, 12000, 8):
        record(sim_dir, i, is_agent_imported=i % 2 == 0)
    E.write_index(str(build_dir), str(sim_dir), "staging")
    index = read_index(build_dir)
    assert set(index["agent_imported"]) <= set(index["simulations"])
    assert index["agent_imported"] == sorted(index["agent_imported"])
    assert index["agent_imported"] == ["MDR00000008", "MDR00000900", "MDR00012000"]


def test_existing_keys_unchanged(dirs):
    # Adding fields only: what consumers already read stays as it was
    build_dir, sim_dir = dirs
    record(sim_dir, 2, is_agent_imported=True)
    E.write_index(str(build_dir), str(sim_dir), "prod")
    index = read_index(build_dir)
    assert list(index)[:5] == ["generated", "server", "count", "visibility", "simulations"]
    assert index["server"] == "prod"
    assert index["count"] == 1
    assert index["visibility"] == (
        "is_public and not is_deprecated and not is_placeholder and not is_embargoed"
    )


def test_a_record_without_the_flag_refuses(dirs):
    build_dir, sim_dir = dirs
    record(sim_dir, 2)
    body = json.loads((sim_dir / "MDR00000002.json").read_text())
    del body["is_agent_imported"]
    (sim_dir / "MDR00000002.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="MDR00000002.json.*too old"):
        E.write_index(str(build_dir), str(sim_dir), "prod")
    assert not (build_dir / "index.json").exists()


@pytest.mark.parametrize("bad", [None, "true", 1, 0, []])
def test_a_flag_that_is_not_a_bool_refuses(dirs, bad):
    build_dir, sim_dir = dirs
    record(sim_dir, 2, is_agent_imported=bad)
    with pytest.raises(ValueError, match="not true/false"):
        E.write_index(str(build_dir), str(sim_dir), "prod")


def test_non_json_files_are_ignored(dirs):
    build_dir, sim_dir = dirs
    record(sim_dir, 2, is_agent_imported=True)
    (sim_dir / "README.txt").write_text("not a record")
    assert E.write_index(str(build_dir), str(sim_dir), "prod") == 1
    assert read_index(build_dir)["agent_imported"] == ["MDR00000002"]


def test_empty_export_writes_an_empty_index(dirs):
    # main() refuses to publish a zero count; write_index itself just says so
    build_dir, sim_dir = dirs
    assert E.write_index(str(build_dir), str(sim_dir), "prod") == 0
    assert read_index(build_dir)["agent_imported"] == []


def test_agent_imported_reads_the_records_directly(dirs):
    _, sim_dir = dirs
    record(sim_dir, 4, is_agent_imported=True)
    record(sim_dir, 6)
    assert E.agent_imported(str(sim_dir), [4, 6]) == [4]
    assert E.agent_imported(str(sim_dir), []) == []
    with pytest.raises(FileNotFoundError):
        E.agent_imported(str(sim_dir), [99])


@pytest.mark.parametrize("text", ["", "{", "not json", "[1, 2]", "null"])
def test_a_damaged_record_refuses_naming_it(dirs, text):
    build_dir, sim_dir = dirs
    record(sim_dir, 2)
    (sim_dir / "MDR00000031.json").write_text(text)
    with pytest.raises(ValueError, match="MDR00000031.json"):
        E.write_index(str(build_dir), str(sim_dir), "prod")
    assert not (build_dir / "index.json").exists()
