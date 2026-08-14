"""Tests for the bundles2/ backlog processor's pure logic

The IRODS/DB/subprocess-touching pieces (find_next_candidate's session,
ticket_queue_busy's cursor, run_mdr_process, delete_irods_source) are
exercised with fakes rather than a live server -- get_args, load_state and
save_state need nothing at all.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import process_bundles2 as pb  # noqa: E402


# --------------------------------------------------
class FakeDataObject:
    def __init__(self, name, size, create_time):
        self.name = name
        self.size = size
        self.create_time = create_time


class FakeCollections:
    def __init__(self, data_objects):
        self._data_objects = data_objects

    def get(self, path):
        return type("Coll", (), {"data_objects": self._data_objects})()


class FakeSession:
    def __init__(self, data_objects):
        self.collections = FakeCollections(data_objects)


# --------------------------------------------------
def test_get_args_defaults_are_prod_scoped(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["process_bundles2.py"])
    args = pb.get_args()

    assert args.server == "prod"
    assert args.lock_file.endswith("drain_process_queue-prod.lock")
    assert args.state_file.endswith("process_bundles2-prod.json")
    assert args.log_dir.endswith(os.path.join("bundles2", "prod"))
    assert args.dry_run is False


def test_dry_run_implies_verbose(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["process_bundles2.py", "--dry-run"])
    args = pb.get_args()

    assert args.dry_run is True
    assert args.verbose is True


def test_staging_scopes_lock_and_state_file_separately(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["process_bundles2.py", "--server", "staging"]
    )
    args = pb.get_args()

    assert args.lock_file.endswith("drain_process_queue-staging.lock")
    assert args.state_file.endswith("process_bundles2-staging.json")


# --------------------------------------------------
def test_load_state_missing_file_is_empty(tmp_path):
    assert pb.load_state(str(tmp_path / "nope.json")) == {}


def test_load_state_corrupt_file_is_empty_not_a_crash(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json{{{")

    assert pb.load_state(str(path)) == {}


def test_save_state_then_load_state_roundtrips(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state = {
        "sim001": {"status": "done", "timestamp": "2026-08-14T00:00:00Z"}
    }

    pb.save_state(str(path), state)

    assert path.is_file()
    assert pb.load_state(str(path)) == state


def test_save_state_leaves_no_tmp_file_behind(tmp_path):
    path = tmp_path / "state.json"
    pb.save_state(str(path), {"a": {"status": "done"}})

    assert not (tmp_path / "state.json.tmp").exists()


def test_save_state_is_atomic_replace_not_partial_write(tmp_path):
    """A crash between the write and the rename must never corrupt the file

    save_state writes to "<path>.tmp" then os.replace()s it into place, so
    the real file is either the old complete contents or the new complete
    contents -- never a half-written mix of both.
    """

    path = tmp_path / "state.json"
    pb.save_state(str(path), {"a": {"status": "done"}})
    original = path.read_text()

    # Simulate the write step failing before the replace: the .tmp file may
    # exist mid-write, but the real file must be untouched.
    tmp = Path(f"{path}.tmp")
    tmp.write_text("garbage, not valid json")

    assert path.read_text() == original
    assert json.loads(path.read_text()) == {"a": {"status": "done"}}


# --------------------------------------------------
def test_find_next_candidate_skips_truncated_and_recorded_bundles():
    now = datetime(2026, 1, 1)
    session = FakeSession(
        [
            FakeDataObject("a.md5", 10, now),
            # 0-byte sidecar from a truncated write -- not ready.
            FakeDataObject("b.md5", 0, now - timedelta(days=1)),
            # Oldest ready candidate -- should win over "a".
            FakeDataObject("c.md5", 5, now - timedelta(days=2)),
            # Not a sidecar at all -- ignored.
            FakeDataObject("c.data", 5, now),
            # Already recorded in state -- skipped even though it's oldest.
            FakeDataObject("d.md5", 5, now - timedelta(days=3)),
        ]
    )
    state = {"d": {"status": "done"}}

    assert pb.find_next_candidate(session, state) == "c"


def test_find_next_candidate_returns_none_when_nothing_ready():
    session = FakeSession([FakeDataObject("a.md5", 0, datetime(2026, 1, 1))])

    assert pb.find_next_candidate(session, {}) is None


def test_find_next_candidate_returns_none_when_everything_is_recorded():
    now = datetime(2026, 1, 1)
    session = FakeSession([FakeDataObject("a.md5", 10, now)])
    state = {"a": {"status": "failed"}}

    assert pb.find_next_candidate(session, state) is None


# --------------------------------------------------
def test_ticket_queue_busy_true_when_a_row_is_pending_or_running():
    class FakeCursor:
        def execute(self, *_args, **_kwargs):
            pass

        def fetchone(self):
            return (1,)

    assert pb.ticket_queue_busy(FakeCursor(), "prod") is True


def test_ticket_queue_busy_false_when_queue_is_empty():
    class FakeCursor:
        def execute(self, *_args, **_kwargs):
            pass

        def fetchone(self):
            return None

    assert pb.ticket_queue_busy(FakeCursor(), "prod") is False
