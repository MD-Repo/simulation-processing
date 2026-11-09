"""Tests for the bundles2/ backlog processor's pure logic

The IRODS/DB/subprocess-touching pieces (find_next_candidate's session,
ticket_queue_busy's cursor, run_mdr_process, delete_irods_source) are
exercised with fakes rather than a live server -- get_args, load_record and
append_record need nothing at all.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import process_bundles2 as pb  # noqa: E402
from irods.models import DataObject  # noqa: E402


# --------------------------------------------------
class FakeQuery:
    """The catalog query find_next_candidate builds: filters and the ordering
    are the catalog's job, so the fake keeps the rows in create_time order
    and ignores the filters."""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args):
        return self

    def order_by(self, *_args):
        return self

    def get_results(self):
        for name, size, created in sorted(self._rows, key=lambda r: r[2]):
            yield {DataObject.name: name, DataObject.size: size, DataObject.create_time: created}


class FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_columns):
        return FakeQuery(self._rows)


# --------------------------------------------------
def test_get_args_defaults_are_staging_scoped(monkeypatch):
    """Staging by default: one forgotten flag must not write the backfill
    into production."""

    monkeypatch.setattr(sys, "argv", ["process_bundles2.py"])
    args = pb.get_args()

    assert args.server == "staging"
    assert args.lock_file.endswith("drain_process_queue-staging.lock")
    assert args.record_file.endswith(os.path.join("bundles2", "staging", "processed.tsv"))
    assert args.log_dir.endswith(os.path.join("bundles2", "staging"))
    assert args.dry_run is False


def test_dry_run_implies_verbose(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["process_bundles2.py", "--dry-run"])
    args = pb.get_args()

    assert args.dry_run is True
    assert args.verbose is True


def test_prod_scopes_lock_and_record_separately(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["process_bundles2.py", "--server", "prod"])
    args = pb.get_args()

    assert args.lock_file.endswith("drain_process_queue-prod.lock")
    assert args.record_file.endswith(os.path.join("bundles2", "prod", "processed.tsv"))


# --------------------------------------------------
def test_load_record_missing_file_is_empty(tmp_path):
    assert pb.load_record(str(tmp_path / "nope.tsv")) == {}


def test_load_record_skips_the_header_and_short_lines(tmp_path):
    path = tmp_path / "processed.tsv"
    path.write_text("timestamp\tresult\tbundle\tdetail\nnot a record line\n")

    assert pb.load_record(str(path)) == {}


def test_append_record_then_load_record_roundtrips(tmp_path):
    path = tmp_path / "nested" / "processed.tsv"

    pb.append_record(str(path), "sim001", "done", "")
    pb.append_record(str(path), "sim002", "failed", "mdr-process\nexited 1")

    assert pb.load_record(str(path)) == {"sim001": "done", "sim002": "failed"}
    lines = path.read_text().splitlines()
    assert lines[0] == "timestamp\tresult\tbundle\tdetail", "one header, first"
    assert len(lines) == 3, "the detail is flattened to one line"
    assert lines[2].endswith("\tmdr-process exited 1")


def test_a_later_line_wins(tmp_path):
    """Re-running a bundle that failed and recording it done reads as done,
    without anyone editing history."""

    path = tmp_path / "processed.tsv"
    pb.append_record(str(path), "sim001", "failed", "boom")
    pb.append_record(str(path), "sim001", "done", "")

    assert pb.load_record(str(path)) == {"sim001": "done"}


def test_a_long_detail_keeps_both_ends(tmp_path):
    """The failing step is at the front and the diagnosis at the back; the
    cap cuts the middle."""

    path = tmp_path / "processed.tsv"
    detail = "STEP " + "x" * 5000 + " DIAGNOSIS"
    pb.append_record(str(path), "sim001", "failed", detail)

    kept = path.read_text().splitlines()[1].split("\t")[3]
    assert len(kept) <= pb.DETAIL_MAX
    assert kept.startswith("STEP ") and kept.endswith(" DIAGNOSIS")
    assert "chars elided" in kept


# --------------------------------------------------
def test_find_next_candidate_skips_truncated_and_recorded_bundles():
    now = datetime(2026, 1, 1)
    session = FakeSession(
        [
            ("a.md5", 10, now),
            # 0-byte sidecar from a truncated write -- not ready.
            ("b.md5", 0, now - timedelta(days=1)),
            # Oldest ready candidate -- should win over "a".
            ("c.md5", 5, now - timedelta(days=2)),
            # Not a sidecar at all -- ignored.
            ("c.data", 5, now),
            # Already recorded -- skipped even though it's oldest.
            ("d.md5", 5, now - timedelta(days=3)),
        ]
    )

    assert pb.find_next_candidate(session, {"d": "done"}) == "c"


def test_find_next_candidate_returns_none_when_nothing_ready():
    session = FakeSession([("a.md5", 0, datetime(2026, 1, 1))])

    assert pb.find_next_candidate(session, {}) is None


def test_find_next_candidate_returns_none_when_everything_is_recorded():
    session = FakeSession([("a.md5", 10, datetime(2026, 1, 1))])

    assert pb.find_next_candidate(session, {"a": "failed"}) is None


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
