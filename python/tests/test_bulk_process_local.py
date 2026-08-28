"""Tests for bulk_process_local.py's outage guards.

These exist because both guards were got WRONG in production on 2026-08-28,
in ways that only showed up against a real prod run.

The circuit breaker was written with a consecutive-fault rule copied from
merge_replicate_groups.py. Then batch 1 lost **7 of 8 bundles** to IRODS and
never tripped it: four faults, one success reset the count, three more. With
several workers in flight a single success landing between failures hides an
outage indefinitely, and the run would have ground through all 6,625 bundles
minting a hidden placeholder row for each. Hence the second rule -- N of the
last M -- and hence `test_window_rule_catches_what_consecutive_misses`, which
replays batch 1's actual shape.

The reap was written to delete every failed bundle's files, because an outage
fails everything and ~4,400 kept failures fill the volume. That threw away the
cheap retry: push_sim_files.py skips any file whose md5 already matches, so a
re-push moves only what is missing, and all 7 of batch 1's failures had to be
reprocessed from scratch instead. Hence is_irods_fault() gating the reap.

The loop is driven through main() rather than by testing helpers in isolation,
because the bug both times was in the loop's control flow, not in the
predicates. The pool is swapped for threads so process_one can be stubbed
in-process; everything else is the real code path.
"""

import concurrent.futures
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bulk_process_local as B  # noqa: E402

PUSH_FAIL = ("failed", "Command failed: push_sim_files.py")
DATA_FAIL = ("failed", 'short_description: value "..." length must be <= 300')
OK = ("dry-run-ok", "")


def run_loop(outcomes, max_faults=5, fault_wait=0, parallel=2,
             window=10, win_faults=5):
    """Drive main()'s real loop over len(outcomes) stub bundles.

    Returns (recorded rows, exit code). wait_for_irods is stubbed to False so
    the outage always outlasts us; the waiting path is a sleep loop and is not
    what these tests are about.
    """

    tmp = tempfile.mkdtemp()
    survey = os.path.join(tmp, "survey.tsv")
    record = os.path.join(tmp, "processed.tsv")
    with open(survey, "w") as fh:
        fh.write("bundle\tclassification\tdetail\n")
        for i in range(len(outcomes)):
            fh.write(f"b{i:03d}\tgo\tx\n")

    seq = list(outcomes)

    def stub(args, name):
        return (name, *seq[int(name[1:])])

    saved = (concurrent.futures.ProcessPoolExecutor, B.process_one,
             B.wait_for_irods, sys.argv)
    concurrent.futures.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor
    B.process_one = stub
    B.wait_for_irods = lambda a: False
    sys.argv = [
        "bulk_process_local.py", "--survey-tsv", survey, "--go-classes", "go",
        "--record", record, "--work-dir", os.path.join(tmp, "work"),
        "--log-dir", os.path.join(tmp, "logs"), "--dry-run",
        "--parallel", str(parallel),
        "--max-consecutive-faults", str(max_faults),
        "--fault-wait", str(fault_wait),
        "--fault-window", str(window),
        "--max-window-faults", str(win_faults),
    ]

    code = 0
    try:
        B.main()
    except SystemExit as e:
        code = e.code or 0
    finally:
        (concurrent.futures.ProcessPoolExecutor, B.process_one,
         B.wait_for_irods, sys.argv) = saved

    with open(record) as fh:
        rows = [ln.split("\t") for ln in fh.read().splitlines()[1:]]
    return rows, code


# ---- is_irods_fault -------------------------------------------------------

@pytest.mark.parametrize("result,detail,expected", [
    ("failed", "Command failed: push_sim_files.py", True),
    ("failed", "NetworkException: could not receive server response", True),
    ("failed", "HIERARCHY_ERROR on chksum", True),
    ("failed", "UNIX_FILE_OPEN_ERR", True),
    ("failed", 'short_description: value "..." length must be <= 300', False),
    ("fetch-failed", "corrupt or truncated tarball", False),
    ("failed", "Command failed: sample_trajectory.py", False),
    ("done", "", False),
    ("done-flag", "ligand[0] tetrahedral stereo", False),
])
def test_only_storage_failures_count_as_faults(result, detail, expected):
    """The whole point of a fault count is that it means the SERVER is gone.

    A run of short_description overflows or corrupt tarballs is bad data and
    must not stop a wave; a run of push_sim_files.py failures is CyVerse.
    Conflating them would either stop the run on a data problem or fail to
    stop it on an outage.
    """

    assert B.is_irods_fault(result, detail) is expected


# ---- the loop still works --------------------------------------------------

def test_healthy_run_processes_every_bundle_once():
    """The breaker restructured the executor to submit a window at a time.

    The previous loop queued all 6,625 futures up front, which is why it could
    not stop early. The replacement must still process everything exactly once.
    """

    rows, code = run_loop([OK] * 12)
    assert len(rows) == 12
    assert len({r[2] for r in rows}) == 12, "a bundle was processed twice"
    assert code == 0


# ---- consecutive rule ------------------------------------------------------

@pytest.mark.parametrize("parallel,total", [(1, 40), (4, 100)])
def test_sustained_outage_stops_the_run(parallel, total):
    """An outage fails every bundle, so the consecutive count climbs and trips.

    Checked at two concurrencies because the count is kept in completion order,
    and more workers in flight means more chances for the ordering to differ.
    """

    outcomes = [OK] * 5 + [PUSH_FAIL] * (total - 5)
    rows, code = run_loop(outcomes, parallel=parallel, window=0, win_faults=0)
    assert len(rows) < total // 2, "ground on through the outage"
    assert code == 1, "a stopped run must exit non-zero"


def test_data_failures_never_stop_the_run():
    """15 consecutive short_description failures is bad data, not an outage."""

    rows, code = run_loop([DATA_FAIL] * 15 + [OK] * 5)
    assert len(rows) == 20
    assert code == 0


def test_consecutive_rule_counts_consecutively():
    """Four faults, a success, repeat -- never five in a row.

    With the window rule off this must run to completion; that is what
    "consecutive" means. It is also exactly the pattern that let batch 1
    through, which is why the window rule exists -- see the next test.
    """

    outcomes = ([PUSH_FAIL] * 4 + [OK]) * 6
    rows, code = run_loop(outcomes, parallel=1, window=0, win_faults=0)
    assert len(rows) == len(outcomes)
    assert code == 0


def test_zero_disables_both_rules():
    """0 has to mean off, or there is no way back to the old behaviour."""

    rows, _ = run_loop([PUSH_FAIL] * 30, max_faults=0, window=0, win_faults=0)
    assert len(rows) == 30


# ---- window rule -----------------------------------------------------------

def test_window_rule_catches_what_consecutive_misses():
    """Batch 1's real shape, 2026-08-28: 7 of 8 lost, breaker never tripped.

    Faults interleaved with a success that resets the consecutive count. The
    consecutive rule alone runs to the end; adding 5-of-the-last-10 stops it
    early. This is the regression that the second rule was written for.
    """

    outcomes = ([PUSH_FAIL] * 4 + [("done", "")]) * 10

    rows, code = run_loop(outcomes, parallel=1, window=0, win_faults=0)
    assert len(rows) == len(outcomes), "consecutive rule alone should not trip"
    assert code == 0

    rows, code = run_loop(outcomes, parallel=1, window=10, win_faults=5)
    assert len(rows) < 20, "window rule failed to stop an 80% failure rate"
    assert code == 1


def test_window_rule_tolerates_a_healthy_failure_rate():
    """A healthy wave fails ~3.5% of bundles. That must never trip anything.

    Three scattered faults in 40 is roughly double the observed rate and still
    has to run clean, or the guard costs more than it saves on a six-day run.
    """

    outcomes = [("done", "")] * 40
    for i in (7, 19, 31):
        outcomes[i] = PUSH_FAIL
    rows, code = run_loop(outcomes)
    assert len(rows) == 40
    assert code == 0
