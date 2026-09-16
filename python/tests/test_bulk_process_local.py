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
             window=10, win_faults=5, max_trips=3, probe=False):
    """Drive main()'s real loop over len(outcomes) stub bundles.

    Returns (recorded rows, exit code). wait_for_irods is stubbed to `probe`:
    False (the default) means the outage always outlasts us, True means the
    health probe keeps insisting the server is fine -- which is the
    2026-09-16 shape and what --max-trips exists to survive.
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
    B.wait_for_irods = lambda a: probe
    sys.argv = [
        "bulk_process_local.py", "--survey-tsv", survey, "--go-classes", "go",
        "--record", record, "--work-dir", os.path.join(tmp, "work"),
        "--log-dir", os.path.join(tmp, "logs"), "--dry-run",
        "--parallel", str(parallel),
        "--max-consecutive-faults", str(max_faults),
        "--fault-wait", str(fault_wait),
        "--fault-window", str(window),
        "--max-window-faults", str(win_faults),
        "--max-trips", str(max_trips),
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


# ---- the 2026-09-16 regression: a probe that is wrong all day -------------

def test_a_lying_health_probe_cannot_keep_the_run_alive():
    """The breaker must stop even when the health probe says the server is up.

    On 2026-09-16 the breaker tripped 70 times over 11 hours and the probe
    cleared it every single time, so the wave pushed on and left 355 failed
    bundles and ~354 hidden placeholder rows. The probe was not lying: it
    wrote ONE object, and the fault failed files independently at ~8.5%, so
    one object passed 91.5% of the time while a 19-file bundle passed 18.6%.

    A breaker that a probe can veto indefinitely is not a breaker. After
    --max-trips trips the probe has been demonstrated wrong and is not
    consulted again.
    """

    # Every bundle is a storage fault, and the probe always says "fine".
    rows, code = run_loop([PUSH_FAIL] * 200, fault_wait=600,
                          max_trips=3, probe=True)

    assert code == 1, "a run this broken must exit non-zero"
    assert len(rows) < 200, "the run must stop, not grind through every bundle"
    # 3 trips, each needing max_faults(5) faults to arm: comfortably under 60.
    assert len(rows) <= 60, (
        f"stopped only after {len(rows)} bundles -- the trip cap is not "
        f"bounding the damage")


def test_max_trips_zero_restores_the_old_unbounded_behaviour():
    """0 means "never stop on trip count", which is what shipped before.

    Kept switchable because the trip cap is a policy, not a fact: a run that
    genuinely expects a long flaky patch may want the probe to keep deciding.
    """

    rows, code = run_loop([PUSH_FAIL] * 40, fault_wait=600,
                          max_trips=0, probe=True)

    assert code == 0, "with no trip cap the probe keeps clearing the breaker"
    assert len(rows) == 40, "every bundle is attempted, as it did on 09-16"


def test_a_probe_that_reports_recovery_still_resumes():
    """The cap must not break the case the waiting path exists for.

    A genuine ten-minute blip at hour twenty should not need a human, so one
    or two trips followed by a real recovery must resume normally.
    """

    # Five faults arm the breaker once, then the server comes back for good.
    rows, code = run_loop([PUSH_FAIL] * 5 + [OK] * 30, fault_wait=600,
                          max_trips=3, probe=True)

    assert code == 0, "one trip then recovery is not an outage"
    assert len(rows) == 35, "the run should complete every bundle"


# ---- the probe samples as hard as a real push ----------------------------

def test_health_probe_writes_a_bundles_worth_of_objects(monkeypatch, tmp_path):
    """One object cannot clear a partial outage -- that is the 09-16 bug.

    The probe must fail if ANY of its writes fail, and must write about as
    many as a bundle pushes, or it is systematically blinder than the thing
    it is vouching for.
    """

    canary = tmp_path / "irods_write_canary.py"
    canary.write_text("")
    calls = []

    class Proc:
        def __init__(self, rc):
            self.returncode = rc

    # Healthy: every write succeeds.
    monkeypatch.setattr(B.subprocess, "run",
                        lambda *a, **k: (calls.append(1), Proc(0))[1])
    args = B.Args(*([None] * len(B.Args._fields)))._replace(
        script_dir=str(tmp_path), server="prod", health_probe_objects=19)
    assert B.irods_healthy(args) is True
    assert len(calls) == 19, "the probe must write a bundle's worth, not one"

    # Partial outage: the 7th write fails, as HIERARCHY_ERROR did.
    calls.clear()
    monkeypatch.setattr(
        B.subprocess, "run",
        lambda *a, **k: (calls.append(1), Proc(0 if len(calls) < 7 else 1))[1])
    assert B.irods_healthy(args) is False, (
        "one failed object among many must condemn the whole probe")
