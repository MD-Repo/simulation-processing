"""Tests for the replica check in backup_database.py

WHY THIS EXISTS. On 2026-10-21 all four nightly backup objects (prod and
staging) were clean single replicas whose corral4 copy never arrived; CyVerse's
replication simply did not run. A forced replica to taccRes landed at once, so
check_replicas now forces one when the wait runs out short. These pin down when
it forces, when it does not, and that a forced replica is checked like any
other.

A fake session stands in for IRODS; the wait is set to zero so nothing sleeps.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backup_database  # noqa: E402
from backup_database import check_replicas  # noqa: E402

SIZE = 1000
MD5 = "c2bc1be8b88b473b4ba6ec9b03ae632f"
PATH = "/iplant/home/mdadm/pg_dumps/mdrepo.org/mdrepo.21.sql.gz"


def replica(number, resc, hier, size=SIZE, md5=MD5, status="1"):
    return SimpleNamespace(number=number, resource_name=resc, resc_hier=hier,
                           size=size, checksum=md5, status=status)


PRIMARY = replica(0, "ds10", "CyVerseRes;ds10Broker;ds10")
TACC = replica(1, "corral4", "taccRes;corral4")


class FakeDataObjects:
    def __init__(self, replicas, on_replicate=None):
        self.replicas = list(replicas)
        self.on_replicate = on_replicate
        self.replicate_calls = []

    def get(self, path):
        return SimpleNamespace(size=SIZE, replicas=list(self.replicas))

    def replicate(self, path, resource=None):
        self.replicate_calls.append(resource)
        if self.on_replicate:
            self.on_replicate(self)


def session(replicas, on_replicate=None):
    return SimpleNamespace(data_objects=FakeDataObjects(replicas, on_replicate))


def add_tacc(objs, md5=MD5):
    objs.replicas.append(replica(1, "corral4", "taccRes;corral4", md5=md5))


def quiet(_message):
    pass


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(backup_database, "REPLICA_WAIT_SECONDS", 0)


def test_two_good_replicas_force_nothing():
    s = session([PRIMARY, TACC])
    assert check_replicas(s, PATH, SIZE, MD5, quiet) == ([], False)
    assert s.data_objects.replicate_calls == []


def test_missing_replica_is_forced_onto_the_passthru_root():
    s = session([PRIMARY], on_replicate=add_tacc)
    assert check_replicas(s, PATH, SIZE, MD5, quiet) == ([], True)
    assert s.data_objects.replicate_calls == ["taccRes"]


def test_no_force_keeps_the_old_failure():
    s = session([PRIMARY], on_replicate=add_tacc)
    problems, forced = check_replicas(s, PATH, SIZE, MD5, quiet, force=False)
    assert not forced
    assert s.data_objects.replicate_calls == []
    assert problems == ["only 1 replica(s) after 0s, expected 2"]


def test_forced_replica_is_checked_like_any_other():
    s = session([PRIMARY], on_replicate=lambda o: add_tacc(o, md5="0" * 32))
    problems, forced = check_replicas(s, PATH, SIZE, MD5, quiet)
    assert forced
    assert problems == [f"replica 1 on corral4 md5 {'0' * 32} != local {MD5}"]


def test_replicate_that_raises_is_a_problem_not_a_crash():
    def boom(_objs):
        raise RuntimeError("SYS_RESC_IS_DOWN")

    s = session([PRIMARY], on_replicate=boom)
    problems, forced = check_replicas(s, PATH, SIZE, MD5, quiet)
    assert not forced
    assert problems[0].startswith("forcing a replica onto taccRes failed:")
    assert problems[1] == "only 1 replica(s) after 0s, expected 2"


def test_lone_replica_already_on_taccres_is_not_forced():
    # The CyVerse side is the one missing; replicating to taccRes again would
    # not fill it.
    s = session([TACC])
    problems, forced = check_replicas(s, PATH, SIZE, MD5, quiet)
    assert not forced
    assert s.data_objects.replicate_calls == []
    assert problems == ["only 1 replica(s) after 0s, expected 2"]
