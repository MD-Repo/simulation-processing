"""Tests for fetch_uploads

The interesting behaviour is all in phase two -- the part that runs on worker
threads -- so that is what these cover: how a fetch is batched into gocmd
calls, what counts as a verified file, and how a failure gets back to the
main thread without killing the process from inside a thread.

The live test at the bottom talks to production iRODS and is skipped unless
FETCH_LIVE_TEST=1.
"""

import hashlib
import os
import re
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetch_uploads as fup  # noqa: E402
from drain_process_queue import is_retryable_irods_error  # noqa: E402


def make_object(tmp_path: Path, name: str, content: bytes) -> fup.RemoteObject:
    """A RemoteObject describing `content`, without writing it anywhere"""

    return fup.RemoteObject(
        path=f"/iplant/home/shared/mdrepo/prod/landing/{name}",
        name=name,
        size=len(content),
        md5=hashlib.md5(content).hexdigest(),
    )


def make_part(tmp_path: Path, objects) -> fup.Part:
    dest = tmp_path / "landing-a"
    dest.mkdir(exist_ok=True)
    return fup.Part(
        ticket_id=1718,
        part_num=1,
        landing_dir="/iplant/home/shared/mdrepo/prod/landing/landing-a",
        dest_dir=str(dest),
        objects=list(objects),
    )


# -------------------------------------------------- chunked
def test_chunked_splits_evenly():
    assert fup.chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunked_keeps_remainder():
    assert fup.chunked([1, 2, 3], 2) == [[1, 2], [3]]


def test_chunked_empty():
    assert fup.chunked([], 10) == []


# -------------------------------------------------- needs_fetch
def test_verify_flags_missing_file(tmp_path):
    obj = make_object(tmp_path, "sim.xtc", b"trajectory")
    assert fup.needs_fetch(str(tmp_path), obj) is True


def test_verify_flags_wrong_size(tmp_path):
    obj = make_object(tmp_path, "sim.xtc", b"trajectory")
    (tmp_path / "sim.xtc").write_bytes(b"traj")
    assert fup.needs_fetch(str(tmp_path), obj) is True


def test_verify_flags_wrong_md5(tmp_path):
    obj = make_object(tmp_path, "sim.xtc", b"trajectory")
    # Same length, different bytes: only the hash can tell these apart.
    (tmp_path / "sim.xtc").write_bytes(b"TRAJECTORY")
    assert fup.needs_fetch(str(tmp_path), obj) is True


def test_verify_accepts_good_file(tmp_path):
    obj = make_object(tmp_path, "sim.xtc", b"trajectory")
    (tmp_path / "sim.xtc").write_bytes(b"trajectory")
    assert fup.needs_fetch(str(tmp_path), obj) is False


# -------------------------------------------------- run_gocmd
def test_run_gocmd_batches_every_path_into_one_call(monkeypatch):
    """The whole point of the rewrite: N objects, one invocation."""

    seen = []
    monkeypatch.setattr(fup, "getstatusoutput", lambda cmd: seen.append(cmd) or (0, ""))
    fup.run_gocmd(["/a/one.xtc", "/a/two.xtc", "/a/three.xtc"], "/local/dest")

    assert len(seen) == 1
    assert seen[0].count("gocmd get") == 1
    for path in ("/a/one.xtc", "/a/two.xtc", "/a/three.xtc"):
        assert path in seen[0]
    assert seen[0].endswith("/local/dest")


def test_run_gocmd_uses_diff_to_skip_existing(monkeypatch):
    seen = []
    monkeypatch.setattr(fup, "getstatusoutput", lambda cmd: seen.append(cmd) or (0, ""))
    fup.run_gocmd(["/a/one.xtc"], "/local/dest")
    assert "--diff" in seen[0]


def test_run_gocmd_quotes_paths(monkeypatch):
    seen = []
    monkeypatch.setattr(fup, "getstatusoutput", lambda cmd: seen.append(cmd) or (0, ""))
    fup.run_gocmd(["/a/odd name.xtc"], "/local/dest")
    assert "'/a/odd name.xtc'" in seen[0]


def test_run_gocmd_raises_rather_than_exits(monkeypatch):
    """A worker thread must not call sys.exit."""

    monkeypatch.setattr(fup, "getstatusoutput", lambda cmd: (1, "boom"))
    with pytest.raises(fup.FetchError) as excinfo:
        fup.run_gocmd(["/a/one.xtc"], "/local/dest")
    assert "boom" in str(excinfo.value)


# -------------------------------------------------- the queue contract
def test_irods_outage_message_stays_retryable(monkeypatch):
    """drain_process_queue reads gocmd's text to decide whether to requeue.

    Batching changed the command but must not change what a connection
    failure looks like by the time it reaches the queue.
    """

    gocmd_says = "\x1b[31mFailed to establish a connection to iRODS server\x1b[0m"
    monkeypatch.setattr(fup, "getstatusoutput", lambda cmd: (1, gocmd_says))
    with pytest.raises(fup.FetchError) as excinfo:
        fup.run_gocmd(["/a/one.xtc"], "/local/dest")

    assert is_retryable_irods_error(str(excinfo.value))


# -------------------------------------------------- fetch_part
def test_fetch_part_succeeds_when_files_land(tmp_path, monkeypatch):
    objects = [
        make_object(tmp_path, "one.xtc", b"aaaa"),
        make_object(tmp_path, "two.xtc", b"bbbbbb"),
    ]
    part = make_part(tmp_path, objects)
    calls = []

    def fake_gocmd(paths, dest_dir):
        calls.append(paths)
        for obj in objects:
            if obj.path in paths:
                content = b"aaaa" if obj.name == "one.xtc" else b"bbbbbb"
                Path(dest_dir, obj.name).write_bytes(content)
        return ""

    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    result = fup.fetch_part(part)

    assert len(calls) == 1, "both objects should go in one gocmd call"
    assert result.bytes_fetched == 10


def test_fetch_part_retries_only_the_bad_object(tmp_path, monkeypatch):
    """A file that arrives corrupt is re-fetched; its neighbours are not."""

    objects = [
        make_object(tmp_path, "good.xtc", b"aaaa"),
        make_object(tmp_path, "bad.xtc", b"bbbb"),
    ]
    part = make_part(tmp_path, objects)
    calls = []

    def fake_gocmd(paths, dest_dir):
        calls.append(list(paths))
        Path(dest_dir, "good.xtc").write_bytes(b"aaaa")
        # Truncated on the first pass, correct on the retry.
        content = b"xx" if len(calls) == 1 else b"bbbb"
        Path(dest_dir, "bad.xtc").write_bytes(content)
        return ""

    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    result = fup.fetch_part(part)

    assert len(calls) == 2
    assert calls[1] == [objects[1].path], "retry should carry only the bad object"
    assert result.bytes_fetched == 8


def test_fetch_part_raises_when_verification_keeps_failing(tmp_path, monkeypatch):
    objects = [make_object(tmp_path, "sim.xtc", b"trajectory")]
    part = make_part(tmp_path, objects)

    def fake_gocmd(paths, dest_dir):
        Path(dest_dir, "sim.xtc").write_bytes(b"corrupt")
        return ""

    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    with pytest.raises(fup.FetchError) as excinfo:
        fup.fetch_part(part)

    assert "failed to verify" in str(excinfo.value)
    assert "sim.xtc" in str(excinfo.value)


def test_fetch_part_removes_bad_copy_before_retrying(tmp_path, monkeypatch):
    """--diff compares hashes, so a bad file must be deleted or it is kept."""

    objects = [make_object(tmp_path, "sim.xtc", b"trajectory")]
    part = make_part(tmp_path, objects)
    existed_at_retry = []
    calls = []

    def fake_gocmd(paths, dest_dir):
        calls.append(list(paths))
        if len(calls) > 1:
            existed_at_retry.append(os.path.isfile(os.path.join(dest_dir, "sim.xtc")))
        Path(dest_dir, "sim.xtc").write_bytes(
            b"corrupt!!!" if len(calls) == 1 else b"trajectory"
        )
        return ""

    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    fup.fetch_part(part)

    assert existed_at_retry == [False]


def test_fetch_part_chunks_huge_object_lists(tmp_path, monkeypatch):
    objects = [make_object(tmp_path, f"f{i}.dat", b"x") for i in range(250)]
    part = make_part(tmp_path, objects)
    calls = []

    def fake_gocmd(paths, dest_dir):
        calls.append(list(paths))
        for obj in objects:
            if obj.path in paths:
                Path(dest_dir, obj.name).write_bytes(b"x")
        return ""

    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    fup.fetch_part(part)

    assert [len(c) for c in calls] == [100, 100, 50]


# -------------------------------------------------- SessionPool
def test_session_pool_gives_each_thread_its_own_session(monkeypatch):
    """An iRODSSession cannot be shared, so two threads must never get one."""

    import threading

    class FakeSession:
        def __init__(self, irods_env_file):
            self.closed = False

        def cleanup(self):
            self.closed = True

    monkeypatch.setattr(fup, "iRODSSession", FakeSession)
    pool = fup.SessionPool("/irods/env.json")
    seen = {}

    def grab(name):
        # Twice, to prove a thread reuses the session it already has.
        seen[name] = (pool.get(), pool.get())

    threads = [threading.Thread(target=grab, args=(n,)) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["a"][0] is seen["a"][1], "a thread should reuse its session"
    assert seen["a"][0] is not seen["b"][0], "threads must not share a session"

    pool.close()
    assert all(session.closed for session, _ in seen.values())


# -------------------------------------------------- resolve/process
def test_resolve_part_rejects_malformed_irods_ticket(tmp_path):
    ref = fup.PartRef(
        ticket_id=1728,
        part_num=3,
        irods_ticket="no-colon-here",
        ticket_dir=str(tmp_path),
    )
    part, output = fup.resolve_part(session=None, ref=ref, args=None)

    assert part is None
    assert "Invalid IRODS ticket" in "\n".join(output)


def test_process_batch_skips_fetch_when_nothing_resolves(tmp_path, monkeypatch):
    """Unresolvable parts are reported, not fetched, and are not an error."""

    called = []
    monkeypatch.setattr(fup, "fetch_parts", lambda parts, args: called.append(parts))
    batch = [
        fup.PartRef(
            ticket_id=1728, part_num=n, irods_ticket="bad", ticket_dir=str(tmp_path)
        )
        for n in (1, 2)
    ]

    class Pool:
        def get(self):
            return None

    result = fup.process_batch(batch, args=None, sessions=Pool())

    assert called == []
    assert result.bytes_fetched == 0
    assert result.output.count("  Ticket 1728 part 1: Invalid IRODS ticket 'bad'") == 1


# -------------------------------------------------- batching
def make_args(threads=8, pattern=None):
    return fup.Args(
        out_dir="/out",
        server="prod",
        irods_env="/env.json",
        landing_dirs=[],
        ticket_ids=[],
        pattern=pattern,
        threads=threads,
    )


def make_ref(ticket_id, part_num, ticket_dir="/land/ticket-1"):
    return fup.PartRef(
        ticket_id=ticket_id,
        part_num=part_num,
        irods_ticket=f"tok:/iplant/landing/L{part_num}",
        ticket_dir=ticket_dir,
    )


def test_batches_aim_for_one_per_worker():
    refs = [make_ref(1728, n) for n in range(200)]
    batches = fup.make_batches(refs, make_args(threads=8))

    assert len(batches) == 8
    assert sum(len(b) for b in batches) == 200
    assert max(len(b) for b in batches) == 25


def test_batches_are_capped():
    """A huge ticket on few threads must not become one enormous batch."""

    refs = [make_ref(1728, n) for n in range(300)]
    batches = fup.make_batches(refs, make_args(threads=2))

    assert max(len(b) for b in batches) == fup.MAX_COLLECTIONS_PER_CALL


def test_batches_never_span_tickets():
    """gocmd names local dirs by basename; only within a ticket are they unique."""

    refs = [make_ref(1728, n, "/land/ticket-1728") for n in range(4)]
    refs += [make_ref(1729, n, "/land/ticket-1729") for n in range(4)]
    batches = fup.make_batches(refs, make_args(threads=1))

    for batch in batches:
        assert len({ref.ticket_id for ref in batch}) == 1


# -------------------------------------------------- fetch_parts
def make_resolved(tmp_path, name, objects, ticket_dir=None):
    ticket_dir = ticket_dir or str(tmp_path / "ticket-1728")
    dest = os.path.join(ticket_dir, name)
    os.makedirs(dest, exist_ok=True)
    return fup.Part(
        ticket_id=1728,
        part_num=1,
        landing_dir=f"/iplant/landing/{name}",
        dest_dir=dest,
        objects=list(objects),
    )


def test_fetch_parts_uses_one_call_for_the_whole_batch(tmp_path, monkeypatch):
    parts = []
    for name in ("L1", "L2", "L3"):
        obj = make_object(tmp_path, f"{name}.xtc", b"data")
        parts.append(make_resolved(tmp_path, name, [obj]))

    calls = []

    def fake_gocmd(paths, dest):
        calls.append((list(paths), dest))
        for part in parts:
            Path(part.dest_dir, f"{os.path.basename(part.dest_dir)}.xtc").write_bytes(
                b"data"
            )

    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    result = fup.fetch_parts(parts, make_args())

    assert len(calls) == 1, "three landing dirs should cost one gocmd call"
    assert calls[0][0] == [p.landing_dir for p in parts]
    assert result.bytes_fetched == 12


def test_fetch_parts_falls_back_when_basenames_collide(tmp_path, monkeypatch):
    """gocmd merges same-named collections; 15 files became 7 when measured."""

    obj = make_object(tmp_path, "a.xtc", b"data")
    parts = [
        make_resolved(tmp_path, "same", [obj], str(tmp_path / "t1")),
        make_resolved(tmp_path, "same", [obj], str(tmp_path / "t2")),
    ]
    per_part = []
    monkeypatch.setattr(
        fup,
        "fetch_part",
        lambda part: per_part.append(part)
        or fup.PartResult(output=[], bytes_fetched=4),
    )
    monkeypatch.setattr(
        fup, "run_gocmd", lambda *a: pytest.fail("must not batch colliding names")
    )

    result = fup.fetch_parts(parts, make_args())

    assert len(per_part) == 2
    assert result.bytes_fetched == 8


def test_fetch_parts_falls_back_with_a_pattern(tmp_path, monkeypatch):
    """A collection get cannot filter, so --pattern must name objects."""

    obj = make_object(tmp_path, "a.xtc", b"data")
    parts = [make_resolved(tmp_path, "L1", [obj])]
    monkeypatch.setattr(
        fup, "fetch_part", lambda part: fup.PartResult(output=[], bytes_fetched=4)
    )
    monkeypatch.setattr(
        fup, "run_gocmd", lambda *a: pytest.fail("must not batch with a pattern")
    )

    fup.fetch_parts(parts, make_args(pattern=re.compile(r"\.xtc$")))


def test_fetch_parts_refetches_only_the_unverified_part(tmp_path, monkeypatch):
    """One bad landing dir must not condemn the rest of the batch."""

    parts = []
    for name in ("good", "bad"):
        obj = make_object(tmp_path, f"{name}.xtc", b"data")
        parts.append(make_resolved(tmp_path, name, [obj]))

    def fake_gocmd(paths, dest):
        # Only "good" lands correctly; "bad" arrives corrupt.
        Path(parts[0].dest_dir, "good.xtc").write_bytes(b"data")
        Path(parts[1].dest_dir, "bad.xtc").write_bytes(b"XXXX")

    retried = []
    monkeypatch.setattr(fup, "run_gocmd", fake_gocmd)
    monkeypatch.setattr(
        fup,
        "fetch_part",
        lambda part: retried.append(part) or fup.PartResult(output=[], bytes_fetched=4),
    )

    fup.fetch_parts(parts, make_args())

    assert [p.dest_dir for p in retried] == [parts[1].dest_dir]


# -------------------------------------------------- live
@pytest.mark.skipif(
    os.environ.get("FETCH_LIVE_TEST") != "1",
    reason="hits production iRODS; set FETCH_LIVE_TEST=1 to run",
)
def test_fetch_part_against_real_irods(tmp_path):
    """End-to-end against a released simulation: real gocmd, real MD5s."""

    from irods.session import iRODSSession

    coll_path = "/iplant/home/shared/mdrepo/prod/release/MDR00032893/original"
    env = os.path.expanduser("~/.irods/irods_environment.json")
    with iRODSSession(irods_env_file=env) as session:
        coll = session.collections.get(coll_path)
        small = sorted(coll.data_objects, key=lambda o: o.size)[:4]
        objects = [
            fup.RemoteObject(
                path=obj.path, name=obj.name, size=obj.size, md5=obj.chksum()
            )
            for obj in small
        ]

    part = make_part(tmp_path, objects)
    result = fup.fetch_part(part)

    assert result.bytes_fetched == sum(o.size for o in objects)
    for obj in objects:
        assert fup.needs_fetch(part.dest_dir, obj) is False
