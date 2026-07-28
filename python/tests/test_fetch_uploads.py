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


def test_process_part_skips_fetch_when_unresolvable(tmp_path, monkeypatch):
    """An unresolvable part is reported, not fetched, and is not an error."""

    called = []
    monkeypatch.setattr(fup, "fetch_part", lambda part: called.append(part))
    ref = fup.PartRef(
        ticket_id=1728, part_num=3, irods_ticket="bad", ticket_dir=str(tmp_path)
    )

    class Pool:
        def get(self):
            return None

    result = fup.process_part(ref, args=None, sessions=Pool())

    assert called == []
    assert result.bytes_fetched == 0
    assert "Invalid IRODS ticket" in "\n".join(result.output)


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
