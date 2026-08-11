"""Tests for irods_write_canary.py

The happy path is verified against the live zone by running the script; what
cannot be provoked against a healthy zone is everything interesting. A canary
that reports green when the write did not really land is worse than no canary,
so the three verification branches -- short object, wrong checksum, wrong bytes
read back -- are stubbed here rather than left untested until an outage.

Every case also asserts the canary object was removed. A canary that litters on
failure would accumulate exactly the kind of stuck object a CyVerse admin had to
clear by hand on 2026-08-07.
"""

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irods_write_canary as c  # noqa: E402


# --------------------------------------------------
class FakeObj:
    def __init__(self, size, checksum):
        self.size = size
        self._checksum = checksum

    def chksum(self):
        return self._checksum


# --------------------------------------------------
class FakeHandle:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


# --------------------------------------------------
class FakeDataObjects:
    """Records what happened so the tests can assert on cleanup"""

    def __init__(self, parent):
        self.parent = parent

    def put(self, local_path, remote_path, **_kw):
        if self.parent.put_raises:
            raise self.parent.put_raises
        with open(local_path, "rb") as fh:
            self.parent.stored = fh.read()
        self.parent.remote_path = remote_path

    def get(self, _remote_path):
        stored = self.parent.stored
        size = self.parent.size_override
        if size is None:
            size = len(stored)
        checksum = self.parent.checksum_override
        if checksum is None:
            checksum = hashlib.md5(stored).hexdigest()
        return FakeObj(size, checksum)

    def open(self, _remote_path, _mode):
        payload = self.parent.readback_override
        if payload is None:
            payload = self.parent.stored
        return FakeHandle(payload)

    def unlink(self, remote_path, force=False):
        self.parent.unlinked.append((remote_path, force))


# --------------------------------------------------
class FakeCollections:
    def __init__(self, parent):
        self.parent = parent

    def exists(self, _path):
        return self.parent.collection_exists

    def create(self, path):
        self.parent.created.append(path)


# --------------------------------------------------
class FakeSession:
    def __init__(self, **kwargs):
        self.stored = b""
        self.remote_path = None
        self.unlinked = []
        self.created = []
        self.collection_exists = True
        self.size_override = None
        self.checksum_override = None
        self.readback_override = None
        self.put_raises = None
        self.data_objects = FakeDataObjects(self)
        self.collections = FakeCollections(self)

    def cleanup(self):
        pass


# --------------------------------------------------
@pytest.fixture
def fake_session(monkeypatch):
    """Install a fake iRODSSession, optionally pre-broken, and expose it

    Returns an installer so each test can describe the fault it wants in one
    line. Everything goes through monkeypatch so the real iRODSSession is
    always restored -- assigning c.iRODSSession directly in a test body works
    by accident and is the kind of thing that leaks into the next test.
    """

    holder = {}

    def install(**overrides):
        def factory(**kwargs):
            s = FakeSession(**kwargs)
            for key, value in overrides.items():
                setattr(s, key, value)
            holder["session"] = s
            return s

        monkeypatch.setattr(c, "iRODSSession", factory)
        return holder

    return install


# --------------------------------------------------
def run(tmp_path, size_mb=1):
    return c.run_canary(
        server="staging",
        irods_env="/dev/null",
        size_mb=size_mb,
        timeout=30,
        tmp_dir=str(tmp_path),
    )


# --------------------------------------------------
def test_happy_path(fake_session, tmp_path):
    """A write that lands intact passes and cleans up after itself"""

    holder = fake_session()
    result = run(tmp_path)

    assert result.ok, result.detail
    assert "verified twice" in result.detail
    assert len(holder["session"].unlinked) == 1


# --------------------------------------------------
def test_short_object_fails(fake_session, tmp_path):
    """The catalog reporting fewer bytes than we put is a failure

    This is ticket 1355's truncated object and the 0-byte replicas of the
    August outage: the call returns, the object exists, and it is not what was
    written.
    """

    holder = fake_session(size_override=12)
    result = run(tmp_path)

    assert not result.ok
    assert "size mismatch" in result.detail
    assert len(holder["session"].unlinked) == 1


# --------------------------------------------------
def test_checksum_mismatch_fails(fake_session, tmp_path):
    """Right length, wrong content -- the divergent-replica signature

    Explicitly the case a size check cannot see, which is how the DB backup
    stayed broken: corral4 held the old content at the new size, flagged Good.
    """

    holder = fake_session(checksum_override="0" * 32)
    result = run(tmp_path)

    assert not result.ok
    assert "checksum mismatch" in result.detail
    assert len(holder["session"].unlinked) == 1


# --------------------------------------------------
def test_missing_checksum_fails(fake_session, tmp_path):
    """An object with no registered checksum does not get the benefit of the doubt"""

    fake_session(checksum_override="")
    result = run(tmp_path)

    assert not result.ok
    assert "checksum mismatch" in result.detail
    assert "(none)" in result.detail


# --------------------------------------------------
def test_readback_mismatch_fails(fake_session, tmp_path):
    """Server checksum agrees but the bytes served differ

    Not redundant with the checksum test: this is a stale replica being served
    in place of what was just written, where the catalog is confident and
    wrong. The read-back is what a consumer would actually get.
    """

    fake_session(readback_override=b"not what was written")
    result = run(tmp_path)

    assert not result.ok
    assert "read-back mismatch" in result.detail


# --------------------------------------------------
def test_put_failure_is_reported_not_raised(fake_session, tmp_path):
    """A raising put comes back as ok=False, never as an exception

    The drain calls this before claiming a job; an exception escaping here
    would turn a grid outage into a drain crash.
    """

    holder = fake_session(put_raises=RuntimeError("HIERARCHY_ERROR"))
    result = run(tmp_path)

    assert not result.ok
    assert "RuntimeError" in result.detail
    assert "HIERARCHY_ERROR" in result.detail
    # Nothing was written, so nothing should have been removed
    assert holder["session"].unlinked == []


# --------------------------------------------------
def test_local_temp_file_is_always_removed(fake_session, tmp_path):
    """No payload is left behind on disk, on success or failure"""

    run(tmp_path)
    assert list(tmp_path.iterdir()) == []

    fake_session(put_raises=RuntimeError("boom"))
    run(tmp_path)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------
def test_collection_is_created_when_absent(fake_session, tmp_path):
    """Creating the collection is part of what is tested, not setup around it

    push_sim_files.py creates the per-simulation collection before putting into
    it, and "mkdir + put into prod/release" is the pattern recorded as failing
    and later recovering on 2026-08-07.
    """

    holder = fake_session(collection_exists=False)
    result = run(tmp_path)

    assert result.ok, result.detail
    assert holder["session"].created == [c.canary_collection("staging")]


# --------------------------------------------------
def test_unlink_forces_rather_than_trashing():
    """A canary in the trash is one object per run for 30-40 days"""

    session = FakeSession()
    c._cleanup(session, "/some/path", "/nonexistent/local", lambda _m: None)

    assert session.unlinked == [("/some/path", True)]


# --------------------------------------------------
def test_cleanup_failure_does_not_raise():
    """Cleanup must never turn a healthy canary into a failed one"""

    class Exploding(FakeSession):
        def __init__(self):
            super().__init__()

            class DO:
                def unlink(self, *_a, **_k):
                    raise RuntimeError("cannot remove")

            self.data_objects = DO()

    c._cleanup(Exploding(), "/some/path", "/nonexistent/local", lambda _m: None)


# --------------------------------------------------
def test_payload_is_requested_size_and_unique():
    """Size is exact, and two runs never produce the same bytes

    Uniqueness matters: a repeated payload could be satisfied by a cached or
    stale replica, which is the thing the read-back exists to catch.
    """

    a = c._make_payload(1, "aaaa")
    b = c._make_payload(1, "bbbb")

    assert len(a) == len(b) == 1024 * 1024
    assert a != b


# --------------------------------------------------
def test_canary_collection_is_server_scoped():
    """prod and staging must not share a canary collection"""

    prod = c.canary_collection("prod")
    staging = c.canary_collection("staging")

    assert prod != staging
    assert "/prod/release/" in prod
    assert "/staging/release/" in staging
