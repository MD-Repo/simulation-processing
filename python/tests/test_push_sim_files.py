"""Tests for push_sim_files.py

Two functions are covered here, verify_irods() and remote_md5_and_size(), and
for one reason: both ask IRODS for a checksum, and each in turn killed a run by
letting the answer escape as an exception. The second half of this file is the
2026-09-10 repeat; the first half is the original.

verify_irods() is the function that decides whether a pushed simulation is
complete, and on 2026-08-12 and
2026-08-14 it was also the function that killed the run. It called
`obj.chksum()`, which makes the server resolve a resource hierarchy and re-read
the object; that raised HIERARCHY_ERROR on 46 replicate-merge groups, and
because the exception escaped, the whole verification pass died with it. The
files were on disk in IRODS and fine. What was lost was the *record* of that --
so each group was rolled back, its target left `is_placeholder = true`, and 46
public simulations 404'd until someone ran the finish-only pass by hand.

A healthy zone will not produce any of that, which is exactly why it is stubbed
here. The cases that matter are the ones where IRODS answers badly: no
registered checksum, and a checksum call that raises. Neither may be allowed to
propagate, and neither may be allowed to read as verified.
"""

import io
import os
import signal
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import push_sim_files as p  # noqa: E402


GOOD_MD5 = "d41d8cd98f00b204e9800998ecf8427e"


# --------------------------------------------------
class FakeObj:
    def __init__(self, size, checksum, chksum_raises=None, chksum_returns=None):
        self.size = size
        # The catalog value, as python-irodsclient exposes it: a plain
        # attribute, and None when nothing was ever registered.
        self.checksum = checksum
        self._chksum_raises = chksum_raises
        self._chksum_returns = chksum_returns
        self.chksum_calls = 0

    def chksum(self):
        """The expensive, hierarchy-resolving RPC. Counted so a test can prove
        it was not called when the catalog already had the answer."""

        self.chksum_calls += 1
        if self._chksum_raises:
            raise self._chksum_raises
        return self._chksum_returns


# --------------------------------------------------
class FakeDataObjects:
    def __init__(self, obj, exists=True, content=b"", open_raises=None):
        self._obj = obj
        self._exists = exists
        self._content = content
        self._open_raises = open_raises
        self.opens = 0

    def exists(self, _remote_path):
        return self._exists

    def get(self, _remote_path):
        return self._obj

    def open(self, _remote_path, _mode):
        """The read that md5_readback hashes. Counted so a test can prove a
        large file with a registered checksum is not read."""

        self.opens += 1
        if self._open_raises:
            raise self._open_raises
        return io.BytesIO(self._content)


# --------------------------------------------------
class FakeSession:
    def __init__(self, obj, exists=True, content=b"", open_raises=None):
        self.data_objects = FakeDataObjects(obj, exists, content, open_raises)


# --------------------------------------------------
def test_absent_object_is_not_present():
    """A path IRODS does not have is not present, and nothing is asked of it"""

    obj = FakeObj(size=10, checksum=GOOD_MD5)
    session = FakeSession(obj, exists=False)

    assert p.verify_irods(session, "/z/gone", 10) == (False, None, "absent")
    assert session.data_objects.opens == 0 and obj.chksum_calls == 0


# --------------------------------------------------
def test_size_mismatch_is_not_present():
    """A short or stale object fails on size before any checksum work

    Size is the cheap discriminator, so it runs first -- and a size mismatch is
    already a definitive no, whatever the checksum would have said.
    """

    obj = FakeObj(size=17, checksum=GOOD_MD5)
    session = FakeSession(obj)

    assert p.verify_irods(session, "/z/short", 4096) == (False, None, "size mismatch")
    assert session.data_objects.opens == 0


# --------------------------------------------------
def test_a_small_file_is_proved_by_reading_it_back():
    """Up to the read-back limit, the md5 is of the bytes IRODS serves, not
    the catalog's claim about them"""

    obj = FakeObj(size=0, checksum="md5:" + "0" * 32)
    session = FakeSession(obj, content=b"")

    assert p.verify_irods(session, "/z/small", 0, readback_limit=10) == (True, GOOD_MD5, "read back")
    assert session.data_objects.opens == 1


# --------------------------------------------------
def test_registered_checksum_is_used_without_reading_or_computing():
    """Above the read-back limit the catalog value is the answer: no second
    pass over the file, and chksum() is never called for it"""

    obj = FakeObj(size=4096, checksum=GOOD_MD5)
    session = FakeSession(obj)

    assert p.verify_irods(session, "/z/ok", 4096, readback_limit=0) == (True, GOOD_MD5, "registered checksum")
    assert session.data_objects.opens == 0 and obj.chksum_calls == 0


# --------------------------------------------------
@pytest.mark.parametrize("registered", [f"md5:{GOOD_MD5}", f"  {GOOD_MD5.upper()}  "])
def test_checksum_is_normalised(registered):
    """A "md5:" prefix, padding, or upper case still compares to our manifest"""

    obj = FakeObj(size=4096, checksum=registered)
    got = p.verify_irods(FakeSession(obj), "/z/ok", 4096, readback_limit=0)

    assert got == (True, GOOD_MD5, "registered checksum")


# --------------------------------------------------
def test_a_failed_read_back_falls_through_to_the_registered_checksum(capsys):
    """A small file that will not read is recorded, and the catalog may still
    know its checksum"""

    obj = FakeObj(size=4096, checksum=GOOD_MD5)
    session = FakeSession(obj, open_raises=OSError("connection reset"))

    assert p.verify_irods(session, "/z/flaky", 4096) == (True, GOOD_MD5, "registered checksum")
    assert "could not read back" in capsys.readouterr().out


# --------------------------------------------------
def test_missing_registration_falls_back_to_reading():
    """An object with no catalog checksum still gets an answer

    Objects uploaded before REG_CHKSUM_KW, or by something that is not this
    script, have nothing registered. The bytes are read rather than calling
    chksum(), the RPC that resolved a hierarchy and raised HIERARCHY_ERROR.
    """

    obj = FakeObj(size=4096, checksum=None, chksum_returns="f" * 32)
    session = FakeSession(obj, content=b"")

    got = p.verify_irods(session, "/z/old", 4096, readback_limit=0)

    assert got == (True, GOOD_MD5, "read back (no registered checksum)")
    assert obj.chksum_calls == 0


# --------------------------------------------------
def test_an_unreadable_object_is_contained(capsys):
    """A read that raises returns unverified instead of propagating

    The regression this file exists for. `present` stays True because the
    object really is there at the right size -- lying about that would send the
    caller down the "missing" path -- but the md5 is None, which cannot equal
    an expected md5, so the file reads as NOT VERIFIED and the simulation stays
    a placeholder. That is recoverable. A raised exception was not.
    """

    from irods.exception import HIERARCHY_ERROR

    obj = FakeObj(size=4096, checksum=None)
    session = FakeSession(obj, open_raises=HIERARCHY_ERROR(None))

    assert p.verify_irods(session, "/z/broken", 4096, readback_limit=0) == (True, None, "unreadable")
    assert "could not checksum" in capsys.readouterr().out


# --------------------------------------------------
def test_unverifiable_file_never_reads_as_verified():
    """The caller's comparison must reject a None md5 for any expected value

    verify_irods() returning None is only safe if it cannot accidentally
    match. This pins the property at the point the decision is actually made.
    """

    _present, remote_md5, _how = p.verify_irods(
        FakeSession(FakeObj(size=4096, checksum=None), open_raises=RuntimeError("x")),
        "/z/broken",
        4096,
        readback_limit=0,
    )

    assert remote_md5 is None
    assert remote_md5 != GOOD_MD5.lower()


# --------------------------------------------------
# remote_md5_and_size() -- the skip decision, added 2026-09-10.
#
# Same RPC, same failure, other end of the run. verify_irods() was hardened in
# August; this path kept calling obj.chksum() unconditionally and unguarded,
# and on 2026-09-10 it took down job 193 (ticket 2337) as a traceback in 14
# seconds. One 0-byte replica, left intermediate by an interrupted write the
# night before, answered HIERARCHY_ERROR -- and because the call sat in the
# loop that decides what to upload, the run died before it had looked at a
# single file. Nothing was uploaded, nothing was verified, nothing was
# recorded.


# --------------------------------------------------
def test_skip_check_absent_object_is_zero_and_empty():
    """A path IRODS does not have cannot be skipped, and is not asked anything"""

    obj = FakeObj(size=10, checksum=GOOD_MD5)
    size, md5 = p.remote_md5_and_size(FakeSession(obj, exists=False), "/z/gone")

    assert (size, md5) == (0, "")
    assert obj.chksum_calls == 0


# --------------------------------------------------
def test_skip_check_uses_the_registered_checksum():
    """The catalog value is the answer, and chksum() is never called for it

    The normal re-run path: every object this script wrote has its checksum
    registered by put_file(), so the hierarchy-resolving RPC is not touched at
    all. That is the difference between a re-run costing a metadata lookup per
    file and costing a full server-side re-read of every file.
    """

    obj = FakeObj(size=4096, checksum=GOOD_MD5)
    size, md5 = p.remote_md5_and_size(FakeSession(obj), "/z/ok")

    assert (size, md5) == (4096, GOOD_MD5)
    assert obj.chksum_calls == 0


# --------------------------------------------------
@pytest.mark.parametrize("registered", [f"md5:{GOOD_MD5}", f"  {GOOD_MD5.upper()}  "])
def test_skip_check_normalises_the_checksum(registered):
    """A "md5:" prefix, padding, or upper case still compares to our manifest"""

    obj = FakeObj(size=4096, checksum=registered)
    assert p.remote_md5_and_size(FakeSession(obj), "/z/ok") == (4096, GOOD_MD5)


# --------------------------------------------------
def test_skip_check_falls_back_to_computing():
    """An object with nothing registered still gets an answer

    Objects written before REG_CHKSUM_KW, or by something other than this
    script, have no catalog checksum. Skipping them on size alone is what the
    2026-08-10 change set out to stop, so the fallback has to exist.
    """

    obj = FakeObj(size=4096, checksum=None, chksum_returns=GOOD_MD5)
    size, md5 = p.remote_md5_and_size(FakeSession(obj), "/z/old")

    assert (size, md5) == (4096, GOOD_MD5)
    assert obj.chksum_calls == 1


# --------------------------------------------------
def test_skip_check_contains_hierarchy_error(capsys):
    """HIERARCHY_ERROR from chksum() returns no md5 instead of propagating

    The regression. The size still comes back, because the caller's fallback
    test needs it and the catalog really does say 0.
    """

    from irods.exception import HIERARCHY_ERROR

    obj = FakeObj(size=0, checksum=None, chksum_raises=HIERARCHY_ERROR(None))
    size, md5 = p.remote_md5_and_size(FakeSession(obj), "/z/stuck")

    assert (size, md5) == (0, "")
    assert "could not checksum" in capsys.readouterr().out


# --------------------------------------------------
def test_stuck_object_is_queued_rather_than_skipped():
    """The 0-byte intermediate replica must read as "upload this"

    This is the decision the run actually makes, spelled out: an unanswerable
    checksum sends it to the size test, and 188,689 local bytes against 0
    remote ones is not a skip. Anything else would be worse than the crash --
    silently accepting an empty object as the published file.
    """

    from irods.exception import HIERARCHY_ERROR

    local_size, local_md5 = 188689, "4386ca18466183940d42f2119c40e7b2"
    obj = FakeObj(size=0, checksum=None, chksum_raises=HIERARCHY_ERROR(None))
    remote_size, remote_md5 = p.remote_md5_and_size(FakeSession(obj), "/z/stuck")

    if local_md5 and remote_md5:
        can_skip = remote_md5 == local_md5.strip().lower()
    else:
        can_skip = local_size == remote_size

    assert can_skip is False


# --------------------------------------------------
# tighten_deadline (MDR-73): the post-upload stretch gets a shorter deadline.
@pytest.fixture
def no_alarm():
    signal.alarm(0)
    yield
    signal.alarm(0)
    p.ABORT.clear()


@pytest.mark.parametrize(
    "armed, expected",
    [
        (3600, 600),  # an hour left: pulled in
        (100, 100),  # already sooner: never extended
        (0, 0),  # --timeout 0: stays off
    ],
)
def test_tighten_deadline(no_alarm, armed, expected):
    signal.alarm(armed)
    p.tighten_deadline(600)
    assert signal.alarm(0) == expected


def test_tighten_deadline_leaves_the_abort_grace_alone(no_alarm):
    p.ABORT.set()
    signal.alarm(p.ABORT_GRACE)
    p.tighten_deadline(10)
    assert signal.alarm(0) == p.ABORT_GRACE
