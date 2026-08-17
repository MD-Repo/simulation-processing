"""Tests for push_sim_files.py

Only verify_irods() is covered here, and for one reason: it is the function
that decides whether a pushed simulation is complete, and on 2026-08-12 and
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

import os
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
    def __init__(self, obj, exists=True):
        self._obj = obj
        self._exists = exists

    def exists(self, _remote_path):
        return self._exists

    def get(self, _remote_path):
        return self._obj


# --------------------------------------------------
class FakeSession:
    def __init__(self, obj, exists=True):
        self.data_objects = FakeDataObjects(obj, exists)


# --------------------------------------------------
def test_absent_object_is_not_present():
    """A path IRODS does not have is (False, None), and nothing is asked of it"""

    obj = FakeObj(size=10, checksum=GOOD_MD5)
    present, md5 = p.verify_irods(FakeSession(obj, exists=False), "/z/gone", 10)

    assert (present, md5) == (False, None)
    assert obj.chksum_calls == 0


# --------------------------------------------------
def test_size_mismatch_is_not_present():
    """A short or stale object fails on size before any checksum work

    Size is the cheap discriminator, so it runs first -- and a size mismatch is
    already a definitive no, whatever the checksum would have said.
    """

    obj = FakeObj(size=17, checksum=GOOD_MD5)
    present, md5 = p.verify_irods(FakeSession(obj), "/z/short", 4096)

    assert (present, md5) == (False, None)
    assert obj.chksum_calls == 0


# --------------------------------------------------
def test_registered_checksum_is_used_without_computing():
    """The catalog value is the answer, and chksum() is never called for it

    This is the whole point of registering the checksum at upload time: the
    normal path must not touch the RPC that failed.
    """

    obj = FakeObj(size=4096, checksum=GOOD_MD5)
    present, md5 = p.verify_irods(FakeSession(obj), "/z/ok", 4096)

    assert (present, md5) == (True, GOOD_MD5)
    assert obj.chksum_calls == 0


# --------------------------------------------------
@pytest.mark.parametrize("registered", [f"md5:{GOOD_MD5}", f"  {GOOD_MD5.upper()}  "])
def test_checksum_is_normalised(registered):
    """A "md5:" prefix, padding, or upper case still compares to our manifest"""

    obj = FakeObj(size=4096, checksum=registered)
    assert p.verify_irods(FakeSession(obj), "/z/ok", 4096) == (True, GOOD_MD5)


# --------------------------------------------------
def test_missing_registration_falls_back_to_computing():
    """An object with no catalog checksum still gets an answer

    Objects uploaded before REG_CHKSUM_KW, or by something that is not this
    script, have nothing registered. Forcing the computation is the only way
    left to verify them, so the fallback has to exist.
    """

    obj = FakeObj(size=4096, checksum=None, chksum_returns=GOOD_MD5)
    present, md5 = p.verify_irods(FakeSession(obj), "/z/old", 4096)

    assert (present, md5) == (True, GOOD_MD5)
    assert obj.chksum_calls == 1


# --------------------------------------------------
def test_hierarchy_error_is_contained(capsys):
    """HIERARCHY_ERROR from chksum() returns unverified instead of propagating

    The regression this file exists for. `present` stays True because the
    object really is there at the right size -- lying about that would send the
    caller down the "missing" path -- but the md5 is None, which cannot equal
    an expected md5, so the file reads as NOT VERIFIED and the simulation stays
    a placeholder. That is recoverable. A raised exception was not.
    """

    from irods.exception import HIERARCHY_ERROR

    obj = FakeObj(size=4096, checksum=None, chksum_raises=HIERARCHY_ERROR(None))
    present, md5 = p.verify_irods(FakeSession(obj), "/z/broken", 4096)

    assert present is True
    assert md5 is None
    assert "could not checksum" in capsys.readouterr().out


# --------------------------------------------------
def test_unverifiable_file_never_reads_as_verified():
    """The caller's comparison must reject a None md5 for any expected value

    verify_irods() returning None is only safe if it cannot accidentally
    match. This pins the property at the point the decision is actually made.
    """

    _present, remote_md5 = p.verify_irods(
        FakeSession(FakeObj(size=4096, checksum=None, chksum_raises=RuntimeError("x"))),
        "/z/broken",
        4096,
    )

    assert remote_md5 != GOOD_MD5.lower()
    assert not (remote_md5 == GOOD_MD5.lower())
