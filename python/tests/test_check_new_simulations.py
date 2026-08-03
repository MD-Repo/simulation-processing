"""Tests for check_new_simulations

Only `upload_complete` is covered, because it is the whole of the scan's
verdict: everything else in a pass is IRODS iteration and one SQL statement,
while this one function decides whether a ticket is notified and enqueued or
left alone for the next pass.

It is worth pinning because the obvious implementation is wrong in a way that
costs data. Tickets 1501 and 1505 (yang, 2026-07-01) transferred all 52 files
and then wrote a 0-byte `mdrepo-submission.completed.json`. An existence test
called those uploads complete; 32.65 GB then sat unprocessed for a month, and
they were only found by probing the collections by hand on 2026-08-03.
"""

import sys
from pathlib import Path

import pytest
from irods.exception import CollectionDoesNotExist, DataObjectDoesNotExist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_new_simulations import SUBMISSION_COMPLETE, upload_complete  # noqa: E402

LANDING = "/iplant/home/shared/mdrepo/prod/landing/ABC123===_0"


class FakeObject:
    def __init__(self, size: int) -> None:
        self.size = size


class FakeDataObjects:
    """Stands in for session.data_objects, recording what was asked for"""

    def __init__(self, result) -> None:
        self._result = result
        self.asked = []

    def get(self, path: str):
        self.asked.append(path)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeSession:
    def __init__(self, result) -> None:
        self.data_objects = FakeDataObjects(result)


def test_marker_with_content_is_complete():
    assert upload_complete(FakeSession(FakeObject(5039)), LANDING) is True


def test_zero_byte_marker_is_not_complete():
    # The 1501/1505 case: every data file arrived, the manifest did not.
    assert upload_complete(FakeSession(FakeObject(0)), LANDING) is False


def test_missing_marker_is_not_complete():
    session = FakeSession(DataObjectDoesNotExist())
    assert upload_complete(session, LANDING) is False


def test_missing_collection_is_not_complete():
    # get() raises on the collection before it looks for the object, so this
    # is a different exception from a missing marker and must also be caught.
    session = FakeSession(CollectionDoesNotExist())
    assert upload_complete(session, LANDING) is False


def test_probes_the_marker_and_nothing_else():
    # One round trip per landing is the whole cost model of a scan: a pass
    # walks >12,000 landings, and listing the collection instead measured
    # 0.78s against 0.135s.
    session = FakeSession(FakeObject(1))
    upload_complete(session, LANDING)
    assert session.data_objects.asked == [f"{LANDING}/{SUBMISSION_COMPLETE}"]


def test_unexpected_errors_are_not_swallowed():
    # A network failure must reach process_ticket's handler, which counts the
    # ticket as errored and exits non-zero. Reading it as "not complete" would
    # silently defer every ticket on a bad night and report a clean run.
    session = FakeSession(RuntimeError("connection reset"))
    with pytest.raises(RuntimeError):
        upload_complete(session, LANDING)
