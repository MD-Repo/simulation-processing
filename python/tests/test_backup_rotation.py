"""Tests for the backup rotation audit in backup_database.py

WHY THIS EXISTS. `report_stored` audits the objects that ARE in the collection,
which cannot see the one failure that matters most: a slot that is GONE. Both
destinations lost a slot this way within eleven days, by the same mechanism --
the run removes the target slot before uploading to it, so a failed put leaves
nothing behind:

  prod    mdrepo.05.sql.gz  2026-09-05  put failed after the remove; the four
                            nightly verify-only runs that followed all reported
                            "0 problem(s)" while the listing jumped 04 to 06
  staging mdrepo.30.sql.gz  2026-08-30  same shape, "Failed to establish a
                            connection to iRODS server!", unnoticed for ten days

Both were recovered from Swift on 2026-09-09. The check added alongside them is
what turns the next one into an alert instead of an archaeology exercise.

The hard part is not presence, it is knowing which date a slot SHOULD hold.
Slot NN is written on the NNth, so short months make an old object correct: on
5 March the most recent 30th is 30 January, and slot 30 holding a five-week-old
dump is right, not stale. Every case below that looks like an off-by-one is
really that rule.

Pure functions over dates -- no IRODS, no network, no database.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backup_database import (  # noqa: E402
    SLOT_RE,
    check_rotation,
    expected_slot_date,
)


def quiet(_message):
    """A status sink, so the checks under test do not print during a run"""


# --------------------------------------------------
def test_expected_slot_date_ordinary():
    """This month if the day has passed, last month if it has not"""

    assert expected_slot_date(5, date(2026, 9, 9)) == date(2026, 9, 5)
    assert expected_slot_date(9, date(2026, 9, 9)) == date(2026, 9, 9)
    assert expected_slot_date(20, date(2026, 9, 9)) == date(2026, 8, 20)


# --------------------------------------------------
def test_expected_slot_date_skips_short_months():
    """A month without the day is skipped, not clamped to its last day

    Clamping is the tempting bug: it would make slot 30 expect 28 February and
    then report the correct January object as stale every March.
    """

    assert expected_slot_date(30, date(2026, 3, 5)) == date(2026, 1, 30)
    assert expected_slot_date(31, date(2026, 3, 5)) == date(2026, 1, 31)
    assert expected_slot_date(31, date(2026, 5, 5)) == date(2026, 3, 31)


# --------------------------------------------------
def test_expected_slot_date_leap_year():
    """29 February exists only when it exists"""

    assert expected_slot_date(29, date(2024, 3, 5)) == date(2024, 2, 29)
    assert expected_slot_date(29, date(2026, 3, 5)) == date(2026, 1, 29)


# --------------------------------------------------
def test_slot_re_matches_only_rotation_slots():
    """latest and the monthly archives are not part of the day-of-month cycle

    Unanchored or looser, this pulls in archive.mdrepo.2026-09-01.sql.gz and the
    audit starts demanding a rotation slot for every archived month.
    """

    for name in ("mdrepo.01.sql.gz", "mdrepo.05.sql.gz", "mdrepo.31.sql.gz"):
        assert SLOT_RE.match(name), name

    for name in ("mdrepo.latest.sql.gz", "archive.mdrepo.2026-09-01.sql.gz",
                 "mdrepo.00.sql.gz", "mdrepo.32.sql.gz", "mdrepo.5.sql.gz"):
        assert not SLOT_RE.match(name), name


# --------------------------------------------------
def full_rotation(today):
    """Every slot holding exactly the dump it should"""

    return {day: expected_slot_date(day, today) for day in range(1, 32)}


# --------------------------------------------------
def test_full_rotation_is_clean():
    """The healthy case reports nothing"""

    today = date(2026, 9, 9)
    assert check_rotation(full_rotation(today), today, quiet) == 0


# --------------------------------------------------
def test_missing_slot_is_reported():
    """The prod 2026-09-05 case: the slot is simply not there"""

    today = date(2026, 9, 9)
    seen = full_rotation(today)
    del seen[5]

    messages = []
    assert check_rotation(seen, today, messages.append) == 1
    assert "mdrepo.05.sql.gz" in messages[0]
    assert "MISSING" in messages[0]


# --------------------------------------------------
def test_stale_slot_is_reported():
    """The shape delete-then-put cannot produce, but put-then-rename can

    Today's code removes before uploading, so a failed night leaves nothing.
    The planned reorder leaves last month's object in place instead, and that
    reads as healthy to any check that only asks whether the object exists.
    """

    today = date(2026, 9, 9)
    seen = full_rotation(today)
    seen[4] = date(2026, 8, 4)

    messages = []
    assert check_rotation(seen, today, messages.append) == 1
    assert "STALE" in messages[0]


# --------------------------------------------------
def test_todays_slot_may_not_have_run_yet():
    """Between midnight and the cron, today's slot still holds last month

    Without this the audit reports a false problem every day in the window
    before the backup runs -- 00:00 to 00:20 on prod.
    """

    today = date(2026, 9, 9)
    seen = full_rotation(today)
    seen[9] = date(2026, 8, 9)

    assert check_rotation(seen, today, quiet) == 0


# --------------------------------------------------
def test_young_collection_is_not_all_holes():
    """A collection younger than a month has not reached most slots

    The horizon is the oldest date present, so slots whose turn predates the
    collection are not demanded. Otherwise standing up a new deployment reports
    30 missing slots on day one.
    """

    today = date(2026, 9, 9)
    seen = {day: date(2026, 9, day) for day in range(1, 10)}

    assert check_rotation(seen, today, quiet) == 0


# --------------------------------------------------
def test_empty_collection_is_a_problem():
    """No slots at all is not "nothing to check\""""

    assert check_rotation({}, date(2026, 9, 9), quiet) == 1

