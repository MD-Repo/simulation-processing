"""Tests for the drain's retry backoff (TODO 36)

These need a real Postgres and SKIP without one, which is deliberate: the
backoff curve and the eligibility rule live in SQL -- an interval expression
over num_attempts inside claim_job's subquery -- so there is no pure-Python
version of them to test. Mirroring the arithmetic in Python and asserting
against that would only prove the mirror agrees with itself.

Everything runs inside a transaction that is rolled back, under a `server`
value no cron line uses, so the live drain can never see these rows even
momentarily.

What is being pinned: before 2026-08-11 a requeued job kept its created_at and
so returned to the head of the queue, was re-claimed on the very next tick, and
burned all five attempts in five minutes. Measured on ticket 2042 -- attempts
2-5 at 17:17, 17:18, 17:19, 17:20, terminal by 17:21 -- against a transient
that lasted about an hour.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

psycopg2 = pytest.importorskip("psycopg2")
import psycopg2.extras  # noqa: E402

import drain_process_queue as d  # noqa: E402

# No cron line uses this, so a row that somehow escaped the rollback would
# still never be claimed by the real drain.
TEST_SERVER = "backofftest"


# --------------------------------------------------
def _dsn():
    """STAGING_DSN, loading the .env beside the scripts if needed"""

    if os.environ.get("STAGING_DSN"):
        return os.environ["STAGING_DSN"]
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return os.environ.get("STAGING_DSN")


# --------------------------------------------------
@pytest.fixture
def cur():
    """A cursor on staging whose work is always rolled back"""

    dsn = _dsn()
    if not dsn:
        pytest.skip("no STAGING_DSN; backoff logic is SQL and needs a database")

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    try:
        yield cursor
    finally:
        conn.rollback()
        conn.close()


# --------------------------------------------------
@pytest.fixture
def ticket_id(cur):
    cur.execute("select id from md_ticket order by id limit 1")
    row = cur.fetchone()
    if not row:
        pytest.skip("no md_ticket rows to reference")
    return row[0]


# --------------------------------------------------
def add_job(cur, ticket_id, num_attempts, finished_minutes_ago, age_minutes=0):
    """A pending job whose last attempt ended `finished_minutes_ago` (or never)"""

    cur.execute(
        """
        insert into md_process_job (ticket_id, server, status, num_attempts,
                                    finished_at, created_at)
        values (%s, %s, 'pending', %s,
                case when %s is null then null
                     else now() - (%s * interval '1 minute') end,
                now() - (%s * interval '1 minute'))
        returning id
        """,
        (ticket_id, TEST_SERVER, num_attempts, finished_minutes_ago,
         finished_minutes_ago, age_minutes),
    )
    return cur.fetchone()[0]


# --------------------------------------------------
def claim(cur):
    row = d.claim_job(cur, TEST_SERVER)
    return row["id"] if row else None


# --------------------------------------------------
def test_never_run_job_is_claimable(cur, ticket_id):
    """finished_at is null for a job that has never run, and must not gate it"""

    job = add_job(cur, ticket_id, num_attempts=0, finished_minutes_ago=None)
    assert claim(cur) == job


# --------------------------------------------------
@pytest.mark.parametrize(
    "num_attempts,minutes_ago,claimable,why",
    [
        (1, 1, False, "first retry waits 5 minutes"),
        (1, 6, True, "first retry after 5 minutes"),
        (2, 10, False, "second retry waits 20 minutes"),
        (2, 25, True, "second retry after 20 minutes"),
        (3, 30, False, "third retry waits 80 minutes"),
        (3, 90, True, "third retry after 80 minutes"),
        (4, 100, False, "fourth retry waits the 180 minute cap"),
        (4, 200, True, "fourth retry after the cap"),
        (9, 200, True, "the cap holds for any attempt count"),
    ],
)
def test_backoff_curve(cur, ticket_id, num_attempts, minutes_ago, claimable, why):
    """5, 20, 80, then capped at 180 -- ~4.75h to exhaust the budget

    The 4.75h matters: the faults on record are ~1h (the 08-10 read fault) and
    4.5h (the 08-04 CyVerse outage), and the old behaviour covered 5 minutes.
    """

    job = add_job(cur, ticket_id, num_attempts, minutes_ago)
    assert (claim(cur) == job) is claimable, why


# --------------------------------------------------
def test_backed_off_job_does_not_block_a_newer_one(cur, ticket_id):
    """A ticket stuck on a transient must stop monopolising the queue head

    claim_job orders by created_at, so without the eligibility predicate the
    oldest failing row is re-offered ahead of every newer ticket on every tick.
    """

    add_job(cur, ticket_id, num_attempts=2, finished_minutes_ago=5, age_minutes=600)
    newer = add_job(cur, ticket_id, num_attempts=0, finished_minutes_ago=None,
                    age_minutes=1)

    assert claim(cur) == newer


# --------------------------------------------------
def test_requeue_sets_the_pacing_field_and_is_not_instantly_reclaimable(
    cur, ticket_id
):
    """The whole defect in one test

    finished_at is what paces the retry, so a requeue that left it null (as it
    did before 08-11) puts the job straight back at the head of the queue.
    """

    job_id = add_job(cur, ticket_id, num_attempts=1, finished_minutes_ago=None)
    cur.execute("update md_process_job set status = 'running' where id = %s", (job_id,))

    d.requeue_for_retry(
        cur,
        {"id": job_id, "ticket_id": ticket_id, "num_attempts": 1},
        1,
        "NetworkException: test",
        lambda _m: None,
    )

    cur.execute(
        "select status, num_attempts, started_at, finished_at "
        "from md_process_job where id = %s",
        (job_id,),
    )
    row = cur.fetchone()

    assert row["status"] == "pending"
    assert row["num_attempts"] == 2
    assert row["started_at"] is None, "started_at must still mean 'is it running'"
    assert row["finished_at"] is not None, "this is what paces the retry"
    assert claim(cur) is None, "must not be re-claimable on the very next tick"


# --------------------------------------------------
def test_rows_are_rolled_back(cur, ticket_id):
    """The suite must not leave queue rows behind on a live database"""

    add_job(cur, ticket_id, num_attempts=0, finished_minutes_ago=None)
    cur.connection.rollback()

    cur.execute(
        "select count(*) from md_process_job where server = %s", (TEST_SERVER,)
    )
    assert cur.fetchone()[0] == 0
