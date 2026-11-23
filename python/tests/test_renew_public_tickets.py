"""Tests for renew_public_tickets.py and shared_tickets.py (MDR-115)

The rule under test: every ticket string handed out must outlive the
download token that holds it, and a failure part way must never leave a
ticket with no expiry, or a row naming a ticket that ends sooner than the
row says.

iRODS is a fake catalog (a dict). Most tests use a fake cursor that applies
the two UPDATEs the way Postgres would; the SQL itself is run for real in
the last group, against TICKET_TEST_DSN (skipped without it), e.g. a
throwaway postgres:15 container:

    TICKET_TEST_DSN=postgresql://user:pass@127.0.0.1:55487/mdrepo \\
        uv run pytest tests/test_renew_public_tickets.py
"""

import datetime
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import renew_public_tickets as rpt  # noqa: E402
import shared_tickets as st  # noqa: E402
from irods.exception import CAT_TICKET_INVALID, SYS_SOCK_READ_TIMEDOUT  # noqa: E402

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 11, 23, 17, 0, 0, tzinfo=UTC)
DAY = datetime.timedelta(days=1)
SCRIPT = str(Path(__file__).resolve().parent.parent / "renew_public_tickets.py")


# --------------------------------------------------
class Catalog:
    """A fake iRODS ticket catalog, and a Ticket class bound to it"""

    def __init__(self):
        self.tickets: Dict[str, dict] = {}
        self.fail: Set[str] = set()  # "issue", "expire", "uses", "delete"
        self.calls = 0
        self._next = 0
        catalog = self

        class FakeTicket:
            def __init__(self, session, ticket=""):
                if not ticket:
                    catalog._next += 1
                    ticket = f"new{catalog._next:04d}xxxxxxxxx"
                self.string = self.ticket = ticket

            def _check(self, op):
                catalog.calls += 1
                if op in catalog.fail:
                    raise SYS_SOCK_READ_TIMEDOUT()

            def issue(self, permission, path):
                self._check("issue")
                catalog.tickets[self.string] = {
                    "permission": permission,
                    "path": path,
                    "expire": None,
                    "uses": None,
                }
                return self

            def modify(self, what, value):
                self._check(what)
                if self.string not in catalog.tickets:
                    raise CAT_TICKET_INVALID()
                catalog.tickets[self.string][what] = value
                return self

            def delete(self):
                self._check("delete")
                if self.string not in catalog.tickets:
                    raise CAT_TICKET_INVALID()
                del catalog.tickets[self.string]
                return self

        self.Ticket = FakeTicket

    def add(self, string, expire=None):
        self.tickets[string] = {"permission": "read", "path": "x", "expire": expire, "uses": "0"}

    def expiry(self, string) -> Optional[datetime.datetime]:
        value = self.tickets[string]["expire"]
        return None if value is None else datetime.datetime.fromtimestamp(int(value), UTC)


# --------------------------------------------------
class Row(dict):
    pass


class FakeCursor:
    """Applies renew()'s two UPDATEs to in-memory rows, as Postgres would"""

    def __init__(self, rows):
        self.rows = {r["id"]: Row(r) for r in rows}
        self.rowcount = 0
        self.statements = []

    @staticmethod
    def released(r):
        return r["is_public"] and not r["is_embargoed"] and not r["is_placeholder"]

    def execute(self, sql, params):
        self.statements.append(sql)
        self.rowcount = 0
        if "irods_ticket = %s," in sql:  # step 2: store the new ticket
            new, new_end, sim_id, old, old_end = params
            r = self.rows.get(sim_id)
            if r and self.released(r) and r["irods_ticket"] == old and r["irods_ticket_expires_at"] == old_end:
                r["irods_ticket"], r["irods_ticket_expires_at"] = new, new_end
                self.rowcount = 1
        else:  # step 1: record the old ticket's expiry
            grace_end, sim_id, old = params
            r = self.rows.get(sim_id)
            if r and r["irods_ticket"] == old and r["irods_ticket_expires_at"] is None:
                r["irods_ticket_expires_at"] = grace_end
                self.rowcount = 1


def sim(sim_id, ticket=None, expires_at=None, **flags):
    base = {"is_public": True, "is_embargoed": False, "is_placeholder": False}
    base.update(flags)
    return {"id": sim_id, "irods_ticket": ticket, "irods_ticket_expires_at": expires_at, **base}


def args(**over):
    base = dict(
        server="prod",
        dry_run=False,
        limit=10_000,
        sim_ids=[],
        lifetime_days=30,
        renew_within_days=16,
        old_grace_days=11,
        token_days=10,
        irods_env="/dev/null",
    )
    base.update(over)
    return rpt.Args(**base)


@pytest.fixture
def catalog(monkeypatch):
    cat = Catalog()
    monkeypatch.setattr(st, "Ticket", cat.Ticket)
    monkeypatch.setattr(rpt, "Ticket", cat.Ticket)
    return cat


def run(catalog, row, **over):
    cur = FakeCursor([row])
    due = rpt.Due(row["id"], row["irods_ticket"], row["irods_ticket_expires_at"])
    outcome = rpt.renew(None, cur, due, args(**over), NOW)
    return outcome, cur.rows[row["id"]]


# ==================================================
# shared_tickets: pure helpers
# ==================================================
def test_defaults_are_safe_together():
    assert rpt.check_days(st.LIFETIME_DAYS, st.RENEW_WITHIN_DAYS, st.OLD_TICKET_GRACE_DAYS, st.TOKEN_DAYS) is None
    assert (st.LIFETIME_DAYS, st.RENEW_WITHIN_DAYS, st.OLD_TICKET_GRACE_DAYS, st.TOKEN_DAYS) == (30, 16, 11, 10)


def test_expire_value_is_epoch_seconds():
    when = datetime.datetime(2026, 12, 23, 7, 0, 0, tzinfo=UTC)
    assert st.expire_value(when) == "1798009200"
    mst = datetime.timezone(datetime.timedelta(hours=-7))
    assert st.expire_value(when.astimezone(mst)) == "1798009200"


def test_expire_value_refuses_naive_datetime():
    with pytest.raises(ValueError, match="time-zone-aware"):
        st.expire_value(datetime.datetime(2026, 12, 23, 7, 0, 0))


def test_expire_value_matches_irods_client_parsing():
    # The client turns a string into int() first: epoch seconds pass through.
    from irods.ticket import get_epoch_seconds

    when = NOW + 30 * DAY
    assert get_epoch_seconds(st.expire_value(when)) == int(when.timestamp())


def test_sim_path():
    assert st.sim_path("prod", 5532) == "/iplant/home/shared/mdrepo/prod/release/MDR00005532"
    assert st.sim_path("staging", 107428) == "/iplant/home/shared/mdrepo/staging/release/MDR00107428"


def test_label_hides_most_of_the_string():
    assert st.label("abcdefghijklmno") == "abcd…"
    assert st.label(None) == "(none)"
    assert st.label("") == "(none)"


def test_released_sql_names_all_three_flags():
    for flag in ("is_public", "not is_embargoed", "not is_placeholder"):
        assert flag in st.RELEASED_SQL


# ==================================================
# mint_read_ticket
# ==================================================
def test_mint_sets_expiry_and_unlimited_uses(catalog):
    string = st.mint_read_ticket(None, "/r/MDR1", NOW + 30 * DAY)
    t = catalog.tickets[string]
    assert t["permission"] == "read"
    assert t["path"] == "/r/MDR1"
    assert t["uses"] == "0"
    assert catalog.expiry(string) == NOW + 30 * DAY


@pytest.mark.parametrize("op", ["expire", "uses"])
def test_mint_failure_after_issue_deletes_the_ticket(catalog, op):
    catalog.fail = {op}
    with pytest.raises(SYS_SOCK_READ_TIMEDOUT):
        st.mint_read_ticket(None, "/r/MDR1", NOW + 30 * DAY)
    assert catalog.tickets == {}


def test_mint_failure_when_delete_also_fails_raises_the_first_error(catalog):
    catalog.fail = {"expire", "delete"}
    with pytest.raises(SYS_SOCK_READ_TIMEDOUT):
        st.mint_read_ticket(None, "/r/MDR1", NOW + 30 * DAY)


def test_mint_issue_failure_leaves_nothing(catalog):
    catalog.fail = {"issue"}
    with pytest.raises(SYS_SOCK_READ_TIMEDOUT):
        st.mint_read_ticket(None, "/r/MDR1", NOW + 30 * DAY)
    assert catalog.tickets == {}


def test_set_expiry_on_missing_ticket_raises_ticket_invalid(catalog):
    with pytest.raises(CAT_TICKET_INVALID):
        st.set_expiry(None, "gone", NOW)


# ==================================================
# check_days
# ==================================================
@pytest.mark.parametrize(
    "lifetime,renew,grace,token,fragment",
    [
        (30, 10, 11, 10, "--renew-within-days '10' must be more than --token-days '10'"),
        (30, 9, 11, 10, "--renew-within-days '9'"),
        (16, 16, 11, 10, "--lifetime-days '16' must be more than --renew-within-days '16'"),
        (12, 16, 11, 10, "--lifetime-days '12'"),
        (30, 16, 9, 10, "--old-grace-days '9' must be at least --token-days '10'"),
        (30, 16, 11, 0, "--token-days '0' must be positive"),
    ],
)
def test_check_days_refuses_unsafe(lifetime, renew, grace, token, fragment):
    assert fragment in rpt.check_days(lifetime, renew, grace, token)


@pytest.mark.parametrize("lifetime,renew,grace,token", [(30, 16, 11, 10), (17, 16, 10, 10), (60, 30, 20, 14)])
def test_check_days_accepts_safe(lifetime, renew, grace, token):
    assert rpt.check_days(lifetime, renew, grace, token) is None


# ==================================================
# renew(): the old permanent ticket
# ==================================================
def test_old_permanent_ticket_gets_grace_and_is_replaced(catalog):
    catalog.add("oldpermanent123")
    outcome, row = run(catalog, sim(5532, "oldpermanent123"))
    assert outcome == rpt.Outcome(True, "renewed")
    # old: still in iRODS, now ends in 11 days
    assert catalog.expiry("oldpermanent123") == NOW + 11 * DAY
    # new: stored with its 30-day expiry, same instant as in iRODS
    assert row["irods_ticket"] != "oldpermanent123"
    assert row["irods_ticket_expires_at"] == NOW + 30 * DAY
    assert catalog.expiry(row["irods_ticket"]) == NOW + 30 * DAY
    assert catalog.tickets[row["irods_ticket"]]["path"].endswith("/prod/release/MDR00005532")


def test_old_ticket_outlives_any_token_holding_it(catalog):
    # A token made just before the run holds the old string for 10 days.
    catalog.add("oldpermanent123")
    run(catalog, sim(1, "oldpermanent123"))
    assert catalog.expiry("oldpermanent123") - NOW > datetime.timedelta(days=10)


def test_old_ticket_grace_recorded_even_if_new_ticket_fails(catalog):
    catalog.add("oldpermanent123")
    catalog.fail = {"issue"}
    outcome, row = run(catalog, sim(1, "oldpermanent123"))
    assert not outcome.ok
    assert "making the new ticket" in outcome.message
    assert "SYS_SOCK_READ_TIMEDOUT" in outcome.message
    # the row tells the truth: same string, ends in 11 days
    assert row["irods_ticket"] == "oldpermanent123"
    assert row["irods_ticket_expires_at"] == NOW + 11 * DAY
    assert catalog.expiry("oldpermanent123") == NOW + 11 * DAY


def test_rerun_after_partial_failure_finishes_the_job(catalog):
    catalog.add("oldpermanent123")
    catalog.fail = {"issue"}
    _, row = run(catalog, sim(1, "oldpermanent123"))
    catalog.fail = set()
    # next night: the row now has an expiry within 16 days, so it is due
    outcome, row2 = run(catalog, sim(1, row["irods_ticket"], row["irods_ticket_expires_at"]))
    assert outcome.ok
    assert row2["irods_ticket"] != "oldpermanent123"
    assert row2["irods_ticket_expires_at"] == NOW + 30 * DAY
    # the old ticket's grace was not moved again
    assert catalog.expiry("oldpermanent123") == NOW + 11 * DAY


def test_old_ticket_expiry_failure_changes_nothing(catalog):
    catalog.add("oldpermanent123")
    catalog.fail = {"expire"}
    outcome, row = run(catalog, sim(1, "oldpermanent123"))
    assert not outcome.ok
    assert "setting the old ticket's expiry" in outcome.message
    assert row["irods_ticket"] == "oldpermanent123"
    assert row["irods_ticket_expires_at"] is None
    assert catalog.expiry("oldpermanent123") is None
    assert list(catalog.tickets) == ["oldpermanent123"]


def test_old_ticket_missing_in_irods_is_recorded_as_ended(catalog):
    # Nobody can use it, so the row must not claim 11 days.
    catalog.fail = {"issue"}
    outcome, row = run(catalog, sim(1, "alreadygone1234"))
    assert not outcome.ok
    assert row["irods_ticket_expires_at"] == NOW


def test_old_ticket_missing_in_irods_is_replaced(catalog):
    outcome, row = run(catalog, sim(1, "alreadygone1234"))
    assert outcome.ok
    assert row["irods_ticket"] != "alreadygone1234"
    assert row["irods_ticket_expires_at"] == NOW + 30 * DAY


def test_custom_grace_and_lifetime(catalog):
    catalog.add("oldpermanent123")
    _, row = run(catalog, sim(1, "oldpermanent123"), old_grace_days=14, lifetime_days=45)
    assert catalog.expiry("oldpermanent123") == NOW + 14 * DAY
    assert row["irods_ticket_expires_at"] == NOW + 45 * DAY


# ==================================================
# renew(): expiring tickets and missing ones
# ==================================================
def test_expiring_ticket_is_replaced_and_left_alone(catalog):
    end = NOW + 15 * DAY
    catalog.add("expiring1234567", st.expire_value(end))
    outcome, row = run(catalog, sim(1, "expiring1234567", end))
    assert outcome.ok
    assert row["irods_ticket_expires_at"] == NOW + 30 * DAY
    # the old one keeps its own expiry; tokens holding it keep working
    assert catalog.expiry("expiring1234567") == end


def test_expired_ticket_is_replaced(catalog):
    outcome, row = run(catalog, sim(1, "expired12345678", NOW - DAY))
    assert outcome.ok
    assert row["irods_ticket_expires_at"] == NOW + 30 * DAY


def test_no_ticket_gets_one(catalog):
    outcome, row = run(catalog, sim(1))
    assert outcome.ok
    assert row["irods_ticket"].startswith("new")
    assert row["irods_ticket_expires_at"] == NOW + 30 * DAY


def test_catalog_calls_per_simulation(catalog):
    # 4 for an old ticket (expire old; issue, expire, uses), 3 otherwise
    catalog.add("oldpermanent123")
    run(catalog, sim(1, "oldpermanent123"))
    assert catalog.calls == 4
    catalog.calls = 0
    run(catalog, sim(2, "expiring1234567", NOW + DAY))
    assert catalog.calls == 3


# ==================================================
# renew(): the row changed under the run
# ==================================================
class RacingCursor(FakeCursor):
    """Changes the row just before step 2, as the app or an admin might"""

    def __init__(self, rows, change):
        super().__init__(rows)
        self.change = change

    def execute(self, sql, params):
        if "irods_ticket = %s," in sql:
            self.change(self.rows[params[2]])
        super().execute(sql, params)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(irods_ticket="appminted123456", irods_ticket_expires_at=NOW + 30 * DAY),
        lambda r: r.update(is_embargoed=True),
        lambda r: r.update(is_public=False),
        lambda r: r.update(is_placeholder=True),
        lambda r: r.update(irods_ticket=None, irods_ticket_expires_at=None),
    ],
    ids=["app-minted", "embargoed", "unapproved", "placeholder", "revoked"],
)
def test_row_changed_meanwhile_new_ticket_deleted(catalog, change):
    end = NOW + 2 * DAY
    catalog.add("expiring1234567", st.expire_value(end))
    cur = RacingCursor([sim(1, "expiring1234567", end)], change)
    before = dict(cur.rows[1])
    outcome = rpt.renew(None, cur, rpt.Due(1, "expiring1234567", end), args(), NOW)
    assert outcome == rpt.Outcome(True, "row changed meanwhile; left as it is")
    # our new ticket is gone from iRODS; only the old one is left
    assert list(catalog.tickets) == ["expiring1234567"]
    # and the row is as the other writer left it, never our new string
    assert cur.rows[1] != before
    assert not (cur.rows[1]["irods_ticket"] or "").startswith("new")


def test_row_changed_and_delete_fails_is_a_failure(catalog):
    end = NOW + 2 * DAY
    catalog.add("expiring1234567", st.expire_value(end))

    def change(r):
        r.update(irods_ticket="appminted123456")
        catalog.fail = {"delete"}

    cur = RacingCursor([sim(1, "expiring1234567", end)], change)
    outcome = rpt.renew(None, cur, rpt.Due(1, "expiring1234567", end), args(), NOW)
    assert not outcome.ok
    assert "could not be deleted" in outcome.message
    # the message names the leftover ticket by its first four characters only
    leftover = [s for s in catalog.tickets if s.startswith("new")]
    assert len(leftover) == 1
    assert f"{leftover[0][:4]}…" in outcome.message
    assert leftover[0] not in outcome.message


# ==================================================
# describe(): log labels never show a whole string
# ==================================================
def test_describe_labels():
    assert rpt.describe(rpt.Due(5532, None, None)) == "MDR00005532 (no ticket)"
    assert rpt.describe(rpt.Due(5532, "abcdefghijklmno", None)) == "MDR00005532 (old permanent ticket abcd…)"
    assert (
        rpt.describe(rpt.Due(5532, "abcdefghijklmno", datetime.datetime(2026, 12, 9, 17, 0, tzinfo=UTC)))
        == "MDR00005532 (ticket abcd… expires 2026-12-09 17:00Z)"
    )
    for due in (rpt.Due(1, "abcdefghijklmno", None), rpt.Due(1, "abcdefghijklmno", NOW)):
        assert "abcdefghijklmno" not in rpt.describe(due)


# ==================================================
# the command line
# ==================================================
def test_cli_defaults(tmp_path):
    env = tmp_path / "irods.json"
    env.write_text("{}")
    a = rpt.get_args(["--irods-env", str(env)])
    assert (a.server, a.dry_run, a.limit, a.sim_ids) == ("prod", False, 10_000, [])
    assert (a.lifetime_days, a.renew_within_days, a.old_grace_days, a.token_days) == (30, 16, 11, 10)


def test_cli_options(tmp_path):
    env = tmp_path / "irods.json"
    env.write_text("{}")
    a = rpt.get_args(["--irods-env", str(env), "-S", "staging", "-n", "-l", "25", "-s", "5", "6"])
    assert (a.server, a.dry_run, a.limit, a.sim_ids) == ("staging", True, 25, [5, 6])


@pytest.mark.parametrize(
    "argv,fragment",
    [
        (["-l", "0"], "--limit '0' must be positive"),
        (["--renew-within-days", "10"], "must be more than --token-days"),
        (["--lifetime-days", "16"], "must be more than --renew-within-days"),
        (["--old-grace-days", "5"], "must be at least --token-days"),
        (["-S", "dev"], "invalid choice"),
    ],
)
def test_cli_refuses(tmp_path, capsys, argv, fragment):
    env = tmp_path / "irods.json"
    env.write_text("{}")
    with pytest.raises(SystemExit) as e:
        rpt.get_args(["--irods-env", str(env)] + argv)
    assert e.value.code == 2
    assert fragment in capsys.readouterr().err


def test_cli_refuses_missing_irods_env(capsys):
    with pytest.raises(SystemExit):
        rpt.get_args(["--irods-env", "/no/such/file.json"])
    assert "Invalid --irods-env '/no/such/file.json'" in capsys.readouterr().err


def test_cli_help_runs():
    import subprocess

    out = subprocess.run([sys.executable, SCRIPT, "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "--renew-within-days" in out.stdout


# ==================================================
# The SQL, on a real Postgres (TICKET_TEST_DSN)
# ==================================================
@pytest.fixture
def pg():
    dsn = os.environ.get("TICKET_TEST_DSN")
    if not dsn:
        pytest.skip("no TICKET_TEST_DSN; the SQL tests need a throwaway Postgres")
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("create schema if not exists renew_ticket_test")
    cur.execute("set search_path to renew_ticket_test")
    cur.execute("drop table if exists md_simulation")
    cur.execute(
        """
        create table md_simulation (
            id integer primary key,
            is_public boolean not null,
            is_embargoed boolean not null,
            is_placeholder boolean not null,
            irods_ticket varchar(255) unique,
            irods_ticket_expires_at timestamptz
        )
        """
    )
    yield cur
    cur.execute("drop schema renew_ticket_test cascade")
    conn.close()


def insert(cur, *rows):
    for r in rows:
        cur.execute(
            "insert into md_simulation values (%(id)s, %(is_public)s, %(is_embargoed)s, "
            "%(is_placeholder)s, %(irods_ticket)s, %(irods_ticket_expires_at)s)",
            r,
        )


def fetch(cur, sim_id):
    cur.execute("select irods_ticket, irods_ticket_expires_at from md_simulation where id = %s", (sim_id,))
    return cur.fetchone()


def test_sql_finds_due_in_order(pg):
    insert(
        pg,
        sim(1, "permanent000001"),                         # old permanent
        sim(2, "fresh0000000002", NOW + 25 * DAY),          # not due
        sim(3, "soon00000000003", NOW + 10 * DAY),          # due
        sim(4, "expired00000004", NOW - DAY),               # due first
        sim(5),                                             # no ticket
        sim(6, "hidden000000006", is_embargoed=True),       # hidden
        sim(7, "unapproved00007", is_public=False),         # hidden
        sim(8, "placeholder0008", is_placeholder=True),     # hidden
        sim(9, "edge00000000009", NOW + 16 * DAY),          # exactly on the line: due
        sim(10, "edge00000000010", NOW + 16 * DAY + datetime.timedelta(seconds=1)),  # not due
    )
    renew_before = NOW + 16 * DAY
    due = rpt.find_due(pg, renew_before, 100, [])
    assert [d.id for d in due] == [4, 3, 9, 1, 5]
    assert rpt.count_due(pg, renew_before, []) == 5
    assert due[3] == rpt.Due(1, "permanent000001", None)


def test_sql_limit_takes_most_urgent(pg):
    insert(pg, sim(1, "permanent000001"), sim(2, "soon00000000002", NOW + DAY), sim(3))
    due = rpt.find_due(pg, NOW + 16 * DAY, 1, [])
    assert [d.id for d in due] == [2]
    assert rpt.count_due(pg, NOW + 16 * DAY, []) == 3


def test_sql_sim_ids_narrow(pg):
    insert(pg, sim(1, "permanent000001"), sim(2, "permanent000002"), sim(3, "hidden000000003", is_embargoed=True))
    assert [d.id for d in rpt.find_due(pg, NOW + 16 * DAY, 10, [2, 3])] == [2]
    assert rpt.count_due(pg, NOW + 16 * DAY, [2, 3]) == 1


def test_sql_renew_old_permanent(pg, catalog):
    insert(pg, sim(1, "permanent000001"))
    catalog.add("permanent000001")
    (due,) = rpt.find_due(pg, NOW + 16 * DAY, 10, [])
    assert rpt.renew(None, pg, due, args(), NOW) == rpt.Outcome(True, "renewed")
    string, end = fetch(pg, 1)
    assert string != "permanent000001"
    assert end == NOW + 30 * DAY
    assert catalog.expiry("permanent000001") == NOW + 11 * DAY
    # not due again tonight
    assert rpt.find_due(pg, NOW + 16 * DAY, 10, []) == []


def test_sql_partial_failure_records_grace(pg, catalog):
    insert(pg, sim(1, "permanent000001"))
    catalog.add("permanent000001")
    catalog.fail = {"issue"}
    (due,) = rpt.find_due(pg, NOW + 16 * DAY, 10, [])
    assert not rpt.renew(None, pg, due, args(), NOW).ok
    assert fetch(pg, 1) == ("permanent000001", NOW + 11 * DAY)


def test_sql_row_changed_meanwhile(pg, catalog):
    insert(pg, sim(1, "soon00000000001", NOW + DAY))
    (due,) = rpt.find_due(pg, NOW + 16 * DAY, 10, [])
    pg.execute(
        "update md_simulation set irods_ticket = 'appminted000001', irods_ticket_expires_at = %s where id = 1",
        (NOW + 30 * DAY,),
    )
    outcome = rpt.renew(None, pg, due, args(), NOW)
    assert outcome == rpt.Outcome(True, "row changed meanwhile; left as it is")
    assert fetch(pg, 1) == ("appminted000001", NOW + 30 * DAY)
    assert catalog.tickets == {}


def test_sql_hidden_meanwhile(pg, catalog):
    insert(pg, sim(1, "soon00000000001", NOW + DAY))
    (due,) = rpt.find_due(pg, NOW + 16 * DAY, 10, [])
    pg.execute("update md_simulation set is_embargoed = true where id = 1")
    assert rpt.renew(None, pg, due, args(), NOW).message == "row changed meanwhile; left as it is"
    assert fetch(pg, 1) == ("soon00000000001", NOW + DAY)
    assert catalog.tickets == {}


def test_sql_no_ticket(pg, catalog):
    insert(pg, sim(1))
    (due,) = rpt.find_due(pg, NOW + 16 * DAY, 10, [])
    assert rpt.renew(None, pg, due, args(), NOW).ok
    string, end = fetch(pg, 1)
    assert string.startswith("new")
    assert end == NOW + 30 * DAY
