#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-11-23
Purpose: Renew the shared read tickets on released simulations, and retire
         the old ones that never expire (MDR-115, md-repo-app#497).

         Each released simulation has one unlimited-use iRODS read ticket on
         its MDR* collection (md_simulation.irods_ticket), handed to everyone
         who downloads it. Until MDR-115 these never expired, so anyone who
         had collected the strings could read as anonymous forever. Now each
         ticket has an expiry (md_simulation.irods_ticket_expires_at) and
         this job, run nightly, replaces it before it gets near the end.

         For each released simulation whose ticket has no expiry, has
         --renew-within days or fewer left, or is missing:

           1. An old ticket with no expiry gets one --old-grace days out,
              written to irods_ticket_expires_at at once. That is longer
              than a download token lives (10 days), so every token that
              already holds the string keeps working until it runs out.
              Every string handed out before MDR-115 stops working then.
           2. A new ticket is made that expires in --lifetime days and is
              stored with its expiry. The old ticket is not deleted; it
              expires on its own.

         A new download token therefore always gets a string with more
         than --renew-within days left, longer than the token's life. Step 1
         is written to the database before step 2, so if step 2 fails the
         row still names a ticket with a known expiry, and md-repo-app mints
         a replacement itself rather than hand out a string that would end
         before its token (irods_utils.get_download_ticket).

         The rows are written only while the simulation is still released
         and still names the ticket this run read, so a ticket the app
         minted in the meantime is never overwritten; this run's new ticket
         is then deleted.

         --limit caps the simulations handled per run (soonest expiry
         first, then old permanent tickets by id), to spread the load on
         CyVerse's catalog. Each simulation costs 3-4 catalog calls.

         Ticket strings are credentials: the log shows only their first four
         characters.

         Exit status: 0 when every simulation was handled, 1 when any failed.
"""

import argparse
import datetime
import os
import ssl
import sys
import time
from typing import List, NamedTuple, Optional

import psycopg2
import psycopg2.extras
from dotenv import dotenv_values
from irods.exception import CAT_TICKET_INVALID
from irods.session import iRODSSession
from irods.ticket import Ticket

import shared_tickets as st
from common import describe_exc, stamp


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    dry_run: bool
    limit: int
    sim_ids: List[int]
    lifetime_days: int
    renew_within_days: int
    old_grace_days: int
    token_days: int
    irods_env: str


class Due(NamedTuple):
    """A released simulation whose ticket needs work"""

    id: int
    ticket: Optional[str]
    expires_at: Optional[datetime.datetime]


class Outcome(NamedTuple):
    """What happened to one simulation"""

    ok: bool
    message: str


# --------------------------------------------------
def get_args(argv: Optional[List[str]] = None) -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Renew shared read tickets on released simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-S",
        "--server",
        help="Server",
        metavar="STR",
        choices=["prod", "staging"],
        default="prod",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        help="Report what would be done without changing anything",
        action="store_true",
    )

    parser.add_argument(
        "-l",
        "--limit",
        help="Most simulations to handle in one run",
        metavar="INT",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "-s",
        "--sim-id",
        help="Only these simulation ids",
        metavar="INT",
        type=int,
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--lifetime-days",
        help="Days a new ticket lives",
        metavar="INT",
        type=int,
        default=st.LIFETIME_DAYS,
    )

    parser.add_argument(
        "--renew-within-days",
        help="Renew a ticket with this many days or fewer left",
        metavar="INT",
        type=int,
        default=st.RENEW_WITHIN_DAYS,
    )

    parser.add_argument(
        "--old-grace-days",
        help="Days an old permanent ticket keeps working",
        metavar="INT",
        type=int,
        default=st.OLD_TICKET_GRACE_DAYS,
    )

    parser.add_argument(
        "--token-days",
        help="Days a download token lives (md-repo-app TICKET_EXPIRATION_MINUTES)",
        metavar="INT",
        type=int,
        default=st.TOKEN_DAYS,
    )

    parser.add_argument(
        "--irods-env",
        help="iRODS environment file",
        metavar="FILE",
        default=os.environ.get(
            "IRODS_ENVIRONMENT_FILE",
            os.path.expanduser("~/.irods/irods_environment.json"),
        ),
    )

    args = parser.parse_args(argv)

    if error := check_days(
        args.lifetime_days, args.renew_within_days, args.old_grace_days, args.token_days
    ):
        parser.error(error)

    if args.limit < 1:
        parser.error(f"--limit '{args.limit}' must be positive")

    if not os.path.isfile(args.irods_env):
        parser.error(f"Invalid --irods-env '{args.irods_env}'")

    return Args(
        server=args.server,
        dry_run=args.dry_run,
        limit=args.limit,
        sim_ids=args.sim_id,
        lifetime_days=args.lifetime_days,
        renew_within_days=args.renew_within_days,
        old_grace_days=args.old_grace_days,
        token_days=args.token_days,
        irods_env=args.irods_env,
    )


# --------------------------------------------------
def check_days(lifetime: int, renew_within: int, old_grace: int, token: int) -> Optional[str]:
    """Why these day counts are unsafe, or None

    Every string handed out must outlive the token that holds it:
    - a ticket is renewed while it still has more days than a token lives;
    - a new ticket must have more days than the renewal point, or it would
      be due again the moment it is made;
    - an old ticket's grace must cover a token made just before it.
    """

    if token < 1:
        return f"--token-days '{token}' must be positive"
    if renew_within <= token:
        return (
            f"--renew-within-days '{renew_within}' must be more than "
            f"--token-days '{token}'"
        )
    if lifetime <= renew_within:
        return (
            f"--lifetime-days '{lifetime}' must be more than "
            f"--renew-within-days '{renew_within}'"
        )
    if old_grace < token:
        return (
            f"--old-grace-days '{old_grace}' must be at least "
            f"--token-days '{token}'"
        )
    return None


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    env = dotenv_values()
    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dsn = env.get(env_key, os.environ.get(env_key))
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    now = st.now_utc()
    renew_before = now + datetime.timedelta(days=args.renew_within_days)
    total = count_due(cur, renew_before, args.sim_ids)
    due = find_due(cur, renew_before, args.limit, args.sim_ids)
    old = sum(1 for sim in due if sim.ticket and sim.expires_at is None)
    mode = "DRY RUN" if args.dry_run else "RENEWING"
    print(
        f"{stamp()} {mode}: {total:,} released simulation(s) on {args.server} "
        f"due; this run takes {len(due):,} ({old:,} old permanent, "
        f"{sum(1 for sim in due if not sim.ticket):,} with no ticket)",
        flush=True,
    )

    if args.dry_run:
        for sim in due:
            print(f"  would renew {describe(sim)}", flush=True)
        print(f"{stamp()} Done (dry run)", flush=True)
        return

    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    failed = renewed = 0
    start = time.monotonic()

    with iRODSSession(irods_env_file=args.irods_env, ssl_context=ssl_context) as session:
        for n, sim in enumerate(due, 1):
            outcome = renew(session, cur, sim, args, st.now_utc())
            if outcome.ok:
                renewed += 1
                if outcome.message != "renewed":
                    print(f"  {describe(sim)}: {outcome.message}", flush=True)
            else:
                failed += 1
                print(f"  FAILED {describe(sim)}: {outcome.message}", flush=True)

            if n % 1000 == 0 or n == len(due):
                elapsed = time.monotonic() - start
                print(
                    f"{stamp()}   {n:,} of {len(due):,}: {renewed:,} renewed, "
                    f"{failed:,} failed, {elapsed / n:.2f} s each",
                    flush=True,
                )

    print(
        f"{stamp()} Done: {renewed:,} renewed, {failed:,} failure(s), "
        f"{max(total - len(due), 0):,} left for later runs",
        flush=True,
    )
    sys.exit(1 if failed else 0)


# --------------------------------------------------
def due_where(sim_ids: List[int]) -> str:
    """WHERE clause for released simulations whose ticket needs work"""

    sql = f"""
        where  {st.RELEASED_SQL}
        and    (irods_ticket is null
                or irods_ticket_expires_at is null
                or irods_ticket_expires_at <= %(renew_before)s)
    """
    if sim_ids:
        sql += " and id = any(%(sim_ids)s)"
    return sql


# --------------------------------------------------
def count_due(cur, renew_before: datetime.datetime, sim_ids: List[int]) -> int:
    """How many released simulations are due, before --limit"""

    cur.execute(
        "select count(*) from md_simulation " + due_where(sim_ids),
        {"renew_before": renew_before, "sim_ids": sim_ids},
    )
    return cur.fetchone()[0]


# --------------------------------------------------
def find_due(cur, renew_before: datetime.datetime, limit: int, sim_ids: List[int]) -> List[Due]:
    """Released simulations whose ticket needs work, most urgent first

    Tickets that expire soonest come first (an expired one must not wait
    behind 80,000 old permanent ones), then simulations with no ticket or an
    old permanent one, by id.
    """

    cur.execute(
        "select id, irods_ticket, irods_ticket_expires_at from md_simulation "
        + due_where(sim_ids)
        + " order by irods_ticket_expires_at asc nulls last, id limit %(limit)s",
        {"renew_before": renew_before, "sim_ids": sim_ids, "limit": limit},
    )
    return [Due(row[0], row[1], row[2]) for row in cur.fetchall()]


# --------------------------------------------------
def describe(sim: Due) -> str:
    """A log label for a due simulation"""

    if not sim.ticket:
        state = "no ticket"
    elif sim.expires_at is None:
        state = f"old permanent ticket {st.label(sim.ticket)}"
    else:
        state = f"ticket {st.label(sim.ticket)} expires {sim.expires_at:%Y-%m-%d %H:%MZ}"
    return f"MDR{sim.id:08d} ({state})"


# --------------------------------------------------
def renew(session, cur, sim: Due, args: Args, now: datetime.datetime) -> Outcome:
    """Retire an old permanent ticket if there is one, then store a new one"""

    expires_at = sim.expires_at

    # 1. Give an old permanent ticket an expiry, and record it at once.
    if sim.ticket and expires_at is None:
        grace_end = now + datetime.timedelta(days=args.old_grace_days)
        try:
            st.set_expiry(session, sim.ticket, grace_end)
        except CAT_TICKET_INVALID:
            # iRODS no longer has it, so it already works for nobody: record
            # it as ended now, so the app replaces it even if step 2 fails.
            grace_end = now
        except Exception as e:
            return Outcome(False, f"setting the old ticket's expiry: {describe_exc(e)}")

        cur.execute(
            """
            update md_simulation
            set    irods_ticket_expires_at = %s
            where  id = %s
            and    irods_ticket = %s
            and    irods_ticket_expires_at is null
            """,
            (grace_end, sim.id, sim.ticket),
        )
        expires_at = grace_end

    # 2. A new ticket, stored only if the row is as this run read it.
    new_end = now + datetime.timedelta(days=args.lifetime_days)
    try:
        new = st.mint_read_ticket(session, st.sim_path(args.server, sim.id), new_end)
    except Exception as e:
        return Outcome(False, f"making the new ticket: {describe_exc(e)}")

    cur.execute(
        f"""
        update md_simulation
        set    irods_ticket = %s,
               irods_ticket_expires_at = %s
        where  id = %s
        and    {st.RELEASED_SQL}
        and    irods_ticket is not distinct from %s
        and    irods_ticket_expires_at is not distinct from %s
        """,
        (new, new_end, sim.id, sim.ticket, expires_at),
    )
    if cur.rowcount == 1:
        return Outcome(True, "renewed")

    # The row changed under us (the app minted a ticket, or the simulation
    # was hidden): drop ours rather than leave an untracked ticket.
    try:
        Ticket(session, ticket=new).delete()
    except Exception as e:
        return Outcome(
            False,
            f"row changed meanwhile and the unused new ticket {st.label(new)} "
            f"could not be deleted: {describe_exc(e)}",
        )
    return Outcome(True, "row changed meanwhile; left as it is")


# --------------------------------------------------
if __name__ == "__main__":
    main()
