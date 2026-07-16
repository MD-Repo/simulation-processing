#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-14
Purpose: Scan for unprocessed simulation uploads (cron replacement for the
         Django "check_new_simulations" qcluster task)
"""

import argparse
import os
import re
import psycopg2
import psycopg2.extras
import ssl
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from irods.session import iRODSSession
from typing import List, NamedTuple, Optional

from common import FRONTEND_BASE_URLS, send_slack_message

TICKET_RE = re.compile(r"^MDRSubmit_([^:]+):(.+)$")
MAX_DAYS_OLD = 7
SUBMISSION_COMPLETE = "mdrepo-submission.completed.json"

# Tables with a FK to md_ticket, per the Django models (Simulation,
# SimulationUploadInstance, ProcessJob). Postgres reports only the first
# constraint a delete violates, so all of them have to be checked to decide
# whether a ticket is really unreferenced. Keep in sync with the models.
TICKET_REFERENCES = (
    ("md_simulation", "md_repo_ticket_id", "simulation"),
    ("md_upload_instance", "ticket_id", "upload instance"),
    ("md_process_job", "ticket_id", "process job"),
)


class Args(NamedTuple):
    """Command-line arguments"""

    landing_id: Optional[str]
    server: str
    dry_run: bool
    verbose: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Scan for unprocessed simulation uploads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-l", "--landing-id", help="Landing ID", metavar="ID", default=None
    )

    parser.add_argument(
        "-s",
        "--server",
        help="Target server",
        metavar="STR",
        choices=["staging", "prod"],
        default="staging",
    )

    parser.add_argument(
        "--dry-run",
        help="Log intended changes without making them (implies --verbose)",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Verbose", action="store_true")

    args = parser.parse_args()

    return Args(
        landing_id=args.landing_id,
        server=args.server,
        dry_run=args.dry_run,
        verbose=args.verbose or args.dry_run,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    load_dotenv()

    def status(msg: str) -> None:
        if args.verbose:
            print(msg)

    if args.dry_run:
        status("DRY RUN: no changes will be made")

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dsn = os.environ.get(env_key)
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    unprocessed = find_unprocessed_tickets(cur, args.landing_id)

    if not unprocessed:
        status("Found no tickets to process")
        sys.exit(0)

    status(f"Found {len(unprocessed)} unprocessed tickets, getting IRODS session")

    irods_env = os.environ.get(
        "IRODS_ENVIRONMENT_FILE",
        os.path.expanduser("~/.irods/irods_environment.json"),
    )
    ssl_context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None
    )

    with iRODSSession(irods_env_file=irods_env, ssl_context=ssl_context) as session:
        status("Got IRODS session")

        base_url = FRONTEND_BASE_URLS[args.server]

        failed = 0
        for ticket in unprocessed:
            # One unhappy ticket must not strand the rest of the scan: without
            # this, an error here aborts the loop and every later ticket is
            # silently never processed.
            try:
                process_ticket(
                    cur,
                    session,
                    ticket,
                    args.server,
                    base_url,
                    args.dry_run,
                    status,
                )
            except Exception as e:
                failed += 1
                status(f"ERROR on ticket {ticket['id']}, skipping: {e}")

    if failed:
        status(f"FINISHED check_new_simulations ({failed} ticket(s) errored)")
        sys.exit(1)

    status("FINISHED check_new_simulations")


# --------------------------------------------------
def find_unprocessed_tickets(cur, landing_id: Optional[str]) -> List[dict]:
    """Find MDRepo tickets with completed/incomplete IRODS uploads"""

    if landing_id:
        cur.execute(
            """
            select id, created_at, irods_tickets
            from   md_ticket
            where  irods_tickets like %s
            """,
            (f"%{landing_id}%",),
        )
    else:
        cur.execute("""
            select id, created_at, irods_tickets
            from   md_ticket
            where  ticket_type = 'u'
            and    upload_notification_sent = false
            """)

    return cur.fetchall()


# --------------------------------------------------
def ticket_dependents(cur, ticket_id: int) -> str:
    """Describe rows referencing this ticket, or "" if it is unreferenced

    Deleting a ticket here is raw SQL, so it gets none of the on_delete
    behaviour the Django models declare -- that is emulated by the ORM, not by
    the database (the constraints carry no ON DELETE clause). A raw delete of a
    referenced ticket therefore just raises ForeignKeyViolation.
    """

    counts = []
    for table, col, label in TICKET_REFERENCES:
        cur.execute(f"select count(*) from {table} where {col} = %s", (ticket_id,))
        n = cur.fetchone()[0]
        if n:
            counts.append(f"{n} {label}{'s' if n > 1 else ''}")

    return ", ".join(counts)


# --------------------------------------------------
def process_ticket(
    cur,
    session,
    ticket,
    server: str,
    base_url: str,
    dry_run: bool,
    status,
) -> None:
    """Check a single ticket's IRODS collections and act on its status"""

    created = ticket["created_at"]
    now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now(timezone.utc)
    days_old = (now - created).days

    collections = []
    upload_complete = []
    landing_dirs: List[str] = []

    irods_tickets = ticket["irods_tickets"]
    if irods_tickets:
        for irods_ticket in irods_tickets.split(";"):
            matches = TICKET_RE.search(irods_ticket)
            if matches:
                _landing_id, landing_dir = matches.groups()
                landing_dirs.append(landing_dir)

                if session.collections.exists(landing_dir):
                    coll = session.collections.get(landing_dir)
                    collections.append(coll)
                    coll_filenames = [o.name for o in coll.data_objects]
                    upload_complete.append(SUBMISSION_COMPLETE in coll_filenames)
                else:
                    upload_complete.append(False)
            else:
                status(f"Unknown IRODS ticket format: {irods_ticket}")

    # all([]) == True !
    is_complete = bool(upload_complete) and all(upload_complete)

    if is_complete:
        msg = f"New simulation upload {ticket['id']}: {', '.join(landing_dirs)}"

        if dry_run:
            status(
                f"DRY RUN: would notify Slack, mark ticket {ticket['id']} "
                "notified and used for upload, then enqueue an mdr-process job"
            )
        else:
            status(msg)
            send_slack_message(msg, base_url)

            # Mark the ticket and enqueue the job as one atomic statement, so a
            # crash can't leave a ticket marked "notified" but never queued
            # (which no future scan would re-find). Safe under autocommit.
            cur.execute(
                """
                with marked as (
                    update md_ticket
                    set    upload_notification_sent = true,
                           used_for_upload = true
                    where  id = %s
                    returning id
                )
                insert into md_process_job (ticket_id, server, status)
                select id, %s, 'pending' from marked
                """,
                (ticket["id"], server),
            )
            status(f"Enqueued mdr-process job for ticket {ticket['id']}")

    if not is_complete and days_old >= MAX_DAYS_OLD:
        # The reap is for abandoned uploads. A ticket with rows hanging off it
        # produced real data despite never getting its completion marker, so it
        # isn't abandoned -- leave it (and its IRODS collections) alone.
        dependents = ticket_dependents(cur, ticket["id"])
        if dependents:
            status(
                f"Ticket {ticket['id']} is {days_old} days old and incomplete "
                f"but has {dependents}: SKIP (not abandoned)"
            )
        elif dry_run:
            status(
                f"DRY RUN: ticket {ticket['id']} is {days_old} days old and "
                "incomplete: would DELETE"
            )
        else:
            status(
                f"Ticket {ticket['id']} is {days_old} days old and incomplete: DELETE"
            )
            for coll in collections:
                coll.remove()
            cur.execute("delete from md_ticket where id = %s", (ticket["id"],))


# --------------------------------------------------
if __name__ == "__main__":
    main()
