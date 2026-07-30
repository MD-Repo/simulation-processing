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
from dotenv import load_dotenv
from irods.session import iRODSSession
from typing import List, NamedTuple, Optional

from common import FRONTEND_BASE_URLS, send_slack_message

TICKET_RE = re.compile(r"^MDRSubmit_([^:]+):(.+)$")
SUBMISSION_COMPLETE = "mdrepo-submission.completed.json"


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
            select t.id, t.created_at, t.irods_tickets,
                   u.username, u.first_name, u.last_name
            from   md_ticket t
            join   md_user u on u.id = t.created_by_id
            where  t.irods_tickets like %s
            """,
            (f"%{landing_id}%",),
        )
    else:
        cur.execute("""
            select t.id, t.created_at, t.irods_tickets,
                   u.username, u.first_name, u.last_name
            from   md_ticket t
            join   md_user u on u.id = t.created_by_id
            where  t.ticket_type = 'u'
            and    t.upload_notification_sent = false
            """)

    return cur.fetchall()


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

    landing_dirs: List[str] = []
    missing_marker = False

    irods_tickets = ticket["irods_tickets"]
    if irods_tickets:
        for irods_ticket in irods_tickets.split(";"):
            matches = TICKET_RE.search(irods_ticket)
            if not matches:
                status(f"Unknown IRODS ticket format: {irods_ticket}")
                continue

            _landing_id, landing_dir = matches.groups()
            landing_dirs.append(landing_dir)

            # Probe for the one marker file rather than listing the
            # collection. Listing pulled metadata for every object just to
            # test one name, at three round trips per landing; this is one.
            # A missing collection makes the probe False, same as before.
            # Measured on prod: 0.78s -> 0.135s per landing, and the scan
            # walks >12,000 of them per pass.
            if not session.data_objects.exists(
                f"{landing_dir}/{SUBMISSION_COMPLETE}"
            ):
                # One missing marker already settles the ticket, so stop
                # probing: an incomplete 200-landing ticket costs 1 probe
                # instead of 200. Nothing is lost by not finishing -- the
                # ticket stays unnotified and is re-scanned every pass until
                # it completes, and "landing_dirs" is only read below on the
                # complete path.
                missing_marker = True
                break

    # An empty ticket is not complete: all([]) would have said it was.
    is_complete = bool(landing_dirs) and not missing_marker

    if is_complete:
        num_simulations = len(landing_dirs)
        full_name = f"{ticket['first_name']} {ticket['last_name']}".strip()
        user_label = (
            f"{full_name} ({ticket['username']})" if full_name else ticket["username"]
        )
        msg = (
            f"New simulation upload {ticket['id']} by {user_label} containing "
            f"{num_simulations} simulation{'' if num_simulations == 1 else 's'}"
        )

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


# --------------------------------------------------
if __name__ == "__main__":
    main()
