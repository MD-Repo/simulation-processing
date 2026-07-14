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
import requests
import psycopg2
import psycopg2.extras
import ssl
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from irods.session import iRODSSession
from typing import List, NamedTuple, Optional

TICKET_RE = re.compile(r"^MDRSubmit_([^:]+):(.+)$")
MAX_DAYS_OLD = 7
SUBMISSION_COMPLETE = "mdrepo-submission.completed.json"
FRONTEND_BASE_URLS = {
    "staging": "https://staging.mdrepo.org",
    "prod": "https://mdrepo.org",
}


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

        for ticket in unprocessed:
            process_ticket(cur, session, ticket, base_url, args.dry_run, status)

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
def process_ticket(cur, session, ticket, base_url: str, dry_run: bool, status) -> None:
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
                f"DRY RUN: would notify Slack and mark ticket {ticket['id']} "
                "notified and used for upload"
            )
        else:
            status(msg)
            send_slack_message(msg, base_url)

            cur.execute(
                """
                update md_ticket
                set    upload_notification_sent = true,
                       used_for_upload = true
                where  id = %s
                """,
                (ticket["id"],),
            )

    if not is_complete and days_old >= MAX_DAYS_OLD:
        if dry_run:
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
def send_slack_message(
    message: str, base_url: str, channel: str = "mdrepo-alerts"
) -> None:
    """Post a message to Slack (best-effort, mirrors slack_messages.send_message)"""

    token = os.getenv("SLACK_TOKEN")
    if not token:
        print(f'No SLACK_TOKEN, not sending Slack message "{message}"')
        return

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "channel": channel,
                "text": f"{message} ({base_url})",
                "username": "Bot User",
            },
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f'Unable to send Slack message "{message}": {e}')


# --------------------------------------------------
if __name__ == "__main__":
    main()
