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
        help="Log intended changes without making them",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Verbose", action="store_true")

    args = parser.parse_args()

    return Args(
        landing_id=args.landing_id,
        server=args.server,
        dry_run=args.dry_run,
        verbose=args.verbose,
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

        for ticket in unprocessed:
            process_ticket(cur, session, ticket, args.dry_run, status)

    status("FINISHED check_new_simulations")


# --------------------------------------------------
def find_unprocessed_tickets(cur, landing_id: Optional[str]) -> List[dict]:
    """Find MDRepo tickets with completed/incomplete IRODS uploads"""

    if landing_id:
        cur.execute(
            """
            select id, created_at, created_by_id, irods_tickets
            from   md_ticket
            where  irods_tickets like %s
            """,
            (f"%{landing_id}%",),
        )
    else:
        cur.execute(
            """
            select id, created_at, created_by_id, irods_tickets
            from   md_ticket
            where  ticket_type = 'u'
            and    upload_notification_sent = false
            """
        )

    return cur.fetchall()


# --------------------------------------------------
def process_ticket(cur, session, ticket, dry_run: bool, status) -> None:
    """Check a single ticket's IRODS collections and act on its status"""

    created = ticket["created_at"]
    now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now(timezone.utc)
    days_old = (now - created).days

    collections = []
    upload_complete = []
    filenames: List[str] = []
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
                    filenames.extend(coll_filenames)
                    upload_complete.append(SUBMISSION_COMPLETE in coll_filenames)
                else:
                    upload_complete.append(False)
            else:
                status(f"Unknown IRODS ticket format: {irods_ticket}")

    # all([]) == True !
    is_complete = bool(upload_complete) and all(upload_complete)

    if is_complete:
        lead_contributor_orcid = get_orcid(cur, ticket["created_by_id"]) or "NA"
        keep_filenames = [
            f for f in filenames if not f.startswith("mdrepo-submission.")
        ]

        msg = f"New simulation upload {ticket['id']}: {', '.join(landing_dirs)}"

        if dry_run:
            status(
                f"DRY RUN: would create upload instance for ticket {ticket['id']} "
                f"({', '.join(keep_filenames)}), notify Slack, mark notified"
            )
        else:
            cur.execute(
                """
                insert
                into   md_upload_instance
                       (created_on, user_id, ticket_id, lead_contributor_orcid,
                        filenames)
                values (now(), %s, %s, %s, %s)
                returning id
                """,
                (
                    ticket["created_by_id"],
                    ticket["id"],
                    lead_contributor_orcid,
                    ", ".join(keep_filenames),
                ),
            )
            upload_instance_id = cur.fetchone()[0]

            cur.execute(
                """
                insert
                into   md_upload_instance_message
                       (timestamp, message, simulation_upload_id, is_error,
                        is_warning)
                values (now(), %s, %s, false, false)
                """,
                ("Files received, awaiting processing", upload_instance_id),
            )

            status(msg)
            send_slack_message(msg)

            cur.execute(
                """
                update md_ticket
                set    upload_notification_sent = true
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
            status(f"Ticket {ticket['id']} is {days_old} days old and incomplete: DELETE")
            for coll in collections:
                coll.remove()
            cur.execute("delete from md_ticket where id = %s", (ticket["id"],))


# --------------------------------------------------
def get_orcid(cur, user_id: Optional[int]) -> Optional[str]:
    """Get a user's first linked social-account uid (mirrors User.orcid)"""

    if user_id is None:
        return None

    cur.execute(
        """
        select uid
        from   socialaccount_socialaccount
        where  user_id = %s
        order  by id
        limit  1
        """,
        (user_id,),
    )
    res = cur.fetchone()
    return res["uid"] if res else None


# --------------------------------------------------
def send_slack_message(message: str, channel: str = "mdrepo-alerts") -> None:
    """Post a message to Slack (best-effort, mirrors slack_messages.send_message)"""

    try:
        token = os.getenv("SLACK_TOKEN")
        if token:
            domain = os.getenv("FRONTEND_BASE_URL")
            resp = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "channel": channel,
                    "text": f"{message} ({domain})",
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
