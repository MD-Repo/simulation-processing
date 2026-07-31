#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-31
Purpose: Delete per-ticket mdr-process logs past a retention age

Delete by age, never rotate. A per-ticket log is written once by a single
mdr-process run and then never touched again, so logrotate has nothing to do
with it -- rotating would rename a finished file to something nothing looks
for. Worse, md_process_job.log_file records the exact path, so a rename turns
that column back into the lie it was before 2026-07-31, when it was declared
but never written.

The same reasoning is why this nulls log_file for any row whose file is gone:
a path pointing at nothing is worse than a null, because null is honest about
not knowing. That reconciliation runs over every row, not just the ones this
pass deleted, so a log removed by hand is tidied up too.

Disk is not the motivation -- these are ~5 MB each against 2 TB free. The
point is that nothing here was bounded, and an unbounded thing nobody watches
is the shape that has bitten this box repeatedly.

Nothing is deleted without --delete.
"""

import argparse
import os
import sys
import time
from typing import List, NamedTuple

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from common import TICKET_LOG_ROOT, stamp

# Only ever touch files matching this, inside TICKET_LOG_ROOT/<server>. The
# prefix is what keeps a stray file in that directory from being deleted by a
# glob that was only ever meant for mdr-process output.
LOG_PREFIX = "ticket-"
LOG_SUFFIX = ".log"
DEFAULT_DAYS = 90


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    days: int
    delete: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Delete per-ticket mdr-process logs past a retention age",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "-d",
        "--days",
        help="Delete logs last modified more than this many days ago",
        metavar="INT",
        type=int,
        default=DEFAULT_DAYS,
    )

    parser.add_argument(
        "--delete",
        help="Actually delete. Without this nothing is removed",
        action="store_true",
    )

    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")

    return Args(server=args.server, days=args.days, delete=args.delete)


# --------------------------------------------------
def say(msg: str) -> None:
    """Print one timestamped line"""

    print(f"{stamp()} {msg}", flush=True)


# --------------------------------------------------
def human(num: float) -> str:
    """Format a byte count"""

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024

    return f"{num:.1f} PB"


# --------------------------------------------------
def find_expired(log_dir: str, days: int) -> List[str]:
    """Ticket logs in this directory last modified more than "days" ago

    Keyed on mtime rather than the ticket ID in the filename: the ID says when
    the ticket was created, which can be months before it was processed, and a
    reprocess rewrites the file. mtime is when this log was last written, which
    is the thing retention is actually about.
    """

    if not os.path.isdir(log_dir):
        return []

    cutoff = time.time() - (days * 86400)
    expired = []

    for name in sorted(os.listdir(log_dir)):
        if not (name.startswith(LOG_PREFIX) and name.endswith(LOG_SUFFIX)):
            continue

        path = os.path.join(log_dir, name)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            expired.append(path)

    return expired


# --------------------------------------------------
def reconcile(cur, delete: bool) -> int:
    """Null log_file on rows whose file is no longer on disk

    Runs over every populated row, not just what this pass removed, so a log
    deleted by hand is reconciled too. Reading a path out of the database and
    finding nothing there is the failure this avoids.
    """

    cur.execute(
        "select id, log_file from md_process_job where log_file is not null"
    )
    missing = [r["id"] for r in cur.fetchall() if not os.path.isfile(r["log_file"])]

    if missing and delete:
        cur.execute(
            "update md_process_job set log_file = null where id = any(%s)",
            (missing,),
        )

    return len(missing)


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    # Beside this file, so the cron line needs no "cd" (see the note in
    # purge_processed_landings.py -- explicit, not because the cwd would break
    # it).
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dsn = os.environ.get(env_key)
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    log_dir = os.path.join(TICKET_LOG_ROOT, args.server)
    expired = find_expired(log_dir, args.days)
    freed = sum(os.path.getsize(p) for p in expired)

    verb = "Deleting" if args.delete else "DRY RUN (pass --delete to remove):"
    say(f"{verb} {len(expired)} log(s) older than {args.days} days in "
        f"{log_dir}, {human(freed)}")

    removed = 0
    for path in expired:
        if args.delete:
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                say(f"  FAILED to remove {path}: {e}")
                continue
        say(f"  {'removed' if args.delete else 'would remove'} "
            f"{os.path.basename(path)}")

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        stale = reconcile(cur, args.delete)
        if stale:
            say(f"{'Nulled' if args.delete else 'Would null'} log_file on "
                f"{stale} row(s) whose log is no longer on disk")
    finally:
        conn.close()

    if args.delete:
        say(f"Removed {removed} log(s), {human(freed)}")
    else:
        say("Nothing was deleted. Re-run with --delete to act.")


# --------------------------------------------------
if __name__ == "__main__":
    main()
