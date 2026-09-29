#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-13
Purpose: Nightly cron job to delete old md_frontend_download_instance rows
         (and their file associations). These back the "download selected
         files as a zip" button on the web UI; once the browser has streamed
         the zip (or the user never came back to finish the download), the
         row exists only as bookkeeping and nothing ever reads it again.
         Left alone they accumulate indefinitely and have to be manually
         cleared out of the way whenever a simulation's files are deleted
         (see delete_simulation.py), so this sweeps anything past a
         retention window regardless of whether it was ever used.
"""

import argparse
import os
import psycopg2
import psycopg2.extras
import sys
from dotenv import dotenv_values
from typing import NamedTuple


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    days: int
    dry_run: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Delete old md_frontend_download_instance rows",
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
        "-d",
        "--days",
        help="Delete instances created more than this many days ago",
        metavar="INT",
        type=int,
        default=7,
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        help="Show what would be deleted without deleting anything",
        action="store_true",
    )

    args = parser.parse_args()

    return Args(server=args.server, days=args.days, dry_run=args.dry_run)


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

    cur.execute(
        """
        select id
        from   md_frontend_download_instance
        where  created_on < now() - interval '%s days'
        """,
        (args.days,),
    )
    ids = [row["id"] for row in cur.fetchall()]

    if not ids:
        print(f"No download instances older than {args.days} days.")
        return

    print(f"Found {len(ids):,} download instance(s) older than {args.days} days.")

    if args.dry_run:
        sample = ids[:10]
        print(f"  (dry run) sample IDs: {sample}")
        if len(ids) > len(sample):
            print(f"  (dry run) ...and {len(ids) - len(sample):,} more")
        return

    cur.execute(
        """
        delete
        from   md_frontend_download_instance_uploaded_files
        where  frontenddownloadinstance_id = any(%s)
        """,
        (ids,),
    )
    print(f"  Deleted {cur.rowcount:,} row(s) from md_frontend_download_instance_uploaded_files")

    cur.execute(
        """
        delete
        from   md_frontend_download_instance_processed_files
        where  frontenddownloadinstance_id = any(%s)
        """,
        (ids,),
    )
    print(f"  Deleted {cur.rowcount:,} row(s) from md_frontend_download_instance_processed_files")

    cur.execute(
        """
        delete
        from   md_frontend_download_instance
        where  id = any(%s)
        """,
        (ids,),
    )
    print(f"  Deleted {cur.rowcount:,} row(s) from md_frontend_download_instance")

    print("Done.")


# --------------------------------------------------
if __name__ == "__main__":
    main()
