#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-29
Purpose: Delete IRODS landing collections whose data is already in permanent
         storage

Nothing else removes landing collections. The reap in check_new_simulations.py
only ever considered *incomplete* tickets, so a ticket that processed
successfully kept its landing copy forever: 16,241 collections were sitting
under the prod landing area, on the order of a terabyte, all of it already
imported and pushed.

The condition for deleting a ticket's landings is a positive one, not a guess
about abandonment: every simulation on the ticket has `is_placeholder = false`,
which `mdr-process` clears only after verifying the push by MD5 and presence.
Age is never consulted. A ticket with no simulations is never a candidate --
nothing was imported, so there is nothing to have been pushed.

By default this reports and deletes nothing; pass --delete. Deletes go to the
IRODS trash unless --force is given, so a mistaken run is recoverable until the
trash is emptied.
"""

import argparse
import os
import re
import ssl
import sys
from typing import Dict, List, NamedTuple, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from irods.session import iRODSSession

TICKET_RE = re.compile(r"^MDRSubmit_([^:]+):(.+)$")


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    ticket_ids: List[int]
    limit: Optional[int]
    delete: bool
    force: bool
    include_unqueued: bool
    irods_env: str


class Candidate(NamedTuple):
    """A ticket whose landings are eligible for deletion"""

    ticket_id: int
    num_sims: int
    has_job: bool
    landing_dirs: List[str]


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Delete IRODS landing collections already in permanent storage",
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
        "-t",
        "--ticket-id",
        help="Only consider these ticket ID(s)",
        metavar="INT",
        type=int,
        nargs="*",
    )

    parser.add_argument(
        "--limit",
        help="Stop after this many tickets (a bounded first pass)",
        metavar="INT",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--delete",
        help="Actually delete. Without this nothing is removed",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        help="Delete outright instead of moving to the IRODS trash. "
        "Reclaims space immediately and is NOT recoverable",
        action="store_true",
    )

    parser.add_argument(
        "--include-unqueued",
        help="Also consider tickets with no md_process_job row. These predate "
        "the processing queue (2026-07-15), so the only evidence they were "
        "pushed is is_placeholder; review a dry run before trusting it",
        action="store_true",
    )

    parser.add_argument(
        "-e",
        "--irods-env",
        help="IRODS environment file",
        metavar="FILE",
        default=os.environ.get(
            "IRODS_ENVIRONMENT_FILE",
            os.path.expanduser("~/.irods/irods_environment.json"),
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.irods_env):
        parser.error(f"Invalid --irods-env '{args.irods_env}'")

    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit '{args.limit}' must be a positive integer")

    if args.force and not args.delete:
        parser.error("--force is meaningless without --delete")

    return Args(
        server=args.server,
        ticket_ids=list(args.ticket_id or []),
        limit=args.limit,
        delete=args.delete,
        force=args.force,
        include_unqueued=args.include_unqueued,
        irods_env=args.irods_env,
    )


# --------------------------------------------------
def human(num: float) -> str:
    """Format a byte count"""

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024

    return f"{num:.1f} PB"


# --------------------------------------------------
def find_candidates(cur, args: Args) -> List[Candidate]:
    """Find tickets whose every simulation is out of placeholder state

    `is_placeholder = false` is set only after mdr-process verifies the push,
    so "no placeholders left" means the data reached permanent storage. The
    ticket must own at least one simulation: zero simulations means nothing was
    ever imported, which is not the same as everything having been pushed.
    """

    where = ["s.md_repo_ticket_id is not null"]
    params: List = []

    if args.ticket_ids:
        where.append("t.id = any(%s)")
        params.append(args.ticket_ids)

    if not args.include_unqueued:
        where.append(
            "exists (select 1 from md_process_job j "
            "where j.ticket_id = t.id and j.status = 'succeeded')"
        )

    cur.execute(
        f"""
        select t.id,
               t.irods_tickets,
               count(s.id)                             as num_sims,
               exists (select 1 from md_process_job j
                       where j.ticket_id = t.id)       as has_job
        from   md_ticket    t
        join   md_simulation s on s.md_repo_ticket_id = t.id
        where  {' and '.join(where)}
        group  by t.id, t.irods_tickets
        having count(*) filter (where s.is_placeholder) = 0
        order  by t.id
        """,
        params,
    )

    candidates = []
    for row in cur.fetchall():
        landing_dirs = [
            m.groups()[1]
            for it in (row["irods_tickets"] or "").split(";")
            if (m := TICKET_RE.search(it))
        ]

        if not landing_dirs:
            continue

        candidates.append(
            Candidate(
                ticket_id=row["id"],
                num_sims=row["num_sims"],
                has_job=row["has_job"],
                landing_dirs=landing_dirs,
            )
        )

    return candidates


# --------------------------------------------------
def purge_ticket(session, cand: Candidate, args: Args) -> Dict[str, int]:
    """Remove one ticket's landing collections, reporting what was found"""

    present = 0
    freed = 0
    removed = 0

    for landing_dir in cand.landing_dirs:
        if not session.collections.exists(landing_dir):
            continue

        present += 1
        coll = session.collections.get(landing_dir)
        freed += sum(obj.size for obj in coll.data_objects)

        if args.delete:
            # recurse: a landing holds its objects directly, but a partially
            # uploaded one can carry subcollections too.
            coll.remove(recurse=True, force=args.force)
            removed += 1

    return {"present": present, "freed": freed, "removed": removed}


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    load_dotenv()

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dsn = os.environ.get(env_key)
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    conn = psycopg2.connect(dsn)
    # Nothing here writes to Postgres: the ticket and its simulations stay, only
    # the IRODS copies go.
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    candidates = find_candidates(cur, args)

    if not candidates:
        print("No tickets are eligible")
        sys.exit(0)

    if args.limit:
        candidates = candidates[: args.limit]

    mode = "DELETING" if args.delete else "DRY RUN (pass --delete to remove)"
    trash = "" if args.force else " to trash"
    print(f"{mode}{trash if args.delete else ''}: "
          f"{len(candidates)} eligible ticket(s) on {args.server}\n")

    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    totals = {"present": 0, "freed": 0, "removed": 0, "tickets": 0}

    with iRODSSession(
        irods_env_file=args.irods_env, ssl_context=ssl_context
    ) as session:
        for cand in candidates:
            result = purge_ticket(session, cand, args)

            if not result["present"]:
                print(f"ticket {cand.ticket_id}: landings already gone")
                continue

            totals["present"] += result["present"]
            totals["freed"] += result["freed"]
            totals["removed"] += result["removed"]
            totals["tickets"] += 1

            verb = "removed" if args.delete else "would remove"
            note = "" if cand.has_job else "  [no process job -- pre-queue]"
            print(
                f"ticket {cand.ticket_id}: {cand.num_sims} sims, "
                f"{verb} {result['present']} of {len(cand.landing_dirs)} "
                f"landing(s), {human(result['freed'])}{note}"
            )

    verb = "Removed" if args.delete else "Would remove"
    print(
        f"\n{verb} {totals['present']} landing collection(s) across "
        f"{totals['tickets']} ticket(s), {human(totals['freed'])}"
    )

    if not args.delete:
        print("Nothing was deleted. Re-run with --delete to act.")
    elif not args.force:
        print("Deleted to the IRODS trash; empty it to reclaim the space.")


# --------------------------------------------------
if __name__ == "__main__":
    main()
