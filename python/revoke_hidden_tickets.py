#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-10-20
Purpose: Revoke iRODS read tickets on simulations that are not released.

         A simulation's stored read ticket (md_simulation.irods_ticket) has no
         expiry and unlimited uses, and it covers the whole MDR* collection,
         so anyone holding the string can read the files anonymously.
         Embargoing a simulation only sets is_embargoed; the ticket keeps
         working. This deletes the ticket in iRODS for every simulation that
         is not released (not is_public, or is_embargoed, or is_placeholder)
         and then clears the column. Delete first, clear second: if the
         delete fails the row still names a live ticket and the next run
         retries it. A ticket iRODS no longer knows (CAT_TICKET_INVALID) is
         already revoked, so its column is cleared.

         md-repo-app's get_download_ticket() stores a new permanent ticket
         when a contributor or superuser downloads a hidden simulation, so
         this runs nightly to keep that ticket's life to a day.

         --sweep also searches the iRODS catalog for tickets that
         md_simulation does not track (legacy per-request tickets, MDR-83) on
         the hidden simulations' collections and files, and deletes those.
         It looks objects up by exact collection name and tickets by object
         id: a LIKE on the collection name, or a ticket query joined to the
         data-object table, does not return within 60 s on this catalog.
         About 11 s per 50 simulations, so a full sweep takes an hour.

         Ticket strings are credentials: the log shows only their first four
         characters.

         Exit status: 0 when every ticket was revoked, 1 when any failed.
         See MDR-90 (md-repo-app#434).
"""

import argparse
import os
import ssl
import sys
from typing import Dict, List, NamedTuple, Optional

import psycopg2
import psycopg2.extras
from dotenv import dotenv_values
from irods.column import In
from irods.exception import CAT_TICKET_INVALID
from irods.models import Collection, DataObject, TicketQuery
from irods.session import iRODSSession
from irods.ticket import Ticket

from common import describe_exc, stamp

RELEASE_ROOT = "/iplant/home/shared/mdrepo/{server}/release"


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    dry_run: bool
    sweep: bool
    contributor: Optional[int]
    sim_ids: List[int]
    batch_size: int
    irods_env: str


class Hidden(NamedTuple):
    """A simulation that is not released"""

    id: int
    stored_ticket: Optional[str]


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Revoke iRODS read tickets on simulations that are not released",
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
        help="Report what would be revoked without changing anything",
        action="store_true",
    )

    parser.add_argument(
        "--sweep",
        help="Also search the iRODS catalog for tickets md_simulation does not track",
        action="store_true",
    )

    parser.add_argument(
        "-c",
        "--contributor",
        help="Only simulations from this contributor (md_user id)",
        metavar="INT",
        type=int,
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
        "-b",
        "--batch-size",
        help="Simulations per catalog query in --sweep",
        metavar="INT",
        type=int,
        default=50,
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

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(f"--batch-size '{args.batch_size}' must be positive")

    if not os.path.isfile(args.irods_env):
        parser.error(f"Invalid --irods-env '{args.irods_env}'")

    return Args(
        server=args.server,
        dry_run=args.dry_run,
        sweep=args.sweep,
        contributor=args.contributor,
        sim_ids=args.sim_id,
        batch_size=args.batch_size,
        irods_env=args.irods_env,
    )


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

    hidden = find_hidden(cur, args.contributor, args.sim_ids)
    stored = [sim for sim in hidden if sim.stored_ticket]
    mode = "DRY RUN" if args.dry_run else "REVOKING"
    print(
        f"{stamp()} {mode}: {len(hidden):,} hidden simulation(s) on "
        f"{args.server}, {len(stored):,} with a stored ticket",
        flush=True,
    )

    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    failed = 0

    with iRODSSession(irods_env_file=args.irods_env, ssl_context=ssl_context) as session:
        for sim in stored:
            failed += revoke_stored(session, cur, sim, args.dry_run)

        if args.sweep and hidden:
            failed += sweep(session, hidden, args)

    print(
        f"{stamp()} Done: {len(stored):,} stored ticket(s) "
        f"{'to revoke' if args.dry_run else 'handled'}, {failed:,} failure(s)",
        flush=True,
    )
    sys.exit(1 if failed else 0)


# --------------------------------------------------
def find_hidden(cur, contributor: Optional[int], sim_ids: List[int]) -> List[Hidden]:
    """Simulations that are not released, optionally narrowed"""

    sql = """
        select id, irods_ticket
        from   md_simulation
        where  (not is_public or is_embargoed or is_placeholder)
    """
    params: list = []
    if contributor is not None:
        sql += " and contributor_id = %s"
        params.append(contributor)
    if sim_ids:
        sql += " and id = any(%s)"
        params.append(sim_ids)

    cur.execute(sql + " order by id", params)
    return [Hidden(row["id"], row["irods_ticket"]) for row in cur.fetchall()]


# --------------------------------------------------
def revoke_stored(session, cur, sim: Hidden, dry_run: bool) -> int:
    """Delete one simulation's stored ticket, then clear the column

    Returns the number of failures (0 or 1).
    """

    label = f"MDR{sim.id:08d} stored ticket {sim.stored_ticket[:4]}…"
    if dry_run:
        print(f"  would revoke {label}", flush=True)
        return 0

    try:
        Ticket(session, ticket=sim.stored_ticket).delete()
        outcome = "revoked"
    except CAT_TICKET_INVALID:
        outcome = "already gone in iRODS"
    except Exception as e:
        print(f"  FAILED {label}: {describe_exc(e)}", flush=True)
        return 1

    # Match the string too, in case the app stored a new ticket meanwhile.
    cur.execute(
        """
        update md_simulation
        set    irods_ticket = null
        where  id = %s
        and    irods_ticket = %s
        """,
        (sim.id, sim.stored_ticket),
    )
    print(f"  {label}: {outcome}, column cleared ({cur.rowcount} row)", flush=True)
    return 0


# --------------------------------------------------
def sweep(session, hidden: List[Hidden], args: Args) -> int:
    """Find and delete catalog tickets on hidden simulations' objects

    Skips each simulation's stored ticket, which revoke_stored() owns.
    Returns the number of failures.
    """

    root = RELEASE_ROOT.format(server=args.server)
    stored = {sim.stored_ticket for sim in hidden if sim.stored_ticket}
    seen: set = set()
    T = TicketQuery.Ticket
    found = failed = matched = 0

    print(f"{stamp()} Sweeping the catalog in batches of {args.batch_size}", flush=True)
    for start in range(0, len(hidden), args.batch_size):
        batch = hidden[start : start + args.batch_size]
        tops = [f"{root}/MDR{sim.id:08d}" for sim in batch]

        try:
            objects = object_ids(session, tops)
            rows = []
            if objects:
                rows = list(
                    session.query(T.string, T.object_id, T.object_type, T.create_time).filter(
                        In(T.object_id, [str(i) for i in objects])
                    )
                )
        except Exception as e:
            print(
                f"  FAILED batch {tops[0][-11:]}..{tops[-1][-11:]}: {describe_exc(e)}",
                flush=True,
            )
            failed += 1
            continue

        for row in rows:
            string = row[T.string]
            if string in stored or string in seen:
                matched += string in stored and string not in seen
                seen.add(string)
                continue

            seen.add(string)
            found += 1
            label = (
                f"{objects[int(row[T.object_id])].replace(root + '/', '')} "
                f"{row[T.object_type]} ticket {string[:4]}… created {row[T.create_time]}"
            )
            if args.dry_run:
                print(f"  would revoke {label}", flush=True)
                continue

            try:
                Ticket(session, ticket=string).delete()
                print(f"  revoked {label}", flush=True)
            except CAT_TICKET_INVALID:
                print(f"  already gone {label}", flush=True)
            except Exception as e:
                print(f"  FAILED {label}: {describe_exc(e)}", flush=True)
                failed += 1

        done = min(start + args.batch_size, len(hidden))
        if done % 1000 < args.batch_size or done == len(hidden):
            print(f"{stamp()}   {done:,} of {len(hidden):,} swept, {found:,} untracked ticket(s)", flush=True)

    print(
        f"{stamp()} Sweep found {found:,} untracked ticket(s), "
        f"and {matched:,} stored ticket(s) left to revoke_stored()",
        flush=True,
    )
    return failed


# --------------------------------------------------
def object_ids(session, tops: List[str]) -> Dict[int, str]:
    """Catalog ids of these collections and everything beneath them

    Walks down one level per query by exact parent name, which the catalog
    answers quickly, where a LIKE on the collection name does not.
    """

    objects: Dict[int, str] = {}
    level = tops
    collections: List[str] = []
    while level:
        collections.extend(level)
        names = [
            row[Collection.name]
            for row in session.query(Collection.name).filter(In(Collection.parent_name, level))
        ]
        level = names

    for row in session.query(Collection.id, Collection.name).filter(In(Collection.name, collections)):
        objects[row[Collection.id]] = row[Collection.name]

    for row in session.query(DataObject.id, DataObject.name, Collection.name).filter(
        In(Collection.name, collections)
    ):
        objects[row[DataObject.id]] = f"{row[Collection.name]}/{row[DataObject.name]}"

    return objects


# --------------------------------------------------
if __name__ == "__main__":
    main()
