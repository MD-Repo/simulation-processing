#!/usr/bin/env python3
"""
Author : Travis Wheeler <travis@traviswheeler.com>
Date   : 2026-07-09
Purpose: Create an unlimited-use iRODS ticket for a simulation's iRODS
         path and store it in md_simulation.irods_ticket.
"""

import argparse
import os
import psycopg2
import psycopg2.extras
import sys
from dotenv import dotenv_values
from irods.session import iRODSSession
from irods.ticket import Ticket
from typing import NamedTuple


class Args(NamedTuple):
    """Command-line arguments"""

    simulation_id: int
    path: str
    server: str
    permission: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Create an iRODS ticket for a simulation path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--simulation-id",
        help="Simulation ID",
        metavar="INT",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="iRODS path",
        metavar="PATH",
        required=True,
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
        "--permission",
        help="Ticket permission",
        metavar="STR",
        choices=["read", "write"],
        default="read",
    )

    args = parser.parse_args()

    return Args(
        simulation_id=args.simulation_id,
        path=args.path,
        server=args.server,
        permission=args.permission,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dot_env = dotenv_values()
    dsn = dot_env.get(env_key, os.environ.get(env_key, ""))
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    print(f"Connecting to '{args.server}'")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    irods_env = os.environ.get(
        "IRODS_ENVIRONMENT_FILE",
        os.path.expanduser("~/.irods/irods_environment.json"),
    )

    mdrepo_id = f"MDR{args.simulation_id:08d}"
    print(f"{mdrepo_id} -> {args.path}")

    with iRODSSession(irods_env_file=irods_env) as session:
        ticket = Ticket(session)
        ticket.issue(args.permission, args.path)

        # Allow the ticket to be used an unlimited number of times
        # (equivalent to "iticket mod <ticket-string> uses 0"). The value
        # must be a string; a falsy int like 0 is dropped by the client.
        ticket.modify("uses", "0")

    cur.execute(
        """
        update md_simulation
        set    irods_ticket=%s
        where  id=%s
        """,
        (ticket.ticket, args.simulation_id),
    )

    print(f"Done. {mdrepo_id} = {ticket.ticket}")


# --------------------------------------------------
if __name__ == "__main__":
    main()
