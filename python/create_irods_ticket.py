#!/usr/bin/env python3
"""
Author : Travis Wheeler <travis@traviswheeler.com>
Date   : 2026-07-09
Purpose: Create an unlimited-use iRODS ticket for a simulation's iRODS
         path and store it in md_simulation.irods_ticket.

A read ticket expires after --lifetime-days (30 by default) and its expiry is
stored in md_simulation.irods_ticket_expires_at; renew_public_tickets.py
replaces it nightly before it ends (MDR-115). Anyone holding the string can
read the simulation until then, so a read ticket is only issued for a
released simulation (approved, not embargoed, not a placeholder).
"""

import argparse
import datetime
import os
import psycopg2
import psycopg2.extras
import sys
from dotenv import dotenv_values
from irods.session import iRODSSession
from irods.ticket import Ticket
from typing import NamedTuple, Optional

import shared_tickets as st


class Args(NamedTuple):
    """Command-line arguments"""

    simulation_id: int
    path: str
    server: str
    permission: str
    lifetime_days: int


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

    parser.add_argument(
        "--lifetime-days",
        help="Days a read ticket lives",
        metavar="INT",
        type=int,
        default=st.LIFETIME_DAYS,
    )

    args = parser.parse_args()

    if args.lifetime_days <= st.RENEW_WITHIN_DAYS:
        parser.error(
            f"--lifetime-days '{args.lifetime_days}' must be more than "
            f"{st.RENEW_WITHIN_DAYS}, or the nightly renewal replaces it at once"
        )

    return Args(
        simulation_id=args.simulation_id,
        path=args.path,
        server=args.server,
        permission=args.permission,
        lifetime_days=args.lifetime_days,
    )


# --------------------------------------------------
def unreleased_reason(sim: Optional[dict]) -> Optional[str]:
    """Why a simulation may not get a permanent read ticket, or None"""

    if sim is None:
        return "not in md_simulation"
    if not sim["is_public"]:
        return "not approved"
    if sim["is_embargoed"]:
        return "embargoed"
    if sim["is_placeholder"]:
        return "a placeholder"
    return None


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

    if args.permission == "read":
        cur.execute(
            """
            select is_public, is_embargoed, is_placeholder
            from   md_simulation
            where  id=%s
            """,
            (args.simulation_id,),
        )
        if reason := unreleased_reason(cur.fetchone()):
            sys.exit(f"{mdrepo_id} is {reason}: not issuing a read ticket")

    print(f"{mdrepo_id} -> {args.path}")

    expires_at = None
    with iRODSSession(irods_env_file=irods_env) as session:
        if args.permission == "read":
            expires_at = st.now_utc() + datetime.timedelta(days=args.lifetime_days)
            string = st.mint_read_ticket(session, args.path, expires_at)
        else:
            ticket = Ticket(session)
            ticket.issue(args.permission, args.path)

            # Allow the ticket to be used an unlimited number of times
            # (equivalent to "iticket mod <ticket-string> uses 0"). The value
            # must be a string; a falsy int like 0 is dropped by the client.
            ticket.modify("uses", "0")
            string = ticket.ticket

    cur.execute(
        """
        update md_simulation
        set    irods_ticket=%s,
               irods_ticket_expires_at=%s
        where  id=%s
        """,
        (string, expires_at, args.simulation_id),
    )

    until = f", expires {expires_at:%Y-%m-%d %H:%MZ}" if expires_at else ""
    print(f"Done. {mdrepo_id} = {string}{until}")


# --------------------------------------------------
if __name__ == "__main__":
    main()
