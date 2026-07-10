#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-01-22
Purpose: Fetch simulation upload directories
"""

import argparse
import humanize
import json
import os
import psycopg2
import psycopg2.extras
import sys
from datetime import datetime
from dotenv import dotenv_values
from irods.session import iRODSSession
from subprocess import getstatusoutput
from typing import List, NamedTuple, Optional


class Args(NamedTuple):
    """Command-line arguments"""

    out_dir: str
    server: str
    irods_env: str
    landing_dirs: List[str]
    ticket_ids: List[str]


class Ticket(NamedTuple):
    """MDRepo Ticket"""

    ticket_id: int
    token: str
    full_token: str
    irods_tickets: str
    created_at: datetime.datetime
    user_id: int
    first_name: str
    last_name: str
    username: str
    institution: str
    orcid: str
    email: str


class UploadInstance(NamedTuple):
    """UploadInstance"""

    created_on: datetime.datetime
    simulation_id: Optional[int]
    user_id: int
    successful: bool
    lead_contributor_orcid: str
    filenames: str
    ticket_id: int
    landing_id: str


SUBMISSION_COMPLETE = "mdrepo-submission.completed.json"


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Fetch simulation upload directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        help="Output directory",
        metavar="DIR",
        default="",
    )

    parser.add_argument(
        "-s",
        "--server",
        help="Server",
        metavar="STR",
        choices=["prod", "staging"],
        default="prod",
    )

    parser.add_argument(
        "-l",
        "--landing-dir",
        help="Landing directory name(s)",
        metavar="STR",
        nargs="*",
    )

    parser.add_argument(
        "-t",
        "--ticket-id",
        help="Ticket ID(s)",
        metavar="INT",
        type=int,
        nargs="*",
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
        parser.error("Invalid --irods-env '{args.irods_env}'")

    if not args.out_dir:
        args.out_dir = os.path.join("/opt/mdrepo/landing", args.server)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return Args(
        server=args.server,
        out_dir=args.out_dir,
        irods_env=args.irods_env,
        landing_dirs=args.landing_dir or [],
        ticket_ids=args.ticket_id or [],
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
    tickets = find_tickets(cur, args)

    print(f"Found {len(tickets)} MDRepo Tickets")
    with iRODSSession(irods_env_file=args.irods_env) as session:
        for ticket_num, ticket in enumerate(tickets, start=1):
            irods_tickets = ticket.irods_tickets.split(";")
            created = ticket.created_at
            created_by = f"{ticket.first_name} {ticket.last_name} ({ticket.email})"
            ticket_dir = os.path.join(args.out_dir, f"ticket-{ticket.ticket_id}")
            if not os.path.isdir(ticket_dir):
                os.makedirs(ticket_dir)

            print(
                "\n".join(
                    [
                        f">>> {ticket_num}: Ticket {ticket.ticket_id} <<<",
                        f" Created   : {created.strftime('%Y-%m-%d %H:%M')}",
                        f" Created by: {created_by}",
                        f" Token     : {ticket.token}",
                        f" Directory : {ticket_dir}",
                        f" Num parts : {len(irods_tickets)}",
                    ]
                )
            )

            ticket_info = os.path.join(ticket_dir, "ticket.json")
            with open(ticket_info, "wt") as fh:
                json.dump(
                    {
                        "first_name": ticket.first_name,
                        "last_name": ticket.last_name,
                        "email": ticket.email,
                        "orcid": ticket.orcid,
                        "institution": ticket.institution,
                    },
                    fh,
                )

            for irods_num, irods_ticket in enumerate(irods_tickets, start=1):
                print(f"  Part {irods_num}: ", end="")
                if ":" not in irods_ticket:
                    print(f"Invalid IRODS ticket '{irods_ticket}'")
                    continue

                landing_dir = irods_ticket.split(":")[1]

                coll = None
                try:
                    coll = session.collections.get(landing_dir)
                except Exception as e:
                    pass

                if coll is None:
                    print(f"Unable to get landing directory '{landing_dir}'")
                    continue

                dest_dir = os.path.join(ticket_dir, coll.name)
                if not os.path.isdir(dest_dir):
                    os.makedirs(dest_dir)

                # Get the JSON file with the expected MD5 values
                irods_completed = os.path.join(landing_dir, SUBMISSION_COMPLETE)
                if not session.data_objects.exists(irods_completed):
                    sys.exit(f"Missing {irods_completed}")

                local_completed = os.path.join(dest_dir, SUBMISSION_COMPLETE)
                if os.path.isfile(local_completed):
                    os.remove(local_completed)

                session.data_objects.get(irods_completed, local_completed)
                completed = json.load(open(local_completed))
                # NB: despite its name, "irods_path" holds just the basename
                # (e.g. "1_prod.mdc"), so this dict is keyed by basename and
                # matches obj.name in the loop below.
                hash_by_filename = {
                    v["irods_path"]: v["md5_hash"] for v in completed["files"]
                }

                data_objects = coll.data_objects
                if not data_objects:
                    print(f"landing dir '{landing_dir}' is empty")
                    continue

                print(f"{landing_dir} has {len(data_objects)} data objects")

                for obj_num, obj in enumerate(data_objects, start=1):
                    filename = obj.name
                    if filename == SUBMISSION_COMPLETE:
                        continue

                    irods_md5 = obj.chksum()

                    # Read the MD5 from the catalog instead of calling
                    # obj.chksum(), which triggers a server-side operation that
                    # fails with HIERARCHY_ERROR on the cache+archive compound
                    # resource. Good replicas already carry the stored checksum.
                    # checksums = {
                    #     r.checksum
                    #     for r in obj.replicas
                    #     if r.status == "1" and r.checksum
                    # }
                    # if not checksums:
                    #     print(f"    no good replica checksum for '{obj.path}', skipping")
                    #     continue
                    # if len(checksums) > 1:
                    #     print(f"    replicas disagree on checksum for '{obj.path}': {checksums}")
                    #     continue
                    # irods_md5 = checksums.pop()
                    if completed_md5 := hash_by_filename.get(filename):
                        if completed_md5 != irods_md5:
                            print(
                                f"complete JSON MD5 '{completed_md5}' != IRODS MD5 '{irods_md5}'"
                            )
                            continue

                    size = humanize.naturalsize(obj.size)
                    print(f"    {obj_num:2}: {obj.name} [{size}]", end="")
                    sys.stdout.flush()

                    local_path = os.path.join(dest_dir, filename)
                    exists = os.path.isfile(local_path)

                    if exists:
                        if (os.path.getsize(local_path) == obj.size) and (
                            get_local_md5(local_path) == irods_md5
                        ):
                            print(" (already downloaded)", end="")
                        else:
                            print(" (removing incomplete file)", end="")
                            os.remove(local_path)

                    for retry in range(2):
                        if os.path.isfile(local_path):
                            break

                        start = datetime.now()
                        cmd = f"gocmd get {obj.path} {local_path}"
                        rv, out = getstatusoutput(cmd)
                        if rv != 0:
                            sys.exit(f"Error running {cmd}: {out}")

                        elapsed = humanize.precisedelta(datetime.now() - start)
                        print(f" (took {elapsed})", end="")

                        if irods_md5 != get_local_md5(local_path):
                            print(
                                f"Retry {retry + 1}: Bad md5 after download, removing"
                            )
                            os.remove(local_path)

                    print()

    print(f"Done, see '{args.out_dir}'")


# --------------------------------------------------
def find_tickets(cur, args: Args) -> List[Ticket]:
    """Find tickets"""

    ticket_ids = []
    for landing_dir in args.landing_dirs:
        cur.execute(f"""
            select id
            from   md_ticket
            where  irods_tickets like '%{landing_dir}%'
            """)
        for res in cur.fetchall():
            ticket_ids.append(res["id"])

    for ticket_id in args.ticket_ids:
        cur.execute(
            """
            select count(*)
            from   md_ticket
            where  id=%s
            """,
            (ticket_id,),
        )
        count = cur.fetchone()[0]
        if count == 1:
            ticket_ids.append(ticket_id)

    if not args.landing_dirs and not args.ticket_ids:
        cur.execute("""
            select id
            from   md_ticket
            where  ticket_type='u'
            and    irods_tickets is not null
            and    processing_complete='false'
            """)
        ticket_ids = list(map(lambda r: r[0], cur.fetchall()))

    tickets = []
    for ticket_id in ticket_ids:
        cur.execute(
            """
            select t.id, t.token, t.full_token, t.irods_tickets,
                   t.created_at, t.created_by_id,
                   u.first_name, u.last_name, u.username,
                   u.institution, u.email
            from   md_ticket t,
                   md_user u
            where  t.id=%s
            and    t.created_by_id=u.id
            """,
            (ticket_id,),
        )

        if res := cur.fetchone():
            cur.execute(
                """
                select uid
                from   socialaccount_socialaccount
                where  provider='orcid'
                and    user_id=%s
                """,
                (res["created_by_id"],),
            )
            orcid = cur.fetchone()

            tickets.append(
                Ticket(
                    ticket_id=res["id"],
                    token=res["token"],
                    full_token=res["full_token"],
                    irods_tickets=res["irods_tickets"],
                    created_at=res["created_at"],
                    user_id=res["created_by_id"],
                    first_name=res["first_name"],
                    last_name=res["last_name"],
                    username=res["username"],
                    institution=res["institution"],
                    orcid=orcid[0] if orcid else "",
                    email=res["email"],
                )
            )

    return tickets


# --------------------------------------------------
def get_upload_instance(cur, ticket: Ticket, landing_dir: str, filenames: str):
    """Get upload instance"""

    cur.execute(
        """
        select id
        from   md_upload_instance
        where  ticket_id=%s
        and    landing_id=%s
        """,
        (ticket.ticket_id, landing_dir),
    )

    upload_instance_id = None
    if res := cur.fetchone():
        upload_instance_id = res[0]
    else:
        cur.execute(
            """
            insert
            into   md_upload_instance
                   (user_id, lead_contributor_id, filenames, ticket_id, landing_id)
            values (%s, %s, %s, %s, %s)
            returning id
            """,
            (
                Ticket.user_id,
                Ticket.orcid,
                filenames,
                Ticket.ticket_id,
                landing_dir,
            ),
        )
        upload_instance_id = cur.fetchone()[0]

    if not upload_instance_id:
        sys.exit(
            "Failed to create upload instance for ticket {ticket} landing {landing_dir}"
        )

    cur.execute(
        """
        select created_on, simulation_id, user_id, successful, lead_contributor_orcid,
               filenames, ticket_id, landing_id
        from   md_upload_instance
        where  id=%s
        """,
        (upload_instance_id,),
    )

    if res := cur.fetchone():
        return UploadInstance(
            created_on=res["created_on"],
            simulation_id=res["simulation_id"],
            user_id=res["user_id"],
            successful=res["successful"],
            lead_contributor_orcid=res["lead_contributor_orcid"],
            filenames=res["filenames"],
            ticket_id=res["ticket_id"],
            landing_id=res["landing_id"],
        )


# --------------------------------------------------
def get_local_md5(filename: str) -> str:
    """get local md5"""

    cmd = f"md5sum {filename}"
    rv, out = getstatusoutput(cmd)
    if rv != 0:
        sys.exit(f"{cmd}: {out}")

    return out.split()[0]


# --------------------------------------------------
if __name__ == "__main__":
    main()
