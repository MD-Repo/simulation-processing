#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2025-10-26
Purpose: Push simulation files to cat/IRODS
"""

import argparse
import fabric
import json
import os
import psycopg2
import shlex
import sys
from datetime import datetime as dt
import humanize
from dotenv import dotenv_values
from irods.session import iRODSSession
from typing import Dict, List, NamedTuple, TextIO, Optional
from subprocess import getstatusoutput


class Args(NamedTuple):
    """Command-line arguments"""

    file: TextIO
    simulation_id: int
    irods_env: str
    server: str
    data_dir: str
    file_types: List[str]
    out_file: Optional[str]
    remove_processed_dir: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Push simulation files to cat/IRODS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Preprocessed JSON file",
        type=argparse.FileType("rt"),
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--simulation-id",
        help="Simulation ID",
        type=int,
        metavar="INT",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--data-dir",
        help="Local data directory",
        metavar="DIR",
        required=True,
    )

    parser.add_argument(
        "-S",
        "--server",
        help="Server",
        metavar="STR",
        choices=["staging", "prod"],
        default="staging",
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

    parser.add_argument(
        "-t",
        "--file-types",
        help="File types",
        metavar="STR",
        choices=["all", "media", "original", "processed"],
        default="all",
        nargs="*",
    )

    parser.add_argument(
        "-o",
        "--out-file",
        help="Output file",
        metavar="FILE",
    )

    parser.add_argument(
        "--remove-processed-dir",
        help="Remove existing 'processed' dir",
        action="store_true",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        parser.error(f'Invaid --data-dir "{args.data_dir}"')

    if not args.irods_env or not os.path.isfile(args.irods_env):
        parser.error(f'Invaid or missing --irods-env file "{args.irods_env}"')

    return Args(
        file=args.file,
        simulation_id=args.simulation_id,
        irods_env=args.irods_env,
        server=args.server,
        data_dir=args.data_dir,
        file_types=args.file_types,
        out_file=args.out_file,
        remove_processed_dir=args.remove_processed_dir,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    env = dotenv_values()

    def get_env(key):
        if val := env.get(key, os.environ.get(key, None)):
            return val
        else:
            sys.exit(f"Missing env '{key}'")

    dsn = get_env("PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN")
    media_host = get_env("MEDIA_HOST")
    media_port = get_env("MEDIA_PORT")
    media_user = get_env("MEDIA_USER")
    media_pass = get_env("MEDIA_PASSWORD")

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    files = get_files(args)
    mdrepo_id = f"MDR{args.simulation_id:08d}"

    #
    # Media Files
    #
    media_server = fabric.Connection(
        media_host,
        port=media_port,
        user=media_user,
        connect_kwargs={"password": media_pass},
    )
    media_dir = f"/home/web/mdrepo/{args.server}/{mdrepo_id}"
    print(f"Making media dir '{media_dir}'")
    media_server.run(f'mkdir -p "{media_dir}"', warn=False)

    if any(map(lambda t: t in args.file_types, ["all", "media"])):
        for local_path in files["media_files"]:
            if os.path.isfile(local_path):
                remote_path = os.path.join(media_dir, os.path.basename(local_path))
                print(f" {local_path} -> {remote_path}")
                media_server.put(local_path, remote=remote_path)
            else:
                print(f"Invalid path '{local_path}'")

    push = []
    if any(map(lambda t: t in args.file_types, ["all", "original"])):
        push.append(("original", files["original_files"]))
    if any(map(lambda t: t in args.file_types, ["all", "processed"])):
        push.append(("processed", files["processed_files"]))

    #
    # IRODS Files
    #
    irods_root = f"/iplant/home/shared/mdrepo/{args.server}/release/{mdrepo_id}"

    results = []
    with iRODSSession(irods_env_file=args.irods_env) as session:
        if args.remove_processed_dir:
            processed_path = os.path.join(irods_root, "processed")
            if session.collections.exists(processed_path):
                print(f"Removing '{processed_path}'")
                cmd = f"gocmd rm -rf {processed_path}"
                rv, out = getstatusoutput(cmd)
                if rv != 0:
                    sys.exit(f"Error running '{cmd}': {out}")

        for sub_dir, paths in push:
            irods_dir = os.path.join(irods_root, sub_dir)
            print(f"Checking IRODS dir '{irods_dir}'")
            if not session.collections.exists(irods_dir):
                print(f"Making IRODS dir '{irods_dir}'")
                session.collections.create(irods_dir)

            # Gather the files needing upload so they can be pushed with a
            # single "gocmd put" rather than one connection per file
            upload = []
            upload_size = 0
            for local_path in paths:
                if not os.path.isfile(local_path):
                    print(f"Invalid path '{local_path}'")
                    continue

                local_size = os.path.getsize(local_path)
                human_size = humanize.naturalsize(local_size)

                # Check if we can skip
                basename = os.path.basename(local_path)
                remote_path = os.path.join(irods_dir, basename)
                remote_size = 0
                if session.data_objects.exists(remote_path):
                    obj = session.data_objects.get(remote_path)
                    remote_size = obj.size

                if local_size == remote_size:
                    print(f" {local_path} [{human_size}] (already uploaded)")
                else:
                    print(f" {local_path} [{human_size}] ->\n  {irods_dir}")
                    upload.append(local_path)
                    upload_size += local_size

                results.append(
                    {
                        "src": local_path,
                        "dest": remote_path,
                        "size": local_size,
                    }
                )

            if upload:
                human_size = humanize.naturalsize(upload_size)
                print(
                    f"Uploading {len(upload)} file(s) [{human_size}] "
                    f"to '{irods_dir}'",
                    end="",
                )
                sys.stdout.flush()

                start = dt.now()
                sources = " ".join(map(shlex.quote, upload))
                cmd = f"gocmd put --thread-num 10 -f {sources} {shlex.quote(irods_dir)}"
                rv, out = getstatusoutput(cmd)
                if rv != 0:
                    sys.exit(f"Error running '{cmd}': {out}")

                elapsed = humanize.precisedelta(dt.now() - start)
                print(f" (took {elapsed})")

    cur.execute(
        """
        update md_simulation
        set    is_placeholder=False
        where  id=%s
        """,
        (args.simulation_id,),
    )

    print(f"results = {results}")

    if filename := args.out_file:
        print(f"Writing results to '{filename}'")
        with open(filename, "wt") as fh:
            print(json.dumps(results, indent=4), file=fh)

    print("Done")


# --------------------------------------------------
def get_files(args: Args) -> Dict[str, List[str]]:
    """Find local files"""

    metadata = json.loads(args.file.read())
    simulation = metadata["simulation"]

    def mkpath(filename):
        return os.path.join(args.data_dir, filename)

    files = {
        "original_files": [],
        "processed_files": [],
        "media_files": [],
    }

    for file in simulation.get("original_files", []):
        files["original_files"].append(mkpath(file["name"]))

    for file in simulation["processed_files"]:
        local_path = mkpath(os.path.join("processed", file["name"]))
        files["processed_files"].append(local_path)

        if file["name"] in ["thumbnail.png", "minimal.pdb", "sampled.xtc"]:
            files["media_files"].append(local_path)

    return files


# --------------------------------------------------
if __name__ == "__main__":
    main()
