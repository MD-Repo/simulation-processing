#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2025-10-26
Purpose: Push simulation files to cat/IRODS
"""

import argparse
import fabric
import irods.keywords as kw
import json
import os
import psycopg2
import queue
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt, timedelta
import humanize
from dotenv import dotenv_values
from irods.parallel import abort_parallel_transfers
from irods.session import iRODSSession
from typing import Dict, List, NamedTuple, TextIO, Optional
from subprocess import getstatusoutput

# Attempts per file before giving up
NUM_RETRIES = 3

# Serializes output from the upload threads
PRINT_LOCK = threading.Lock()

# Set when a terminating signal asks the uploads to stop
ABORT = threading.Event()


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
    threads: int


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
        "-n",
        "--threads",
        help="Number of concurrent IRODS uploads",
        metavar="INT",
        type=int,
        default=4,
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

    if args.threads < 1:
        parser.error(f'--threads "{args.threads}" must be positive')

    return Args(
        file=args.file,
        simulation_id=args.simulation_id,
        irods_env=args.irods_env,
        server=args.server,
        data_dir=args.data_dir,
        file_types=args.file_types,
        out_file=args.out_file,
        remove_processed_dir=args.remove_processed_dir,
        threads=args.threads,
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

    # A session cannot be shared across threads, so each upload thread borrows
    # a clone from here (see the python-irodsclient README). Filled on first
    # use and shared by both sub dirs so we only authenticate once.
    sessions = queue.Queue()

    with iRODSSession(irods_env_file=args.irods_env) as session:
        if args.remove_processed_dir:
            processed_path = os.path.join(irods_root, "processed")
            if session.collections.exists(processed_path):
                print(f"Removing '{processed_path}'")
                cmd = f"gocmd rm -rf {processed_path}"
                rv, out = getstatusoutput(cmd)
                if rv != 0:
                    sys.exit(f"Error running '{cmd}': {out}")

        try:
            for sub_dir, paths in push:
                irods_dir = os.path.join(irods_root, sub_dir)
                print(f"Checking IRODS dir '{irods_dir}'")
                if not session.collections.exists(irods_dir):
                    print(f"Making IRODS dir '{irods_dir}'")
                    session.collections.create(irods_dir)

                # Gather the files needing upload so they can all be pushed
                # concurrently rather than one at a time
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
                        print(f" {local_path} [{human_size}] (queued)")
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
                    num_threads = min(args.threads, len(upload))
                    print(
                        f"Uploading {len(upload)} file(s) [{human_size}] to "
                        f"'{irods_dir}' using {num_threads} thread(s)"
                    )
                    sys.stdout.flush()

                    # Every thread must be able to take a clone without
                    # waiting, but clones from an earlier sub dir still count
                    while sessions.qsize() < num_threads:
                        sessions.put(session.clone())

                    # A terminating signal must reach the parallel transfer
                    # threads that python-irodsclient spawns for files over
                    # 32M, or they can keep running after the main program
                    # is done
                    prev_handlers = {
                        sig: signal.signal(sig, abort_uploads)
                        for sig in (signal.SIGINT, signal.SIGTERM)
                    }

                    start = dt.now()
                    errors = []
                    reported = set()
                    futures = {}
                    pool = ThreadPoolExecutor(max_workers=num_threads)
                    try:
                        futures = {
                            pool.submit(put_file, sessions, path, irods_dir): path
                            for path in upload
                        }

                        for future in as_completed(futures):
                            local_path = futures[future]
                            basename = os.path.basename(local_path)
                            reported.add(future)
                            try:
                                took = humanize.precisedelta(future.result())
                                message = f" {basename} (took {took})"
                            except Exception as e:
                                message = f" {basename} FAILED: {e}"
                                errors.append(f"{local_path}: {e}")

                            # The upload threads print retries under this lock
                            with PRINT_LOCK:
                                print(message)
                                sys.stdout.flush()

                            if ABORT.is_set():
                                print("Dropping the uploads that have not started")
                                break
                    finally:
                        pool.shutdown(wait=True, cancel_futures=ABORT.is_set())

                        for sig, handler in prev_handlers.items():
                            signal.signal(sig, handler)

                        # Account for the files the loop above broke out of
                        for future, local_path in futures.items():
                            if future in reported:
                                continue
                            if future.cancelled():
                                errors.append(f"{local_path}: not uploaded (aborted)")
                            elif exc := future.exception():
                                errors.append(f"{local_path}: {exc}")

                    if ABORT.is_set():
                        sys.exit("Uploads aborted:\n" + "\n".join(errors))

                    if errors:
                        sys.exit("Upload errors:\n" + "\n".join(errors))

                    elapsed = humanize.precisedelta(dt.now() - start)
                    print(f"Uploaded {len(upload)} file(s) in {elapsed}")
        finally:
            # Leaving clones open causes SYS_HEADER_READ_LEN_ERR
            while not sessions.empty():
                sessions.get().cleanup()

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
def abort_uploads(signum, _frame) -> None:
    """Stop the in-flight IRODS transfers on a terminating signal"""

    # Let a second signal kill the process outright in case a transfer
    # thread refuses to wind down
    signal.signal(signum, signal.SIG_DFL)
    ABORT.set()
    abort_parallel_transfers()


# --------------------------------------------------
def put_file(sessions: queue.Queue, local_path: str, irods_dir: str) -> timedelta:
    """Upload one file to IRODS, retrying on failure"""

    basename = os.path.basename(local_path)
    remote_path = os.path.join(irods_dir, basename)

    # Files over 32M are transferred with multiple threads automatically
    options = {kw.FORCE_FLAG_KW: ""}

    for attempt in range(1, NUM_RETRIES + 1):
        if ABORT.is_set():
            raise RuntimeError("aborted before the upload started")

        # Borrow a session for the length of this attempt only
        session = sessions.get()
        try:
            start = dt.now()
            session.data_objects.put(local_path, remote_path, **options)
            return dt.now() - start
        except Exception as e:
            # A failed transfer can leave a connection mid-protocol, so drop
            # this clone's connections rather than retry over them
            session.cleanup()

            # An aborted transfer raises like any other failure, so check
            # before deciding this one is worth another attempt
            if ABORT.is_set() or attempt == NUM_RETRIES:
                raise

            with PRINT_LOCK:
                print(f" {basename} attempt {attempt} failed: {e}")
                sys.stdout.flush()
        finally:
            sessions.put(session)

        # Backing off out here leaves the session free for another thread.
        # The wait returns True as soon as an abort is signalled.
        if ABORT.wait(2**attempt):
            raise RuntimeError("aborted during the retry backoff")


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
