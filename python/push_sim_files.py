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
import queue
import shlex
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

# Default transfer threads for ONE put. 0 means "let python-irodsclient
# decide", which is 3 for anything over 32 MB -- the behaviour this script had
# before 2026-08-20.
#
# Do NOT hardcode this to 1. It was, briefly, on the strength of a download
# benchmark taken on the post-processing host against a congested server, where
# the server was the bottleneck and extra streams bought nothing. Uploading from
# the processing VM is a different regime: on a high-latency WAN link a single
# TCP stream is capped by the bandwidth-delay product, and num_threads=1 does
# not merely use fewer streams -- it fails should_parallelize_transfer() and
# drops the put into a synchronous chunked write loop over one connection. On
# the VM working the 18k backlog that turned four concurrent pushes into
# something close to sequential.
#
# Tune it per host with --transfer-threads. Lower values trade throughput for
# IRODS connections, which matter because CyVerse's ~500 ceiling is shared
# across all of their users.
DEFAULT_TRANSFER_THREADS = 0
# Nothing bounded a whole invocation of this script. The socket read is
# bounded (python-irodsclient defaults connection_timeout to 120s) and one
# file is bounded by NUM_RETRIES, so the worst case for a single file is about
# six minutes -- yet on 2026-08-18 two of ticket 2243's landings sat in here
# for 1h51m against a degraded IRODS, between them consuming 3 seconds of CPU
# and holding two of mdr-process's four worker slots the entire time. They had
# to be killed by hand before the ticket could finish. A landing that is
# working takes about two minutes, so an hour is thirty times the headroom a
# healthy push needs and still cuts a wedged one loose the same shift.
PUSH_TIMEOUT = 3600  # seconds
# What the graceful abort gets before the process leaves anyway. The wedged
# case is exactly the one where waiting for transfer threads to wind down is
# itself what hangs, so the deadline cannot depend on them cooperating.
ABORT_GRACE = 120  # seconds

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
    transfer_threads: int
    timeout: int


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
        "--transfer-threads",
        help="Threads per single transfer (0 = client default, 3 over 32MB)",
        metavar="INT",
        type=int,
        default=DEFAULT_TRANSFER_THREADS,
    )

    parser.add_argument(
        "--timeout",
        help="Seconds before the whole push gives up. Bounds the invocation, "
        "which neither the socket timeout nor the per-file retries do "
        "(0 disables)",
        metavar="INT",
        type=int,
        default=PUSH_TIMEOUT,
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
        transfer_threads=args.transfer_threads,
        timeout=args.timeout,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    # Arm the wall-clock deadline before anything can block. SIGALRM is
    # delivered to the main thread, which is where this runs and where the
    # 2026-08-18 hang was parked waiting on its worker pool.
    if args.timeout:
        signal.signal(signal.SIGALRM, deadline_reached)
        signal.alarm(args.timeout)

    env = dotenv_values()

    def get_env(key):
        if val := env.get(key, os.environ.get(key, None)):
            return val
        else:
            sys.exit(f"Missing env '{key}'")

    media_host = get_env("MEDIA_HOST")
    media_port = get_env("MEDIA_PORT")
    media_user = get_env("MEDIA_USER")
    media_pass = get_env("MEDIA_PASSWORD")

    files = get_files(args)
    errors: List[str] = []
    file_results: List[dict] = []
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

    do_media = any(t in args.file_types for t in ["all", "media"])
    if do_media:
        for local_path in files["media_files"]:
            if os.path.isfile(local_path):
                remote_path = os.path.join(media_dir, os.path.basename(local_path))
                print(f" {local_path} -> {remote_path}")
                try:
                    media_server.put(local_path, remote=remote_path)
                except Exception as e:
                    errors.append(f"{local_path} -> media: {e}")
            else:
                errors.append(f"Invalid path '{local_path}'")

    push = []
    if any(map(lambda t: t in args.file_types, ["all", "original"])):
        push.append(("original", files["original_files"]))
    if any(map(lambda t: t in args.file_types, ["all", "processed"])):
        push.append(("processed", files["processed_files"]))

    #
    # IRODS Files
    #
    irods_root = f"/iplant/home/shared/mdrepo/{args.server}/release/{mdrepo_id}"

    # Every file this run is responsible for, paired with the remote path and
    # the expected MD5 (from import.json) to verify it against after uploading.
    targets = []
    if do_media:
        for local_path in files["media_files"]:
            meta = files["meta"].get(local_path, {})
            targets.append(
                {
                    "location": "media",
                    "src": local_path,
                    "dest": os.path.join(media_dir, os.path.basename(local_path)),
                    "expected_md5": meta.get("md5", ""),
                    "size": meta.get("size", 0),
                }
            )
    for sub_dir, paths in push:
        for local_path in dict.fromkeys(paths):
            meta = files["meta"].get(local_path, {})
            targets.append(
                {
                    "location": "irods",
                    "src": local_path,
                    "dest": os.path.join(
                        irods_root, sub_dir, os.path.basename(local_path)
                    ),
                    "expected_md5": meta.get("md5", ""),
                    "size": meta.get("size", 0),
                }
            )

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

                # A path listed twice must not be force-put twice concurrently
                for local_path in dict.fromkeys(paths):
                    if not os.path.isfile(local_path):
                        print(f"Invalid path '{local_path}'")
                        continue

                    local_size = os.path.getsize(local_path)
                    human_size = humanize.naturalsize(local_size)

                    # Check if we can skip.
                    #
                    # On the MD5, not the size (changed 2026-08-10). Size is not
                    # an identity: a tar pads to 10,240-byte blocks, so a
                    # regenerated tar with different content routinely lands on
                    # the byte-exact size of the one already in IRODS. Directory
                    # 10gs did exactly that -- four tars (178M-366M) rebuilt on
                    # 08-10 matched the sizes of objects written 2026-07-23, so
                    # all four were skipped as "already uploaded", no bytes
                    # moved, and the run failed post-upload verification with
                    # complete: false. The smaller files in the same push were
                    # byte-identical to July's and so were genuinely fine, which
                    # is what made it look like a size-threshold problem.
                    #
                    # chksum() forces a server-side hash rather than trusting the
                    # catalog, matching verify_irods() below -- catalog metadata
                    # on this zone has been demonstrably stale (see the replica
                    # divergence in the DB backups). That costs server time on a
                    # re-run, but only for objects that already exist: a first
                    # push finds nothing remote and pays none of it.
                    basename = os.path.basename(local_path)
                    remote_path = os.path.join(irods_dir, basename)
                    local_md5 = files["meta"].get(local_path, {}).get("md5", "")
                    remote_size = 0
                    remote_md5 = ""
                    if session.data_objects.exists(remote_path):
                        obj = session.data_objects.get(remote_path)
                        remote_size = obj.size
                        chksum = obj.chksum() or ""
                        remote_md5 = chksum.split(":", 1)[-1].strip().lower()

                    # Fall back to the old size test only when a checksum is
                    # genuinely unavailable on one side, so a manifest without
                    # an md5 behaves as it did before rather than re-uploading
                    # everything.
                    if local_md5 and remote_md5:
                        can_skip = remote_md5 == local_md5.strip().lower()
                        reason = "md5 differs"
                    else:
                        can_skip = local_size == remote_size
                        reason = "size differs, no md5 to compare"

                    if can_skip:
                        print(f" {local_path} [{human_size}] (already uploaded)")
                    else:
                        print(f" {local_path} [{human_size}] (queued: {reason})")
                        upload.append(local_path)
                        upload_size += local_size

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
                    reported = set()
                    futures = {}
                    pool = ThreadPoolExecutor(max_workers=num_threads)
                    try:
                        futures = {
                            pool.submit(
                                put_file,
                                sessions,
                                path,
                                irods_dir,
                                args.transfer_threads,
                            ): path
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
                            basename = os.path.basename(local_path)
                            if future.cancelled():
                                errors.append(f"{local_path}: not uploaded (aborted)")
                            elif exc := future.exception():
                                errors.append(f"{local_path}: {exc}")
                            else:
                                # Finished while the pool was shutting down
                                took = humanize.precisedelta(future.result())
                                print(f" {basename} (took {took})")

                    if ABORT.is_set():
                        message = "Uploads aborted"
                        if errors:
                            message += ":\n" + "\n".join(errors)
                        sys.exit(message)

                    elapsed = humanize.precisedelta(dt.now() - start)
                    print(f"Uploaded {len(upload)} file(s) in {elapsed}")
        finally:
            # Leaving clones open causes SYS_HEADER_READ_LEN_ERR
            while not sessions.empty():
                sessions.get().cleanup()

        # Verify every file landed on its remote with a matching MD5. This runs
        # regardless of upload errors: the is_placeholder flip (now done in
        # Rust) keys off observed remote state, not off whether this particular
        # run uploaded cleanly. A partial push therefore leaves the simulation a
        # placeholder, and a later run that completes it clears the flag.
        print("Verifying uploads")
        for target in targets:
            if target["location"] == "media":
                present, remote_md5 = verify_media(media_server, target["dest"])
            else:
                present, remote_md5 = verify_irods(
                    session, target["dest"], target["size"]
                )

            verified = present and remote_md5 == target["expected_md5"].lower()
            if not verified:
                if not present:
                    why = "missing"
                elif remote_md5 is None:
                    why = "no checksum available"
                else:
                    why = "md5 mismatch"
                print(
                    f" NOT VERIFIED [{target['location']}] {target['dest']} ({why})"
                )

            file_results.append(
                {
                    "location": target["location"],
                    "src": target["src"],
                    "dest": target["dest"],
                    "size": target["size"],
                    "expected_md5": target["expected_md5"].lower(),
                    "remote_md5": remote_md5,
                    "present": present,
                    "verified": verified,
                }
            )

    complete = bool(file_results) and all(f["verified"] for f in file_results)

    result = {
        "simulation_id": args.simulation_id,
        "complete": complete,
        "files": file_results,
        "errors": errors,
    }

    if filename := args.out_file:
        print(f"Writing results to '{filename}'")
        with open(filename, "wt") as fh:
            print(json.dumps(result, indent=4), file=fh)

    n_verified = sum(1 for f in file_results if f["verified"])
    print(f"Verified {n_verified}/{len(file_results)} file(s); complete={complete}")
    if errors:
        print("Upload errors:\n" + "\n".join(errors))
    print("Done")


# --------------------------------------------------
def hard_exit(_signum, _frame) -> None:
    """Leave immediately, without waiting on transfer threads"""

    print(
        f"Deadline exceeded and {ABORT_GRACE}s of grace elapsed; exiting",
        file=sys.stderr,
        flush=True,
    )
    # os._exit rather than sys.exit: the threads this is escaping from are
    # not daemons, so a normal exit would block on the join that is stuck.
    os._exit(1)


# --------------------------------------------------
def deadline_reached(_signum, _frame) -> None:
    """Abort on the wall-clock deadline, then leave whether or not it works

    Reuses the abort path a terminating signal takes, so a timeout winds the
    transfers down the same way a SIGTERM does, and falls back to leaving
    outright after ABORT_GRACE -- the same two-stage shape abort_uploads uses
    for a second signal, and for the same reason.
    """

    signal.signal(signal.SIGALRM, hard_exit)
    signal.alarm(ABORT_GRACE)
    ABORT.set()
    abort_parallel_transfers()
    print("Push deadline reached; aborting uploads", file=sys.stderr, flush=True)


# --------------------------------------------------
def abort_uploads(signum, _frame) -> None:
    """Stop the in-flight IRODS transfers on a terminating signal"""

    # Let a second signal kill the process outright in case a transfer
    # thread refuses to wind down
    signal.signal(signum, signal.SIG_DFL)
    ABORT.set()
    abort_parallel_transfers()


# --------------------------------------------------
def put_file(
    sessions: queue.Queue,
    local_path: str,
    irods_dir: str,
    transfer_threads: int,
) -> timedelta:
    """Upload one file to IRODS, retrying on failure"""

    basename = os.path.basename(local_path)
    remote_path = os.path.join(irods_dir, basename)

    # Files over 32M are transferred with multiple threads automatically.
    #
    # REG_CHKSUM_KW makes the server hash the bytes as it writes them and
    # register the result in the catalog. That is what lets verify_irods()
    # read a checksum back as plain metadata instead of calling chksum(),
    # which asks the server to resolve a resource hierarchy and re-read the
    # whole object -- the RPC that raised HIERARCHY_ERROR on 46 groups across
    # the 08-12 and 08-14 merge runs. Without this, a put registers no
    # checksum at all, so there is nothing for verification to fall back on.
    # It is also the cheaper of the two: hashing during the write replaces a
    # second full read of every file, which matters at 10G per full.tar.
    options = {kw.FORCE_FLAG_KW: "", kw.REG_CHKSUM_KW: ""}

    for attempt in range(1, NUM_RETRIES + 1):
        if ABORT.is_set():
            raise RuntimeError("aborted before the upload started")

        # Borrow a session for the length of this attempt only
        session = sessions.get()
        try:
            start = dt.now()
            session.data_objects.put(
                local_path,
                remote_path,
                num_threads=transfer_threads,
                **options,
            )
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
        # local_path -> {"md5", "size"} from import.json, for post-upload
        # verification
        "meta": {},
    }

    for file in simulation.get("original_files", []):
        local_path = mkpath(file["name"])
        files["original_files"].append(local_path)
        files["meta"][local_path] = {"md5": file["md5_sum"], "size": file["size"]}

    for file in simulation["processed_files"]:
        local_path = mkpath(os.path.join("processed", file["name"]))
        files["processed_files"].append(local_path)
        files["meta"][local_path] = {"md5": file["md5_sum"], "size": file["size"]}

        if file["name"] in ["thumbnail.png", "minimal.pdb", "sampled.xtc"]:
            files["media_files"].append(local_path)

    return files


# --------------------------------------------------
def verify_irods(session, remote_path: str, expected_size: int):
    """Return (present, md5) for an IRODS object. `present` requires the remote
    size to match the local file; `md5` is the object's MD5 as IRODS has it (the
    zone hashes with MD5, so this compares to our manifest directly). A `md5` of
    None means the object is there but could not be checksummed -- the caller
    treats that as unverified, never as a pass."""

    if not session.data_objects.exists(remote_path):
        return (False, None)

    obj = session.data_objects.get(remote_path)
    if obj.size != expected_size:
        return (False, None)

    # Read the checksum the catalog already holds. put_file() registers it at
    # upload time, so this is the normal path, and it is a metadata lookup: no
    # hierarchy to resolve, nothing to re-read. MD5-scheme checksums come back
    # as bare hex; strip any "md5:" prefix defensively.
    chksum = obj.checksum or ""

    if not chksum:
        # Nothing registered -- an object uploaded before REG_CHKSUM_KW, or by
        # something other than this script. Forcing a computation is the only
        # way left to get an answer, and it is exactly the call that returned
        # HIERARCHY_ERROR on 46 merge groups. It stays behind a fallback, and
        # its failure is contained: one unverifiable file is a NOT VERIFIED
        # line and a simulation left a placeholder, which a later push clears.
        # Letting it propagate killed the whole verification pass instead, and
        # with it the run -- so nothing was recorded about the other files.
        try:
            chksum = obj.chksum() or ""
        except Exception as err:
            print(f" could not checksum {remote_path}: {err}")
            return (True, None)

    return (True, chksum.split(":", 1)[-1].strip().lower())


# --------------------------------------------------
def verify_media(media_server, remote_path: str):
    """Return (present, md5) for a file on the media server, via `md5sum` over
    SSH. A missing file makes md5sum fail, which reads as not-present."""

    result = media_server.run(
        f"md5sum -- {shlex.quote(remote_path)}", warn=True, hide=True
    )
    if result.ok and result.stdout.strip():
        return (True, result.stdout.split()[0].strip().lower())
    return (False, None)


# --------------------------------------------------
if __name__ == "__main__":
    main()
