#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-14
Purpose: Process the bundles2/ backlog -- simulation directories a contributor
         is pushing directly into IRODS at
         /iplant/home/shared/mdrepo/bundles2/, outside the ticket-upload flow,
         so nothing in md_ticket / md_process_job knows they exist.

         One invocation claims the oldest ready bundle, downloads it, runs
         "mdr-process process" on it, and cleans up. Meant to run from cron,
         one bundle per tick -- there is deliberately no long-lived loop or
         retry/backoff machinery here, mirroring drain_process_queue.py's
         "one claim per invocation" shape.

         Two things keep this from fighting the live ticket queue for the
         same VM's CPU/GPU/IRODS connections:
           - It takes the exact flock drain_process_queue.py uses, so the two
             never run mdr-process at once.
           - Before claiming a bundle it checks md_process_job for any
             pending/running row on this server and steps aside for the tick
             if so, so a real contributor's ticket gets the next opening
             rather than waiting behind the backlog.

         A bundle is ready once its contributor-written sibling "<name>.md5"
         sidecar exists and is non-empty -- mirrors check_new_simulations.py's
         completion-marker check (upload_complete()): a 0-byte marker from a
         truncated write must not read as "done". The sidecar's contents are
         not otherwise verified; it is a completion signal, not a checksum
         this script checks against.

         "mdr-process process" already treats an Ok exit as verified and
         permanent: process.rs calls push_sim_files.py and only clears
         is_placeholder once every file is confirmed by MD5, bail!ing (Err)
         otherwise. That is the same bar purge_processed_landings.py's
         docstring cites before it will delete a landing collection, so on
         success this deletes the bundles2/ source and its .md5 sidecar too.

         On failure the local scratch copy is still deleted -- there is no
         DB row and no ticket tracking these 15,000 directories, so keeping a
         failed download around locally costs disk (shared with live ticket
         processing) without helping anyone. The bundles2/ source and its
         .md5 are left untouched on failure, so the bundle stays exactly
         where a human can find it, inspect the log, and re-run this script
         with --dry-run against it once the cause is fixed. Nothing here
         retries automatically.
"""

import argparse
import json
import os
import shlex
import shutil
import ssl
import subprocess
import sys
from subprocess import getstatusoutput
from typing import Dict, NamedTuple, Optional, Tuple

import psycopg2
import psycopg2.extras
from irods.exception import DoesNotExist
from irods.session import iRODSSession

from common import (
    CRON_STATE_ROOT,
    FRONTEND_BASE_URLS,
    send_slack_message,
    stamp,
)
from drain_process_queue import (
    KILL_GRACE,
    PROCESS_TIMEOUT,
    acquire_lock,
    kill_process_group,
    load_env,
    tail_file,
)

BUNDLES_COLLECTION = "/iplant/home/shared/mdrepo/bundles2"
ERROR_LINES = 20  # tail of mdr-process output to include in a failure notice


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    lock_file: str
    state_file: str
    work_dir: str
    log_dir: str
    irods_env: str
    dry_run: bool
    verbose: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Process the bundles2/ direct-upload backlog",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--server",
        help="Target server",
        metavar="STR",
        choices=["staging", "prod"],
        default="prod",
    )

    parser.add_argument(
        "--lock-file",
        help="flock path shared with drain_process_queue.py, so the two "
        "never run mdr-process at once (default: the same lock file "
        "drain_process_queue.py uses for this server)",
        metavar="PATH",
        default=None,
    )

    parser.add_argument(
        "--state-file",
        help="JSON file tracking which bundles are done/failed "
        f"(default: {CRON_STATE_ROOT}/process_bundles2-<server>.json)",
        metavar="PATH",
        default=None,
    )

    parser.add_argument(
        "--work-dir",
        help="Scratch directory bundles are downloaded into before "
        "processing (default: /opt/mdrepo/landing/bundles2, alongside "
        "fetch_uploads.py's /opt/mdrepo/landing/<server>)",
        metavar="DIR",
        default="/opt/mdrepo/landing/bundles2",
    )

    parser.add_argument(
        "--log-dir",
        help="Per-bundle mdr-process debug logs "
        "(default: /opt/mdrepo/logs/bundles2/<server>)",
        metavar="DIR",
        default=None,
    )

    parser.add_argument(
        "--irods-env",
        help="IRODS environment file",
        metavar="FILE",
        default=os.environ.get(
            "IRODS_ENVIRONMENT_FILE",
            os.path.expanduser("~/.irods/irods_environment.json"),
        ),
    )

    parser.add_argument(
        "--dry-run",
        help="Download the next candidate and run 'mdr-process process "
        "--dry-run' on it, but never delete the IRODS source or touch the "
        "state file, so the same or a later run can inspect it again "
        "(implies --verbose)",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Verbose", action="store_true")

    args = parser.parse_args()

    lock_file = args.lock_file or os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"drain_process_queue-{args.server}.lock",
    )
    state_file = args.state_file or os.path.join(
        CRON_STATE_ROOT, f"process_bundles2-{args.server}.json"
    )
    log_dir = args.log_dir or os.path.join(
        "/opt/mdrepo/logs/bundles2", args.server
    )

    return Args(
        server=args.server,
        lock_file=lock_file,
        state_file=state_file,
        work_dir=args.work_dir,
        log_dir=log_dir,
        irods_env=args.irods_env,
        dry_run=args.dry_run,
        verbose=args.verbose or args.dry_run,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    load_env()

    def status(msg: str) -> None:
        if args.verbose:
            print(f"{stamp()} {msg}", flush=True)

    if args.dry_run:
        status("DRY RUN: no IRODS deletions or state changes will be made")

    # Shared with drain_process_queue.py: whichever of the two gets here
    # first holds mdr-process for the rest of its run.
    lock_fd = acquire_lock(args.lock_file)
    if lock_fd is None:
        status(f"Another worker holds {args.lock_file}, exiting")
        sys.exit(0)

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dsn = os.environ.get(env_key)
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if ticket_queue_busy(cur, args.server):
        status(
            "Ticket queue has pending/running work, stepping aside this tick"
        )
        sys.exit(0)

    state = load_state(args.state_file)

    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
    base_url = FRONTEND_BASE_URLS[args.server]

    with iRODSSession(
        irods_env_file=args.irods_env, ssl_context=ssl_context
    ) as session:
        name = find_next_candidate(session, state)
        if name is None:
            status("No ready bundles found")
            sys.exit(0)

        status(f"Claimed bundle {name}")

        try:
            local_dir = fetch_bundle(name, args.work_dir)
        except Exception as e:
            # Not recorded in state, and there is deliberately no cleanup of
            # a possibly-partial local download: gocmd's own --diff makes the
            # next tick's retry resume rather than re-pull everything.
            status(f"ERROR fetching {name}: {e}")
            send_slack_message(
                f"bundles2 fetch failed for {name}: {e}", base_url
            )
            sys.exit(1)

        log_file = os.path.join(args.log_dir, f"{name}.log")
        returncode, detail = run_mdr_process(
            local_dir, args.server, log_file, args.dry_run, status
        )

        if args.dry_run:
            status(f"DRY RUN result for {name}: exit {returncode}\n{detail}")
            status(f"Local copy left at {local_dir} for inspection")
            sys.exit(0)

        shutil.rmtree(local_dir, ignore_errors=True)

        if returncode == 0:
            delete_irods_source(session, name)
            state[name] = {"status": "done", "timestamp": stamp()}
            save_state(args.state_file, state)
            status(f"Processed {name}")
        else:
            state[name] = {
                "status": "failed",
                "timestamp": stamp(),
                "error": detail,
            }
            save_state(args.state_file, state)
            status(f"FAILED {name}: {detail}")
            send_slack_message(
                f"bundles2 processing failed for {name} (exit {returncode}). "
                f"Source left in place at {BUNDLES_COLLECTION}/{name} for a "
                f"human to retry after fixing the cause.\nLog: {log_file}",
                base_url,
            )


# --------------------------------------------------
def ticket_queue_busy(cur, server: str) -> bool:
    """Whether md_process_job has any pending/running work for this server"""

    cur.execute(
        "select 1 from md_process_job "
        "where server = %s and status in ('pending', 'running') limit 1",
        (server,),
    )
    return cur.fetchone() is not None


# --------------------------------------------------
def load_state(path: str) -> Dict[str, dict]:
    """Bundle name -> {"status": "done"|"failed", "timestamp", "error"?}"""

    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        # A crash mid-write could in principle leave this truncated. Treating
        # that as "nothing recorded yet" costs at most a few re-downloads of
        # already-done bundles (harmless: mdr-process/import is not re-run
        # against a live simulation without --force), which is cheap next to
        # a cron job that refuses to run at all.
        return {}


# --------------------------------------------------
def save_state(path: str, state: Dict[str, dict]) -> None:
    """Write atomically so a crash mid-save can't corrupt the file"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


# --------------------------------------------------
def find_next_candidate(session, state: Dict[str, dict]) -> Optional[str]:
    """Oldest ready bundle not already recorded in state, or None"""

    coll = session.collections.get(BUNDLES_COLLECTION)
    candidates = []
    for obj in coll.data_objects:
        if not obj.name.endswith(".md5"):
            continue
        if obj.size == 0:
            continue  # truncated/mid-write sidecar -- not ready yet
        name = obj.name[: -len(".md5")]
        if name in state:
            continue
        candidates.append((obj.create_time, name))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# --------------------------------------------------
def fetch_bundle(name: str, work_dir: str) -> str:
    """gocmd get the bundle collection into work_dir, return the local path

    Mirrors fetch_uploads.py's run_gocmd. "--diff" skips files already
    present with a matching hash, so a retry after a partial download or a
    failed mdr-process run does not re-pull everything.
    """

    os.makedirs(work_dir, exist_ok=True)
    src = f"{BUNDLES_COLLECTION}/{name}"
    cmd = f"gocmd get -f --diff {shlex.quote(src)} {shlex.quote(work_dir)}"
    rv, out = getstatusoutput(cmd)
    if rv != 0:
        raise RuntimeError(f"Error running {cmd}: {out}")

    return os.path.join(work_dir, name)


# --------------------------------------------------
def run_mdr_process(
    local_dir: str, server: str, log_file: str, dry_run: bool, status
) -> Tuple[Optional[int], str]:
    """Run "mdr-process process" on a local directory.

    Returns (returncode, detail): detail is empty on success. returncode is
    -1 if the binary was never found, or None if it was killed for exceeding
    PROCESS_TIMEOUT. Reuses drain_process_queue.py's process-group kill and
    log-tail helpers rather than re-implementing that handling here.
    """

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    cmd = [
        "mdr-process",
        "-l",
        "debug",
        "--log-file",
        log_file,
        "process",
        local_dir,
        "--server",
        server,
    ]
    if dry_run:
        cmd.append("--dry-run")

    status(f"Running {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError:
        return -1, "mdr-process not found on PATH"

    timed_out = False
    try:
        out, err = proc.communicate(timeout=PROCESS_TIMEOUT)
    except subprocess.TimeoutExpired:
        timed_out = True
        signals = kill_process_group(proc.pid, status)
        try:
            out, err = proc.communicate(timeout=KILL_GRACE)
        except subprocess.TimeoutExpired:
            out, err = "", ""

    if timed_out:
        hours = PROCESS_TIMEOUT // 3600
        detail = (
            f"KILLED for exceeding the {hours}h time limit "
            f"(PROCESS_TIMEOUT={PROCESS_TIMEOUT}s). Sent {signals} to the "
            f"process group. Whatever it was doing was left unfinished and "
            f"needs a human."
        )
        if log_tail := tail_file(log_file, ERROR_LINES):
            detail += (
                f"\n\nLast {ERROR_LINES} lines of debug log:\n{log_tail}"
            )
        return None, detail

    if proc.returncode == 0:
        return 0, ""

    err = (err or out or "").strip()
    detail = err or "no error output"
    if log_tail := tail_file(log_file, ERROR_LINES):
        detail += f"\n\nLast {ERROR_LINES} lines of debug log:\n{log_tail}"
    return proc.returncode, detail


# --------------------------------------------------
def delete_irods_source(session, name: str) -> None:
    """Delete bundles2/<name>/ and its .md5 sidecar.

    Only called after mdr-process has exited 0, which -- per process.rs --
    means push_sim_files.py already verified every file by MD5 in permanent
    storage. Same bar purge_processed_landings.py uses before it deletes a
    landing collection.
    """

    src_dir = f"{BUNDLES_COLLECTION}/{name}"
    try:
        coll = session.collections.get(src_dir)
        coll.remove(recurse=True, force=True)
    except DoesNotExist:
        pass

    try:
        obj = session.data_objects.get(f"{src_dir}.md5")
        obj.unlink(force=True)
    except DoesNotExist:
        pass


# --------------------------------------------------
if __name__ == "__main__":
    main()
