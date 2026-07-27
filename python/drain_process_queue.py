#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-15
Purpose: Drain the md_process_job queue: run "mdr-process ticket" for each
         pending job and report the outcome to Slack. Meant to run from cron,
         guarded by an flock so overlapping ticks don't double-run.

         Most failures are NOT retried: a non-zero exit / timeout marks the job
         'failed' (terminal) and posts a Slack notice for a human to act on.
         The one exception is a transient iRODS connection failure during the
         fetch step (fetch_uploads.py shells out to "gocmd", which reports
         "Failed to establish a connection to iRODS server") -- fetch_uploads.py
         is the first thing "mdr-process ticket" runs, before anything is
         mutated, so it is safe to requeue. The job goes back to 'pending' and
         bumps num_attempts, up to MAX_RETRY_ATTEMPTS; the next cron tick
         retries it. To avoid hammering iRODS by immediately reclaiming the
         same job in this same tick, a requeue stops this invocation's drain
         loop -- the next attempt waits for the next cron minute.
"""

import argparse
import errno
import fcntl
import os
import psycopg2
import psycopg2.extras
import subprocess
import sys
from typing import NamedTuple, Optional

from common import FRONTEND_BASE_URLS, send_slack_message

PROCESS_TIMEOUT = 60 * 60 * 12  # seconds
ERROR_LINES = 20  # tail of mdr-process output to include in a failure notice
MAX_RETRY_ATTEMPTS = 5  # requeue this many times before treating it as terminal


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    log_dir: str
    lock_file: str
    max_jobs: Optional[int]
    dry_run: bool
    verbose: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Drain the md_process_job queue",
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
        "--log-dir",
        help="Directory for mdr-process logs",
        metavar="DIR",
        default="logs",
    )

    parser.add_argument(
        "--lock-file",
        help="flock path guarding against overlapping workers (per server)",
        metavar="PATH",
        default=None,
    )

    parser.add_argument(
        "-n",
        "--max-jobs",
        help="Stop after this many jobs (default: drain all pending)",
        metavar="INT",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        help="Claim nothing and run nothing; just report pending count "
        "(implies --verbose)",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Verbose", action="store_true")

    args = parser.parse_args()

    lock_file = args.lock_file or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"drain_process_queue-{args.server}.lock"
    )

    return Args(
        server=args.server,
        log_dir=args.log_dir,
        lock_file=lock_file,
        max_jobs=args.max_jobs,
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
            print(msg)

    if args.dry_run:
        status("DRY RUN: no jobs will be claimed or run")

    # Bail immediately if another worker for this server already holds the lock;
    # cron can fire a new tick while a long mdr-process run is still going.
    lock_fd = acquire_lock(args.lock_file)
    if lock_fd is None:
        status(f"Another worker holds {args.lock_file}, exiting")
        sys.exit(0)

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dsn = os.environ.get(env_key)
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    base_url = FRONTEND_BASE_URLS[args.server]

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if args.dry_run:
        cur.execute(
            "select count(*) from md_process_job where status = 'pending' "
            "and server = %s",
            (args.server,),
        )
        status(f"DRY RUN: {cur.fetchone()[0]} pending job(s) for {args.server}")
        sys.exit(0)

    processed = 0
    while args.max_jobs is None or processed < args.max_jobs:
        job = claim_job(cur, args.server)
        if job is None:
            status("No more pending jobs")
            break

        keep_draining = run_job(cur, job, base_url, args.log_dir, status)
        processed += 1
        if not keep_draining:
            status("Job requeued for retry; stopping this drain tick")
            break

    status(f"FINISHED drain_process_queue ({processed} job(s))")


# --------------------------------------------------
def load_env() -> None:
    """Load .env (optional dependency, mirrors check_new_simulations.py)"""

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


# --------------------------------------------------
def acquire_lock(path: str):
    """Take a non-blocking exclusive flock; return the fd, or None if held"""

    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            fd.close()
            return None
        raise
    return fd  # keep the fd open (and thus the lock held) for the process life


# --------------------------------------------------
def claim_job(cur, server: str):
    """Atomically claim the oldest pending job, or None if the queue is empty"""

    # FOR UPDATE SKIP LOCKED lets multiple workers coexist without contention.
    cur.execute(
        """
        update md_process_job
        set    status = 'running', started_at = now()
        where  id = (
            select id
            from   md_process_job
            where  status = 'pending' and server = %s
            order  by created_at
            for update skip locked
            limit  1
        )
        returning id, ticket_id, server, num_attempts
        """,
        (server,),
    )
    return cur.fetchone()


# --------------------------------------------------
def run_job(cur, job, base_url: str, log_dir: str, status) -> bool:
    """Run "mdr-process ticket" for one claimed job and record the outcome.

    Returns False if the job was requeued for retry (the caller should stop
    draining this tick so the retry isn't reclaimed immediately), True
    otherwise.
    """

    ticket_id = job["ticket_id"]
    server = job["server"]

    os.makedirs(log_dir, exist_ok=True)
    # Include the server: ticket IDs are per-database, so prod and staging can
    # share a ticket id and would otherwise clobber each other's log.
    log_file = os.path.join(log_dir, f"ticket-{ticket_id}-{server}.log")

    # "-l debug" writes the log to --log-file, leaving the failure on stderr
    cmd = [
        "mdr-process",
        "-l",
        "debug",
        "--log-file",
        log_file,
        "ticket",
        "--ticket-id",
        str(ticket_id),
        "--server",
        server,
    ]
    status(f"Job {job['id']} (ticket {ticket_id}): running {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PROCESS_TIMEOUT
        )
    except FileNotFoundError:
        finish_failure(cur, job, None, "mdr-process not found", base_url, log_file, status)
        return True
    except subprocess.TimeoutExpired:
        finish_failure(
            cur,
            job,
            None,
            f"timed out after {PROCESS_TIMEOUT} seconds",
            base_url,
            log_file,
            status,
        )
        return True

    if proc.returncode == 0:
        finish_success(cur, job, base_url, status)
        return True

    err = (proc.stderr or proc.stdout or "").strip()
    detail = err or "no error output"
    if log_tail := tail_file(log_file):
        detail += f"\n\nLast {ERROR_LINES} lines of debug log:\n{log_tail}"

    if is_retryable_irods_error(detail) and job["num_attempts"] < MAX_RETRY_ATTEMPTS:
        requeue_for_retry(cur, job, proc.returncode, detail, status)
        return False

    finish_failure(cur, job, proc.returncode, detail, base_url, log_file, status)
    return True


# --------------------------------------------------
def finish_success(cur, job, base_url: str, status) -> None:
    """Mark a job succeeded and post a Slack notice"""

    ticket_id = job["ticket_id"]
    cur.execute(
        """
        update md_process_job
        set    status = 'succeeded', exit_code = 0, finished_at = now()
        where  id = %s
        """,
        (job["id"],),
    )
    status(f"Job {job['id']} (ticket {ticket_id}) SUCCEEDED")
    send_slack_message(f"Ticket {ticket_id} processing SUCCEEDED", base_url)


# --------------------------------------------------
def is_retryable_irods_error(detail: str) -> bool:
    """True if the failure detail looks like a transient iRODS connection issue

    Two distinct code paths in fetch_uploads.py can surface this:
      - It shells out to the "gocmd" binary for each file download (the
        per-file retry loop in main(), plus the completed.json fallback); when
        iRODS itself is unreachable, gocmd writes its own (ANSI-colored)
        message to stdout, which fetch_uploads.py then re-raises via
        "sys.exit(f'Error running {cmd}: {out}')".
      - Its own iRODS session calls (e.g. session.collections.get) can raise
        python-irodsclient's NetworkException directly, which propagates as an
        uncaught traceback -- ending up on stderr with "NetworkException" in
        it.
    Either way, that text is what ends up here as the captured stderr/log tail.
    """

    lowered = detail.lower()
    return (
        "failed to establish a connection to irods server" in lowered
        or "networkexception" in lowered
    )


# --------------------------------------------------
def requeue_for_retry(cur, job, exit_code: Optional[int], detail: str, status) -> None:
    """Reset a retryable-failure job back to 'pending' and bump num_attempts

    fetch_uploads.py is the first thing "mdr-process ticket" runs, before any
    database writes or file processing, so it's safe to requeue and re-run the
    whole ticket from scratch (unlike other failures, which need --force and a
    human's judgment).
    """

    ticket_id = job["ticket_id"]
    next_attempt = job["num_attempts"] + 1
    cur.execute(
        """
        update md_process_job
        set    status = 'pending', num_attempts = %s, exit_code = %s,
               last_error = %s, started_at = null, finished_at = null
        where  id = %s
        """,
        (next_attempt, exit_code, detail, job["id"]),
    )
    status(
        f"Job {job['id']} (ticket {ticket_id}) hit a retryable iRODS connection "
        f"error (attempt {next_attempt}/{MAX_RETRY_ATTEMPTS}); requeued"
    )


# --------------------------------------------------
def finish_failure(
    cur, job, exit_code: Optional[int], detail: str, base_url: str, log_file: str, status
) -> None:
    """Mark a job failed (terminal, no retry) and post a Slack notice for a human"""

    ticket_id = job["ticket_id"]
    cur.execute(
        """
        update md_process_job
        set    status = 'failed', exit_code = %s, last_error = %s, finished_at = now()
        where  id = %s
        """,
        (exit_code, detail, job["id"]),
    )
    status(f"Job {job['id']} (ticket {ticket_id}) FAILED: {detail}")
    exit_note = f" (exit {exit_code})" if exit_code is not None else ""
    send_slack_message(
        f"Ticket {ticket_id} processing FAILED{exit_note}\n"
        f"```\n{detail}\n```\nFull debug log: {log_file}",
        base_url,
    )


# --------------------------------------------------
def tail_file(path: str, num_lines: int = ERROR_LINES) -> str:
    """Last "num_lines" lines of a file, empty string if unreadable"""

    try:
        with open(path) as fh:
            return "".join(fh.readlines()[-num_lines:]).strip()
    except OSError:
        return ""


# --------------------------------------------------
if __name__ == "__main__":
    main()
