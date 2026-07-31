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

That is a Postgres fact, though: it says the push was verified when it
happened, not that the data is still in IRODS now. So before each ticket's
landings go, its permanent copies are sampled and checked for existence and
size -- one file per simulation by default, immediately before that ticket is
deleted. A ticket that fails is refused and the run exits non-zero.

By default this reports and deletes nothing; pass --delete. Deletes go to the
IRODS trash unless --force is given, so a mistaken run is recoverable until the
trash is emptied.

Removal is bound by round trips to the catalog, not by server capacity:
measured on prod, remove() is 97.7% of the per-collection cost (39.0s of
40.0s, ~3.9s for each of ten objects) against a 45.6ms RTT to
data.cyverse.org -- roughly 855 round trips per collection. Concurrency is
therefore close to free: 8 threads measured 6.2x. Hence --threads, which a
serial run of the July 2026 backlog would have needed ~69 hours to finish.
"""

import argparse
import errno
import fcntl
import os
import random
import re
import signal
import ssl
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from irods.exception import NetworkException
from irods.session import iRODSSession

from common import FRONTEND_BASE_URLS, send_slack_message, stamp

TICKET_RE = re.compile(r"^MDRSubmit_([^:]+):(.+)$")
# Where push_sim_files.py puts the permanent copy. local_file_path in the file
# tables is relative to this.
RELEASE_ROOT = "/iplant/home/shared/mdrepo/{server}/release"
# Removing one landing normally takes seconds; an IRODS call has blocked here
# for over half an hour under load. Bound it so one stall cannot wedge a run.
REMOVE_TIMEOUT = 300  # seconds
# A read probe that gets no answer is transient far more often than not: the
# 2026-07-31 cron pass died on one, 20 tickets in, having removed 3,833
# collections. Ride out a blip rather than losing the run. Kept small on
# purpose -- this is not meant to wait out an outage, which should stop the
# run and leave the remaining tickets for the next pass.
PROBE_ATTEMPTS = 3
PROBE_BACKOFF = 2  # seconds, multiplied by the attempt just failed


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    ticket_ids: List[int]
    limit: Optional[int]
    delete: bool
    force: bool
    include_unqueued: bool
    irods_env: str
    # Defaulted so other tools can build an Args for find_candidates without
    # restating the whole CLI -- preflight_permanent.py does exactly that, and
    # adding a field here used to break it. get_args always passes all of them.
    lock_file: str = ""
    threads: int = 1
    full_scan: bool = False
    verify_files: int = 1
    verify_only: bool = False
    skip_verify: bool = False


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

    parser.add_argument(
        "--lock-file",
        help="flock path guarding against overlapping runs (per server)",
        metavar="FILE",
        default=None,
    )

    parser.add_argument(
        "-j",
        "--threads",
        help="Landing collections to remove concurrently. Removal is bound by "
        "round trips to the catalog, not by server capacity, so this scales "
        "nearly linearly. Defaults to 1: concurrency is opt-in on a tool that "
        "deletes",
        metavar="INT",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--verify-files",
        help="Files checked per simulation against permanent storage before "
        "its ticket's landings are deleted (0 = every file)",
        metavar="INT",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--verify-only",
        help="Run the permanent-storage check and report, deleting nothing",
        action="store_true",
    )

    parser.add_argument(
        "--skip-verify",
        help="Delete without checking permanent storage first. Only sensible "
        "when something else just verified it",
        action="store_true",
    )

    parser.add_argument(
        "--full-scan",
        help="Probe every landing of every eligible ticket. Without this, a "
        "ticket whose first and last landing are both already gone is skipped "
        "unprobed",
        action="store_true",
    )

    args = parser.parse_args()

    if args.threads < 1:
        parser.error(f"--threads '{args.threads}' must be a positive integer")

    if not os.path.isfile(args.irods_env):
        parser.error(f"Invalid --irods-env '{args.irods_env}'")

    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit '{args.limit}' must be a positive integer")

    if args.force and not args.delete:
        parser.error("--force is meaningless without --delete")

    if args.verify_files < 0:
        parser.error(f"--verify-files '{args.verify_files}' cannot be negative")

    if args.skip_verify and args.verify_only:
        parser.error("--skip-verify and --verify-only contradict each other")

    lock_file = args.lock_file or os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"purge_processed_landings-{args.server}.lock",
    )

    return Args(
        server=args.server,
        ticket_ids=list(args.ticket_id or []),
        limit=args.limit,
        delete=args.delete,
        force=args.force,
        include_unqueued=args.include_unqueued,
        irods_env=args.irods_env,
        lock_file=lock_file,
        threads=args.threads,
        full_scan=args.full_scan,
        verify_files=args.verify_files,
        verify_only=args.verify_only,
        skip_verify=args.skip_verify,
    )


# --------------------------------------------------
def say(msg: str) -> None:
    """Print one timestamped line

    Nearly every line here is an event with a duration attached: a removal
    that took 40 seconds, a per-ticket verify, a pass that ran for hours.
    Without a clock in the file the only way to recover a rate is to count
    lines and compare against the file's mtime, which is exactly what had to
    be done to work out what the 2026-07-31 run managed before it died.

    Indented detail lines and the end-of-run summary deliberately do not go
    through this: the first are continuations of the line above, and the
    second all print within the same instant, so stamping each would add
    noise rather than information.
    """

    print(f"{stamp()} {msg}", flush=True)


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
class SessionPool:
    """One iRODSSession per worker thread

    A session cannot be shared across threads, and opening one per collection
    would pay the connect every time, so each thread keeps its own for the life
    of the run. connection_timeout bounds a blocked socket read, which is the
    stall this tool has actually hit; SIGALRM cannot do that off the main
    thread.
    """

    def __init__(self, irods_env: str, timeout: int) -> None:
        self._irods_env = irods_env
        self._timeout = timeout
        self._ssl = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        self._local = threading.local()
        self._sessions: List[iRODSSession] = []
        self._lock = threading.Lock()

    def get(self) -> iRODSSession:
        session = getattr(self._local, "session", None)
        if session is None:
            session = iRODSSession(
                irods_env_file=self._irods_env,
                ssl_context=self._ssl,
                connection_timeout=self._timeout,
            )
            self._local.session = session
            with self._lock:
                self._sessions.append(session)

        return session

    def discard(self) -> None:
        """Drop this thread's session so the next get() builds a fresh one

        A session that has just raised on the wire may be holding a dead
        connection, and it is cached for the life of the run -- so without
        this, one network blip can poison every later call on that thread.
        """

        session = getattr(self._local, "session", None)
        if session is None:
            return

        self._local.session = None
        with self._lock:
            if session in self._sessions:
                self._sessions.remove(session)

        try:
            session.cleanup()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            for session in self._sessions:
                try:
                    session.cleanup()
                except Exception:
                    pass
            self._sessions = []


# --------------------------------------------------
def probe(sessions: SessionPool, func: Callable, *args):
    """Run one IRODS read probe, riding out a transient network error

    Re-raises the last error once the attempts are spent rather than returning
    a default, because the right reading of an unanswered probe depends on the
    caller and none of them may treat it as good news: under --force, taking
    "could not check" for "verified" deletes the only other copy of the data.
    """

    for attempt in range(1, PROBE_ATTEMPTS + 1):
        try:
            return func(sessions.get(), *args)
        except (TimeoutError, NetworkException):
            sessions.discard()
            if attempt == PROBE_ATTEMPTS:
                raise
            time.sleep(PROBE_BACKOFF * attempt)


# --------------------------------------------------
def remove_landing(sessions: SessionPool, landing_dir: str, args: Args) -> Dict:
    """Remove one landing collection, reporting what it found and freed"""

    nothing = {"present": 0, "freed": 0, "removed": 0, "stalled": 0}
    session = sessions.get()

    if not session.collections.exists(landing_dir):
        return nothing

    coll = session.collections.get(landing_dir)
    size = sum(obj.size for obj in coll.data_objects)

    if not args.delete:
        return {"present": 1, "freed": size, "removed": 0, "stalled": 0}

    try:
        # remove() can block indefinitely -- an IRODS call has hung here for
        # over half an hour under load, with no timeout of its own. Bound it
        # so one stall cannot wedge a run: single-threaded via SIGALRM as
        # before, and in the pool via the session's connection_timeout, since
        # SIGALRM only works on the main thread.
        if args.threads == 1:
            with time_limit(REMOVE_TIMEOUT):
                coll.remove(recurse=True, force=args.force)
        else:
            coll.remove(recurse=True, force=args.force)
    except (TimeoutError, NetworkException):
        return {"present": 1, "freed": 0, "removed": 0, "stalled": 1}

    return {"present": 1, "freed": size, "removed": 1, "stalled": 0}


# --------------------------------------------------
def verify_permanent(
    cur, sessions: SessionPool, cand: Candidate, args: Args
) -> Tuple[List[str], List[str]]:
    """Check this ticket's permanent copies exist before deleting the landings

    Eligibility is a Postgres fact -- is_placeholder = false, which mdr-process
    clears only after verifying the push. That says the push was verified when
    it happened, not that the data is still there now, and the landing copy is
    the only other one. So look.

    Runs per ticket, immediately before that ticket is deleted, rather than as
    a bulk pass up front: a ticket can become eligible while a bulk pass is
    running and would then be deleted having never been checked.

    Coverage is one file per simulation by default, so no simulation goes
    unverified. --verify-files raises that (0 checks every file). A sample
    catches a ticket that never pushed or a release collection that has gone
    missing; it will not catch one absent file among thousands.

    Returns (problems, unchecked), which the caller must keep apart. A problem
    is an answer from IRODS that the data is wrong or gone -- the safety net
    firing. An unchecked file is no answer at all, which is evidence about the
    network and none whatsoever about the data. Reporting the second as the
    first cries wolf on a healthy ticket; treating it as verified deletes on a
    guess.
    """

    cur.execute(
        """
        select s.id as sim_id, f.local_file_path, f.file_size_bytes
        from   md_processed_file f
        join   md_simulation s on s.id = f.simulation_id
        where  s.md_repo_ticket_id = %s
        union all
        select s.id, f.local_file_path, f.file_size_bytes
        from   md_uploaded_file f
        join   md_simulation s on s.id = f.simulation_id
        where  s.md_repo_ticket_id = %s
        """,
        (cand.ticket_id, cand.ticket_id),
    )

    by_sim = defaultdict(list)
    for row in cur.fetchall():
        by_sim[row["sim_id"]].append(
            (row["local_file_path"], row["file_size_bytes"])
        )

    if not by_sim:
        # Eligible, but nothing to verify against: refuse rather than guess.
        return ([f"ticket {cand.ticket_id}: no file rows, cannot verify"], [])

    sample = []
    for files in by_sim.values():
        if args.verify_files and len(files) > args.verify_files:
            sample.extend(random.sample(files, args.verify_files))
        else:
            sample.extend(files)

    root = RELEASE_ROOT.format(server=args.server)

    def look(session, remote: str, size: Optional[int]) -> Optional[str]:
        if not session.data_objects.exists(remote):
            return f"MISSING {remote}"
        if size is not None:
            actual = session.data_objects.get(remote).size
            if actual != size:
                return f"SIZE {remote} want={size} got={actual}"
        return None

    def check(item) -> Optional[Tuple[str, str]]:
        path, size = item
        remote = f"{root}/{path}"

        try:
            problem = probe(sessions, look, remote, size)
        except (TimeoutError, NetworkException) as e:
            return ("unchecked",
                    f"UNREACHABLE {remote}: {type(e).__name__}: {e}")

        return ("problem", problem) if problem else None

    if args.threads == 1:
        results = [check(i) for i in sample]
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            results = list(pool.map(check, sample))

    found = [r for r in results if r]

    return ([m for kind, m in found if kind == "problem"],
            [m for kind, m in found if kind == "unchecked"])


# --------------------------------------------------
def already_purged(sessions: SessionPool, cand: Candidate) -> bool:
    """Whether a ticket looks fully purged, from two probes

    The eligible set only grows and is never re-checked against what is left,
    so without this a daily run re-probes every landing it has ever deleted --
    at ~0.07s each that is minutes per pass, rising forever.

    First *and* last, not just first: removal walks the list in order, so an
    interrupted run leaves a purged prefix and a present tail. Testing only the
    first would skip that tail permanently. A collection stranded in the middle
    by a stall is still possible, hence --full-scan.

    An unanswered probe reads as "not purged", which costs a verify pass that
    will meet the same outage and skip the ticket properly. The other way round
    would quietly drop a ticket from the run on a network blip.
    """

    ends = {cand.landing_dirs[0], cand.landing_dirs[-1]}

    try:
        return not any(
            probe(sessions, lambda s, d: s.collections.exists(d), d)
            for d in ends
        )
    except (TimeoutError, NetworkException):
        return False


# --------------------------------------------------
def purge_ticket(sessions: SessionPool, cand: Candidate, args: Args) -> Dict[str, int]:
    """Remove one ticket's landing collections, reporting what was found"""

    totals = {"present": 0, "freed": 0, "removed": 0, "stalled": 0}
    total = len(cand.landing_dirs)
    done = 0
    lock = threading.Lock()

    def one(landing_dir: str) -> Dict:
        result = remove_landing(sessions, landing_dir, args)

        # Progress per collection, not per ticket: a 200-landing ticket
        # otherwise prints nothing for minutes, and a stall inside remove()
        # looks identical to slow work. Ordering is by completion once
        # threaded, so the count is progress, not position.
        nonlocal done
        with lock:
            done += 1
            if args.delete and result["present"]:
                note = " STALLED" if result["stalled"] else ""
                say(f"    [{cand.ticket_id}] {done}/{total} "
                    f"{landing_dir.rsplit('/', 1)[-1]}{note}")

        return result

    if args.threads == 1:
        results = [one(d) for d in cand.landing_dirs]
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            results = list(pool.map(one, cand.landing_dirs))

    for result in results:
        for key in totals:
            totals[key] += result[key]

    return totals


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
@contextmanager
def time_limit(seconds: int):
    """Raise TimeoutError if the body outlasts `seconds`

    SIGALRM rather than a thread, because the call being bounded blocks in a
    socket read: the signal interrupts it and the exception propagates. Only
    usable on the main thread, which is where this runs.
    """

    def handler(_signum, _frame):
        raise TimeoutError(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def notify(args: Args, message: str) -> None:
    """Post to Slack, but only for runs that actually changed something

    Dry runs and --verify-only stay silent: this runs nightly, and a daily
    "nothing to do" would train everyone to ignore the channel, which is the
    one thing a failure alert cannot afford.
    """

    say(message)

    if not args.delete or args.verify_only:
        return

    send_slack_message(message, FRONTEND_BASE_URLS[args.server])


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    # Name the .env beside this file rather than leaning on bare load_dotenv().
    # Not because the cwd would break it -- find_dotenv() defaults to
    # usecwd=False and walks up from the *calling file*, so a bare call already
    # resolves here from any cwd. It falls back to the cwd only in a REPL,
    # under a debugger, or in a frozen build. Being explicit just keeps that
    # subtlety from being load-bearing when the cron line drops its "cd".
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    # Bail if another run for this server is already going. Taken for dry runs
    # too, not just --delete: a second pass walks the same collections and
    # doubles the IRODS load for nothing, and every phase here is bound by
    # round trips to a remote catalog.
    lock_fd = acquire_lock(args.lock_file)
    if lock_fd is None:
        say(f"Another run holds {args.lock_file}, exiting")
        sys.exit(0)

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
        say("No tickets are eligible")
        sys.exit(0)

    if args.limit:
        candidates = candidates[: args.limit]

    if args.verify_only:
        mode = "VERIFY ONLY (nothing will be deleted)"
    elif args.delete:
        mode = "DELETING"
    else:
        mode = "DRY RUN (pass --delete to remove)"

    trash = "" if args.force or args.verify_only else " to trash"
    say(f"{mode}{trash if args.delete else ''}: "
        f"{len(candidates)} eligible ticket(s) on {args.server}, "
        f"{args.threads} thread(s)\n")

    totals = {"present": 0, "freed": 0, "removed": 0, "stalled": 0,
              "tickets": 0, "skipped": 0, "unverified": 0, "verified": 0,
              "unchecked": 0}
    sessions = SessionPool(args.irods_env, REMOVE_TIMEOUT)

    try:
        for cand in candidates:
            if not args.full_scan and already_purged(sessions, cand):
                totals["skipped"] += 1
                continue

            # Verify this ticket right before deleting it, not in a bulk pass
            # up front: a ticket can become eligible mid-pass and would then be
            # deleted having never been checked, which --force makes permanent.
            if not args.skip_verify:
                problems, unchecked = verify_permanent(cur, sessions, cand, args)
                if problems:
                    totals["unverified"] += 1
                    say(f"ticket {cand.ticket_id}: NOT VERIFIED, refusing to "
                        f"delete ({len(problems)} problem(s))")
                    for p in problems[:5]:
                        print(f"    {p}", flush=True)
                    if len(problems) > 5:
                        print(f"    ... and {len(problems) - 5} more", flush=True)
                    continue

                if unchecked:
                    # IRODS did not answer, which says nothing about the data.
                    # Skip this ticket and carry on rather than abort the run:
                    # one such probe killed the 2026-07-31 pass 20 tickets in.
                    # The ticket stays eligible, so the next pass retries it.
                    totals["unchecked"] += 1
                    say(f"ticket {cand.ticket_id}: NOT CHECKED, skipping "
                        f"({len(unchecked)} file(s) unreachable after "
                        f"{PROBE_ATTEMPTS} attempts)")
                    for u in unchecked[:5]:
                        print(f"    {u}", flush=True)
                    if len(unchecked) > 5:
                        print(f"    ... and {len(unchecked) - 5} more", flush=True)
                    continue

                totals["verified"] += 1

            if args.verify_only:
                say(f"ticket {cand.ticket_id}: verified")
                continue

            result = purge_ticket(sessions, cand, args)

            if not result["present"]:
                say(f"ticket {cand.ticket_id}: landings already gone")
                continue

            totals["present"] += result["present"]
            totals["freed"] += result["freed"]
            totals["removed"] += result["removed"]
            totals["stalled"] += result["stalled"]
            totals["tickets"] += 1

            verb = "removed" if args.delete else "would remove"
            note = "" if cand.has_job else "  [no process job -- pre-queue]"
            if result["stalled"]:
                note += f"  [{result['stalled']} STALLED, re-run to retry]"
            say(
                f"ticket {cand.ticket_id}: {cand.num_sims} sims, "
                f"{verb} {result['present']} of {len(cand.landing_dirs)} "
                f"landing(s), {human(result['freed'])}{note}"
            )
    except Exception as e:
        # A traceback in a log nobody reads is not a signal. Say so in Slack,
        # then re-raise so the log still gets the traceback and the exit is
        # non-zero.
        notify(args, f"Landing purge ({args.server}) FAILED after removing "
                     f"{totals['removed']} collection(s): {type(e).__name__}: {e}")
        raise
    finally:
        sessions.close()

    verb = "Removed" if args.delete else "Would remove"
    print(
        f"\n{verb} {totals['removed'] if args.delete else totals['present']} "
        f"landing collection(s) across {totals['tickets']} ticket(s), "
        f"{human(totals['freed'])}"
    )

    if totals["skipped"]:
        print(f"Skipped {totals['skipped']} ticket(s) already purged "
              f"(--full-scan to probe them anyway)")

    if totals["verified"]:
        print(f"Verified {totals['verified']} ticket(s) against permanent "
              f"storage first")

    if totals["unchecked"]:
        # Deliberately not a non-zero exit, unlike a refusal. This is the
        # transient case and it self-heals on the next pass, the Slack summary
        # already carries it, and the exit code is nobody's signal here -- if
        # this job is ever wrapped in cron_notify.py, exiting 1 would just
        # duplicate an alert the script has already sent.
        print(f"Skipped {totals['unchecked']} ticket(s) IRODS did not answer "
              f"for; they stay eligible and the next run retries them")

    if totals["unverified"]:
        # Non-zero exit, and say it out loud: this is the safety net firing.
        notify(args, f"Landing purge ({args.server}): REFUSED "
                     f"{totals['unverified']} ticket(s) whose permanent copies "
                     f"did not verify -- landings NOT deleted. Removed "
                     f"{totals['removed']} collection(s) from the rest, "
                     f"{human(totals['freed'])}. Investigate before re-running.")
        sys.exit(1)

    # Only report a run that did something. A nightly "nothing to do" would
    # train the channel to be ignored. An unreachable ticket counts as
    # something: it is the one outcome where the run looks clean and silently
    # did less than it was asked to.
    if totals["removed"] or totals["stalled"] or totals["unchecked"]:
        summary = (f"Landing purge ({args.server}): removed "
                   f"{totals['removed']} landing collection(s) across "
                   f"{totals['tickets']} ticket(s), {human(totals['freed'])}")
        if totals["stalled"]:
            summary += (f"; {totals['stalled']} collection(s) stalled past "
                        f"{REMOVE_TIMEOUT}s and will be retried next run")
        if totals["unchecked"]:
            summary += (f"; skipped {totals['unchecked']} ticket(s) IRODS did "
                        f"not answer for, retried next run")
        notify(args, summary)

    if totals["stalled"]:
        print(f"{totals['stalled']} collection(s) STALLED past "
              f"{REMOVE_TIMEOUT}s and were skipped. Re-running retries them; "
              f"already-removed ones are skipped, so it is safe to repeat.")

    if not args.delete:
        print("Nothing was deleted. Re-run with --delete to act.")
    elif not args.force:
        print("Deleted to the IRODS trash; empty it to reclaim the space.")


# --------------------------------------------------
if __name__ == "__main__":
    main()
