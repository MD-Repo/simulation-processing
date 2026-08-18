#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-29
Purpose: Delete landing directories whose data is already in permanent
         storage, in IRODS and (with --local) on local disk

Nothing else removes landing collections. The reap in check_new_simulations.py
only ever considered *incomplete* tickets, so a ticket that processed
successfully kept its landing copy forever: 16,241 collections were sitting
under the prod landing area, on the order of a terabyte, all of it already
imported and pushed.

The unit is the landing directory, not the ticket. Each one answers for itself
from two independent signals that have to agree: `md_upload_instance.
successful`, which is the pipeline's opinion of its own run, and
`is_placeholder = false` on the simulation it produced, which `mdr-process`
clears only after verifying the push by MD5 and presence. Age is never
consulted, and abandonment is never inferred -- a landing that fails either
test is held for a human, however old it is.

The landing is the unit because it is the only granularity at which
verification and deletion are the same thing. Deleting per ticket while
verifying per simulation is how a landing that produced no simulation got
deleted anyway, unverified, contributing nothing to the check that cleared it
(MDR-45; ticket 2175 is 16 landing directories against 15 simulations). Per
landing that asymmetry cannot be expressed.

Nothing is lost by working this finely. Every ticket from 1350 onward records
one upload instance per landing; the earlier convention was one per ticket, and
those 5,533 landings across 82 tickets (all id <= 837) were cleared by the July
2026 backlog run -- 25 sampled, 0 still present, none with a local directory.

Both copies, not just the IRODS one. Local cleanup used to live entirely in
mdr-process (ticket.rs, fs::remove_dir_all gated on all_ok), so a ticket that
did not finish clean kept its local directory forever even after its IRODS
landings were purged -- 387 GB across four tickets when this was written.
--local removes it here, gated on the same verify pass, and "already purged"
means purged in both places or the addition would have been a silent no-op.
Once a ticket's last landing directory is gone, the `ticket-NNNN` shell that
held them is reaped too: it carries nothing but `ticket.json`.

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
import shutil
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
# Where mdr-process puts the local copy, built the same way ticket.rs builds it
# (MDREPO_WORK_DIR/landing/<server>/ticket-<id>) so the two cannot drift. Its
# sub-directories are named by landing_id, which is verbatim the basename of
# the IRODS landing collection.
LANDING_ROOT = "{work_dir}/landing/{server}"
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
# Job states that mean mdr-process is working on a ticket, or is about to be:
# the drain picks a pending job up within the minute. Not terminal, so not
# safe to delete under.
IN_FLIGHT = ("pending", "running")
# What one landing's removal reports, in both places at once. IRODS and local
# counts are never added together: they free different disks, and only one of
# them is the one that fills.
EMPTY = {"present": 0, "freed": 0, "removed": 0, "stalled": 0,
         "local_present": 0, "local_freed": 0, "local_removed": 0,
         "local_failed": 0}


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    ticket_ids: List[int]
    limit: Optional[int]
    delete: bool
    force: bool
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
    local: bool = False
    local_root: str = ""


class Landing(NamedTuple):
    """One landing directory, in both of the places it exists

    `landing_id` is the join: md_upload_instance records it, the IRODS
    collection is named by it, and so is the local sub-directory. Either side
    can be absent -- an IRODS collection with no upload instance is a landing
    that produced no simulation, and a local directory can outlive its IRODS
    twin (or the reverse) because until now only one of the two was ever
    cleaned up.
    """

    landing_id: str
    irods_dir: Optional[str]
    local_dir: Optional[str]
    sim_id: Optional[int]


class Candidate(NamedTuple):
    """A ticket with at least one landing eligible for deletion"""

    ticket_id: int
    num_sims: int
    has_job: bool
    landings: List[Landing]
    # Landings on the ticket that are NOT eligible, so the report can say what
    # is being held rather than silently narrowing the ticket.
    held: int = 0

    @property
    def landing_dirs(self) -> List[str]:
        """The IRODS collections, which is what the ticket-mode paths want"""

        return [l.irods_dir for l in self.landings if l.irods_dir]


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Delete landing directories already in permanent storage",
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
        "--local",
        help="Also delete the landing directory on local disk. Gated on the "
        "same verify pass as the IRODS copy, and there is no local --force: "
        "rmtree is never recoverable",
        action="store_true",
    )

    parser.add_argument(
        "--local-root",
        help="Landing root on local disk (default: MDREPO_WORK_DIR/landing/"
        "<server>, the same path mdr-process writes)",
        metavar="DIR",
        default=None,
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

    if args.local and args.skip_verify:
        # The IRODS copy can go to the trash and come back; a local rmtree
        # cannot. --skip-verify exists for "something else just verified it",
        # which is not a claim this can check, so refuse the combination
        # rather than take its word for an unrecoverable delete.
        parser.error("--local will not run with --skip-verify")

    if args.local_root and not args.local:
        parser.error("--local-root is meaningless without --local")

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
        irods_env=args.irods_env,
        lock_file=lock_file,
        threads=args.threads,
        full_scan=args.full_scan,
        verify_files=args.verify_files,
        verify_only=args.verify_only,
        skip_verify=args.skip_verify,
        local=args.local,
        # Resolved in main(), which is where .env has been read.
        local_root=args.local_root or "",
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
def parse_irods_tickets(raw: Optional[str]) -> Dict[str, str]:
    """Map landing_id -> IRODS collection, from md_ticket.irods_tickets

    Each entry is `MDRSubmit_<landing_id>:<collection>`, and the collection's
    basename is that same landing_id -- checked against ticket 2183 on prod,
    2026-08-18, all 200 entries. So the ticket string, the IRODS collection and
    the local sub-directory all agree on one name, which is what makes a
    per-landing join possible without any schema work.
    """

    out = {}
    for entry in (raw or "").split(";"):
        if m := TICKET_RE.search(entry):
            landing_id, collection = m.groups()
            out[landing_id] = collection

    return out


# --------------------------------------------------
def local_dir_for(args: Args, ticket_id: int, landing_id: str) -> Optional[str]:
    """Where mdr-process would have put this landing on local disk"""

    if not args.local_root:
        return None

    return os.path.join(args.local_root, f"ticket-{ticket_id}", landing_id)


# --------------------------------------------------
def find_candidates(cur, args: Args) -> List[Candidate]:
    """Find landing directories whose data is already in permanent storage

    Eligibility is decided for each landing on its own, from two independent
    signals that have to agree -- `md_upload_instance.successful`, which is the
    pipeline's opinion of its own run, and `is_placeholder = false` on the
    simulation that landing produced, which is a verified fact about the push.
    Two sources agreeing is worth the extra join on a tool that deletes.

    A ticket appears here if any of its landings qualify; the rest are counted
    into `held` and reported, never silently dropped. That is the only way to
    reach a ticket that partly failed, and partly-failed is most of what is
    stranded: emme's 20 tickets carry 3,878 landings that imported and pushed
    cleanly beside 122 that never will.

    Job status does not decide *eligibility*, and its absence there is
    deliberate. `md_process_job.status = 'succeeded'` was a ticket-shaped proxy
    for "the run finished" -- reasonable when the alternative was guessing, and
    wrong at this grain twice over: emme's 20 jobs are all `failed`, correctly,
    while each owns ~194 finished landings, and a job row hand-edited to
    `succeeded` says nothing about any particular landing.

    It does still decide *safety*, which is a different question and was easy
    to lose sight of when the succeeded-job test went away. That test was also
    doing concurrency control by accident: a ticket being processed has no
    succeeded job, so the old query could never reach one. Per landing, both
    signals are set by the running job itself, one landing at a time -- a
    landing that finished ten seconds ago looks exactly like one that finished
    last month. Deleting it would pull the directory out from under a live
    mdr-process. Observed, not imagined: job 161 on ticket 2227 was writing
    those very flags while the 2026-08-18 dry run was reading them, and 15
    landings changed answer between the query and the report.

    So tickets with a pending or running job are excluded here, and checked
    again immediately before deletion -- the same reason verify_permanent runs
    per ticket rather than as a bulk pass up front.

    A landing collection with no upload instance at all is never eligible: it
    produced no simulation, so nothing about it was ever verified. Those are
    held for a human, which is what the abandonment flag is for.
    """

    where = [
        "u.landing_id is not null",
        "not exists (select 1 from md_process_job j "
        "where j.ticket_id = u.ticket_id and j.status = any(%s))",
    ]
    params: List = [list(IN_FLIGHT)]

    if args.ticket_ids:
        where.append("u.ticket_id = any(%s)")
        params.append(args.ticket_ids)

    cur.execute(
        f"""
        select u.ticket_id,
               u.landing_id,
               u.simulation_id,
               u.successful,
               s.is_placeholder,
               t.irods_tickets,
               exists (select 1 from md_process_job j
                       where j.ticket_id = u.ticket_id) as has_job
        from   md_upload_instance u
        join   md_ticket     t on t.id = u.ticket_id
        left   join md_simulation s on s.id = u.simulation_id
        where  {' and '.join(where)}
        order  by u.ticket_id, u.landing_id
        """,
        params,
    )

    by_ticket: Dict[int, Dict] = {}
    for row in cur.fetchall():
        ticket = by_ticket.setdefault(
            row["ticket_id"],
            {"irods": parse_irods_tickets(row["irods_tickets"]),
             "has_job": row["has_job"], "landings": [], "held": 0},
        )

        # Both signals, and both have to be positive. `successful` alone is the
        # pipeline grading its own homework; is_placeholder alone cannot tell a
        # landing that failed from one not processed yet.
        eligible = (
            row["successful"]
            and row["simulation_id"] is not None
            and row["is_placeholder"] is False
        )

        if not eligible:
            ticket["held"] += 1
            continue

        ticket["landings"].append(
            Landing(
                landing_id=row["landing_id"],
                irods_dir=ticket["irods"].get(row["landing_id"]),
                local_dir=local_dir_for(args, row["ticket_id"],
                                        row["landing_id"]),
                sim_id=row["simulation_id"],
            )
        )

    return [
        Candidate(
            ticket_id=ticket_id,
            num_sims=len(info["landings"]),
            has_job=info["has_job"],
            landings=info["landings"],
            held=info["held"],
        )
        for ticket_id, info in sorted(by_ticket.items())
        if info["landings"]
    ]


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
def remove_landing(sessions: SessionPool, landing: Landing, args: Args) -> Dict:
    """Remove one landing from both places, reporting what it found and freed

    IRODS and local disk are counted separately all the way through, because
    they free different disks and only one of them is the one that fills.
    """

    result = dict(EMPTY)

    if landing.irods_dir:
        result.update(remove_irods_landing(sessions, landing.irods_dir, args))

    if args.local and landing.local_dir:
        result.update(remove_local_landing(landing.local_dir, args))

    return result


# --------------------------------------------------
def remove_irods_landing(
    sessions: SessionPool, landing_dir: str, args: Args
) -> Dict:
    """Remove one landing collection from IRODS"""

    session = sessions.get()

    if not session.collections.exists(landing_dir):
        return {}

    coll = session.collections.get(landing_dir)
    size = sum(obj.size for obj in coll.data_objects)

    if not args.delete:
        return {"present": 1, "freed": size}

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
        return {"present": 1, "stalled": 1}

    return {"present": 1, "freed": size, "removed": 1}


# --------------------------------------------------
def remove_local_landing(local_dir: str, args: Args) -> Dict:
    """Remove one landing directory from local disk

    There is no local --force. The IRODS side can go to the trash and be
    recovered until it is emptied; rmtree cannot, ever. What stands in for it
    is the verify pass, which every landing has already passed by the time this
    is called -- the permanent copy was checked to exist at the right size
    minutes ago, and this deletes the second copy of that.
    """

    if not os.path.isdir(local_dir):
        return {}

    size = 0
    for dirpath, _dirnames, filenames in os.walk(local_dir):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                # Apparent size, not blocks: it is what the IRODS side reports
                # and what a landing's manifest records, so the two columns of
                # the report mean the same thing.
                size += os.lstat(path).st_size
            except OSError:
                pass

    if not args.delete:
        return {"local_present": 1, "local_freed": size}

    try:
        shutil.rmtree(local_dir)
    except OSError as e:
        say(f"    local removal FAILED {local_dir}: {e}")
        return {"local_present": 1, "local_failed": 1}

    return {"local_present": 1, "local_freed": size, "local_removed": 1}


# --------------------------------------------------
def verify_permanent(
    cur, sessions: SessionPool, cand: Candidate, args: Args
) -> Tuple[Dict[Optional[int], List[str]], Dict[Optional[int], List[str]]]:
    """Check the permanent copies exist before deleting anything

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

    **This is also what earns the right to delete the local copy**, which is
    why --local has no analogue of --force: rmtree is always unrecoverable, and
    a local directory whose IRODS twin is already gone does not get a free pass
    for having outlived it. It is checked against release like any other.

    Returns (problems, unchecked), each keyed by simulation id, which the
    caller must keep apart. A problem is an answer from IRODS that the data is
    wrong or gone -- the safety net firing. An unchecked file is no answer at
    all, which is evidence about the network and none whatsoever about the
    data. Reporting the second as the first cries wolf on a healthy ticket;
    treating it as verified deletes on a guess. Keying by simulation is what
    holds back just the landings that failed to verify instead of the whole
    ticket; a refusal that belongs to no simulation (nothing to verify against
    at all) is keyed None.
    """

    # Verify exactly the simulations whose landings we are about to delete,
    # not the ticket's other ones, which we are not touching.
    sim_ids = [l.sim_id for l in cand.landings]

    cur.execute(
        """
        select s.id as sim_id, f.local_file_path, f.file_size_bytes
        from   md_processed_file f
        join   md_simulation s on s.id = f.simulation_id
        where  s.id = any(%s)
        union all
        select s.id, f.local_file_path, f.file_size_bytes
        from   md_uploaded_file f
        join   md_simulation s on s.id = f.simulation_id
        where  s.id = any(%s)
        """,
        (sim_ids, sim_ids),
    )

    by_sim = defaultdict(list)
    for row in cur.fetchall():
        by_sim[row["sim_id"]].append(
            (row["local_file_path"], row["file_size_bytes"])
        )

    if not by_sim:
        # Eligible, but nothing to verify against: refuse rather than guess.
        return ({None: [f"ticket {cand.ticket_id}: no file rows, cannot verify"]},
                {})

    # A landing whose simulation has no file rows at all is the same refusal at
    # a finer grain -- it would otherwise be deleted having been verified by
    # the other landings' files.
    missing_rows = [s for s in sim_ids if s not in by_sim]

    sample = []
    for sim_id, files in by_sim.items():
        if args.verify_files and len(files) > args.verify_files:
            chosen = random.sample(files, args.verify_files)
        else:
            chosen = files
        sample.extend((sim_id, path, size) for path, size in chosen)

    root = RELEASE_ROOT.format(server=args.server)

    def look(session, remote: str, size: Optional[int]) -> Optional[str]:
        if not session.data_objects.exists(remote):
            return f"MISSING {remote}"
        if size is not None:
            actual = session.data_objects.get(remote).size
            if actual != size:
                return f"SIZE {remote} want={size} got={actual}"
        return None

    def check(item) -> Optional[Tuple[str, Optional[int], str]]:
        sim_id, path, size = item
        remote = f"{root}/{path}"

        try:
            problem = probe(sessions, look, remote, size)
        except (TimeoutError, NetworkException) as e:
            return ("unchecked", sim_id,
                    f"UNREACHABLE {remote}: {type(e).__name__}: {e}")

        return ("problem", sim_id, problem) if problem else None

    if args.threads == 1:
        results = [check(i) for i in sample]
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            results = list(pool.map(check, sample))

    problems: Dict[Optional[int], List[str]] = defaultdict(list)
    unchecked: Dict[Optional[int], List[str]] = defaultdict(list)

    for sim_id in missing_rows:
        problems[sim_id].append(
            f"simulation {sim_id}: no file rows, cannot verify"
        )

    for found in results:
        if not found:
            continue
        kind, sim_id, message = found
        (problems if kind == "problem" else unchecked)[sim_id].append(message)

    return (dict(problems), dict(unchecked))


# --------------------------------------------------
def already_purged(sessions: SessionPool, cand: Candidate, args: Args) -> bool:
    """Whether this ticket is purged *in both places*

    The eligible set only grows and is never re-checked against what is left,
    so without this a daily run re-probes every landing it has ever deleted --
    at ~0.07s each that is minutes per pass, rising forever.

    Answering for IRODS alone is what stranded 387 GB. Tickets 2019, 2020, 2042
    and 2175 have no landing collections left and 387 GB of landing directories
    still on local disk, so an IRODS-only probe counts them in "Skipped N
    ticket(s) already purged" every night, forever, and adding local removal
    below this check would have been a silent no-op. So the question this
    answers is about the ticket, not about IRODS.

    IRODS is probed first *and* last, not just first: removal walks the list in
    order, so an interrupted run leaves a purged prefix and a present tail.
    Testing only the first would skip that tail permanently. A collection
    stranded in the middle by a stall is still possible, hence --full-scan.
    Local disk is checked in full -- it is a stat per directory against a local
    filesystem, so there is nothing to save by sampling it.

    An unanswered probe reads as "not purged", which costs a verify pass that
    will meet the same outage and skip the ticket properly. The other way round
    would quietly drop a ticket from the run on a network blip.
    """

    if args.local and any(
        l.local_dir and os.path.isdir(l.local_dir) for l in cand.landings
    ):
        return False

    dirs = cand.landing_dirs
    if not dirs:
        # Nothing on the IRODS side to probe. In --local mode we have just
        # established there is nothing on disk either; without it, a ticket
        # with no collections was never this tool's business.
        return True

    ends = {dirs[0], dirs[-1]}

    try:
        return not any(
            probe(sessions, lambda s, d: s.collections.exists(d), d)
            for d in ends
        )
    except (TimeoutError, NetworkException):
        return False


# --------------------------------------------------
def purge_ticket(
    sessions: SessionPool, cand: Candidate, args: Args
) -> Dict[str, int]:
    """Remove this ticket's eligible landings, reporting what was found"""

    totals = dict(EMPTY)
    total = len(cand.landings)
    done = 0
    lock = threading.Lock()

    def one(landing: Landing) -> Dict:
        result = remove_landing(sessions, landing, args)

        # Progress per landing, not per ticket: a 200-landing ticket otherwise
        # prints nothing for minutes, and a stall inside remove() looks
        # identical to slow work. Ordering is by completion once threaded, so
        # the count is progress, not position.
        nonlocal done
        with lock:
            done += 1
            if args.delete and (result["present"] or result["local_present"]):
                note = " STALLED" if result["stalled"] else ""
                where = "".join(
                    (
                        "i" if result["present"] else "-",
                        "l" if result["local_present"] else "-",
                    )
                )
                say(f"    [{cand.ticket_id}] {done}/{total} [{where}] "
                    f"{landing.landing_id}{note}")

        return result

    if args.threads == 1:
        results = [one(l) for l in cand.landings]
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            results = list(pool.map(one, cand.landings))

    for result in results:
        for key in totals:
            totals[key] += result[key]

    return totals


# --------------------------------------------------
def job_in_flight(cur, ticket_id: int) -> bool:
    """Whether mdr-process is working on this ticket, or is about to be

    Asked again immediately before deleting, not just when the candidates were
    chosen. A full pass takes half an hour; the drain starts a job every
    minute, and processing a ticket that already has landings in permanent
    storage is an ordinary thing to do -- a re-run, a reprocess, a correction
    arriving as a new ticket over the same landings. Between the candidate
    query and the deletion is easily long enough for one to start.
    """

    cur.execute(
        "select 1 from md_process_job "
        "where ticket_id = %s and status = any(%s) limit 1",
        (ticket_id, list(IN_FLIGHT)),
    )

    return cur.fetchone() is not None


# --------------------------------------------------
def reap_ticket_dir(cand: Candidate, args: Args) -> int:
    """Remove the `ticket-NNNN` shell once its last landing is gone

    The landing directory is the unit of work, so this is not a second kind of
    eligibility -- it is bookkeeping. A ticket directory that holds no landing
    directories holds nothing anyone can use: the only other thing in it is
    `ticket.json`, checked across every ticket directory on prod 2026-08-18.
    So "empty" has to mean "no landing sub-directories left" rather than a
    literal emptiness test, which `ticket.json` would defeat every time.

    A held landing leaves its directory in place, which leaves the shell
    non-empty, which is exactly right: the ticket is not finished with.

    IRODS has no ticket-level collection to match this -- landings are flat
    under <server>/landing/<landing_id> -- so this is local disk only.
    """

    if not args.local or not args.delete:
        return 0

    ticket_dir = os.path.join(args.local_root, f"ticket-{cand.ticket_id}")

    if not os.path.isdir(ticket_dir):
        return 0

    try:
        if any(entry.is_dir() for entry in os.scandir(ticket_dir)):
            return 0
    except OSError:
        return 0

    try:
        shutil.rmtree(ticket_dir)
    except OSError as e:
        say(f"    ticket directory removal FAILED {ticket_dir}: {e}")
        return 0

    return 1


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
    if args.local and not args.local_root:
        work_dir = os.environ.get("MDREPO_WORK_DIR")
        if not work_dir:
            sys.exit("--local needs MDREPO_WORK_DIR or --local-root")
        args = args._replace(
            local_root=LANDING_ROOT.format(work_dir=work_dir,
                                           server=args.server)
        )

    if args.local and not os.path.isdir(args.local_root):
        sys.exit(f"Local landing root '{args.local_root}' is not a directory")

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

    totals = dict(EMPTY)
    totals.update({"tickets": 0, "skipped": 0, "unverified": 0, "verified": 0,
                   "unchecked": 0, "held_unverified": 0, "held_unchecked": 0,
                   "reaped": 0, "in_flight": 0})
    sessions = SessionPool(args.irods_env, REMOVE_TIMEOUT)

    try:
        for cand in candidates:
            if not args.full_scan and already_purged(sessions, cand, args):
                totals["skipped"] += 1
                # Still reap: a previous run can leave the shell behind if it
                # was interrupted between the last landing and the directory,
                # and nothing else would ever come back for it.
                totals["reaped"] += reap_ticket_dir(cand, args)
                continue

            if job_in_flight(cur, cand.ticket_id):
                # Started since the candidate query. Leave every landing alone
                # and let the next pass have it: the run will still be there.
                totals["in_flight"] += 1
                say(f"ticket {cand.ticket_id}: a job is in flight, skipping")
                continue

            # Verify this ticket right before deleting it, not in a bulk pass
            # up front: a ticket can become eligible mid-pass and would then be
            # deleted having never been checked, which --force makes permanent
            # on the IRODS side and rmtree makes permanent on the local one.
            if not args.skip_verify:
                problems, unchecked = verify_permanent(cur, sessions, cand, args)

                # A refusal that belongs to no simulation is about the ticket
                # itself -- nothing to verify against at all -- so it refuses
                # the whole ticket rather than any one landing.
                if None in problems:
                    totals["unverified"] += 1
                    flat = [m for msgs in problems.values() for m in msgs]
                    say(f"ticket {cand.ticket_id}: NOT VERIFIED, refusing to "
                        f"delete ({len(flat)} problem(s))")
                    for p in flat[:5]:
                        print(f"    {p}", flush=True)
                    if len(flat) > 5:
                        print(f"    ... and {len(flat) - 5} more", flush=True)
                    continue

                if problems or unchecked:
                    # One simulation that did not verify holds back its own
                    # landing and nothing else. Both reasons are counted, and
                    # they mean different things: a problem is the safety net
                    # firing, an unchecked file is the network saying nothing.
                    # An unchecked landing stays eligible, so the next pass
                    # retries it -- one such probe killed the 2026-07-31 pass
                    # 20 tickets in, back when it aborted the whole ticket.
                    bad = set(problems) | set(unchecked)
                    kept = [l for l in cand.landings if l.sim_id not in bad]
                    totals["held_unverified"] += len(problems)
                    totals["held_unchecked"] += len(unchecked)
                    say(f"ticket {cand.ticket_id}: holding "
                        f"{len(cand.landings) - len(kept)} landing(s) -- "
                        f"{len(problems)} did not verify, {len(unchecked)} "
                        f"unreachable")
                    for m in [m for msgs in problems.values() for m in msgs][:5]:
                        print(f"    {m}", flush=True)
                    cand = cand._replace(
                        landings=kept, num_sims=len(kept)
                    )
                    if not kept:
                        continue

                totals["verified"] += 1

            if args.verify_only:
                say(f"ticket {cand.ticket_id}: verified "
                    f"({len(cand.landings)} landing(s))")
                continue

            result = purge_ticket(sessions, cand, args)

            if not (result["present"] or result["local_present"]):
                say(f"ticket {cand.ticket_id}: landings already gone")
                continue

            for key in result:
                totals[key] += result[key]
            totals["tickets"] += 1

            reaped = reap_ticket_dir(cand, args)
            totals["reaped"] += reaped

            verb = "removed" if args.delete else "would remove"
            note = "" if cand.has_job else "  [no process job -- pre-queue]"
            if reaped:
                note += "  [ticket directory reaped]"
            if cand.held:
                note += f"  [{cand.held} landing(s) held, not eligible]"
            if result["stalled"]:
                note += f"  [{result['stalled']} STALLED, re-run to retry]"
            if result["local_failed"]:
                note += f"  [{result['local_failed']} local removal(s) FAILED]"

            where = f"{human(result['freed'])} IRODS"
            if args.local:
                where += f" + {human(result['local_freed'])} local"

            say(
                f"ticket {cand.ticket_id}: {cand.num_sims} sims, "
                f"{verb} {result['present']} IRODS + "
                f"{result['local_present']} local of "
                f"{len(cand.landings)} landing(s), {where}{note}"
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
    # IRODS and local are never added together. They free different disks, one
    # of them is the one that fills, and a single number would hide which.
    print(
        f"\n{verb} {totals['removed'] if args.delete else totals['present']} "
        f"landing collection(s) across {totals['tickets']} ticket(s), "
        f"{human(totals['freed'])} in IRODS"
    )

    if args.local:
        print(
            f"{verb} {totals['local_removed'] if args.delete else totals['local_present']} "
            f"local landing director(ies), {human(totals['local_freed'])} on "
            f"{args.local_root}"
        )

    if totals["in_flight"]:
        print(f"Skipped {totals['in_flight']} ticket(s) with a job in flight; "
              f"they stay eligible and the next run retries them")

    if totals["reaped"]:
        print(f"Reaped {totals['reaped']} empty ticket director(ies)")

    if totals["local_failed"]:
        print(f"{totals['local_failed']} local removal(s) FAILED; the IRODS "
              f"side of those landings is gone, so re-run with --local to "
              f"finish them")

    if totals["held_unverified"] or totals["held_unchecked"]:
        print(f"Held {totals['held_unverified']} landing(s) that did not "
              f"verify and {totals['held_unchecked']} IRODS did not answer "
              f"for; the rest of their tickets was purged normally")

    if totals["skipped"]:
        both = " in both places" if args.local else " in IRODS"
        print(f"Skipped {totals['skipped']} ticket(s) already purged{both} "
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
    if (totals["removed"] or totals["local_removed"] or totals["stalled"]
            or totals["unchecked"]):
        summary = (f"Landing purge ({args.server}): removed "
                   f"{totals['removed']} landing collection(s) across "
                   f"{totals['tickets']} ticket(s), {human(totals['freed'])}")
        if args.local:
            summary += (f"; {totals['local_removed']} local director(ies), "
                        f"{human(totals['local_freed'])}")
        if totals["held_unverified"] or totals["held_unchecked"]:
            summary += (f"; held {totals['held_unverified']} unverified and "
                        f"{totals['held_unchecked']} unreachable landing(s)")
        if totals["local_failed"]:
            summary += f"; {totals['local_failed']} local removal(s) failed"
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
        print("Deleted to the IRODS trash; empty it to reclaim the space. "
              "Local directories, if any, are gone outright.")


# --------------------------------------------------
if __name__ == "__main__":
    main()
