#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-28
Purpose: Fetch simulation upload directories

Fetches each landing directory of an upload ticket, in parallel.

NOT YET IN PRODUCTION. This is fetch_uploads.py plus collection batching
(change 3 below), unit-tested but never run end to end against iRODS. Prove
it on a real ticket into a scratch --out-dir before moving it over the live
fetch_uploads.py. A snapshot also exists on branch "batched-fetch".

This used to fetch one file per "gocmd get" invocation. Each invocation
costs ~11 seconds of connection/authentication handshake regardless of the
file's size (measured: 11.0s wall, 0.05s CPU, for a 2,794-byte file), so on
ticket 1718 -- 2,257 files, 33.4 GB -- fixed cost was 94% of the 6h48m the
fetch took. Fitting the per-file times against size gives

    time ~= 10.2s fixed + size / 24.6 MB/s

i.e. the link is fine; the handshakes are the bill. Three changes follow:

  1. One "gocmd get" per landing directory, passing every wanted object as a
     source, instead of one per file. 2,257 handshakes -> 200.
  2. Landing directories are fetched concurrently (--threads).
  3. A batch of landing directories shares a single invocation, passing each
     as a collection. An extra collection inside an existing call costs ~3s
     against ~11s to start a new one, so 200 invocations become ~8.

A worker owns a batch end to end: it looks each directory up in iRODS, reads
the manifests, then fetches the whole batch and verifies each directory
against its own manifest. Resolving a directory costs ~3s of session calls
(mostly reading the manifest), so on a 200-part ticket that is ~10 minutes --
too much to leave on the main thread while the workers wait. An iRODSSession
cannot be shared across threads, so each worker keeps its own for the life of
the run (see SessionPool); only the ticket directory and ticket.json, which
need no iRODS at all, are prepared up front.
"""

import argparse
import humanize
import json
import math
import os
import psycopg2
import psycopg2.extras
import re
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import dotenv_values
from irods.exception import CollectionDoesNotExist
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
    pattern: Optional[re.Pattern]
    threads: int


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


class RemoteObject(NamedTuple):
    """An iRODS data object to fetch, with the MD5 it must have on arrival"""

    path: str
    name: str
    size: int
    md5: str


class PartRef(NamedTuple):
    """One landing directory of a ticket, before iRODS has been consulted

    This is what a worker claims; it resolves the reference into a Part
    itself, so the lookups for one part overlap another part's transfers.
    """

    ticket_id: int
    part_num: int
    irods_ticket: str
    ticket_dir: str


class Part(NamedTuple):
    """One landing directory of a ticket, resolved and ready to fetch"""

    ticket_id: int
    part_num: int
    landing_dir: str
    dest_dir: str
    objects: List[RemoteObject]


class PartResult(NamedTuple):
    """What a worker did with a Part

    `bytes_fetched` is the size of the part, not what crossed the network:
    on a re-run gocmd's --diff skips files that are already here and
    correct, and those still count. Read the summary rate as a lower bound.
    """

    output: List[str]
    bytes_fetched: int


class FetchError(Exception):
    """A part could not be fetched

    Raised instead of exiting so that a failure in a worker thread reaches
    the main thread, which owns the exit. The message keeps gocmd's own
    output verbatim: drain_process_queue.is_retryable_irods_error matches on
    that text to tell a transient iRODS outage (requeue the ticket) from a
    real failure (needs a human).
    """


SUBMISSION_COMPLETE = "mdrepo-submission.completed.json"

# Cap on how many object paths go into a single "gocmd get". Landing
# directories run to a dozen or so files, so this never bites in practice;
# it is here so that a pathological submission cannot build a command line
# long enough to hit ARG_MAX.
MAX_OBJECTS_PER_CALL = 100

# Cap on how many landing directories share one "gocmd get". A batch is
# all-or-nothing on its first attempt and reports nothing until it finishes,
# so this trades a little of the saving for progress visibility and blast
# radius. Measured: one invocation costs ~11s of handshake, each extra
# collection inside it ~3s.
MAX_COLLECTIONS_PER_CALL = 25


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
        "-T",
        "--ticket-file",
        help="File of ticket IDs, one per line",
        metavar="FILE",
        type=argparse.FileType("rt"),
    )

    parser.add_argument(
        "-p",
        "--pattern",
        help="Only download files whose name matches this regex",
        metavar="REGEX",
        default="",
    )

    parser.add_argument(
        "-j",
        "--threads",
        help="Landing directories to fetch concurrently",
        metavar="INT",
        type=int,
        default=8,
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

    if args.threads < 1:
        parser.error(f"--threads '{args.threads}' must be a positive integer")

    ticket_ids = list(args.ticket_id or [])
    if args.ticket_file:
        for line_num, line in enumerate(args.ticket_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ticket_ids.append(int(line))
            except ValueError:
                parser.error(
                    f"Invalid ticket ID '{line}' on line {line_num} "
                    f"of --ticket-file '{args.ticket_file.name}'"
                )

    pattern = None
    if args.pattern:
        try:
            pattern = re.compile(args.pattern)
        except re.error as e:
            parser.error(f"Invalid --pattern '{args.pattern}': {e}")

    if not args.out_dir:
        args.out_dir = os.path.join("/opt/mdrepo/landing", args.server)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    return Args(
        server=args.server,
        out_dir=args.out_dir,
        irods_env=args.irods_env,
        landing_dirs=args.landing_dir or [],
        ticket_ids=ticket_ids,
        pattern=pattern,
        threads=args.threads,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    # This runs under "mdr-process ticket", which captures stdout through a
    # pipe -- so Python block-buffers it and a long fetch looks hung until it
    # finishes. Line buffering makes the log readable while it runs.
    sys.stdout.reconfigure(line_buffering=True)

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

    # Everything that needs no iRODS: the ticket directory and ticket.json,
    # and the split of each ticket into the parts the workers will claim.
    refs = []
    for ticket_num, ticket in enumerate(tickets, start=1):
        refs.extend(prepare_ticket(ticket, args, ticket_num))

    if not refs:
        print(f"Nothing to fetch, see '{args.out_dir}'")
        return

    batches = make_batches(refs, args)
    print(
        f"\nFetching {len(refs)} landing director(ies) in {len(batches)} "
        f"batch(es) using {args.threads} thread(s)"
    )

    # Each worker resolves its own parts and then fetches them, so the iRODS
    # lookups for one batch overlap the transfers of another. Resolving a
    # part costs ~3s of session calls, which is why it is not left on the
    # main thread: 200 parts would be ~10 minutes of dead time.
    start = datetime.now()
    sessions = SessionPool(args.irods_env)
    fetched = 0
    error = None
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {
                pool.submit(process_batch, batch, args, sessions): batch
                for batch in batches
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                except FetchError as e:
                    # Fail fast, as the serial version did: when iRODS is
                    # unreachable every remaining part fails the same way,
                    # and the queue's retry is what should handle it. Parts
                    # already running cannot be cancelled and will finish.
                    error = str(e)
                    cancel_pending(futures)
                    break
                except Exception:
                    # Anything else (a NetworkException out of a session
                    # call, say) still ends the run -- but cancel first, or
                    # the pool drains all 200 doomed parts on the way out.
                    cancel_pending(futures)
                    raise

                fetched += result.bytes_fetched
                print("\n".join(result.output))
    finally:
        sessions.close()

    if error:
        sys.exit(error)

    elapsed = (datetime.now() - start).total_seconds()
    rate = humanize.naturalsize(fetched / elapsed) if elapsed else "n/a"
    print(
        f"\nFetched {humanize.naturalsize(fetched)} in "
        f"{humanize.precisedelta(datetime.now() - start)} ({rate}/s)"
    )
    print(f"Done, see '{args.out_dir}'")


# --------------------------------------------------
def cancel_pending(futures) -> None:
    """Drop work that has not started; running parts cannot be cancelled"""

    for future in futures:
        future.cancel()


# --------------------------------------------------
class SessionPool:
    """One iRODSSession per worker thread

    A session cannot be shared across threads, and opening one per part
    would pay the ~5s connect every time, so each thread keeps its own for
    the life of the run and they are all closed at the end.
    """

    def __init__(self, irods_env: str) -> None:
        self._irods_env = irods_env
        self._local = threading.local()
        self._sessions = []
        self._lock = threading.Lock()

    def get(self) -> iRODSSession:
        session = getattr(self._local, "session", None)
        if session is None:
            session = iRODSSession(irods_env_file=self._irods_env)
            self._local.session = session
            with self._lock:
                self._sessions.append(session)

        return session

    def close(self) -> None:
        with self._lock:
            for session in self._sessions:
                session.cleanup()
            self._sessions = []


# --------------------------------------------------
def prepare_ticket(ticket: Ticket, args: Args, ticket_num: int) -> List[PartRef]:
    """Set up a ticket's directory and split it into parts to be claimed

    Nothing here touches iRODS, so it stays on the main thread: the workers
    can then create their own directories underneath without racing over
    the ticket directory or ticket.json.
    """

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
                "username": ticket.username,
                "institution": ticket.institution,
            },
            fh,
        )

    return [
        PartRef(
            ticket_id=ticket.ticket_id,
            part_num=irods_num,
            irods_ticket=irods_ticket,
            ticket_dir=ticket_dir,
        )
        for irods_num, irods_ticket in enumerate(irods_tickets, start=1)
    ]


# --------------------------------------------------
def make_batches(refs: List[PartRef], args: Args) -> List[List[PartRef]]:
    """Group parts so that one "gocmd get" can serve several of them

    A gocmd invocation costs ~11s of connect/auth before it moves a byte,
    while an extra collection inside an existing invocation costs ~3s, so
    the fewer invocations the better -- up to a point: a batch is
    all-or-nothing on its first attempt and a long one delays any report of
    progress.

    Batches never span tickets. gocmd names the local directory after the
    collection's LAST path component, so two collections sharing a basename
    land on top of each other; landing directory names are unique within a
    ticket but nothing guarantees that across tickets.
    """

    by_ticket: dict = {}
    for ref in refs:
        by_ticket.setdefault(ref.ticket_id, []).append(ref)

    batches = []
    for ticket_refs in by_ticket.values():
        # Aim for one batch per worker, capped so a single failure cannot
        # take an unbounded amount of work down with it.
        per_batch = max(1, math.ceil(len(ticket_refs) / max(1, args.threads)))
        per_batch = min(per_batch, MAX_COLLECTIONS_PER_CALL)
        batches.extend(chunked(ticket_refs, per_batch))

    return batches


# --------------------------------------------------
def process_batch(
    batch: List[PartRef], args: Args, sessions: SessionPool
) -> PartResult:
    """Resolve a batch of parts and fetch them

    The whole life of these parts on one worker thread, so their iRODS
    lookups overlap another batch's transfers.
    """

    session = sessions.get()
    output: List[str] = []
    parts: List[Part] = []
    for ref in batch:
        part, messages = resolve_part(session, ref, args)
        output.extend(messages)
        if part is not None:
            parts.append(part)

    if not parts:
        return PartResult(output=output, bytes_fetched=0)

    result = fetch_parts(parts, args)

    return PartResult(output=output + result.output, bytes_fetched=result.bytes_fetched)


# --------------------------------------------------
def fetch_parts(parts: List[Part], args: Args) -> PartResult:
    """Fetch several landing directories, then verify each against its manifest

    One "gocmd get" takes every collection at once, trading N connect
    handshakes for one. Two cases fall back to fetching each part on its
    own, naming its objects explicitly:

      - a --pattern is in play, since a collection get cannot filter
      - two collections in the batch share a basename, which gocmd merges
        into one local directory (measured: three collections named
        "original" produced one directory holding 7 of the 15 files)

    Whatever fails verification is retried per part, so one bad landing
    directory does not condemn the rest of the batch.
    """

    ticket_dirs = {os.path.dirname(part.dest_dir) for part in parts}
    names = [os.path.basename(part.dest_dir) for part in parts]

    if args.pattern or len(ticket_dirs) != 1 or len(set(names)) != len(names):
        output = [f"    fetching {len(parts)} part(s) individually"]
        total = 0
        for part in parts:
            result = fetch_part(part)
            output.extend(result.output)
            total += result.bytes_fetched
        return PartResult(output=output, bytes_fetched=total)

    start = datetime.now()
    dest = ticket_dirs.pop()
    size = sum(obj.size for part in parts for obj in part.objects)
    output = [
        f"    fetching {len(parts)} landing dir(s), "
        f"{humanize.naturalsize(size)}, in one gocmd call"
    ]

    run_gocmd([part.landing_dir for part in parts], dest)

    # Verify each part against its own manifest and re-fetch only the
    # stragglers; fetch_part re-runs the same verify/retry cycle for one
    # part, so a partial batch costs one extra invocation per bad part.
    fetched = 0
    for part in parts:
        if any(needs_fetch(part.dest_dir, obj) for obj in part.objects):
            output.append(
                f"    {os.path.basename(part.dest_dir)}: unverified, refetching"
            )
            result = fetch_part(part)
            output.extend(result.output)
        fetched += sum(obj.size for obj in part.objects)

    output.append(f"    batch done in {humanize.precisedelta(datetime.now() - start)}")

    return PartResult(output=output, bytes_fetched=fetched)


# --------------------------------------------------
def resolve_part(session: iRODSSession, ref: PartRef, args: Args):
    """Look up a part in iRODS and decide what to fetch from it

    Returns the Part (or None if there is nothing to fetch) and the lines to
    report. A part that cannot be resolved is reported and skipped, exactly
    as before -- only a hard iRODS failure raises.
    """

    label = f"  Ticket {ref.ticket_id} part {ref.part_num}: "
    if ":" not in ref.irods_ticket:
        return None, [f"{label}Invalid IRODS ticket '{ref.irods_ticket}'"]

    landing_dir = ref.irods_ticket.split(":")[1]

    # Only swallow "genuinely doesn't exist" -- anything else
    # (e.g. a NetworkException from iRODS being unreachable) must
    # propagate so the job fails loudly instead of silently
    # skipping this part and possibly finishing "successfully".
    coll = None
    try:
        coll = session.collections.get(landing_dir)
    except CollectionDoesNotExist:
        pass

    if coll is None:
        return None, [f"{label}Unable to get landing directory '{landing_dir}'"]

    dest_dir = os.path.join(ref.ticket_dir, coll.name)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    output = []
    hash_by_filename = fetch_manifest(session, landing_dir, dest_dir, output)
    if hash_by_filename is None:
        return None, [label.rstrip() + f" no manifest for '{landing_dir}'"] + output

    data_objects = coll.data_objects
    if not data_objects:
        return None, [f"{label}landing dir '{landing_dir}' is empty"]

    objects = []
    for obj in data_objects:
        filename = obj.name
        if filename == SUBMISSION_COMPLETE:
            continue

        if args.pattern and not args.pattern.search(filename):
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
        if completed_md5 := hash_by_filename.get(filename):
            if completed_md5 != irods_md5:
                output.append(
                    f"    complete JSON MD5 '{completed_md5}' != "
                    f"IRODS MD5 '{irods_md5}' for '{filename}'"
                )
                continue

        objects.append(
            RemoteObject(path=obj.path, name=filename, size=obj.size, md5=irods_md5)
        )

    output.insert(
        0,
        f"{label}{landing_dir} has {len(data_objects)} data objects, "
        f"{len(objects)} to fetch",
    )

    if not objects:
        return None, output

    return (
        Part(
            ticket_id=ref.ticket_id,
            part_num=ref.part_num,
            landing_dir=landing_dir,
            dest_dir=dest_dir,
            objects=objects,
        ),
        output,
    )


# --------------------------------------------------
def fetch_manifest(
    session: iRODSSession, landing_dir: str, dest_dir: str, output: List[str]
) -> Optional[dict]:
    """Download and parse the manifest, returning MD5s keyed by filename

    Retries (and falls back to gocmd) because session.data_objects.get
    occasionally writes an empty/truncated file on the compound resource.
    Returns None if the part has no usable manifest and must be skipped.
    Appends to `output` rather than printing: this runs on a worker thread,
    where interleaved prints would shred the log.
    """

    irods_completed = os.path.join(landing_dir, SUBMISSION_COMPLETE)
    if not session.data_objects.exists(irods_completed):
        output.append(f"    Warning: Missing {SUBMISSION_COMPLETE}")
        return None

    local_completed = os.path.join(dest_dir, SUBMISSION_COMPLETE)

    completed = None
    for retry in range(3):
        if os.path.isfile(local_completed):
            os.remove(local_completed)

        if retry == 0:
            session.data_objects.get(irods_completed, local_completed)
        else:
            cmd = f"gocmd get {irods_completed} {local_completed}"
            rv, out = getstatusoutput(cmd)
            if rv != 0:
                output.append(f"    Retry {retry}: error running {cmd}: {out}")
                continue

        try:
            with open(local_completed) as fh:
                completed = json.load(fh)
            break
        except (json.JSONDecodeError, OSError) as e:
            output.append(f"    Retry {retry}: bad {SUBMISSION_COMPLETE}: {e}")

    if completed is None:
        output.append(f"    Unable to fetch {SUBMISSION_COMPLETE}, skipping part")
        return None

    # NB: despite its name, "irods_path" holds just the basename
    # (e.g. "1_prod.mdc"), so this dict is keyed by basename and
    # matches obj.name in the loop below.
    return {v["irods_path"]: v["md5_hash"] for v in completed["files"]}


# --------------------------------------------------
def fetch_part(part: Part) -> PartResult:
    """Fetch one landing directory and verify what landed

    Runs on a worker thread, so it touches no iRODS session: "gocmd" moves
    the bytes and md5sum checks them. Raises FetchError rather than exiting
    so the main thread decides what a failure means.
    """

    start = datetime.now()
    output = [
        f"    fetching {len(part.objects)} object(s), "
        f"{humanize.naturalsize(sum(o.size for o in part.objects))}"
    ]

    # Two passes at most: the first fetches everything (gocmd's --diff skips
    # what is already here and correct), the second re-fetches only what
    # failed verification. A file that is wrong twice is a hard failure.
    todo = list(part.objects)
    for attempt in range(1, 3):
        if not todo:
            break

        if attempt > 1:
            output.append(
                f"    retrying {len(todo)} object(s) that failed verification"
            )
            # Remove the bad copies so --diff cannot decide they are fine.
            for obj in todo:
                local_path = os.path.join(part.dest_dir, obj.name)
                if os.path.isfile(local_path):
                    os.remove(local_path)

        for chunk in chunked(todo, MAX_OBJECTS_PER_CALL):
            run_gocmd([obj.path for obj in chunk], part.dest_dir)

        todo = [obj for obj in todo if needs_fetch(part.dest_dir, obj)]

    if todo:
        raise FetchError(
            f"Ticket {part.ticket_id} part {part.part_num} "
            f"({part.landing_dir}): {len(todo)} object(s) failed to verify "
            f"after 2 attempts: {', '.join(obj.name for obj in todo)}"
        )

    output.append(f"    done in {humanize.precisedelta(datetime.now() - start)}")

    return PartResult(
        output=output, bytes_fetched=sum(obj.size for obj in part.objects)
    )


# --------------------------------------------------
def run_gocmd(paths: List[str], dest_dir: str) -> str:
    """Fetch data objects into a local directory with a single gocmd call

    One invocation costs ~11s of handshake no matter how many sources it is
    given, which is the whole reason this takes a list. "--diff" skips
    files already present with the same hash, so a re-run is cheap.
    """

    quoted = " ".join(shlex.quote(path) for path in paths)
    cmd = f"gocmd get -f --diff {quoted} {shlex.quote(dest_dir)}"
    rv, out = getstatusoutput(cmd)
    if rv != 0:
        # Keep gocmd's text verbatim: the queue reads it to decide whether
        # this was a transient iRODS outage worth retrying.
        raise FetchError(f"Error running {cmd}: {out}")

    return out


# --------------------------------------------------
def needs_fetch(dest_dir: str, obj: RemoteObject) -> bool:
    """True if the local copy is missing or does not match iRODS"""

    local_path = os.path.join(dest_dir, obj.name)
    if not os.path.isfile(local_path):
        return True

    if os.path.getsize(local_path) != obj.size:
        return True

    return get_local_md5(local_path) != obj.md5


# --------------------------------------------------
def chunked(items: List, size: int) -> List[List]:
    """Split a list into chunks of at most `size`"""

    return [items[i : i + size] for i in range(0, len(items), size)]


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
def get_local_md5(filename: str) -> str:
    """get local md5"""

    cmd = f"md5sum {shlex.quote(filename)}"
    rv, out = getstatusoutput(cmd)
    if rv != 0:
        # Raise rather than exit: this runs on worker threads.
        raise FetchError(f"{cmd}: {out}")

    return out.split()[0]


# --------------------------------------------------
if __name__ == "__main__":
    main()
