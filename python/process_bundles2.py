#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-14
Purpose: Process the bundles2/ backlog -- simulation directories a contributor
         is pushing directly into IRODS at
         /iplant/home/shared/mdrepo/bundles2/, outside the ticket-upload flow,
         so nothing in md_ticket / md_process_job knows they exist.

         One invocation claims the oldest ready bundle, downloads it,
         extracts it, runs "mdr-process process" on it, and cleans up the
         local copy. Meant to run from cron, one bundle per tick -- there is
         deliberately no long-lived loop or retry/backoff machinery here,
         mirroring drain_process_queue.py's "one claim per invocation" shape.

         A bundle is a TARBALL, not a collection. Measured 2026-08-14:
         bundles2/ holds 481 "<name>.tgz" objects and 473 "<name>.tgz.md5"
         sidecars, and ZERO subcollections. So this downloads one object,
         unpacks it into the scratch directory, and processes the directory
         that comes out.

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

         NOTHING IN IRODS IS EVER DELETED BY THIS SCRIPT. An earlier version
         removed the source and its sidecar once mdr-process exited 0, on the
         same "verified and permanent" argument purge_processed_landings.py
         uses. That was dropped 2026-08-14 (Ken): the contributor's tarballs
         are the only copy of the pre-ingest form of this data, and deletion
         is irreversible (unlink --force leaves no trash), so the storage
         saved is not worth the class of mistake it enables.

         The consequence is that "already done" can no longer be inferred
         from the source being gone, so it is recorded instead -- an
         append-only log, see load_record()/append_record(). Losing that log
         costs re-processing, not corruption: mdr-process matches an existing
         simulation by unique_file_hash_string and updates it rather than
         inserting a duplicate.

         On failure the unpacked local copy is KEPT, under the scratch
         directory, so a human can see what is wrong, fix it in place and
         re-run without pulling a quarter-gigabyte tarball again. Nothing
         reaps those, so watch disk if failures pile up -- the scratch
         directory shares a filesystem with live ticket processing. On
         success it is removed.

         Nothing here retries automatically; a failed bundle is recorded and
         skipped until a human removes its line from the log.
"""

import argparse
import os
import shlex
import shutil
import ssl
import subprocess
import tarfile
import sys
from subprocess import getstatusoutput
from typing import Dict, NamedTuple, Optional, Tuple

import psycopg2
import psycopg2.extras
from irods.models import Collection, DataObject
from irods.session import iRODSSession

from common import (
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


class Fetched(NamedTuple):
    """Where to run mdr-process, and what to delete afterwards.

    Two paths, not one: a bundle usually unpacks to a single top-level
    directory, so the directory handed to mdr-process is a level BELOW the
    one that has to be removed. Returning only the former leaks the
    extraction root -- one empty directory per bundle at best, and real bytes
    whenever a tarball unpacks to more than one entry.
    """

    process_dir: str
    root: str


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    lock_file: str
    record_file: str
    work_dir: str
    log_dir: str
    irods_env: str
    num_threads: int
    smiles_table: str
    fix_smiles: str
    keep_local: bool
    max_bundles: int
    dry_run: bool
    no_slack: bool
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
        help="Target server. Defaults to staging, deliberately: everything "
        "processed out of bundles2 so far went to staging, nothing has been "
        "promoted to prod, and a default of prod meant one forgotten flag "
        "wrote a 15,000-bundle backfill straight into production",
        metavar="STR",
        choices=["staging", "prod"],
        default="staging",
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
        "--record",
        help="Append-only log of which bundles are done/failed. This is what "
        "makes a bundle 'already processed' now that the IRODS source is "
        "never deleted (default: "
        "/opt/mdrepo/logs/bundles2/<server>/processed.tsv)",
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
        "--num-threads",
        help="Passed to mdr-process as a global --num-threads. It defaults to "
        "num_cpus (32 here), which is how one invocation came to hold ~320 "
        "IRODS connections and collapse (MDR-33); the drain's ticket line has "
        "been pinned at 4 ever since. A bundle pushes once rather than per "
        "landing, so the connection count was never the danger here -- the "
        "CPU was, against everything else sharing this box",
        metavar="INT",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--smiles-table",
        help="pdb_id/ligand_name/canonical_smiles TSV used to fill in a "
        "ligand SMILES the submitter omitted. Skipped silently if absent",
        metavar="FILE",
        default=os.path.expanduser("~/pdbbind_ligand_smiles.tsv"),
    )

    parser.add_argument(
        "--fix-smiles",
        help="Path to fix_ligand_smiles.py. Invoked as a subprocess rather "
        "than imported because it lives in the internal utils repo while "
        "this file is in the public one; a missing script is a skip, not an "
        "error, so this repo does not hard-depend on that one",
        metavar="PATH",
        default="/opt/mdrepo/utils/python/fix_ligand_smiles.py",
    )

    parser.add_argument(
        "--keep-local",
        help="Keep the unpacked copy after a SUCCESSFUL run too (a failure "
        "always keeps it). For a staging backfill that will be re-run "
        "against prod: the tarball does not have to be pulled twice",
        action="store_true",
    )

    parser.add_argument(
        "--max-bundles",
        help="Process up to this many bundles before exiting, mirroring "
        "drain_process_queue.py's --max-jobs. Keep it at 1 on cron",
        metavar="INT",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dry-run",
        help="Download and unpack the next candidate and run 'mdr-process "
        "process --dry-run' on it, but write nothing to the record, so the "
        "same bundle is picked again next time (implies --verbose)",
        action="store_true",
    )

    parser.add_argument(
        "--no-slack",
        help="Print the failure notice instead of posting it to "
        "#mdrepo-alerts. For hand-running and testing: a real run of this "
        "script alerts on every failed bundle, which is correct on cron and "
        "pure noise when a human is iterating on one bundle",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Verbose", action="store_true")

    args = parser.parse_args()

    lock_file = args.lock_file or os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"drain_process_queue-{args.server}.lock",
    )
    # Deliberately NOT under CRON_STATE_ROOT (/opt/mdrepo/state). That
    # directory is cron_notify.py's failing-right-now signal, and the standing
    # sweep's first check is "ls /opt/mdrepo/state/ -- non-empty means
    # something is failing". A permanent record file there would make that
    # check useless forever.
    record_file = args.record or os.path.join(
        "/opt/mdrepo/logs/bundles2", args.server, "processed.tsv"
    )
    log_dir = args.log_dir or os.path.join(
        "/opt/mdrepo/logs/bundles2", args.server
    )

    return Args(
        server=args.server,
        lock_file=lock_file,
        record_file=record_file,
        work_dir=args.work_dir,
        log_dir=log_dir,
        irods_env=args.irods_env,
        num_threads=args.num_threads,
        smiles_table=args.smiles_table,
        fix_smiles=args.fix_smiles,
        keep_local=args.keep_local,
        max_bundles=args.max_bundles,
        dry_run=args.dry_run,
        no_slack=args.no_slack,
        verbose=args.verbose or args.dry_run,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    load_env()

    base_url = FRONTEND_BASE_URLS[args.server]

    def status(msg: str) -> None:
        if args.verbose:
            print(f"{stamp()} {msg}", flush=True)

    def notify(message: str) -> None:
        """Alert, unless asked not to -- but always say what would have gone.

        Suppressing silently would make a hand-run look like it alerted when
        it did not, which is the same class of lie as a state file recording
        "we tried" for a message that never arrived (see send_slack_message).
        """

        if args.no_slack:
            print(f"{stamp()} [--no-slack] would have posted to Slack:\n"
                  f"{message}", flush=True)
        else:
            send_slack_message(message, base_url)

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

    record = load_record(args.record_file)
    status(f"{len(record):,} bundle(s) already recorded")

    ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    with iRODSSession(
        irods_env_file=args.irods_env, ssl_context=ssl_context
    ) as session:
        for n in range(1, args.max_bundles + 1):
            name = find_next_candidate(session, record)
            if name is None:
                status("No ready bundles found")
                break

            status(f"Claimed bundle {name} ({n} of at most {args.max_bundles})")
            result = process_one(
                session, args, name, record, status, notify
            )
            record[name] = result
            if args.dry_run:
                break

    return


# --------------------------------------------------
def process_one(session, args, name, record, status, notify) -> str:
    """Fetch, fix, process and record one bundle. Returns its result."""

    try:
        fetched = fetch_bundle(name, args.work_dir, status)
        local_dir = fetched.process_dir
    except Exception as e:
        # Not recorded, and there is deliberately no cleanup of a
        # possibly-partial local download: gocmd's own --diff makes the next
        # tick's retry resume rather than re-pull everything.
        status(f"ERROR fetching {name}: {e}")
        notify(f"bundles2 fetch failed for {name}: {e}")
        return "fetch-failed"

    fix_missing_smiles(local_dir, args, status)

    log_file = os.path.join(args.log_dir, f"{name}.log")
    returncode, detail = run_mdr_process(
        local_dir, args.server, log_file, args.dry_run, status,
        args.num_threads,
    )

    if args.dry_run:
        status(f"DRY RUN result for {name}: exit {returncode}\n{detail}")
        status(f"Local copy left at {fetched.root} for inspection")
        return "dry-run"

    # The IRODS source is NEVER touched, on success or failure. The record
    # below is the only thing that stops this bundle being picked again.
    if returncode == 0:
        if args.keep_local:
            status(f"Local copy KEPT at {fetched.root} (--keep-local)")
        else:
            # The extraction root, not just the directory processed: see
            # Fetched. A failure keeps it regardless, below.
            shutil.rmtree(fetched.root, ignore_errors=True)
        append_record(args.record_file, name, "done", "")
        status(f"Processed {name}")
        return "done"
    else:
            # Kept on purpose: the unpacked copy is what a human needs to see
            # what is wrong, fix it, and re-run against the directory without
            # re-downloading a quarter-gigabyte tarball. It is the operator's
            # to remove once the bundle is dealt with; nothing here reaps it,
            # so watch disk if failures pile up (they share the filesystem
            # with live ticket processing).
        # Kept on purpose: the unpacked copy is what a human needs to see what
        # is wrong, fix it, and re-run against the directory without
        # re-downloading a quarter-gigabyte tarball. It is the operator's to
        # remove once the bundle is dealt with; nothing here reaps it, so
        # watch disk if failures pile up (they share the filesystem with live
        # ticket processing).
        append_record(args.record_file, name, "failed", detail)
        status(f"FAILED {name}: {detail}")
        status(f"Local copy KEPT at {fetched.root} for inspection")
        notify(
            f"bundles2 processing failed for {name} (exit {returncode}). "
            f"The IRODS source is untouched at "
            f"{BUNDLES_COLLECTION}/{name}, and the unpacked copy is kept "
            f"at {fetched.root} so it can be fixed and re-run without "
            f"downloading again. It is recorded as failed and will be "
            f"skipped until that line is removed from "
            f"{args.record_file}.\nLog: {log_file}"
        )
        return "failed"


# --------------------------------------------------
def fix_missing_smiles(local_dir: str, args, status) -> None:
    """Fill in a ligand SMILES the submitter omitted, before processing.

    Prophylactic rather than reactive (Ken, 2026-08-14). Reactive means every
    such bundle costs a download, a failed run, a Slack message and a manual
    fix, which does not scale to the 473 bundles sitting in bundles2/. There
    is no cheaper option upstream: these are gzipped tarballs, so the TOML
    cannot be read without pulling the whole object, which rules out the
    check-before-the-fetch approach TODO MDR-44 argues for elsewhere.

    The cost is that a third-party value ends up in a file pushed verbatim to
    release/<MDR>/original/. fix_ligand_smiles.py writes a comment above the
    key saying where it came from and that the submitter did not supply it,
    which is what keeps the archive honest.

    Best-effort and non-fatal: the script refuses on a name mismatch, an
    empty table entry or several ligands missing SMILES, and those all leave
    mdr-process to fail on its own terms with its own message.
    """

    if not os.path.isfile(args.fix_smiles):
        status(f"No {args.fix_smiles}, not filling in SMILES")
        return
    if not os.path.isfile(args.smiles_table):
        status(f"No {args.smiles_table}, not filling in SMILES")
        return

    cmd = [
        sys.executable, args.fix_smiles, local_dir,
        "--table", args.smiles_table,
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        if line.strip() and not line.startswith(("19,", "DRY RUN")):
            status(f"  smiles: {line.strip()}")
    if proc.returncode != 0:
        status(f"  smiles: fixer exited {proc.returncode}: {proc.stderr[:200]}")


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
def load_record(path: str) -> Dict[str, str]:
    """Bundle name -> "done" | "failed", from the append-only log.

    Append-only rather than a rewritten JSON blob for two reasons: with 481
    bundles today and 15,000 eventually, rewriting the whole file every tick
    is the operation most likely to lose the lot to a crash; and a plain TSV
    is greppable and hand-editable, which is how a failed bundle gets retried
    (delete its line).

    A later line wins, so re-running a bundle that once failed and recording
    it "done" reads correctly without anyone editing history.
    """

    record: Dict[str, str] = {}
    try:
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[0] != "timestamp":
                    record[parts[2]] = parts[1]
    except FileNotFoundError:
        pass
    return record


# --------------------------------------------------
def append_record(path: str, name: str, result: str, detail: str) -> None:
    """One line per outcome: timestamp, result, bundle, detail"""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    flat = " ".join((detail or "").split())[:500]
    with open(path, "a") as fh:
        if new:
            fh.write("timestamp\tresult\tbundle\tdetail\n")
        fh.write(f"{stamp()}\t{result}\t{name}\t{flat}\n")
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------
def find_next_candidate(session, record: Dict[str, str]) -> Optional[str]:
    """Oldest ready bundle with no line in the record, or None.

    Ordered by the SIDECAR's create_time, ascending, and ordered by the
    catalog rather than here: results are consumed lazily and the first
    unrecorded candidate wins, so a backlog of 15,000 does not have to be
    materialised to pick one. Sorting client-side would also work but would
    pull every row every tick, forever.

    The sidecar, not the tarball, is the thing to sort on and the thing to
    test: it is written after the upload finishes, so its timestamp is when
    the bundle became ready. A zero-byte sidecar is a truncated or in-flight
    write and is not ready -- the same check check_new_simulations.py makes on
    its completion marker, and the same trap TODO MDR-4 records, where every
    check tested that the marker existed rather than that it had content.
    """

    q = (
        session.query(DataObject.name, DataObject.size, DataObject.create_time)
        .filter(Collection.name == BUNDLES_COLLECTION)
        .filter(DataObject.replica_number == 0)
        .order_by(DataObject.create_time)
    )

    for row in q.get_results():
        name = row[DataObject.name]
        if not name.endswith(".md5"):
            continue
        if row[DataObject.size] == 0:
            continue
        bundle = name[: -len(".md5")]
        if bundle in record:
            continue
        return bundle

    return None


# --------------------------------------------------
def fetch_bundle(name: str, work_dir: str, status) -> Fetched:
    """Download <name> and unpack it, returning the directory to process.

    Mirrors fetch_uploads.py's run_gocmd. "--diff" skips a file already
    present with a matching hash, so a retry after a failed run does not
    re-pull it.

    The unpack step exists because a bundle is a tarball, not a collection
    (see the module docstring). Extraction doubles as the integrity check
    this script would otherwise lack: a truncated .tgz fails to unpack rather
    than being handed to mdr-process as a short directory. The .md5 sidecar
    is still only a readiness signal and is not verified against the bytes.

    Extraction ALWAYS names an explicit destination, so a tarball that
    unpacks its contents loose rather than under a single top-level directory
    spills into that destination and not into the working directory.
    Verified 2026-08-14 against both shapes: a loose archive gives
    process_dir == root, a tidy one descends a level. Never extract relative
    to the cwd -- cron's cwd is the invoking directory, which here is a repo.
    """

    os.makedirs(work_dir, exist_ok=True)
    src = f"{BUNDLES_COLLECTION}/{name}"
    local = os.path.join(work_dir, name)

    status(f"Downloading {src}")
    cmd = f"gocmd get -f --diff {shlex.quote(src)} {shlex.quote(work_dir)}"
    rv, out = getstatusoutput(cmd)
    if rv != 0:
        raise RuntimeError(f"Error running {cmd}: {out}")
    if not os.path.isfile(local):
        raise RuntimeError(f"{cmd} left no file at {local}")

    root = os.path.join(work_dir, name[: -len(".tgz")] if
                        name.endswith(".tgz") else f"{name}.d")
    dest = root
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

    status(f"Unpacking {local} ({os.path.getsize(local):,} bytes)")
    with tarfile.open(local) as tar:
        # filter="data" refuses absolute paths, "..", symlinks out of the
        # tree and device nodes. This is a contributor-supplied archive being
        # unpacked on the machine that also runs the live pipeline.
        tar.extractall(dest, filter="data")

    os.remove(local)  # the tarball itself is not input to mdr-process

    # A well-formed bundle unpacks to exactly one top-level directory; take
    # it, so mdr-process gets the simulation directory rather than a wrapper.
    entries = [e for e in os.listdir(dest) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest, entries[0])):
        return Fetched(os.path.join(dest, entries[0]), root)

    return Fetched(dest, root)


# --------------------------------------------------
def run_mdr_process(
    local_dir: str, server: str, log_file: str, dry_run: bool, status,
    num_threads: int = 4,
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
        # Global, so it must precede the subcommand -- same shape as the
        # drain's "mdr-process --num-threads 4 ticket ...".
        "--num-threads",
        str(num_threads),
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
if __name__ == "__main__":
    main()
