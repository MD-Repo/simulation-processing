#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-05
Purpose: Dump the MDRepo database and store it in IRODS and Jetstream Swift

This replaces /home/ubuntu/db-backup.sh on the k3s master, which ansible
installed as a plain crontab entry (playbook.yaml, "Add cron for db backup").
It was never a Kubernetes CronJob -- there are no CronJob resources anywhere in
terraform-mdrepo -- so "move it off Kubernetes" really means "move it off the
database VM and onto the processing box, next to the rest of the scheduled
pipeline".

WHAT THE OLD SCRIPT GOT RIGHT, and is kept here: the day-of-month rotation
(mdrepo.01..31.sql.gz, overwritten each month) plus mdrepo.latest.sql.gz, two
destinations, and a size check on the Swift upload. The restore path in
terraform-mdrepo (roles/mdrepo-db) fetches mdrepo.latest.sql.gz by name, so
that name is load-bearing and does not change.

WHAT IS FIXED HERE
------------------
1. The dump can no longer be silently partial. The old script ran

       sudo su - postgres -c "pg_dump $DB_NAME | gzip > $DB_DUMP"

   and `set -o pipefail` in the outer script does NOT reach the shell that
   `su` starts. So a pg_dump that died partway still left gzip exiting 0, and
   the truncated dump was uploaded as if it were good. Here pg_dump's exit
   status is checked directly, and the artifact is verified before it is sent:
   gzip integrity plus the "-- PostgreSQL database dump complete" trailer that
   only a finished pg_dump writes.

2. The IRODS upload is verified. The old script size-checked the Swift copy
   and nothing at all in IRODS -- the same blind spot that let ticket 1355 sit
   behind a truncated 889 MB object that exists() happily reported as present.

3. Every replica is checked, not just the object. On 2026-08-05, 28 of the 32
   objects in the prod collection had two replicas of DIFFERENT sizes, both
   flagged good (status 1): mdrepo.latest.sql.gz was a 469 MB dump from Aug 4
   on ds13 and a 194 MB dump from Jul 3 on corral4. Restores read whichever
   replica IRODS picks, so that is a coin flip between yesterday's database and
   a month-old one. This cannot fix the replication itself -- that is
   CyVerse-side -- but it refuses to call a backup good while it is true.

4. The upload time is recorded in metadata we control. The catalog's own
   timestamps proved untrustworthy: on 2026-08-05 `gocmd ls` showed
   mdrepo.04.sql.gz last modified 2026-07-04 while the object contained
   tickets created 2026-08-03. Every upload here also sets an AVU
   (BACKUP_TIME_AVU) so "when was this actually written" has an answer that
   does not depend on the catalog's mtime. --verify-only reports it.

5. The two destinations are checked independently, because they have failed
   independently. Swift's copy of 2026-07-08 exists, is 187,065,941 bytes and
   is stamped 00:00:34; IRODS still holds a June 8 dump in that slot. So the
   dump ran and the Swift upload worked while the IRODS put silently did not,
   and for a month nothing knew. Swift is also the destination whose listing
   can be trusted -- correct timestamps, no replicas -- which is how the IRODS
   timestamp problem was confirmed rather than merely suspected.

6. Failures are visible. The old job wrote /var/log/db-backup.log on a host
   nobody watches and mailed cron output into a void. This exits non-zero and
   the cron line wraps it in cron_notify.py, which is what puts it in Slack.
   Deliberately no send_slack_message() call in here: one alert per failure,
   from one place.

WHY THE DUMP CAN RUN FROM HERE AT ALL: the app user in .env reaches Postgres
over the network (pg_hba.conf on the DB host allows this box), and the ansible
role makes mduser the owner of every table in the public schema, so its dump is
complete. Measured 2026-08-05: 1m18s for a 513 MB gzipped dump of the 1.2 GB
production database. The upload, not the dump, is the long pole.

VERSION MATTERS. The server is PostgreSQL 14 and the restore path uses the
PG 14 client, so this defaults to the 14 client (--pg-dump), not the 16 one
Ubuntu 24.04 ships. Dumping an older server with a newer client is supported;
restoring a newer client's output into an older server is where it bites.
"""

import argparse
import errno
import fcntl
import gzip
import os
import shutil
import ssl
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, NamedTuple

from dotenv import load_dotenv
from irods.session import iRODSSession
from irods.meta import iRODSMeta

from common import stamp

# The restore path in terraform-mdrepo (ansible/roles/mdrepo-db/tasks/main.yaml)
# fetches this name. Do not rename it.
LATEST_NAME = "mdrepo.latest.sql.gz"

# The old script derived the collection from `hostname | sed s/-/./g`, which
# worked only because it ran ON the database VM. Run anywhere else and that
# silently writes to a collection named after the wrong machine, so the
# mapping is explicit here.
IRODS_COLLECTIONS = {
    "prod": "/iplant/home/mdadm/pg_dumps/mdrepo.org",
    "staging": "/iplant/home/mdadm/pg_dumps/staging.mdrepo.org",
}

DSN_KEYS = {"prod": "PRODUCTION_DSN", "staging": "STAGING_DSN"}

# Jetstream Swift, the second destination -- and currently the healthier of the
# two: its object listing is complete and its timestamps are trustworthy, which
# is what made the IRODS timestamp problem legible at all.
#
# The ansible version had a single JETSTREAM_BACKUP_DIR set per deployment in
# that deployment's terraform.tfvars, which works only because each deployment
# backed up its own database from its own host. One host backing up both
# databases needs the mapping instead. These are container names, not secrets,
# so they live here rather than in .env; --swift-container overrides.
SWIFT_CONTAINERS = {
    "prod": "mdrepo_db_backups",
    "staging": "mdrepo_staging_db_backups",
}

# The PG 14 client, matching the server; see the module docstring.
DEFAULT_PG_DUMP = "/usr/lib/postgresql/14/bin/pg_dump"

DEFAULT_WORK_DIR = "/opt/mdrepo/backups"

# Only a finished pg_dump writes this, so its presence in the last few KB
# separates "the dump ended" from "the dump stopped".
COMPLETE_MARKER = b"PostgreSQL database dump complete"

# The monthly archive: the first backup of each calendar month is ALSO stored
# under a full date, mdrepo.YYYY-MM-DD.sql.gz, and nothing ever overwrites it.
#
# The rotation is 31 slots reused every month, so a dump survives at most ~31
# days: the July 6 dump is overwritten on August 6, and by the end of a month
# every trace of the previous one is gone. That is fine for "restore what broke
# last night" and useless for "what did this table look like in June". The two
# digits also cannot be widened -- mdrepo.32.sql.gz is not a day -- so the
# archive is a separate name space rather than a longer rotation.
#
# The trigger is "no archive exists for this month yet" rather than "today is
# the 1st", because the 1st can fail: 2026-08-05 produced no backup at all in
# either destination. Under this rule the month's archive is simply its first
# *successful* backup. Each destination is checked separately, since the two
# have failed independently before.
#
# It grows without bound by design: twelve objects a year, ~0.5 GB each and
# rising. Against a 10 TiB Swift quota holding 9.5 GB today that is decades of
# headroom, but it is deliberate accumulation, not a leak.
ARCHIVE_TEMPLATE = "mdrepo.%Y-%m-%d.sql.gz"
ARCHIVE_MONTH_PREFIX = "mdrepo.%Y-%m"

# Our own upload timestamp, because the catalog's mtime lies here.
BACKUP_TIME_AVU = "mdrepo_backup_time"
BACKUP_BYTES_AVU = "mdrepo_backup_bytes"
BACKUP_HOST_AVU = "mdrepo_backup_host"


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    work_dir: str
    collection: str
    pg_dump: str
    swift_container: str
    no_swift: bool
    no_irods: bool
    no_archive: bool
    put_timeout: int
    keep_local: bool
    verify_only: bool
    dry_run: bool
    lock_file: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Dump the MDRepo database to IRODS and Swift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--server",
        help="Which database to back up",
        metavar="STR",
        choices=sorted(DSN_KEYS),
        default="prod",
    )

    parser.add_argument(
        "-w",
        "--work-dir",
        help="Local scratch directory for the dump",
        metavar="DIR",
        default=DEFAULT_WORK_DIR,
    )

    parser.add_argument(
        "-c",
        "--collection",
        help="IRODS collection to write to (default: per --server)",
        metavar="PATH",
        default=None,
    )

    parser.add_argument(
        "--pg-dump",
        help="pg_dump binary; should match the server's major version",
        metavar="PATH",
        default=DEFAULT_PG_DUMP,
    )

    parser.add_argument(
        "--swift-container",
        help="Swift container (default: per --server)",
        metavar="STR",
        default=None,
    )

    parser.add_argument(
        "--no-swift",
        help="Skip the Swift copy instead of failing when it cannot be made",
        action="store_true",
    )

    parser.add_argument(
        "--no-archive",
        help="Skip the once-a-month dated snapshot",
        action="store_true",
    )

    parser.add_argument(
        "--put-timeout",
        help="Seconds a single IRODS put may make no progress before it is killed",
        metavar="INT",
        type=int,
        default=1800,
    )

    parser.add_argument(
        "--no-irods",
        help="Skip the IRODS copy (its writes go down; Swift's do not)",
        action="store_true",
    )

    parser.add_argument(
        "--keep-local",
        help="Keep the local dump instead of removing it when done",
        action="store_true",
    )

    parser.add_argument(
        "--verify-only",
        help="Report on what is already stored; dump nothing",
        action="store_true",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        help="Dump and verify locally, but upload nothing",
        action="store_true",
    )

    parser.add_argument(
        "--lock-file",
        help="Lock file preventing overlapping runs",
        metavar="PATH",
        default=None,
    )

    args = parser.parse_args()

    if args.no_swift and args.no_irods:
        parser.error("--no-swift and --no-irods together leave nowhere to put it")

    return Args(
        server=args.server,
        work_dir=args.work_dir,
        collection=args.collection or IRODS_COLLECTIONS[args.server],
        pg_dump=args.pg_dump,
        swift_container=args.swift_container or SWIFT_CONTAINERS[args.server],
        no_swift=args.no_swift,
        no_irods=args.no_irods,
        no_archive=args.no_archive,
        put_timeout=args.put_timeout,
        keep_local=args.keep_local,
        verify_only=args.verify_only,
        dry_run=args.dry_run,
        lock_file=args.lock_file or f"/tmp/backup_database-{args.server}.lock",
    )


# --------------------------------------------------
def acquire_lock(path: str):
    """Take a non-blocking exclusive flock; return the fd, or None if held

    In-script rather than on the cron line, as the purge does and for the same
    reason: this is run by hand too, and a cron-line flock would not cover that.
    """

    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            fd.close()
            return None
        raise
    return fd  # held for the life of the process


# --------------------------------------------------
def irods_session() -> iRODSSession:
    """Open an IRODS session the same way the rest of the pipeline does"""

    env_file = os.environ.get(
        "IRODS_ENVIRONMENT_FILE",
        os.path.expanduser("~/.irods/irods_environment.json"),
    )

    return iRODSSession(
        irods_env_file=env_file,
        ssl_context=ssl.create_default_context(),
    )


# --------------------------------------------------
def dump_database(pg_dump: str, dsn: str, dest: str) -> int:
    """Write a gzipped pg_dump to `dest`; return its size in bytes

    pg_dump's stdout is compressed here rather than in a shell pipeline, so its
    exit status is actually checked. The shell version could not: `su` starts a
    new shell that does not inherit pipefail, so gzip's success masked
    pg_dump's failure and a truncated dump shipped as a good one.
    """

    proc = subprocess.Popen(
        [pg_dump, "--dbname", dsn],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    tail = b""
    try:
        with gzip.open(dest, "wb") as out:
            while True:
                chunk = proc.stdout.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                # Free: the trailer check below needs the end of the *input*,
                # and we are already holding it.
                tail = (tail + chunk)[-8192:]
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        proc.wait()

    if proc.returncode != 0:
        os.path.exists(dest) and os.remove(dest)
        raise RuntimeError(
            f"pg_dump exited {proc.returncode}: {stderr.strip() or 'no output'}"
        )

    if COMPLETE_MARKER not in tail:
        os.path.exists(dest) and os.remove(dest)
        raise RuntimeError(
            "pg_dump exited 0 but its output does not end with "
            f'"{COMPLETE_MARKER.decode()}" -- refusing to upload a partial dump'
        )

    return os.path.getsize(dest)


# --------------------------------------------------
def verify_gzip(path: str) -> None:
    """Decompress the file to confirm it is intact, discarding the output

    Cheap next to the dump itself (~15s for 500 MB) and it is the only check
    that reads every byte that will be uploaded.
    """

    with gzip.open(path, "rb") as fh:
        while fh.read(1 << 22):
            pass


# --------------------------------------------------
def put_to_irods(local: str, collection: str, timeout: int, status) -> None:
    """Upload with gocmd, verifying the checksum on the way in

    gocmd rather than python-irodsclient because it is what the rest of the
    pipeline uses for bulk transfer and it is markedly faster on a file this
    size. -k makes it verify the checksum after transfer; --retry survives the
    kind of transient CyVerse error the scanner sees regularly.

    The timeout is the important part. A gocmd put has no internal bound, and
    on 2026-08-05 CyVerse writes stalled such that a put neither progressed nor
    failed -- one ran over an hour moving ~11 KB/s with no data connections
    open. Unbounded, that is the worst failure mode this job has: it holds the
    flock so the next night's run exits, and it never exits, so cron_notify
    never sees a non-zero status and never says anything. A hang would look
    exactly like a healthy silent night, indefinitely.
    """

    name = os.path.basename(local)
    status(f"Uploading {name} to {collection}")

    try:
        proc = subprocess.run(
            [
                "gocmd", "put", "-f", "-k",
                "--retry", "3",
                "--retry_interval", "30",
                local,
                collection + "/",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"gocmd put of {name} made no progress in {timeout}s and was "
            "killed. The target may be left with a stale or 0-byte replica "
            "that blocks the next attempt with SYS_INTERNAL_ERR (-154000); "
            "unlink it before retrying."
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"gocmd put of {name} exited {proc.returncode}: "
            f"{proc.stdout.strip() or 'no output'}"
        )


# --------------------------------------------------
def check_replicas(session, path: str, expected: int, status) -> List[str]:
    """Return a list of complaints about the stored object; empty means good

    Both halves matter. The object's own size catches a put that died partway
    -- which is not hypothetical: an interrupted upload on 2026-08-05 left
    mdrepo.05.sql.gz as a 0-byte replica. The per-replica check catches the
    divergence described in the module docstring, where a stale copy sits
    beside a fresh one and both claim to be good.
    """

    obj = session.data_objects.get(path)
    problems = []

    if obj.size != expected:
        problems.append(f"object size {obj.size:,} != local {expected:,}")

    for repl in obj.replicas:
        where = f"replica {repl.number} on {repl.resource_name}"
        if int(repl.status) != 1:
            problems.append(f"{where} is not good (status {repl.status})")
        if repl.size != expected:
            problems.append(f"{where} size {repl.size:,} != local {expected:,}")

    status(
        f"{os.path.basename(path)}: {len(obj.replicas)} replica(s), "
        + ", ".join(
            f"#{r.number} {r.size:,} on {r.resource_name} (status {r.status})"
            for r in obj.replicas
        )
    )

    return problems


# --------------------------------------------------
def stamp_metadata(session, path: str, size: int, when: datetime) -> None:
    """Record when this object was really written

    The catalog's mtime is not reliable here (see the module docstring), so the
    only trustworthy answer to "how old is this backup" is one we write
    ourselves. AVUs survive an overwrite, hence the explicit set.
    """

    obj = session.data_objects.get(path)
    for key, value in (
        (BACKUP_TIME_AVU, when.strftime("%Y-%m-%dT%H:%M:%SZ")),
        (BACKUP_BYTES_AVU, str(size)),
        (BACKUP_HOST_AVU, os.uname().nodename),
    ):
        obj.metadata[key] = iRODSMeta(key, value)


# --------------------------------------------------
def swift_names(container: str, prefix: str) -> List[str]:
    """Object names in `container` starting with `prefix`"""

    proc = subprocess.run(
        ["swift", "list", container, "--prefix", prefix],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"swift list of {container} failed: {proc.stdout.strip()}")

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


# --------------------------------------------------
def needs_month_archive(existing: List[str], when: datetime) -> bool:
    """True if this month has no archived snapshot yet

    Kept separate from the listing so the rule can be read at a glance: the
    month is archived if nothing already carries this month's prefix.
    """

    return not any(n.startswith(when.strftime(ARCHIVE_MONTH_PREFIX)) for n in existing)


# --------------------------------------------------
def upload_to_swift(local: str, container: str, name: str, size: int, status) -> None:
    """Copy to Jetstream Swift and confirm the stored length

    A straight port of the old script's swift_upload_and_verify, kept because
    it is the only copy of the backup that does not live in CyVerse -- and the
    CyVerse copy is the one with the replica problem.
    """

    status(f"Uploading {name} to Swift container {container}")
    up = subprocess.run(
        ["swift", "upload", container, local, "--object-name", name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if up.returncode != 0:
        raise RuntimeError(f"swift upload of {name} failed: {up.stdout.strip()}")

    stat = subprocess.run(
        ["swift", "stat", container, name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if stat.returncode != 0:
        raise RuntimeError(f"swift stat of {name} failed: {stat.stdout.strip()}")

    remote = None
    for line in stat.stdout.splitlines():
        if "Content Length:" in line:
            remote = line.split()[-1]

    if remote != str(size):
        raise RuntimeError(
            f"Swift size mismatch for {name}: local={size:,} remote={remote}"
        )

    status(f"Swift upload verified: {name} ({size:,} bytes)")


# --------------------------------------------------
def report_stored(session, collection: str, status) -> int:
    """Describe what is in the collection now; return a problem count

    This is the audit the old arrangement had no way to do. It reads the AVU
    this script writes rather than the catalog's mtime, so a listing that looks
    a month stale can be told apart from a backup that IS a month stale.
    """

    coll = session.collections.get(collection)
    problems = 0

    status(f"{collection}: {len(coll.data_objects)} object(s)")
    for obj in sorted(coll.data_objects, key=lambda d: d.name):
        sizes = {r.size for r in obj.replicas}
        written = obj.metadata.get_all(BACKUP_TIME_AVU)
        when = written[0].value if written else "unknown (no AVU; pre-dates this script)"
        flag = ""
        if len(sizes) > 1:
            flag = "  <-- REPLICAS DISAGREE"
            problems += 1
        if any(int(r.status) != 1 for r in obj.replicas):
            flag += "  <-- REPLICA NOT GOOD"
            problems += 1
        status(
            f"  {obj.name:24} {obj.size:>13,}  written {when}"
            f"  catalog mtime {obj.modify_time:%Y-%m-%d %H:%M}{flag}"
        )

    return problems


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    def status(msg: str) -> None:
        print(f"{stamp()} {msg}", flush=True)

    lock = acquire_lock(args.lock_file)
    if lock is None:
        status(f"Another backup holds {args.lock_file}, exiting")
        sys.exit(0)

    if args.verify_only:
        with irods_session() as session:
            problems = report_stored(session, args.collection, status)
        status(f"FINISHED verify-only ({problems} problem(s))")
        sys.exit(1 if problems else 0)

    dsn = os.environ.get(DSN_KEYS[args.server])
    if not dsn:
        sys.exit(f"No {DSN_KEYS[args.server]} in the environment")

    if not os.path.exists(args.pg_dump):
        sys.exit(
            f"No pg_dump at {args.pg_dump} -- install postgresql-client-14 or "
            "pass --pg-dump"
        )

    os.makedirs(args.work_dir, exist_ok=True)
    started = datetime.now(timezone.utc)
    day = started.strftime("%d")
    dump_name = f"mdrepo.{day}.sql.gz"
    dump_path = os.path.join(args.work_dir, dump_name)
    latest_path = os.path.join(args.work_dir, LATEST_NAME)

    status(f"Starting backup of {args.server} to {dump_name}")

    try:
        size = dump_database(args.pg_dump, dsn, dump_path)
        status(f"Dump written: {dump_path} ({size:,} bytes)")

        verify_gzip(dump_path)
        status("Dump verified: gzip intact and pg_dump trailer present")

        # The same bytes under both names, so latest is never a different
        # dump from the day's file -- and never an older one, which is how
        # mdrepo.latest.sql.gz came to disagree with its own day slot.
        shutil.copyfile(dump_path, latest_path)

        if args.dry_run:
            status("DRY RUN: not uploading")
            return

        archive_name = started.strftime(ARCHIVE_TEMPLATE)
        failures = []

        # Swift first, and each destination inside its own try. The two have
        # failed independently -- Swift holds a 2026-07-08 dump IRODS never
        # received, and on 2026-08-05 CyVerse writes stalled while Swift ran at
        # ~51 MB/s -- so letting one abort the other would mean a CyVerse
        # outage costs us the copy that WAS reachable. The run still exits
        # non-zero, so the night is reported as failed either way; what changes
        # is that the backup exists somewhere while we deal with it.
        if args.no_swift:
            status("Skipping the Swift copy (--no-swift)")
        else:
            try:
                swift_targets = [(dump_path, dump_name), (latest_path, LATEST_NAME)]
                if not args.no_archive:
                    # The month prefix itself, not a looser "mdrepo.20" -- that
                    # also matches the day-20 rotation slot.
                    existing = swift_names(
                        args.swift_container, started.strftime(ARCHIVE_MONTH_PREFIX)
                    )
                    if needs_month_archive(existing, started):
                        status(f"First Swift backup of {started:%B %Y}: also "
                               f"archiving as {archive_name}")
                        swift_targets.append((dump_path, archive_name))
                    else:
                        status(f"{started:%B %Y} is already archived in Swift")

                for path, name in swift_targets:
                    upload_to_swift(path, args.swift_container, name, size, status)
            except Exception as e:
                status(f"SWIFT FAILED: {e}")
                failures.append(f"Swift: {e}")

        if args.no_irods:
            status("Skipping the IRODS copy (--no-irods)")
        else:
            try:
                problems = []
                with irods_session() as session:
                    names = [o.name for o in
                             session.collections.get(args.collection).data_objects]
                    irods_targets = [dump_path, latest_path]
                    if args.no_archive:
                        pass
                    elif needs_month_archive(names, started):
                        status(f"First IRODS backup of {started:%B %Y}: also "
                               f"archiving as {archive_name}")
                        archive_path = os.path.join(args.work_dir, archive_name)
                        shutil.copyfile(dump_path, archive_path)
                        irods_targets.append(archive_path)
                    else:
                        status(f"{started:%B %Y} is already archived in IRODS")

                    # The day slot goes first on purpose: if it fails, latest is
                    # left holding yesterday's good dump rather than a broken
                    # write. That is not hypothetical -- an upload interrupted
                    # during the 2026-08-05 CyVerse outage left latest with a
                    # 0-byte replica and no usable copy. The month's archive goes
                    # last: it is the copy nothing depends on for a restore.
                    for path in irods_targets:
                        name = os.path.basename(path)
                        put_to_irods(path, args.collection, args.put_timeout, status)
                        remote = f"{args.collection}/{name}"
                        problems += [f"{name}: {p}" for p in
                                     check_replicas(session, remote, size, status)]
                        stamp_metadata(session, remote, size, started)

                if problems:
                    raise RuntimeError(
                        "copies did not verify:\n  " + "\n  ".join(problems)
                    )
            except Exception as e:
                status(f"IRODS FAILED: {e}")
                failures.append(f"IRODS: {e}")

    finally:
        if not args.keep_local:
            for path in (dump_path, latest_path,
                         os.path.join(args.work_dir, started.strftime(ARCHIVE_TEMPLATE))):
                if os.path.exists(path):
                    os.remove(path)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    if failures:
        status(
            f"FAILED backup_database ({len(failures)} of the destinations): "
            + "; ".join(failures)
        )
        sys.exit(1)

    status(f"FINISHED backup_database ({size:,} bytes in {elapsed:.0f}s)")


# --------------------------------------------------
if __name__ == "__main__":
    main()
