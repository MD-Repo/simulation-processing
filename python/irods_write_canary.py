#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-11
Purpose: Prove that IRODS will accept a real write before the pipeline commits
         hours of work to a ticket.

         Writes a payload into the same collection tree push_sim_files.py
         writes to, using the same client library and the same native
         protocol, verifies it two independent ways, and removes it.

         Two things this is FOR, and one it is not:

           - Not burning ~2h of BLAST and gmx before discovering the zone is
             down. mdr-process fetches, converts and BLASTs long before it
             pushes, so a write outage is currently discovered at the very end
             of the most expensive part of the run.
           - Telling "our pipeline broke" apart from "CyVerse broke". On
             2026-08-04 CyVerse was down 4.5 hours and the only evidence was
             ~50 failing scanner passes nobody was watching.

         It is NOT here to catch silent corruption. push_sim_files.py already
         MD5-verifies every file, writes complete: false on a mismatch, leaves
         is_placeholder true so a broken simulation is never publicly visible,
         and reports FAILED to Slack. Do not re-justify this script on silent
         failure -- that argument is spent, and it is why this was deliberately
         not built in August.

         Exit codes: 0 healthy, 1 usage/setup error, 75 (EX_TEMPFAIL) the
         canary failed -- IRODS would not take the write.
"""

import argparse
import hashlib
import os
import signal
import sys
from datetime import datetime, timezone
from typing import NamedTuple, Optional

import irods.keywords as kw
from irods.session import iRODSSession

from common import EX_TEMPFAIL, stamp

# 5 MiB, and the size is load-bearing -- do not shrink it to "keep the canary
# cheap". During the 2026-08-05 outage a 2-byte put SUCCEEDED while a 5 MB put
# hung indefinitely: bulk transfer was dead while the control channel still
# worked. A tiny canary would have reported green through the whole thing. 5
# MiB is the smallest size with direct evidence of discriminating, and a 1 MB
# put was separately observed returning HIERARCHY_ERROR, so the floor is
# somewhere under this and well over two bytes.
#
# Above 32 MiB python-irodsclient switches to a parallel multi-connection
# transfer, which is the path large trajectory tars actually take. Raising
# --size-mb past 32 exercises that instead; the default stays under it so the
# canary tests one connection's worth of bulk transfer and stays cheap.
DEFAULT_SIZE_MB = 5

# A hang is this script's worst failure mode, not an error. python-irodsclient
# has no internal timeout and the observed outage signature was an indefinite
# stall, so without this the canary becomes the thing that wedges: the drain
# calls it while holding the flock, so a permanent hang would stop every later
# tick from ever running while never exiting non-zero for cron_notify to see.
# Exactly the failure backup_database.py added --put-timeout for (item 27).
DEFAULT_TIMEOUT = 180


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    irods_env: str
    size_mb: int
    timeout: int
    verbose: bool


class CanaryResult(NamedTuple):
    """Outcome of one canary run"""

    ok: bool
    detail: str
    elapsed: float


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Prove IRODS will accept a write before committing work",
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
        "--irods-env",
        help="IRODS environment file",
        metavar="FILE",
        default=os.environ.get(
            "IRODS_ENVIRONMENT_FILE",
            os.path.expanduser("~/.irods/irods_environment.json"),
        ),
    )

    parser.add_argument(
        "--size-mb",
        help="Payload size in MiB (see DEFAULT_SIZE_MB before lowering)",
        metavar="INT",
        type=int,
        default=DEFAULT_SIZE_MB,
    )

    parser.add_argument(
        "--timeout",
        help="Hard limit for the whole run, in seconds",
        metavar="INT",
        type=int,
        default=DEFAULT_TIMEOUT,
    )

    parser.add_argument("--verbose", help="Verbose", action="store_true")

    args = parser.parse_args()

    if not args.irods_env or not os.path.isfile(args.irods_env):
        parser.error(f'Invalid or missing --irods-env file "{args.irods_env}"')

    if args.size_mb < 1:
        parser.error("--size-mb must be at least 1")

    return Args(
        server=args.server,
        irods_env=args.irods_env,
        size_mb=args.size_mb,
        timeout=args.timeout,
        verbose=args.verbose,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    def status(msg: str) -> None:
        if args.verbose:
            print(f"{stamp()} {msg}", flush=True)

    result = run_canary(
        server=args.server,
        irods_env=args.irods_env,
        size_mb=args.size_mb,
        timeout=args.timeout,
        status=status,
    )

    if result.ok:
        status(f"IRODS write canary OK ({result.detail}) in {result.elapsed:.1f}s")
        sys.exit(0)

    # Printed unconditionally, not through status(): a caller running without
    # --verbose still needs the reason, and cron_notify quotes the tail of this
    # output into the Slack alert.
    print(
        f"{stamp()} IRODS write canary FAILED after {result.elapsed:.1f}s: "
        f"{result.detail}",
        flush=True,
    )
    sys.exit(EX_TEMPFAIL)


# --------------------------------------------------
def canary_collection(server: str) -> str:
    """Where the canary writes

    Deliberately inside the release tree rather than somewhere tidy like
    /iplant/home/mdadm. The push writes here, and at one point in the
    2026-08-06 investigation the fault looked target-specific
    (HIERARCHY_ERROR is what resource resolution failing for a particular
    target looks like), so a canary that writes somewhere else can pass while
    every push fails. Same tree, same resource hierarchy, same account.

    The name is verbose on purpose: anyone finding this collection in a
    listing should be able to tell it is not simulation data without asking.
    """

    return f"/iplant/home/shared/mdrepo/{server}/release/_mdrepo_write_canary"


# --------------------------------------------------
def run_canary(
    server: str,
    irods_env: str,
    size_mb: int = DEFAULT_SIZE_MB,
    timeout: int = DEFAULT_TIMEOUT,
    status=None,
    tmp_dir: str = "/tmp",
) -> CanaryResult:
    """Write, verify twice, and remove a payload. Never raises.

    Importable: drain_process_queue.py calls this before claiming a job, so a
    raised exception here would turn a grid outage into a drain crash. Every
    failure comes back as ok=False with a human-readable detail instead.

    **Call this from the main thread only.** The timeout is SIGALRM, which is
    main-thread-only and would silently never fire from a worker thread,
    leaving the hang this exists to bound completely unguarded. True today for
    both callers (the CLI below, and the drain, which is single-threaded at the
    point it calls this). purge_processed_landings.py hit the same constraint
    and bounds its threaded path with the session's connection_timeout instead
    -- do that rather than this if it ever needs to run under a pool.
    """

    status = status or (lambda _msg: None)
    started = datetime.now(timezone.utc)

    def elapsed() -> float:
        return (datetime.now(timezone.utc) - started).total_seconds()

    # Unique per run, so two hosts or an overlapping tick cannot collide on the
    # same object and read each other's bytes back as their own.
    nonce = os.urandom(8).hex()
    name = f"canary-{started.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}-{nonce}.bin"
    local_path = os.path.join(tmp_dir, name)
    collection = canary_collection(server)
    remote_path = os.path.join(collection, name)

    session = None
    wrote_remote = False

    # Recorded rather than inferred from the exception type. SIGALRM interrupts
    # whatever syscall is in flight, and python-irodsclient catches that and
    # re-raises it as NetworkException("Could not receive server response") --
    # so the TimeoutError never reaches the handler below and a timeout reports
    # itself as a network fault. Measured, not theorised. Since telling one
    # kind of failure from another is half of why this script exists, the alarm
    # records that it fired and that reading wins.
    timed_out = {"fired": False}

    def on_alarm(_signum, _frame):
        timed_out["fired"] = True
        raise TimeoutError("canary timed out")

    old_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(timeout)

    try:
        payload = _make_payload(size_mb, nonce)
        local_md5 = hashlib.md5(payload).hexdigest()
        with open(local_path, "wb") as fh:
            fh.write(payload)

        status(f"Canary payload {len(payload):,} bytes, md5 {local_md5[:8]}")

        session = iRODSSession(irods_env_file=irods_env)

        # collections.create is part of what is being tested, not setup around
        # it: push_sim_files.py creates the per-simulation collection before
        # putting into it, and "mkdir + put into prod/release" is the exact
        # pattern recorded as failing and later recovering on 2026-08-07.
        if not session.collections.exists(collection):
            status(f"Creating {collection}")
            session.collections.create(collection)

        status(f"Putting {remote_path}")
        session.data_objects.put(local_path, remote_path, **{kw.FORCE_FLAG_KW: ""})
        wrote_remote = True

        # Verification 1: the server's own checksum. Catches bytes that never
        # landed, or landed short.
        obj = session.data_objects.get(remote_path)
        if obj.size != len(payload):
            return CanaryResult(
                False,
                f"size mismatch: put {len(payload):,} bytes, "
                f"catalog reports {obj.size:,}",
                elapsed(),
            )

        chksum = obj.chksum() or ""
        remote_md5 = chksum.split(":", 1)[-1].strip().lower()
        if remote_md5 != local_md5:
            return CanaryResult(
                False,
                f"checksum mismatch: local {local_md5}, remote {remote_md5 or '(none)'}",
                elapsed(),
            )

        # Verification 2: read the bytes back. Not redundant with the checksum
        # -- this whole item exists because a write path returned success while
        # the bytes did not change (the WebDAV 204s of 2026-08-07), and because
        # a stale replica can be served in place of what was just written. A
        # server-computed checksum and a client-side read-back can disagree,
        # and it is the read-back that matches what a consumer would get.
        with session.data_objects.open(remote_path, "r") as fh:
            readback = fh.read()

        if hashlib.md5(readback).hexdigest() != local_md5:
            return CanaryResult(
                False,
                f"read-back mismatch: wrote {len(payload):,} bytes, "
                f"read {len(readback):,} that hash differently",
                elapsed(),
            )

        return CanaryResult(
            True, f"{len(payload):,} bytes, md5 {local_md5[:8]}, verified twice",
            elapsed(),
        )

    except Exception as e:
        if timed_out["fired"]:
            return CanaryResult(
                False,
                f"timed out after {timeout}s -- the observed outage signature "
                f"is an indefinite stall, so treat this as the zone being "
                f"unwritable (surfaced as {type(e).__name__})",
                elapsed(),
            )
        return CanaryResult(False, f"{type(e).__name__}: {e}", elapsed())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        _cleanup(session, remote_path if wrote_remote else None, local_path, status)


# --------------------------------------------------
def _make_payload(size_mb: int, nonce: str) -> bytes:
    """Incompressible bytes of the requested size

    Random rather than zeros on purpose: a run of zeros is exactly the thing a
    broken transfer produces by accident, so a zero-filled canary that came
    back as zeros would verify. os.urandom also means the payload differs
    every run, which stops a cached or stale replica from satisfying the
    read-back.
    """

    return (f"mdrepo-write-canary {nonce}\n".encode() + os.urandom(size_mb * 1024 * 1024))[
        : size_mb * 1024 * 1024
    ]


# --------------------------------------------------
def _cleanup(
    session, remote_path: Optional[str], local_path: str, status
) -> None:
    """Best-effort removal of both copies

    Never raises: cleanup failing must not turn a healthy canary into a failed
    one. It is reported loudly, though -- a canary object we could not remove
    is itself evidence, since stuck objects that could not be deleted are what
    the CyVerse admin had to clear by hand on 2026-08-07.
    """

    if remote_path is not None and session is not None:
        try:
            # force=True: straight delete, no trash. Trashing a canary would
            # accumulate one object per run for the 30-40 days CyVerse takes to
            # expire it, for bytes nobody will ever want back.
            session.data_objects.unlink(remote_path, force=True)
            status(f"Removed {remote_path}")
        except Exception as e:
            print(
                f"{stamp()} WARNING: could not remove canary object "
                f"{remote_path}: {type(e).__name__}: {e}",
                flush=True,
            )

    if session is not None:
        try:
            session.cleanup()
        except Exception:
            pass

    try:
        os.unlink(local_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"{stamp()} WARNING: could not remove {local_path}: {e}", flush=True)


# --------------------------------------------------
if __name__ == "__main__":
    main()
