#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-07-30
Purpose: Run a command and report its failure to Slack

Cron on this box cannot report anything. There is no MAILTO in the crontab,
/var/mail is empty, and postfix is installed but masked -- so everything cron
would have mailed is discarded. That is one of the three faults that kept
export_mapping_file.py silently broken from 2026-02-13, and it covers every job
here, not just that one.

A redirect does not save you either: in

    cd /some/dir && ./thing.py >> log 2>&1

the redirect binds to thing.py, not to the cd, so a failed cd writes to the
mailer and is lost. Absolute paths with no cd avoid that, and wrapping the
whole line here catches it regardless.

Wrap a cron line with this and a non-zero exit becomes a Slack message:

    cron_notify.py --label "export mapping (uniprot)" -- ./export_mapping_file.py uniprot

Output is passed through unchanged, so the log keeps everything; Slack gets the
label, the exit status and the tail. Success is silent -- a nightly "it worked"
trains the channel to be ignored, which is the one thing a failure alert cannot
afford.

DEDUPLICATION, and why a high-frequency job needs it
----------------------------------------------------
Alerting on every failing run is only safe for a job that runs daily. The
scanner runs every 5 minutes and the drain every minute, so one stuck ticket
unwrapped is up to 288 messages a day from a single fault -- which is how a
channel gets muted, and a muted channel is worse than no alerting at all.

So a small JSON state file per label (see common.CRON_STATE_ROOT) carries three
decisions across runs:

1. Alert only after --threshold *consecutive* failing runs. Transient IRODS
   errors are a standing background rate here, not a novelty, and the scanner
   exits 1 for as long as any single ticket errors. Alerting on the first
   failure spends a message (and later a recovery message) on a blip that this
   file elsewhere says to ignore unless it stops self-healing. Measured on the
   prod scanner log: a transient has never survived more than ONE consecutive
   pass, while the 2026-08-04 CyVerse outage (15:35-19:55Z, "Could not connect
   to specified host and port: data.cyverse.org:1247") failed ~50 in a row.
   The two populations do not overlap, so any k in 2..5 separates them; 3 is
   the default, which stays silent through a blip and still escalates a real
   outage within k*5 = 15 minutes. Daily and weekly jobs should pass -k 1:
   with -k 3 a daily job would take three days to say anything.

2. Do not repeat the same alert -- but do not go silent forever either. After
   the first alert the failure is re-reported at most once per --repeat-after
   hours, so a fault that outlives everyone's memory keeps surfacing without
   filling the channel.

3. Say when it clears. A recovery message is sent on the first success after an
   alert, so the channel answers "is it still broken" without anyone reading a
   log.

The state advances only on a *delivered* message, never on an attempted one:
send_slack_message is best-effort and returns False on a rejected post, so
recording the attempt would let a Slack outage suppress every retry of the
alert -- converting a temporary outage into permanent silence for whatever
broke during it. An undelivered alert leaves the state untouched, and the next
failing run tries again.

The state file is an optimisation of *messaging*, not a record of truth. If it
is lost, corrupt, or unwritable, the run still reports normally (at worst a
duplicate message) and the wrapped command's exit code is never affected.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, NamedTuple, Optional

from dotenv import load_dotenv

from common import CRON_STATE_ROOT, FRONTEND_BASE_URLS, send_slack_message

# Slack messages have limits and a wall of log helps nobody: enough lines to
# see what broke, with the log named for the rest.
TAIL_LINES = 15

# Consecutive failing runs before the first alert. See the module docstring for
# where 3 comes from; it is only right for the */5 and * * * * * jobs, and every
# daily or weekly cron line should pass -k 1.
DEFAULT_THRESHOLD = 3

# Hours before an alert that is still failing is repeated.
DEFAULT_REPEAT_AFTER = 24.0

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Args(NamedTuple):
    """Command-line arguments"""

    label: str
    server: str
    threshold: int
    repeat_after: float
    state_dir: str
    command: List[str]


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Run a command, reporting failure to Slack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--label",
        help="Human name for this job, used in the Slack message",
        metavar="STR",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--server",
        help="Target server (selects the URL quoted in the message)",
        metavar="STR",
        choices=["staging", "prod"],
        default="prod",
    )

    parser.add_argument(
        "-k",
        "--threshold",
        help="Consecutive failing runs before alerting (1 for daily/weekly jobs)",
        metavar="INT",
        type=int,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "-r",
        "--repeat-after",
        help="Hours before repeating an alert that is still failing (0 to never)",
        metavar="HOURS",
        type=float,
        default=DEFAULT_REPEAT_AFTER,
    )

    parser.add_argument(
        "--state-dir",
        help="Where the per-label state files live",
        metavar="DIR",
        default=CRON_STATE_ROOT,
    )

    parser.add_argument(
        "command",
        help="Command to run, after a '--'",
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("No command given (put it after '--')")

    if args.threshold < 1:
        parser.error(f"--threshold must be at least 1, not {args.threshold}")

    if args.repeat_after < 0:
        parser.error(f"--repeat-after cannot be negative ({args.repeat_after})")

    return Args(
        label=args.label,
        server=args.server,
        threshold=args.threshold,
        repeat_after=args.repeat_after,
        state_dir=args.state_dir,
        command=command,
    )


# --------------------------------------------------
def state_path(state_dir: str, label: str) -> str:
    """Path of the state file for this label

    Keyed on the label because that is the only name a cron line gives a job;
    two lines sharing a label share a state file, which is why the labels in
    utils/cron/crontab are distinct per server.
    """

    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "job"

    return os.path.join(state_dir, f"cron_notify-{slug}.json")


# --------------------------------------------------
def read_state(path: str) -> Dict[str, Any]:
    """Load the state file, treating anything unreadable as a fresh start

    Never raises. A corrupt or unreadable state file must cost at most one
    duplicate Slack message -- it must not stop the run reporting, and it must
    not change the exit code of the wrapped command.
    """

    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Ignoring unreadable state file {path}: {e}")
        return {}

    if not isinstance(state, dict):
        print(f"Ignoring malformed state file {path}: not an object")
        return {}

    return state


# --------------------------------------------------
def write_state(path: str, state: Dict[str, Any]) -> None:
    """Save the state file, or say why it could not be saved"""

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except Exception as e:
        print(f"Unable to write state file {path}: {e}")


# --------------------------------------------------
def clear_state(path: str) -> None:
    """Remove the state file, the representation of "nothing is wrong" """

    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Unable to remove state file {path}: {e}")


# --------------------------------------------------
def parse_time(value: Optional[str]) -> Optional[datetime]:
    """Read a timestamp written by a previous run, if it is readable"""

    if not value:
        return None

    try:
        return datetime.strptime(value, TIME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# --------------------------------------------------
def should_alert(
    state: Dict[str, Any], failures: int, threshold: int, repeat_after: float, now: datetime
) -> bool:
    """Decide whether this failing run is worth a message

    Three gates, in order: not enough consecutive failures yet; already
    reported and not yet due to repeat; otherwise report.
    """

    if failures < threshold:
        return False

    alerted_at = parse_time(state.get("alerted_at"))
    if alerted_at is None:
        # Either nothing has been delivered for this episode, or the previous
        # attempt failed to reach Slack. Both mean: try now.
        return True

    if repeat_after <= 0:
        return False

    return now - alerted_at >= timedelta(hours=repeat_after)


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    try:
        proc = subprocess.run(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output, code = proc.stdout, proc.returncode
    except OSError as e:
        # The command could not be started at all -- the exact failure that
        # made export_mapping_file.py invisible ("not found", wrong PATH).
        output, code = f"could not run {args.command[0]}: {e}", 127

    # Pass the output through first, so the log is complete even if Slack fails.
    if output:
        sys.stdout.write(output if output.endswith("\n") else output + "\n")
        sys.stdout.flush()

    path = state_path(args.state_dir, args.label)
    state = read_state(path)
    base_url = FRONTEND_BASE_URLS[args.server]
    now = datetime.now(timezone.utc)

    if code == 0:
        failures = int(state.get("consecutive_failures", 0) or 0)

        # Only worth a word if somebody was told it was broken. A run that
        # succeeds after a couple of sub-threshold failures says nothing,
        # because nothing was ever said about them.
        if state.get("alerted_at"):
            # A previous success may already have zeroed the count while its
            # own recovery message failed to send; that run parked the number
            # here so this one can still report it.
            failures = int(state.get("failures_at_recovery", failures) or failures)
            since = state.get("first_failure", "unknown")
            message = (
                f"CRON RECOVERED: {args.label} succeeded after "
                f"{failures} failing run(s) since {since}"
            )
            if not send_slack_message(message, base_url):
                # Keep the episode open so the next success tries again. The
                # failure alert was delivered, so silence here would leave the
                # channel believing the job is still down.
                state["consecutive_failures"] = 0
                state["failures_at_recovery"] = failures
                write_state(path, state)
                sys.exit(0)

        clear_state(path)
        sys.exit(0)

    # A recovery that never reached Slack is abandoned here rather than carried
    # forward: the job worked in between, so this is a new episode. Leaving the
    # old alerted_at in place would instead hold the new failure silent until
    # --repeat-after came round, which is the one thing this must not do.
    if "failures_at_recovery" in state:
        print(f"{args.label}: recovered and failed again, starting a new episode")
        state = {}

    failures = int(state.get("consecutive_failures", 0) or 0) + 1
    state["consecutive_failures"] = failures
    state["label"] = args.label
    state["last_failure"] = now.strftime(TIME_FORMAT)
    state["last_exit_code"] = code

    # A continuing episode keeps the timestamp it started at -- that is the
    # "since" the messages quote.
    state.setdefault("first_failure", state["last_failure"])

    if should_alert(state, failures, args.threshold, args.repeat_after, now):
        tail = "\n".join(output.strip().splitlines()[-TAIL_LINES:])
        message = f"CRON FAILED: {args.label} (exit {code})"
        if failures > 1:
            message += (
                f", {failures} consecutive failing runs "
                f"since {state['first_failure']}"
            )
        if tail:
            message += f"\n```\n{tail}\n```"

        # Delivered, not attempted: a rejected post leaves alerted_at unset, so
        # the next failing run tries again rather than being deduplicated away.
        if send_slack_message(message, base_url):
            state["alerted_at"] = now.strftime(TIME_FORMAT)
    elif failures < args.threshold:
        print(
            f"{args.label}: failing run {failures} of {args.threshold} "
            "before alerting, no Slack message sent"
        )
    else:
        print(
            f"{args.label}: failing run {failures}, already reported at "
            f"{state['alerted_at']}, no Slack message sent"
        )

    write_state(path, state)
    sys.exit(code)


# --------------------------------------------------
if __name__ == "__main__":
    main()
