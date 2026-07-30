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
"""

import argparse
import os
import subprocess
import sys
from typing import List, NamedTuple

from dotenv import load_dotenv

from common import FRONTEND_BASE_URLS, send_slack_message

# Slack messages have limits and a wall of log helps nobody: enough lines to
# see what broke, with the log named for the rest.
TAIL_LINES = 15


class Args(NamedTuple):
    """Command-line arguments"""

    label: str
    server: str
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

    return Args(label=args.label, server=args.server, command=command)


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

    if code == 0:
        sys.exit(0)

    tail = "\n".join(output.strip().splitlines()[-TAIL_LINES:])
    message = f"CRON FAILED: {args.label} (exit {code})"
    if tail:
        message += f"\n```\n{tail}\n```"

    send_slack_message(message, FRONTEND_BASE_URLS[args.server])
    sys.exit(code)


# --------------------------------------------------
if __name__ == "__main__":
    main()
