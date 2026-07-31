"""Shared helpers for the simulation-processing scanner and queue worker."""

import os
from datetime import datetime, timezone

import requests

FRONTEND_BASE_URLS = {
    "staging": "https://staging.mdrepo.org",
    "prod": "https://mdrepo.org",
}

# Per-ticket mdr-process logs, under <root>/<server>. Absolute and outside the
# repo: these used to default to a relative "logs", which meant the right thing
# only while the cron line still had a "cd" in front of it, and which put ~165
# MB of debug log inside the *public* simulation-processing checkout.
#
# Shared here rather than owned by the drain, because prune_ticket_logs.py has
# to agree with it exactly -- it deletes from this directory, so a second
# definition that drifted would either miss the files or point somewhere it
# has no business deleting from.
TICKET_LOG_ROOT = "/opt/mdrepo/logs/tickets"


# --------------------------------------------------
def stamp() -> str:
    """UTC timestamp for a log line

    A cron log with no clock in it cannot answer the first question anyone
    asks of it -- when did this happen, and how long did it take. A scanner
    pass takes tens of seconds and runs every 5 minutes, so without this there
    is no telling a slow pass from a stuck one.

    The format matches what mdr-process writes into the per-ticket logs, so a
    ticket's own log and the drain line that started it read the same way.

    UTC, not local: everything else here (Postgres timestamps, IRODS, the
    mdr-process logs) is UTC, and a log that mixes the two is worse than one
    that picks the less friendly zone.
    """

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------
def send_slack_message(
    message: str, base_url: str, channel: str = "mdrepo-alerts"
) -> None:
    """Post a message to Slack (best-effort, mirrors slack_messages.send_message)"""

    token = os.getenv("SLACK_TOKEN")
    if not token:
        print(f'No SLACK_TOKEN, not sending Slack message "{message}"')
        return

    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "channel": channel,
                "text": f"{message} ({base_url})",
                "username": "Bot User",
            },
            timeout=10,
        )
        resp.raise_for_status()

        # Slack answers 200 with {"ok": false} for a rejected post -- a renamed
        # or archived channel, the bot removed from it, a rotated token. Status
        # alone therefore reports success for a message nobody received, which
        # would silently disable every alert here. Raise into the handler below
        # so it prints like any other send failure.
        body = resp.json()
        if not body.get("ok"):
            raise RuntimeError(
                f"Slack rejected the post: {body.get('error', 'unknown')}"
            )
    except Exception as e:
        print(f'Unable to send Slack message "{message}": {e}')
