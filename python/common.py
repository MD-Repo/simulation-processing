"""Shared helpers for the simulation-processing scanner and queue worker."""

import os

import requests

FRONTEND_BASE_URLS = {
    "staging": "https://staging.mdrepo.org",
    "prod": "https://mdrepo.org",
}


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
