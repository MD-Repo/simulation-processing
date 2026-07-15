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
    except Exception as e:
        print(f'Unable to send Slack message "{message}": {e}')
