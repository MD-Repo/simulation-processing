"""
Purpose: Shared read tickets on released simulations (MDR-115).

         A released simulation has one unlimited-use iRODS read ticket on its
         MDR* collection, stored in md_simulation.irods_ticket and handed to
         everyone who downloads it. Since MDR-115 each ticket expires
         (md_simulation.irods_ticket_expires_at) and renew_public_tickets.py
         replaces it nightly before it gets near the end. md-repo-app's
         irods_utils.get_download_ticket uses the same 30-day default and
         mints a replacement itself if a stored ticket would end before the
         download token it is building.

         Used by create_irods_ticket.py and renew_public_tickets.py.
"""

import datetime
from typing import Optional

from irods.ticket import Ticket

RELEASE_ROOT = "/iplant/home/shared/mdrepo/{server}/release"

# A new ticket lives this long (md-repo-app SHARED_TICKET_LIFETIME_DAYS).
LIFETIME_DAYS = 30

# Renew a ticket once it has this many days or fewer left. Must be longer
# than a download token's life, so a new token always gets a string that
# outlives it.
RENEW_WITHIN_DAYS = 16

# A download token lives this long (md-repo-app TICKET_EXPIRATION_MINUTES,
# 14,400 = 10 days), and keeps the ticket strings it was given.
TOKEN_DAYS = 10

# An old permanent ticket gets this long before it stops working: longer
# than any token that already holds it can live.
OLD_TICKET_GRACE_DAYS = 11

# SQL for "released": approved, not embargoed, not a placeholder. Matches
# md-repo-app's Simulation.is_visible_to(AnonymousUser()). Deprecated
# simulations count as released.
RELEASED_SQL = "is_public and not is_embargoed and not is_placeholder"


# --------------------------------------------------
def now_utc() -> datetime.datetime:
    """The current time, time-zone aware"""

    return datetime.datetime.now(datetime.timezone.utc)


# --------------------------------------------------
def expire_value(when: datetime.datetime) -> str:
    """A ticket expiry for Ticket.modify("expire", ...): epoch seconds

    Epoch seconds name the same instant whatever time zone this host or the
    iRODS server is in. The client wants strings for "mod" arguments.
    A naive datetime is refused: its instant depends on the host's zone.
    """

    if when.tzinfo is None:
        raise ValueError("expire_value needs a time-zone-aware datetime")
    return str(int(when.timestamp()))


# --------------------------------------------------
def sim_path(server: str, sim_id: int) -> str:
    """A simulation's release collection in iRODS"""

    return f"{RELEASE_ROOT.format(server=server)}/MDR{sim_id:08d}"


# --------------------------------------------------
def label(string: Optional[str]) -> str:
    """A ticket string safe for a log: the first four characters"""

    return f"{string[:4]}…" if string else "(none)"


# --------------------------------------------------
def mint_read_ticket(session, path: str, expires_at: datetime.datetime) -> str:
    """Issue an unlimited-use read ticket on path that ends at expires_at

    The expiry is set straight after the ticket is made. If anything after
    the issue fails, the new ticket is deleted, so a failure never leaves a
    ticket behind with no expiry. Returns the ticket string.
    """

    value = expire_value(expires_at)
    ticket = Ticket(session)
    ticket.issue("read", path)
    try:
        ticket.modify("expire", value)
        # Unlimited uses ("iticket mod <string> uses 0"). The value must be
        # a string: the client drops a falsy int like 0.
        ticket.modify("uses", "0")
    except Exception:
        try:
            ticket.delete()
        except Exception:
            pass
        raise
    return ticket.string


# --------------------------------------------------
def set_expiry(session, string: str, expires_at: datetime.datetime) -> None:
    """Give an existing ticket an expiry"""

    Ticket(session, ticket=string).modify("expire", expire_value(expires_at))
