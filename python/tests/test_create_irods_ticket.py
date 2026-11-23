"""Tests for create_irods_ticket.unreleased_reason

The read ticket this script stores never expires, and nothing revokes it when
a simulation is embargoed, so it must only go to released simulations
(md-repo-app#434).

Pure function -- no IRODS, no network, no database.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from create_irods_ticket import unreleased_reason  # noqa: E402

RELEASED = {"is_public": True, "is_embargoed": False, "is_placeholder": False}


# --------------------------------------------------
def test_released_gets_a_ticket():
    assert unreleased_reason(RELEASED) is None


# --------------------------------------------------
def test_hidden_simulations_refused():
    assert unreleased_reason(None) == "not in md_simulation"
    assert unreleased_reason({**RELEASED, "is_public": False}) == "not approved"
    assert unreleased_reason({**RELEASED, "is_embargoed": True}) == "embargoed"
    assert (
        unreleased_reason({**RELEASED, "is_placeholder": True})
        == "a placeholder"
    )


# --------------------------------------------------
def _run(*argv):
    import subprocess

    script = Path(__file__).resolve().parent.parent / "create_irods_ticket.py"
    return subprocess.run(
        [sys.executable, str(script), *argv], capture_output=True, text=True
    )


# --------------------------------------------------
def test_lifetime_must_outlast_the_renewal_point():
    # A ticket born with 16 days or fewer would be renewed the same night
    # (MDR-115), so the option refuses it before touching the database.
    out = _run("--simulation-id", "1", "-p", "/x", "--lifetime-days", "16")
    assert out.returncode == 2
    assert "--lifetime-days '16' must be more than 16" in out.stderr


# --------------------------------------------------
def test_help_shows_lifetime_default():
    out = _run("--help")
    assert out.returncode == 0
    assert "--lifetime-days" in out.stdout
    assert "(default: 30)" in out.stdout
