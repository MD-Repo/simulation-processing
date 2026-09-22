"""Tests for common.describe_exc

WHY THIS EXISTS. python-irodsclient decodes the error text the server sends
with a failing status in connection.recv(). When that decode raises TypeError
it swallows it, sets the message to None, and builds the exception as
exc_class(None) -- and str() of that object is the literal string "None". A
handler that logs only str(e) therefore records the word "None" for a real
server error and loses the class and code, which are the only things that name
it.

That has cost three diagnoses (2026-09-05, 2026-09-15, 2026-09-22), and the
helper was independently written three times before it was shared. The case
below that asserts str(e) == "None" is the trap itself, pinned so a future
client release that starts decoding those messages shows up here as a failure
rather than as a silent change in what the logs mean.

Pure function -- no IRODS, no network, no database.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import describe_exc  # noqa: E402


# --------------------------------------------------
def test_ordinary_message_is_kept_and_labelled():
    """Anything the scripts raise themselves already says enough

    The class name is still prepended. It costs one word and it is the
    difference between "the put failed" and "the put timed out".
    """

    e = RuntimeError("gocmd put of mdrepo.22.sql.gz exited 1: no output")

    assert describe_exc(e) == (
        "RuntimeError: gocmd put of mdrepo.22.sql.gz exited 1: no output"
    )


# --------------------------------------------------
def test_messageless_irods_error_still_names_itself():
    """The 2026-09-22 case: a real server error that stringifies as "None\""""

    from irods.exception import get_exception_by_code

    e = get_exception_by_code(-816000, None)

    assert str(e) == "None"  # the trap this helper exists for
    assert describe_exc(e) == "CAT_INVALID_ARGUMENT(-816000)"


# --------------------------------------------------
def test_locked_data_object_is_the_2026_09_15_case():
    """The failure an admin had to identify by hand, now in one line"""

    from irods.exception import get_exception_by_code

    e = get_exception_by_code(-406000, None)

    assert describe_exc(e) == "LOCKED_DATA_OBJECT_ACCESS(-406000)"


# --------------------------------------------------
def test_decodable_irods_error_keeps_both():
    """A server error WITH text keeps its message as well as its identity"""

    from irods.exception import get_exception_by_code

    e = get_exception_by_code(-818000, "user mdadm lacks write access")

    assert describe_exc(e) == (
        "CAT_NO_ACCESS_PERMISSION(-818000): user mdadm lacks write access"
    )


# --------------------------------------------------
def test_bare_exception_has_no_message_at_all():
    """An exception raised with no arguments has an empty str(), not the word"""

    class Boom(Exception):
        pass

    assert describe_exc(Boom()) == "Boom"


# --------------------------------------------------
def test_needs_no_irods_import():
    """It reads .code with getattr, so a non-IRODS caller can use it

    drain_process_queue.py deliberately avoids importing python-irodsclient at
    module scope so a missing client library cannot stop the drain from
    starting. Anything it shares from common has to hold to the same rule.
    """

    import ast

    source = Path(__file__).resolve().parent.parent / "common.py"
    tree = ast.parse(source.read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "irods" not in imported
