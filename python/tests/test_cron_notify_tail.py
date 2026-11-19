"""Tests for cron_notify's Slack tail: tracebacks collapse to their error line"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cron_notify import TAIL_LINES, collapse_tracebacks, message_tail

# What check_new_simulations printed on 2026-11-19 when prod Postgres was full.
CONNECT_ERROR = (
    'psycopg2.OperationalError: connection to server at "mdrepo.org" '
    "(149.165.172.102), port 5432 failed: FATAL:  remaining connection "
    "slots are reserved for non-replication superuser connections"
)
SCANNER_OUTPUT = f"""2026-11-19T11:30:01Z Found 75 unprocessed tickets
Traceback (most recent call last):
  File "/opt/mdrepo/simulation-processing/python/check_new_simulations.py", line 330, in <module>
    main()
    ~~~~^^
  File "/opt/mdrepo/simulation-processing/python/common.py", line 210, in get_db_connection
    conn = psycopg2.connect(dsn)
  File "/opt/mdrepo/.venv/lib/python3.14/site-packages/psycopg2/__init__.py", line 122, in connect
    conn = _connect(dsn, connection_factory=connection_factory, **kwasync)
{CONNECT_ERROR}
"""


def test_traceback_becomes_its_error_line() -> None:
    """Only the error line survives, and the lines around it are kept"""

    assert collapse_tracebacks(SCANNER_OUTPUT) == (
        f"2026-11-19T11:30:01Z Found 75 unprocessed tickets\n{CONNECT_ERROR}"
    )


def test_chained_traceback_keeps_both_errors() -> None:
    """Each traceback in a chain collapses; the joining line stays"""

    text = """Traceback (most recent call last):
  File "a.py", line 1, in <module>
    f()
KeyError: 'x'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "a.py", line 3, in <module>
    g()
ValueError: bad
"""
    assert collapse_tracebacks(text).split("\n") == [
        "KeyError: 'x'",
        "",
        "During handling of the above exception, another exception occurred:",
        "",
        "ValueError: bad",
    ]


def test_output_without_traceback_is_unchanged() -> None:
    """Ordinary output, indented or not, passes through"""

    text = "line one\n  indented detail\nline three"
    assert collapse_tracebacks(text) == text


def test_message_tail_is_short_after_collapse() -> None:
    """Three failed runs of tracebacks fit easily in the tail window"""

    tail = message_tail(SCANNER_OUTPUT * 3)
    assert len(tail.splitlines()) <= TAIL_LINES
    assert tail.splitlines()[-1] == CONNECT_ERROR
    assert "File " not in tail
