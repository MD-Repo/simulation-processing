"""Tests for verify_simulation_files.py's row selection.

Only `load_rows` is covered, and deliberately so: it is the one place the
sweep decides what it will look at, and a filter that quietly selects nothing
is the failure mode this codebase keeps re-learning. `--exclude-filename`
exists so one diagnosed, benign class (the ~21,300 `mdrepo-metadata.toml`
rows left describing the pre-migration file) cannot bury real damage. If it
ever over-matched, the sweep would come back clean while skipping the rows it
was run to check -- the same shape as the HETATM-only element check that
examined nothing and reported every bundle good.

No database: the cursor is a stub that records the SQL and parameters, so
these pin the query the tool builds rather than what Postgres does with it.
The end-to-end behaviour was checked against prod on 2026-09-04 over
simulations 20000-20050: 969 rows unfiltered, 918 with the TOML excluded,
exactly the 51 metadata rows removed and nothing else.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import verify_simulation_files as V  # noqa: E402


class FakeCursor:
    """Records what would have been executed; returns no rows."""

    def __init__(self):
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((" ".join(sql.split()), params))

    def fetchall(self):
        return []


def make_args(**over):
    base = dict(
        server="prod",
        simulation_ids=[1, 2],
        id_range=None,
        collection=None,
        record=None,
        placeholder_only=False,
        exclude_placeholder=False,
        exclude_filenames=(),
        tables=("md_uploaded_file",),
        threads=8,
        deep=False,
        out=None,
        manifest_out=None,
    )
    base.update(over)
    return V.Args(**base)


def run(**over):
    cur = FakeCursor()
    V.load_rows(cur, make_args(**over), [1, 2])
    return cur.calls


def test_no_exclusion_adds_no_clause_and_no_parameter():
    """The default path must be exactly what it was before the flag existed."""

    (sql, params), = run()
    assert "filename <> all" not in sql
    assert params == ([1, 2],)


def test_exclusion_filters_in_sql_not_in_python():
    """Carrying tens of thousands of excluded rows into Python is the bug the
    SQL clause avoids, so pin that the clause is actually emitted."""

    (sql, params), = run(exclude_filenames=("mdrepo-metadata.toml",))
    assert "filename <> all(%s)" in sql
    assert params == ([1, 2], ["mdrepo-metadata.toml"])


def test_exclusion_is_by_exact_name_not_a_pattern():
    """`<> all` is an equality test. A regression to LIKE would also drop
    mdrepo-metadata.v1.toml and anything else sharing the stem, silently
    widening the blind spot."""

    (sql, _), = run(exclude_filenames=("mdrepo-metadata.toml",))
    assert "like" not in sql.lower()


def test_several_names_are_all_passed_through():
    (_, params), = run(exclude_filenames=("a.toml", "b.txt"))
    assert params[1] == ["a.toml", "b.txt"]


def test_the_clause_is_applied_to_every_table():
    """The sweep's whole point is covering both file tables; an exclusion that
    reached only the first would leave the other's rows in the results and
    look like a partial repair."""

    calls = run(
        tables=("md_uploaded_file", "md_processed_file"),
        exclude_filenames=("mdrepo-metadata.toml",),
    )
    assert len(calls) == 2
    assert all("filename <> all(%s)" in sql for sql, _ in calls)
    assert {"md_uploaded_file", "md_processed_file"} == {
        t for sql, _ in calls for t in ("md_uploaded_file", "md_processed_file")
        if f"from {t}" in sql
    }
