"""Tests for export_mapping_file.py's InChIKey mapping

WHAT THESE GUARD, and why the guards are worth having.

**The URL shape.** All three carry a "view" segment. Getting this wrong is
uniquely hard to notice: mdrepo.org is a single-page app and answers **HTTP
200 for every path**, including nonsense, so a wrong URL publishes ~18,500
links that satisfy any checker and render Not Found to a person. The authority
is the Elm page layout, and for what is actually live, the deployed bundle --
which encodes this route as `"inchikey","view"`, exactly as it does
`"uniprot","view"`. Both checked 2026-09-09.

This one moved under us: the route shipped that morning as `/inchikey/:id` and
was renamed to `/inchikey/view/:id` the same day (elm-mdrepo `88bf2c6`), while
this exporter was being written against the first spelling. Re-check it rather
than trusting either this docstring or an old note.

**Only visible simulations.** The mapping files go to PDB, UniProt and now
MDDB, so a leak here is a leak to a third party. `VISIBLE_SIM` is the same
four-condition predicate the Django app uses, and `is_public` alone is not it:
embargo outlives approval.

**No dedup table.** `md_pdb` and `md_uniprot` are one row per identifier with a
join table, which is what lets those two queries keep a row with an empty
`simulations` cell for an identifier nothing visible uses. `md_ligand` carries
`simulation_id` directly, so the InChIKey query groups instead, and such a row
cannot exist. That is a real difference between the three files and someone
diffing them should find it stated rather than infer it.

These are string and predicate tests -- no database and no network. The live
agreement between this export and the site's own `/api/v1/inchikey/<key>` count
was checked separately at the time it was written: 12 of 12 sampled keys, plus
ammonia correctly absent because its only two simulations are not yet public.
"""

import re
import sys
from pathlib import Path

import pytest

PY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_DIR))

pytest.importorskip("psycopg2")
pytest.importorskip("fabric")

import export_mapping_file as emf  # noqa: E402


# --------------------------------------------------
def test_inchikey_is_an_accepted_mapping():
    """The CLI's third choice, alongside uniprot and pdb"""

    source = (PY_DIR / "export_mapping_file.py").read_text()
    assert 'choices=["uniprot", "pdb", "inchikey"]' in source


# --------------------------------------------------
def test_inchikey_url_has_the_view_segment():
    """THE ONE THAT MATTERS, and it changed once already.

    An SPA answers 200 for any path, so no status check catches a wrong URL --
    the links fail only for a person who clicks one. Renamed from
    `/inchikey/:id` on 2026-09-09 (elm-mdrepo 88bf2c6).
    """

    source = (PY_DIR / "export_mapping_file.py").read_text()
    url = "'https://mdrepo.org/inchikey/view/' || l.inchikey"

    assert url in source
    # and not the pre-rename spelling, which would now be a dead link
    assert "'https://mdrepo.org/inchikey/' || l.inchikey" not in source


# --------------------------------------------------
def test_all_three_urls_use_the_same_shape():
    """Since the 88bf2c6 rename the three agree; keep them that way"""

    source = (PY_DIR / "export_mapping_file.py").read_text()

    assert "'https://mdrepo.org/uniprot/view/' || u.uniprot_id" in source
    assert "'https://mdrepo.org/pdb/view/' || p.pdb_id" in source


# --------------------------------------------------
def test_visible_sim_requires_all_four_conditions():
    """is_public alone is not "publicly visible" -- embargo outlives approval

    These files are ingested by third parties, so this predicate is the whole
    thing standing between an embargoed simulation and PDB, UniProt or MDDB.
    """

    predicate = " ".join(emf.VISIBLE_SIM.split())

    assert predicate == (
        "s.is_public and not s.is_deprecated and not s.is_placeholder "
        "and not s.is_embargoed"
    )


# --------------------------------------------------
def test_inchikey_query_filters_on_visibility():
    """The predicate is interpolated into the InChIKey branch too

    Easy to add a third query and forget it; the leak would be silent.
    """

    source = (PY_DIR / "export_mapping_file.py").read_text()
    branch = source[source.index('elif args.mapping == "inchikey"'):
                    source.index("fieldnames = [\"inchikey\"")]

    assert "{VISIBLE_SIM}" in branch
    assert "join md_simulation s on s.id = l.simulation_id" in branch


# --------------------------------------------------
def test_inchikey_query_excludes_unkeyed_ligands():
    """32 rows had no computable key as of 2026-09-09

    They have no identifier to publish. Without this they would group into a
    single NULL row carrying every one of their simulations.
    """

    source = (PY_DIR / "export_mapping_file.py").read_text()
    assert "l.inchikey is not null" in source


# --------------------------------------------------
def test_inchikey_query_deduplicates_slugs_within_a_cell():
    """One simulation can carry the same molecule as two ligand rows

    Without DISTINCT its slug appears twice in that key's cell.
    """

    source = (PY_DIR / "export_mapping_file.py").read_text()
    assert re.search(r"string_agg\(\s*distinct", source)


# --------------------------------------------------
def test_all_three_mappings_share_the_same_three_columns():
    """MDDB, PDB and UniProt read the same shape from all three files"""

    source = (PY_DIR / "export_mapping_file.py").read_text()

    for expected in ('fieldnames = ["uniprot_id", "url", "simulations"]',
                     'fieldnames = ["pdb_id", "url", "simulations"]',
                     'fieldnames = ["inchikey", "url", "simulations"]'):
        assert expected in source, expected
