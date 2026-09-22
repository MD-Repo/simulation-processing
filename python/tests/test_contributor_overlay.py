"""Tests for overlay_contributor_metadata().

WHY THIS EXISTS. 842 batch-1 bundles are held out as `name_mismatch`: the
tarball's ligand name disagrees with the PDBbind table's, so
fix_ligand_smiles.py:387 refuses to take the table's SMILES. For 278 of
them the contributor's 2026-09-08 delivery carries a name that matches the
table exactly, so overlaying it first lets those pass the name check
HONESTLY rather than through --allow-name-mismatch. That is worth a third
of the population, and it is the one step in the wave-3 plan that edits a
bundle's metadata before anything looks at it.

The cases below are the ways it could go wrong, and two of them are real
traps found while writing it rather than invented for coverage:

  - a [[solutes]] table has a `name` key too, so a walker that matches
    `name = ` across the whole file renames the SOLVENT. That is silent and
    would publish.
  - the delivery carries the same exporter quoting bug as the tarballs
    (6szp, 2026-09-15), so the overlay has to read through an unparseable
    file to get at a name, and must not write one back out.

The .orig assertions matter because this runs BEFORE fix_smiles, so it is
the first thing to touch the file and .orig has to end up holding the
submitter's true original rather than our edit.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest
import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bulk_process_local as B  # noqa: E402


BUNDLE = """\
pdb_id = "3d6q"
lead_contributor_orcid = "0000-0000-0000-0000"
integration_timestep_fs = 2.0
short_description = "a bundle"

[[ligands]]
name = "WRONG NAME"

[[solutes]]
name = "K+"
concentration_mol_liter = 0.15
"""

DELIVERY = """\
pdb_id = "3d6q"

[[ligands]]
name = "RIGHT NAME"

[[solutes]]
name = "Na+"
"""


class FakeArgs:
    """Only the two fields the overlay reads."""

    def __init__(self, toml_fix_dir):
        self.toml_fix_dir = toml_fix_dir


def build(tmp_path, bundle_text=BUNDLE, delivery_text=DELIVERY,
          name="3d6q"):
    """Returns (local_dir, args). Writes both sides to disk."""

    local_dir = tmp_path / "work" / name
    local_dir.mkdir(parents=True)
    (local_dir / B.METADATA_NAME).write_text(bundle_text)

    fix_dir = tmp_path / "toml_fix"
    if delivery_text is not None:
        (fix_dir / name).mkdir(parents=True)
        (fix_dir / name / B.METADATA_NAME).write_text(delivery_text)
    else:
        fix_dir.mkdir(parents=True)

    return str(local_dir), FakeArgs(str(fix_dir))


def ligand_names(path):
    meta = toml.loads(Path(path).read_text())
    return [l.get("name") for l in meta.get("ligands") or []]


def solute_names(path):
    meta = toml.loads(Path(path).read_text())
    return [s.get("name") for s in meta.get("solutes") or []]


# --------------------------------------------------
def test_takes_the_delivery_name(tmp_path):
    local_dir, args = build(tmp_path)
    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    meta_path = os.path.join(local_dir, B.METADATA_NAME)
    assert ligand_names(meta_path) == ["RIGHT NAME"]
    assert changed and "RIGHT NAME" in changed[0]
    assert skipped == []


def test_does_not_rename_the_solvent(tmp_path):
    """[[solutes]] has a `name` key too. This is the trap."""

    local_dir, args = build(tmp_path)
    B.overlay_contributor_metadata(local_dir, "3d6q", args)

    meta_path = os.path.join(local_dir, B.METADATA_NAME)
    assert solute_names(meta_path) == ["K+"], \
        "the solute was renamed from the delivery"


def test_a_blank_delivery_name_never_overwrites(tmp_path):
    """The delivery holds 585 blank ligand names corpus-wide. A blank is
    absence of evidence, not a correction."""

    delivery = DELIVERY.replace('name = "RIGHT NAME"', 'name = ""')
    local_dir, args = build(tmp_path, delivery_text=delivery)
    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    meta_path = os.path.join(local_dir, B.METADATA_NAME)
    assert ligand_names(meta_path) == ["WRONG NAME"]
    assert changed == []
    assert not os.path.exists(meta_path + ".orig"), \
        "nothing changed, so nothing should have been backed up"


def test_no_toml_fix_dir_is_a_no_op(tmp_path):
    local_dir, _ = build(tmp_path)
    changed, skipped = B.overlay_contributor_metadata(
        local_dir, "3d6q", FakeArgs(None))

    assert (changed, skipped) == ([], [])
    assert ligand_names(os.path.join(local_dir, B.METADATA_NAME)) == \
        ["WRONG NAME"]


def test_missing_from_the_delivery_is_reported_not_silent(tmp_path):
    local_dir, args = build(tmp_path, delivery_text=None)
    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    assert changed == []
    assert skipped and "delivery" in skipped[0]


def test_ligand_count_mismatch_refuses(tmp_path):
    """Positional mapping, so a count disagreement would rename the wrong
    ligand. All 278 carry exactly one, but the 550 have not been checked."""

    delivery = DELIVERY + '\n[[ligands]]\nname = "SECOND"\n'
    local_dir, args = build(tmp_path, delivery_text=delivery)
    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    assert changed == []
    assert skipped and "ligand count" in skipped[0]
    assert ligand_names(os.path.join(local_dir, B.METADATA_NAME)) == \
        ["WRONG NAME"]


def test_reads_through_the_exporter_quoting_bug(tmp_path):
    """The delivery carries the same unescaped chemistry prime that broke
    1tsl, 1xa5 and 6szp. Reading it must work; the delivery must not be
    rewritten."""

    delivery = DELIVERY.replace(
        'name = "RIGHT NAME"', 'name = "3\'-3"-DICHLOROPHENOL"')
    local_dir, args = build(tmp_path, delivery_text=delivery)
    before = (Path(str(args.toml_fix_dir)) / "3d6q" /
              B.METADATA_NAME).read_text()

    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    meta_path = os.path.join(local_dir, B.METADATA_NAME)
    assert ligand_names(meta_path) == ['3\'-3"-DICHLOROPHENOL']
    assert changed and skipped == []

    after = (Path(str(args.toml_fix_dir)) / "3d6q" /
             B.METADATA_NAME).read_text()
    assert after == before, "the delivery was modified; it is read-only"


def test_a_quote_in_the_name_is_escaped_so_the_file_still_parses(tmp_path):
    delivery = DELIVERY.replace(
        'name = "RIGHT NAME"', 'name = "A\\"B"')
    local_dir, args = build(tmp_path, delivery_text=delivery)
    B.overlay_contributor_metadata(local_dir, "3d6q", args)

    meta_path = os.path.join(local_dir, B.METADATA_NAME)
    toml.loads(Path(meta_path).read_text())   # must not raise
    assert ligand_names(meta_path) == ['A"B']


def test_backup_holds_the_original_and_is_never_overwritten(tmp_path):
    local_dir, args = build(tmp_path)
    meta_path = os.path.join(local_dir, B.METADATA_NAME)

    B.overlay_contributor_metadata(local_dir, "3d6q", args)
    assert ligand_names(meta_path + ".orig") == ["WRONG NAME"]

    # A later stage's edit must not be captured as the "original".
    shutil.copy2(meta_path, meta_path + ".sentinel")
    B.overlay_contributor_metadata(local_dir, "3d6q", args)
    assert ligand_names(meta_path + ".orig") == ["WRONG NAME"]


def test_is_idempotent(tmp_path):
    local_dir, args = build(tmp_path)
    meta_path = os.path.join(local_dir, B.METADATA_NAME)

    first, _ = B.overlay_contributor_metadata(local_dir, "3d6q", args)
    text_after_first = Path(meta_path).read_text()

    second, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)
    assert first and second == [] and skipped == []
    assert Path(meta_path).read_text() == text_after_first


def test_keeps_provenance_comments_and_key_order(tmp_path):
    """Line-based for this reason: a TOML round trip reorders keys and drops
    fix_ligand_smiles.py's provenance comments, so every line reads as
    changed when one key did."""

    bundle = BUNDLE.replace(
        "[[ligands]]\nname = \"WRONG NAME\"",
        "[[ligands]]\n# smiles added by fix_ligand_smiles.py on 2026-09-01\n"
        "name = \"WRONG NAME\"")
    local_dir, args = build(tmp_path, bundle_text=bundle)
    B.overlay_contributor_metadata(local_dir, "3d6q", args)

    text = Path(os.path.join(local_dir, B.METADATA_NAME)).read_text()
    assert "# smiles added by fix_ligand_smiles.py" in text
    assert text.index("pdb_id") < text.index("integration_timestep_fs")


def test_adds_a_name_when_the_bundle_has_none(tmp_path):
    bundle = BUNDLE.replace('[[ligands]]\nname = "WRONG NAME"',
                            "[[ligands]]")
    local_dir, args = build(tmp_path, bundle_text=bundle)
    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    meta_path = os.path.join(local_dir, B.METADATA_NAME)
    assert ligand_names(meta_path) == ["RIGHT NAME"]
    assert changed and "added" in changed[0]
    assert solute_names(meta_path) == ["K+"]


def test_an_unparseable_bundle_is_left_alone(tmp_path):
    """repair_metadata_quotes() has already run, so this is a shape it could
    not help. Leave it for mdr-process to refuse on its own terms."""

    bundle = BUNDLE + '\nbroken = [unquoted\n'
    local_dir, args = build(tmp_path, bundle_text=bundle)
    changed, skipped = B.overlay_contributor_metadata(local_dir, "3d6q", args)

    assert changed == []
    assert skipped and "will not parse" in skipped[0]
    assert Path(os.path.join(local_dir, B.METADATA_NAME)).read_text() == bundle


# --------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    ("plain", '"plain"'),
    ('has "quote"', '"has \\"quote\\""'),
    ("back\\slash", '"back\\\\slash"'),
    ("3'-prime", '"3\'-prime"'),
])
def test_toml_basic_quoting(value, expected):
    assert B.toml_basic(value) == expected
    assert toml.loads("name = " + B.toml_basic(value))["name"] == value


# --------------------------------------------------
# escape_stray_quotes() was factored out of repair_metadata_quotes() so the
# overlay could reuse it -- the delivery carries the same exporter bug as the
# tarballs. repair_metadata_quotes() had no test of its own, so these cover
# the rule at its new home, using the three shapes the corpus actually
# produced rather than invented ones.

QUOTE_CASES = [
    # 1tsl / 6szp: 3 stray quotes, the chemistry prime inside a quoted value
    ('name = "3\'-3"-DICHLOROPHENOL"\n', 1),
    # a value with nothing wrong is left exactly alone
    ('name = "PLAIN NAME"\n', 0),
]


@pytest.mark.parametrize("text,expected_escapes", QUOTE_CASES)
def test_escape_stray_quotes_counts(text, expected_escapes):
    _fixed, escaped = B.escape_stray_quotes(text)
    assert escaped == expected_escapes


def test_escape_stray_quotes_makes_it_parse_and_keeps_the_name():
    """The repair is lossless: the recovered name is character-for-character
    what was intended. This is 6szp's actual ligand name."""

    want = '(1~{S})-~{N}\'-(4-azanylbutyl)-~{N}"-(2-methoxyethyl)methanetriamine'
    text = f'name = "{want}"\n'
    with pytest.raises(Exception):
        toml.loads(text)

    # One stray quote per line carrying the name. The real 6szp needs 3
    # because its exporter templates that name into `name`,
    # `short_description` and `description` alike -- measured on the
    # delivery copy 2026-09-21, which still carries the bug.
    fixed, escaped = B.escape_stray_quotes(text)
    assert escaped == 1
    assert toml.loads(fixed)["name"] == want


def test_escape_stray_quotes_leaves_triple_quoted_values_alone():
    """Multi-line values are skipped -- the naive rule would corrupt their
    delimiters. This corpus has none, which is why it is safe to skip."""

    text = 'desc = """a "quoted" word"""\n'
    _fixed, escaped = B.escape_stray_quotes(text)
    assert escaped == 0


def test_repair_metadata_quotes_never_touches_a_file_that_parses(tmp_path):
    """A well-formed file is never rewritten, so this cannot damage the
    8,000-odd bundles that are fine."""

    path = tmp_path / B.METADATA_NAME
    path.write_text(BUNDLE)
    assert B.repair_metadata_quotes(str(tmp_path)) == 0
    assert path.read_text() == BUNDLE
    assert not (tmp_path / (B.METADATA_NAME + ".orig")).exists()


def test_repair_metadata_quotes_repairs_and_backs_up(tmp_path):
    broken = BUNDLE.replace('name = "WRONG NAME"', 'name = "3\'-3"-DICHLORO"')
    path = tmp_path / B.METADATA_NAME
    path.write_text(broken)

    escaped = B.repair_metadata_quotes(str(tmp_path))
    assert escaped == 1
    assert toml.loads(path.read_text())["ligands"][0]["name"] == \
        '3\'-3"-DICHLORO'
    assert (tmp_path / (B.METADATA_NAME + ".orig")).read_text() == broken
