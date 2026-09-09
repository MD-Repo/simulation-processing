"""Tests for resolve_ligand_identity.py

WHY THIS EXISTS. `mdr-meta` accepts a ligand declared by `smiles`, by `inchi`,
or by both, but `md_ligand.smiles` is NOT NULL -- so until this script existed
an InChI-only ligand passed `mdr-meta check` and then failed at import, and
`inchikey` and `identity_software` were columns nothing wrote.

The pair of properties worth pinning hardest, because both fail silently:

  - a submitted value is NEVER rewritten. The submitter's own smiles is
    canonicalized (as it always was), but their inchi is stored verbatim.
  - a MALFORMED InChI must not acquire a well-formed key. Chem.InchiToInchiKey
    is a hash with no validation behind it and will happily key `InChI=1S/NOPE`.

Real molecules throughout, with values measured against RDKit 2026.03.3 rather
than predicted. Verified 2026-09-09 to reproduce the production backfill
exactly over 63 rows -- identical inchi, inchikey and identity_software -- so
a reprocess does not change what is already stored.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PY_DIR = HERE.parent
sys.path.insert(0, str(PY_DIR))

pytest.importorskip("rdkit")
pytest.importorskip("openbabel")

import resolve_ligand_identity as rli  # noqa: E402

# Acetaldehyde, the smallest molecule that still has a connectivity layer.
SMILES = "CC=O"
INCHI = "InChI=1S/C2H4O/c1-2-3/h2H,1H3"
KEY = "IKHGUXGNUITLKF-UHFFFAOYSA-N"

# Obeticholic acid, from simulation 98331. Its InChI round-trips to a DIFFERENT
# SMILES string than the submitter wrote -- different atom ordering, different
# ring-closure numbers -- while its InChIKey is unchanged. That is the case the
# derived-SMILES test is about.
OCA_INCHI = (
    "InChI=1S/C26H44O4/c1-5-17-21-14-16(27)10-12-26(21,4)20-11-13-25(3)18(15(2)6-9-22(28)29)7-8-19(25)23("
    "20)24(17)30/h15-21,23-24,27,30H,5-14H2,1-4H3,(H,28,29)/t15-,16-,17-,18-,19+,20+,21+,23+,24-,25-,26-/m1/s1"
)
OCA_SUBMITTED_SMILES = (
    "CC[C@H]1[C@@H](O)[C@H]2[C@@H]3CC[C@@H]([C@@]3(C)CC[C@@H]2[C@@]2([C@H]1"
    "C[C@H](O)CC2)C)[C@@H](CCC(=O)O)C"
)


# --------------------------------------------------
def test_smiles_only_derives_inchi_and_key():
    """The ordinary case: everything but the SMILES is ours"""

    got = rli.resolve({"name": "acetaldehyde", "smiles": SMILES, "inchi": None})

    assert got["smiles"] == SMILES
    assert got["inchi"] == INCHI
    assert got["inchikey"] == KEY
    assert got["identity_software"] == "rdkit"


# --------------------------------------------------
def test_inchi_only_derives_smiles():
    """The case that used to fail at import for want of a NOT NULL column"""

    got = rli.resolve({"name": "acetaldehyde", "smiles": None, "inchi": INCHI})

    assert got["smiles"] == SMILES
    assert got["inchi"] == INCHI
    assert got["inchikey"] == KEY
    assert got["identity_software"] == "rdkit"


# --------------------------------------------------
def test_inchi_only_derived_smiles_need_not_match_the_submitters():
    """SMILES has no canonical form; InChI does. The KEY is what must agree.

    Asserting the derived string equals the submitted one would be asserting
    something false, and would break the moment RDKit changed its output
    ordering. What must hold is that the molecule is the same.
    """

    got = rli.resolve({"name": "obeticholic acid", "smiles": None,
                       "inchi": OCA_INCHI})

    assert got["smiles"] != OCA_SUBMITTED_SMILES
    assert got["inchikey"] == "ZXERDUOLZKYMJM-ZWECCWDJSA-N"
    assert got["inchi"] == OCA_INCHI  # verbatim, not recomputed


# --------------------------------------------------
def test_both_declared_keeps_both_and_derives_nothing():
    """identity_software names the toolkit behind a value WE produced

    When the submitter supplies both, we produce neither, so it stays null.
    A row reading "rdkit" here would claim provenance we do not have.
    """

    got = rli.resolve({"name": "acetaldehyde", "smiles": SMILES,
                       "inchi": INCHI})

    assert got["smiles"] == SMILES
    assert got["inchi"] == INCHI
    assert got["inchikey"] == KEY
    assert got["identity_software"] is None


# --------------------------------------------------
def test_both_declared_and_disagreeing_is_refused():
    """Two notations for two different molecules is a person contradicting
    themselves inside one file, and it stops the ligand"""

    with pytest.raises(SystemExit) as err:
        rli.resolve({"name": "wrong", "smiles": "CCO", "inchi": INCHI})

    assert "different molecules" in str(err.value)


# --------------------------------------------------
def test_malformed_inchi_gets_no_key():
    """Chem.InchiToInchiKey will hash anything; MolFromInchi is the gate

    Without that gate `InChI=1S/NOPE` acquires DIAJDFQKHMMVGV-UHFFFAOYSA-N and
    every consumer downstream reads it as a real identifier.
    """

    assert rli.inchikey_of("InChI=1S/NOPE") is None
    assert rli.inchikey_of(INCHI) == KEY


# --------------------------------------------------
def test_unreadable_inchi_alone_is_fatal():
    """It is the only structure given, so nothing can fill the NOT NULL column"""

    with pytest.raises(SystemExit) as err:
        rli.resolve({"name": "bad", "smiles": None, "inchi": "InChI=1S/NOPE"})

    assert "cannot read" in str(err.value)


# --------------------------------------------------
def test_invalid_smiles_is_fatal():
    """Unchanged from canonicalize_smiles.py, which this replaces"""

    with pytest.raises(SystemExit) as err:
        rli.resolve({"name": "bad", "smiles": "not a smiles", "inchi": None})

    assert "invalid smiles" in str(err.value).lower()


# --------------------------------------------------
def test_neither_notation_is_fatal():
    """libmdrepo rejects this first; named here so a change there is visible"""

    with pytest.raises(SystemExit) as err:
        rli.resolve({"name": "empty", "smiles": None, "inchi": None})

    assert "neither" in str(err.value)


# --------------------------------------------------
def test_blank_strings_count_as_absent():
    """An empty smiles is not a smiles; it must take the inchi path"""

    got = rli.resolve({"name": "acetaldehyde", "smiles": "  ", "inchi": INCHI})

    assert got["smiles"] == SMILES
    assert got["identity_software"] == "rdkit"


# --------------------------------------------------
def test_an_unkeyable_smiles_is_stored_rather_than_refused():
    """RDKit's valence rules must not turn a column gap into a failed upload

    RDKit refused 32 of the 121,240 rows backfilled on 2026-09-09, about half
    of them valence cases OpenBabel accepts. `[NH2]` on a neutral nitrogen with
    two more bonds is that shape. OpenBabel parses it, so the ligand is
    storable, and it is stored.
    """

    got = rli.resolve({"name": "unknown",
                       "smiles": "O=C[C@@H]1CCC(=O)[NH2]1", "inchi": None})

    assert got["smiles"]  # the NOT NULL column is filled
    assert got["inchikey"] is None
    assert got["identity_software"] is None


# --------------------------------------------------
def test_only_an_rdkit_inchi_is_stored():
    """A stored InChI must be single-version, unlike a compared one

    compare_smiles falls back to OpenBabel 1.04 deliberately, because a
    comparison puts both sides through the same library and the version
    cancels. A stored value has no other side: it is matched against whatever
    an outside database computed, and InChI is canonical only per version.

    This molecule is one OpenBabel will build an InChI for and RDKit will not
    -- a morpholine nitrogen carrying four bonds and no charge, the shape that
    accounts for the declared-side refusals. Allowing the fallback here
    recovered 5 rows in 100 and made identity_software bi-toolkit, and made a
    reprocess disagree with the production backfill. Storing nothing is right.
    """

    smiles = "Fc1ccc(cc1)C(=O)Nc2c[nH]nc2c3[nH]c4cc(C[NH]5CCOCC5)ccc4n3"
    got = rli.resolve({"name": "L0C", "smiles": smiles, "inchi": None})

    assert got["smiles"]
    assert got["inchi"] is None
    assert got["inchikey"] is None
    assert got["identity_software"] is None


# --------------------------------------------------
def test_results_come_back_in_input_order():
    """The Rust caller matches results to ligands BY POSITION"""

    payload = [
        {"name": "a", "smiles": "N", "inchi": None},
        {"name": "b", "smiles": None, "inchi": INCHI},
        {"name": "c", "smiles": "O", "inchi": None},
    ]
    proc = subprocess.run(
        [sys.executable, str(PY_DIR / "resolve_ligand_identity.py")],
        input=json.dumps(payload), capture_output=True, text=True,
        cwd=str(PY_DIR),
    )

    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert [g["smiles"] for g in got] == ["N", SMILES, "O"]
