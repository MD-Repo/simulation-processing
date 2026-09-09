"""Tests for ligand_check.py — the DDD wave's ligand verdict.

WHY THIS EXISTS. This module decides whether the molecule in a simulation is
the one its metadata claims, for 6,784 bundles, and until now it had no tests
at all. Its bar is a curation policy that was argued out and signed off, not an
implementation detail: block only on a different molecular skeleton, record
protonation and stereochemistry differences rather than treating them as
defects. The measurement behind that -- exact match passes 14.8% of the corpus,
a strict identifier rule flags 3,599 bundles to find 26 real problems -- is what
makes the looser bar correct rather than lazy.

So these tests are as much about the bar as the code. Three of them would fail
if someone tightened it to exact match, and one (`3in3`) would fail if someone
loosened the fragment rescue into a blanket amnesty. That pair of directions is
the point.

The golden cases are real declared/inferred pairs from bundles on disk, with
their verdicts and detail strings measured rather than predicted. Detail
strings are asserted in full deliberately: the 2026-09-04 note records the
detail printing `C7H11NO7P2 vs C7H11NO7P2` -- the same formula on both sides --
for a real finding, and a verdict-only assertion would not have caught it.

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_ligand_check.py -q

`tmp_path` writes under /tmp, which on the ddd host has under 3 GB free on a
96%-full root. Pass `--basetemp` somewhere on /media/volume if you add a test
that writes anything substantial.
"""

import json
import pathlib

import pytest

import compare_smiles
import ligand_check
from ligand_check import BLOCK, FLAG, PASS

GOLDEN = json.loads(
    (pathlib.Path(__file__).parent / "inputs/ligands/declared_vs_inferred.json")
    .read_text()
)["cases"]


@pytest.mark.parametrize("case", GOLDEN, ids=[c["id"] for c in GOLDEN])
def test_golden_verdicts(case):
    """Every regression in the bar, caught at once.

    2lk1, 1a28 or 10gs moving to block means the bar tightened toward exact
    match. 1yq7 or 4dhm moving to block means the fragment-artifact revision
    was lost. 3in3 moving to flag means the rescue over-fired and a genuinely
    wrong molecule started publishing.
    """

    verdict, detail = ligand_check.check([case["declared"]], case["inferred"])

    assert verdict == case["verdict"], case["why"]
    assert detail == case["detail"]


def test_the_toolkit_is_recorded_and_is_the_same_for_both_sides():
    """An InChI is canonical only within one software version, so a comparison
    that took one side from RDKit and the other from OpenBabel would report a
    version difference as a molecular one. 10gs is the case that exercises the
    fallback: its inferred side has a neutral four-bonded nitrogen, which RDKit
    refuses, so the pair drops to OpenBabel wholesale.
    """

    for case in GOLDEN:
        result = compare_smiles.compare(case["declared"], case["inferred"][0])
        expected = case.get("inchi_software", "rdkit")
        assert result["inchi_software"] == expected, case["id"]


def test_no_declared_ligand_is_a_block_not_a_pass():
    """The "nothing to compare, so nothing is wrong" inversion.

    resolve_ligands' else-branch adopts whatever it inferred without checking
    it, and this verdict is the only place that is recorded as a defect.
    """

    assert ligand_check.check([], ["CC"]) == (
        BLOCK,
        "no declared ligand smiles; inference adopted unchecked",
    )


def test_no_inferred_candidates_is_also_a_block():
    """Otherwise every bundle whose structure yielded nothing would look
    verified."""

    assert ligand_check.check(["CC"], []) == (
        BLOCK,
        "no inferred structure smiles to compare against",
    )


def test_an_absent_connectivity_layer_reports_unverifiable_not_a_mismatch():
    """A single atom has no /c layer by construction rather than by failure.

    The verdict is block either way, so nothing has published wrongly, but the
    RECORDED REASON matters -- it is what a contributor report would quote. See
    the known defect noted in TODO.md: for two DIFFERENT monatomic species this
    says "could not check" when the formulas plainly differ and the answer is
    knowable. Asserted here as it behaves today.
    """

    verdict, detail = ligand_check.check(["[Zn+2]"], ["[Zn]"])

    assert verdict == BLOCK
    assert detail == "ligand[0] unverifiable: no inchi connectivity layer"


def test_an_identical_molecule_passes_before_connectivity_is_consulted():
    """compare() reports same_connectivity=False for methane against itself,
    because neither side has a /c layer. check() gets the right answer only
    because it tests exact_match first. Anything reading same_connectivity
    without inchi_available inherits the trap.
    """

    assert ligand_check.check(["C"], ["C"]) == (PASS, "")

    result = compare_smiles.compare("C", "C")
    assert result["exact_match"] is True
    assert result["same_connectivity"] is False
    assert result["inchi_available"] is False


def test_best_comparison_prefers_a_real_match_over_a_fragment_artifact():
    """A bundle can infer several ligands and only one is the right partner.
    Taking the first would attach a plausible-looking verdict to the wrong
    molecule, silently.
    """

    candidates = ["CCC", "CCO.[HH]", "CCO"]

    for ordering in (candidates, list(reversed(candidates))):
        best = ligand_check.best_comparison("CCO", ordering)
        assert best["exact_match"] is True, ordering


def test_worst_takes_the_most_severe_and_an_empty_bundle_passes():
    assert ligand_check.worst([]) == PASS
    assert ligand_check.worst(["pass", "block", "flag"]) == BLOCK
    assert ligand_check.worst(["flag", "pass"]) == FLAG


def test_a_valence_error_our_stack_cannot_see_still_passes():
    """`C[NH]5CCOCC5` is a morpholine nitrogen with four bonds and no charge --
    a molecule that cannot exist. It is the published SMILES of 2w1c and 2w1e.

    OpenBabel accepts it, and libmdrepo's is_valid_smiles accepts it too, since
    purr does not check valence and says so in its own comment. RDKit refuses
    it. Asserted as it behaves today so the blast radius of adopting RDKit for
    parsing -- as opposed to for InChI, which is already done -- is written
    down: 2 declared ligands in 6,664, and 1,963 of 6,659 inferred ones.
    """

    from rdkit import Chem

    bad = "C[NH]5CCOCC5"

    assert ligand_check.check([bad], [bad]) == (PASS, "")
    assert Chem.MolFromSmiles(bad) is None


ELEMENT_COLUMN_PDB = """\
ATOM      1  CL6 LIG A   1      11.104   6.134   7.298  1.00  0.00           C
ATOM      2  BR1 LIG A   1      12.104   6.134   7.298  1.00  0.00           B
ATOM      3  SE2 LIG A   1      13.104   6.134   7.298  1.00  0.00           S
ATOM      4  CA  ALA A   2      14.104   6.134   7.298  1.00  0.00           C
HETATM    5  O   HOH A   3      15.104   6.134   7.298  1.00  0.00           O
END
"""


@pytest.fixture
def pdb_file(tmp_path):
    """Write a PDB into pytest's temp dir and hand back its path."""

    def write(text: str, name: str = "ligand.pdb") -> pathlib.Path:
        path = tmp_path / name
        path.write_text(text)
        return path

    return write


def test_element_column_disagreements_reads_atom_records_not_only_hetatm(pdb_file):
    """The DDD contributor's structures put the ligand in ATOM records, and a
    HETATM-only scan examined nothing and pronounced every bundle good.

    This is the 1,025-package finding: chlorine written as C, bromine as B.
    Synthesized rather than committed, because the function reads text lines
    and the real files are 800 KB.
    """

    path = pdb_file(ELEMENT_COLUMN_PDB)
    bad = ligand_check.element_column_disagreements(str(path))

    assert sorted(bad) == ["BR1:B!=BR", "CL6:C!=CL", "SE2:S!=SE"]
    assert not any(item.startswith("CA") for item in bad), (
        "an alpha-carbon in a protein residue is not a ligand element error"
    )


def test_write_element_corrected_never_touches_the_source(pdb_file):
    """This function's docstring promises it writes to a new path. In-place
    rewriting is the bug that broke the manifest check on every ticket with
    ligands, recorded in test_canonicalize_smiles.py's own docstring, and
    nothing enforced the promise.
    """

    src = pdb_file(ELEMENT_COLUMN_PDB)
    dst = src.with_name("corrected.pdb")
    before = src.read_bytes()

    fixed = ligand_check.write_element_corrected(str(src), str(dst))

    assert src.read_bytes() == before, "the source must be untouched"
    assert sorted(fixed) == ["BR1:B->BR", "CL6:C->CL", "SE2:S->SE"]

    corrected = dst.read_text().splitlines()
    assert corrected[0].endswith("CL")
    assert corrected[1].endswith("BR")
    assert corrected[2].endswith("SE")
    # Everything but the element column is carried through byte for byte.
    assert [line[:76] for line in corrected[:3]] == [
        line[:76] for line in ELEMENT_COLUMN_PDB.splitlines()[:3]
    ]
