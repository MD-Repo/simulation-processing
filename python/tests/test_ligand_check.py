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


def atom(serial, name, resname, resseq, x, y, z, element, record="ATOM  "):
    """One fixed-column PDB atom line (element in columns 77-78)."""

    return (
        f"{record}{serial:5d} {name:<4} {resname:>3} A{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )


# Real bond lengths: C-Cl 1.75 A, C-Br 1.94 A, Se-C 1.95 A, C-H 1.09 A.
# Each name-implied element sits in its own residue with its one carbon.
ELEMENT_COLUMN_PDB = "\n".join([
    atom(1, "CL6", "LIG", 1, 0.0, 0.0, 0.0, "C"),
    atom(2, "BR1", "LIG", 2, 10.0, 0.0, 0.0, "B"),
    atom(3, "SE2", "LIG", 3, 20.0, 0.0, 0.0, "S"),
    atom(4, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C"),
    atom(5, "C2", "LIG", 2, 11.94, 0.0, 0.0, "C"),
    atom(6, "C3", "LIG", 3, 21.95, 0.0, 0.0, "C"),
    atom(7, "C4", "LIG", 3, 18.05, 0.0, 0.0, "C"),
    atom(8, "CA", "ALA", 4, 30.0, 0.0, 0.0, "C"),
    atom(9, "O", "HOH", 5, 40.0, 0.0, 0.0, "O", "HETATM"),
    "END",
]) + "\n"


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


# --------------------------------------------------
# 2026-11-16: a carbon merely NAMED CL... or CU... stays carbon.
# Some DDD bundles name a ligand's carbons alphabetically (CA, CB, ... CL,
# ... CU). Trusting the name alone made 18 of 50 corrections in the 499
# structure-wins bundles wrong (2pcp, 4gh6, 3oc0, ...).


def corrections(*lines):
    """element_corrections over these lines, as {atom name: (col, implied)}"""

    lines = [line + "\n" for line in lines]
    return {
        lines[i][12:16].strip(): v
        for i, v in ligand_check.element_corrections(lines).items()
    }


def test_a_real_chlorine_one_carbon_no_hydrogen_is_corrected():
    assert corrections(
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C"),
        atom(3, "H1", "LIG", 1, 2.3, 0.9, 0.0, "H"),  # on C1, 2.4 A from Cl
    ) == {"CL1": ("C", "CL")}


def test_a_real_bromine_is_corrected():
    assert corrections(
        atom(1, "BR", "LIG", 1, 0.0, 0.0, 0.0, "B"),
        atom(2, "C7", "LIG", 1, 0.0, 1.94, 0.0, "C"),
    ) == {"BR": ("B", "BR")}


def test_a_methyl_carbon_named_cl1_stays_carbon():
    """4gh6, 3l9m, 3mg7...: CL1 is a methyl, three H at 1.09 A."""

    assert corrections(
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C2", "LIG", 1, 1.53, 0.0, 0.0, "C"),
        atom(3, "H1", "LIG", 1, -0.36, 1.03, 0.0, "H"),
        atom(4, "H2", "LIG", 1, -0.36, -0.51, 0.89, "H"),
        atom(5, "H3", "LIG", 1, -0.36, -0.51, -0.89, "H"),
    ) == {}


def test_a_ring_carbon_named_cl_stays_carbon():
    """2pcp: phencyclidine's ring carbon CL, two C neighbours and two H."""

    assert corrections(
        atom(1, "CL", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "CK", "LIG", 1, 1.53, 0.0, 0.0, "C"),
        atom(3, "CG", "LIG", 1, -0.5, 1.45, 0.0, "C"),
        atom(4, "H24", "LIG", 1, -0.4, -0.5, 0.9, "H"),
        atom(5, "H25", "LIG", 1, -0.4, -0.5, -0.9, "H"),
    ) == {}


def test_an_aromatic_carbon_named_cl_with_no_hydrogen_stays_carbon():
    """3gur: CL has no H but three heavy neighbours at ~1.4 A."""

    assert corrections(
        atom(1, "CL", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "CK", "LIG", 1, 1.40, 0.0, 0.0, "C"),
        atom(3, "CM", "LIG", 1, -0.70, 1.21, 0.0, "C"),
        atom(4, "N1", "LIG", 1, -0.70, -1.21, 0.0, "N"),
    ) == {}


def test_a_carbon_named_cu_stays_carbon_and_a_real_copper_is_corrected():
    """3oc0: CU is a CH carbon. A copper sits >= 1.9 A from its ligands."""

    assert corrections(
        atom(1, "CU", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "CT", "LIG", 1, 1.52, 0.0, 0.0, "C"),
        atom(3, "H9", "LIG", 1, -0.4, 1.0, 0.0, "H"),
    ) == {}
    assert corrections(
        atom(1, "CU1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "N1", "LIG", 1, 2.0, 0.0, 0.0, "N"),
        atom(3, "N2", "LIG", 1, -2.0, 0.0, 0.0, "N"),
        atom(4, "N3", "LIG", 1, 0.0, 2.0, 0.0, "N"),
    ) == {"CU1": ("C", "CU")}


def test_a_metal_named_atom_with_a_short_heavy_bond_stays():
    assert corrections(
        atom(1, "ZN", "LIG", 1, 0.0, 0.0, 0.0, "Z"),
        atom(2, "N1", "LIG", 1, 1.47, 0.0, 0.0, "N"),
    ) == {}


def test_a_selenium_between_two_carbons_is_corrected():
    assert corrections(
        atom(1, "SE1", "LIG", 1, 0.0, 0.0, 0.0, "S"),
        atom(2, "C1", "LIG", 1, 1.95, 0.0, 0.0, "C"),
        atom(3, "C2", "LIG", 1, -1.95, 0.0, 0.0, "C"),
    ) == {"SE1": ("S", "SE")}


def test_two_chlorines_on_one_carbon_are_both_corrected():
    """Each Cl has one heavy neighbour; the other Cl is 2.9 A away."""

    assert corrections(
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "CL2", "LIG", 1, 2.9, 0.0, 0.0, "C"),
        atom(3, "C1", "LIG", 1, 1.45, 0.95, 0.0, "C"),
    ) == {"CL1": ("C", "CL"), "CL2": ("C", "CL")}


def test_hydrogen_is_recognised_by_name_when_the_column_is_empty():
    assert corrections(
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C2", "LIG", 1, 1.53, 0.0, 0.0, "C"),
        atom(3, "H1", "LIG", 1, -0.36, 1.03, 0.0, ""),
    ) == {}


def test_neighbours_count_only_within_the_same_residue():
    """A protein carbon 1.5 A away (a clash) does not make a Cl a carbon."""

    assert corrections(
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C"),
        atom(3, "CB", "ALA", 2, -1.5, 0.0, 0.0, "C"),
    ) == {"CL1": ("C", "CL")}


def test_every_model_is_corrected_with_its_own_neighbours():
    lines = [
        "MODEL        1",
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C"),
        "ENDMDL",
        "MODEL        2",
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C"),
        "ENDMDL",
    ]
    fixes = ligand_check.element_corrections([x + "\n" for x in lines])
    assert sorted(fixes) == [1, 5]


def test_unreadable_coordinates_are_not_corrected():
    line = atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C")
    line = line[:30] + "   xxxxx" + line[38:]
    assert corrections(line, atom(2, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C")) == {}


def test_disagreements_and_the_corrected_file_agree(pdb_file):
    """The preflight flag and the correction now make the same decision."""

    text = "\n".join([
        atom(1, "CL1", "LIG", 1, 0.0, 0.0, 0.0, "C"),
        atom(2, "C1", "LIG", 1, 1.75, 0.0, 0.0, "C"),
        atom(3, "CL", "LIG", 2, 10.0, 0.0, 0.0, "C"),
        atom(4, "CK", "LIG", 2, 11.53, 0.0, 0.0, "C"),
        atom(5, "CM", "LIG", 2, 9.5, 1.45, 0.0, "C"),
    ]) + "\n"
    src = pdb_file(text)
    dst = src.with_name("corrected.pdb")

    assert ligand_check.element_column_disagreements(str(src)) == ["CL1:C!=CL"]
    assert ligand_check.write_element_corrected(str(src), str(dst)) == [
        "CL1:C->CL"
    ]
    out = dst.read_text().splitlines()
    assert out[0].endswith("CL") and out[2].endswith(" C")


# --------------------------------------------------
# Real bundles (the 2026-11-16 survey copies on the ddd host); skipped
# elsewhere.
SURVEY = pathlib.Path("/mdrepotmp/ddd/work-structure-wins/survey-work")


@pytest.mark.skipif(not SURVEY.is_dir(), reason="ddd host survey copies")
@pytest.mark.parametrize(
    "bundle,expected",
    [
        ("2pcp", []),  # ring carbon CL
        ("4gh6", []),  # methyl CL1
        ("3oc0", []),  # CH carbons CU and CL
        ("1bxq", []),  # methyls CL1, CL2
        ("3gur", []),  # aromatic CL, no H
    ],
)
def test_real_carbons_named_cl_or_cu_are_left_alone(bundle, expected):
    pdb = SURVEY / bundle / "Pro_lig.pdb"
    assert ligand_check.element_column_disagreements(str(pdb)) == expected


@pytest.mark.skipif(not SURVEY.is_dir(), reason="ddd host survey copies")
def test_real_chlorines_and_bromines_are_all_still_corrected():
    """The 32 corrections in the 499 that ARE halogens (no H, one heavy
    neighbour; elemfix-check/corrections.tsv) are all still made, and the
    18 carbons none."""

    table = SURVEY.parent / "elemfix-check" / "corrections.tsv"
    if not table.is_file():
        pytest.skip("elemfix-check/corrections.tsv not present")
    real, carbon = set(), set()
    for row in table.read_text().splitlines()[1:]:
        f = row.split("\t")
        key = (f[0], f[1].upper())
        (carbon if int(f[4]) or int(f[5]) > 1 else real).add(key)
    assert len(real) == 32 and len(carbon) == 18

    found = set()
    for bundle in {b for b, _ in real | carbon}:
        for item in ligand_check.element_column_disagreements(
            str(SURVEY / bundle / "Pro_lig.pdb")
        ):
            found.add((bundle, item.split(":")[0]))
    assert found == real
