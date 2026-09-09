"""Tests for ligand_agreement.py — a declared smiles and a declared inchi that
must describe the same molecule.

WHY THIS EXISTS. `cba350b` made a ligand's structure optional in either
notation, requiring at least one. Nothing yet checks the case where a submitter
supplies both, and that is the one case where the archive can catch a
contradiction using only what it was given -- no coordinates, no reference
table, no perception. It is also the only ligand check whose two sides come
from the same person, which is why its bar is stricter than the connectivity
bar used everywhere else. See ligand_agreement's module docstring.

The rows below are the agreement rule as designed, one test per row, plus the
two tests that pin the interaction the written rule left ambiguous.

Every InChI string here was generated on this host rather than typed, so a
toolkit change makes these fail loudly rather than silently pass.

Run from `simulation-processing/python`, not from the repo root:

    ./.venv/bin/python -m pytest tests/test_ligand_agreement.py -q
"""

import pytest

import ligand_agreement as la
from ligand_agreement import ACCEPT, STOP, UNVERIFIABLE

# Real InChIs, computed here 2026-09-09 with RDKit (InChI 1.07.3).
PROPANE = "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3"
D_ALANINE = "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m1/s1"
L_ALANINE = "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1"
ALANINE_FLAT = "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)"
PHENOL = "InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H"
ZINC = "InChI=1S/Zn/q+2"

# The 2lk1 pair, from a real bundle: same skeleton, two hydrogens apart. Under
# the connectivity bar this publishes as a flag; here it is a contradiction.
ANTHRAQUINONE_SULFONIC_SMILES = "c1ccc2c(c1)C(=O)c3ccc(cc3C2=O)S(=O)(=O)O"
ANTHRAQUINONE_SULFONATE_INCHI = (
    "InChI=1S/C14H10O5S/c15-13-9-3-1-2-4-10(9)14(16)12-7-8(20(17,18)19)5-6-11(12)13"
    "/h1-7,17-19H"
)


def test_row1_a_different_formula_stops():
    """The failure this whole function is for: a pasted wrong molecule."""

    verdict, why = la.agree("CCO", PROPANE)

    assert verdict == STOP
    assert "formula" in why


def test_row1_b_same_formula_same_skeleton_different_hydrogens_stops():
    """Tautomers: pentane-2,4-dione as keto and as enol.

    Formula C5H8O2 and connectivity `c1-4(6)3-5(2)7` on both sides -- only the
    /h layer moves, because InChI's connection table records which atoms are
    bonded and not their bond orders. So this row is only reachable once
    formula and /c have both agreed, and it is the one that makes this a
    protonation-sensitive check rather than a skeleton one.

    It is also the class the archive has actually been bitten by: MDR-55, where
    4390-4393 simulated a neutral 3,6-dihydropyridine against a declared +1
    pyridinium.
    """

    verdict, why = la.agree("CC(=O)CC(=O)C", "InChI=1S/C5H8O2/c1-4(6)3-5(2)7/h3,6H,1-2H3")

    assert verdict == STOP
    assert "hydrogen" in why


def test_this_bar_is_deliberately_stricter_than_the_connectivity_bar():
    """Asserted side by side so neither bar can drift onto the other.

    The 2lk1 pair: two hydrogens apart on an identical skeleton. Here it is a
    contradiction and stops; measured against coordinates it is a flag and
    publishes. Same molecules, two bars, on purpose.
    """

    import ligand_check

    declared_pair = la.agree(
        ANTHRAQUINONE_SULFONIC_SMILES, ANTHRAQUINONE_SULFONATE_INCHI
    )
    against_structure = ligand_check.check(
        [ANTHRAQUINONE_SULFONIC_SMILES],
        ["O=C1c2cc(ccc2C(=O)c2c1cccc2)S(O)(O)O"],
    )

    assert declared_pair[0] == STOP
    assert against_structure[0] == ligand_check.FLAG


def test_row2_conflicting_stereochemistry_stops():
    """Enantiomers. Same formula, same /c, same /h; only /m differs."""

    verdict, why = la.agree("C[C@@H](N)C(=O)O", L_ALANINE)

    assert verdict == STOP
    assert "stereo" in why


def test_row3_one_side_carrying_stereo_is_accepted():
    """Not a contradiction -- one side is simply less specific.

    This is the corpus's normal shape, not a corner: all 6,664 declared ligands
    in collection 5 carry no stereochemistry, hashing to the UHFFFAOYSA
    empty-stereo block.
    """

    verdict, why = la.agree("CC(N)C(=O)O", L_ALANINE)

    assert verdict == ACCEPT
    assert "inchi" in why, "the richer side must be named, so it can be kept"


def test_row3_is_symmetric_about_which_side_is_richer():
    verdict, why = la.agree("C[C@H](N)C(=O)O", ALANINE_FLAT)

    assert verdict == ACCEPT
    assert "smiles" in why


def test_row4_neither_side_carrying_stereo_is_accepted():
    verdict, _ = la.agree("Oc1ccccc1", PHENOL)
    assert verdict == ACCEPT


def test_row4_agreement_is_not_sensitive_to_smiles_spelling():
    """Two spellings of phenol must land identically, or the rule is comparing
    text rather than layers."""

    assert la.agree("Oc1ccccc1", PHENOL)[0] == ACCEPT
    assert la.agree("c1ccccc1O", PHENOL)[0] == ACCEPT


def test_row5_an_absent_connectivity_layer_is_unverifiable_not_a_mismatch():
    """A single atom has no /c layer by construction, not by failure.

    This is the load-bearing row. Treating an empty layer as evidence of
    disagreement is the mistake compare_smiles and ligand_check both carry
    explicit comments about.
    """

    verdict, why = la.agree("[Zn+2]", ZINC)

    assert verdict == UNVERIFIABLE
    assert "connectivity" in why


def test_row5_does_not_swallow_a_decidable_formula_difference():
    """Sodium declared against zinc's InChI.

    Rows 1 and 5 BOTH match this input -- the formulas differ, and both /c
    layers are empty. Taking row 5 literally would answer "we could not check"
    for a pair where the answer is obvious. The rule is evaluated in order and
    the formula test comes first; this test is what pins that, and it is an
    amendment to the rule as originally written down.
    """

    verdict, why = la.agree("[Na+]", ZINC)

    assert verdict == STOP
    assert "formula" in why


def test_an_unparseable_smiles_stops_rather_than_raising():
    verdict, why = la.agree("not-a-smiles", PHENOL)

    assert verdict == STOP
    assert "parse" in why


def test_a_declared_inchi_without_a_formula_layer_is_unverifiable():
    """`is_valid_inchi` in libmdrepo accepts `InChI=1S/xyzzy` -- it checks the
    shape of the string, not the chemistry. So a nonsense-but-well-formed InChI
    reaches here, and must not be reported as a molecular disagreement."""

    verdict, _ = la.agree("CCO", "InChI=1S/")

    assert verdict == UNVERIFIABLE


@pytest.mark.parametrize(
    "smiles, inchi, expected",
    [
        ("CCO", PROPANE, STOP),
        ("C[C@@H](N)C(=O)O", L_ALANINE, STOP),
        ("CC(N)C(=O)O", L_ALANINE, ACCEPT),
        ("Oc1ccccc1", PHENOL, ACCEPT),
        ("[Zn+2]", ZINC, UNVERIFIABLE),
        ("[Na+]", ZINC, STOP),
    ],
)
def test_the_verdict_does_not_depend_on_argument_order(smiles, inchi, expected):
    """The rule is about two descriptions of one molecule, so recomputing the
    declared InChI's own SMILES and swapping the sides must not change the
    answer. A rule that reads one side preferentially fails here."""

    import compare_smiles

    forward = la.agree(smiles, inchi)[0]
    assert forward == expected

    # Round-trip the declared InChI back to a SMILES and put it on the other
    # side. Only meaningful where the InChI is convertible.
    from rdkit import Chem

    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        pytest.skip("declared InChI is not convertible back to a molecule")
    reversed_smiles = Chem.MolToSmiles(mol)
    ours = compare_smiles.to_inchi(compare_smiles.parse_smiles(smiles))

    assert la.agree(reversed_smiles, ours)[0] == expected
