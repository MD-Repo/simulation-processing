"""Tests for compare_smiles.py — the layer comparison every ligand verdict
rests on, and the toolkit that computes it.

WHY THIS EXISTS NOW. On 2026-09-09 InChI generation moved from OpenBabel, which
bundles InChI 1.04 from 2011, to RDKit, which bundles 1.07.3. That is a change
to the substrate of every ligand comparison in the archive, and it was made
with no tests underneath it at all. These pin both halves: that the layer
comparison means what it says, and that the toolkit policy the switch
introduced cannot quietly come apart.

THE POLICY, because it is the part that is easy to get wrong later: RDKit
computes the InChI unless it refuses the molecule, in which case OpenBabel does
BOTH SIDES of that comparison. Never one of each. RDKit checks valence and
OpenBabel does not, and these molecules are frequently perceived from simulated
coordinates -- measured, RDKit rejects 1,963 of 6,659 inferred structures,
29.5%. Refusing them would have turned nearly a third of the corpus into
"unverifiable", and mixing versions within one comparison would report a
toolkit difference as a molecular one.

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_compare_smiles.py -q
"""

import pytest

import compare_smiles as cs

# A neutral nitrogen with four bonds. Real: the published SMILES of 2w1c/2w1e.
IMPOSSIBLE_VALENCE = "C[NH]5CCOCC5"


def test_inchi_comes_from_rdkit_not_openbabel():
    """The switch itself. 1.07.3 and 1.04 agree on this molecule, so the test
    cannot assert a differing string -- it asserts the route instead."""

    mol = cs.parse_smiles("CCO")

    assert cs._inchi_rdkit(cs.to_canonical(mol)) == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
    assert cs.to_inchi(mol) == cs._inchi_rdkit(cs.to_canonical(mol))


def test_rdkit_refuses_a_molecule_openbabel_will_build():
    """The reason the fallback exists, stated as a fact about the toolkits."""

    mol = cs.parse_smiles(IMPOSSIBLE_VALENCE)

    assert cs._inchi_rdkit(cs.to_canonical(mol)) == ""
    assert cs._inchi_openbabel(mol).startswith("InChI=1S/")


def test_to_inchi_falls_back_rather_than_returning_nothing():
    """A single molecule always gets an InChI if either toolkit can give one.

    Returning "" here would empty `inchi_available` for 29.5% of inferred
    structures and turn their verdicts into "unverifiable".
    """

    assert cs.to_inchi(cs.parse_smiles(IMPOSSIBLE_VALENCE)).startswith("InChI=1S/")


def test_a_pair_is_never_computed_by_two_different_toolkits():
    """The load-bearing invariant of the switch.

    If one side came from 1.07.3 and the other from 1.04, the layer comparison
    would be reading a version difference as a molecular one -- and it would do
    so precisely on the pairs where one side was perceived from coordinates,
    which is where real findings live.
    """

    good, bad = "CCO", IMPOSSIBLE_VALENCE

    assert cs.inchi_pair(cs.parse_smiles(good), cs.parse_smiles(good))[2] == cs.RDKIT

    for left, right in ((good, bad), (bad, good), (bad, bad)):
        i1, i2, software = cs.inchi_pair(cs.parse_smiles(left), cs.parse_smiles(right))
        assert software == cs.OPENBABEL, (left, right)
        assert i1.startswith("InChI=1S/") and i2.startswith("InChI=1S/")
        # And the fallback really did recompute BOTH, not just the failing one.
        assert i1 == cs._inchi_openbabel(cs.parse_smiles(left))
        assert i2 == cs._inchi_openbabel(cs.parse_smiles(right))


def test_compare_reports_which_toolkit_it_used():
    """Callers, and anyone reading a stored verdict later, need to know which
    InChI version the layers are expressed in."""

    assert cs.compare("CCO", "CCC")["inchi_software"] == cs.RDKIT
    assert (
        cs.compare(IMPOSSIBLE_VALENCE, "CCC")["inchi_software"] == cs.OPENBABEL
    )


def test_the_fallback_does_not_change_a_verdict():
    """2w1c against itself. Whichever toolkit is used, identical molecules are
    identical."""

    result = cs.compare(IMPOSSIBLE_VALENCE, IMPOSSIBLE_VALENCE)

    assert result["inchi_software"] == cs.OPENBABEL
    assert result["exact_match"] is True
    assert result["same_connectivity"] is True


def test_openbabel_emitted_a_stereo_layer_of_pure_unknowns_and_rdkit_does_not():
    """The one verdict detail that the switch actually changed, corpus-wide.

    2q2n's inferred structure got a /b layer from OpenBabel consisting entirely
    of undefined markers (`19-9?`, `20-10?`, ...), which the layer comparison
    then read as a genuine E/Z stereo difference and printed in the detail
    string. RDKit omits an all-unknown layer, so the spurious claim is gone.

    Measured over all 6,659 bundles: this was the ONLY change of any kind, and
    it changed a reason, not a verdict.
    """

    declared = (
        "Cc1c2[nH]c(c1CCC(=O)O)C=C3C(=C(C(=Cc4c(c(c([nH]4)C=C5C(=C(C(=C2)N5)"
        "S(=O)(=O)O)C)S(=O)(=O)O)C)N3)C)CCC(=O)O"
    )
    inferred = (
        "OC(=O)CCC1=C(C)C2=NC1=Cc1[nH]c(c(c1CCC(=O)O)C)C=C1N=C(/C=c/3\\[nH]"
        "c(=C2)c(C)c3S(O)(O)O)C(=C1S(O)(O)O)C"
    )

    result = cs.compare(declared, inferred)

    assert result["inchi_software"] == cs.RDKIT
    assert "E/Z stereo" not in result["differences"]
    assert "hydrogen attachment" in result["differences"]

    ob_layers = cs.inchi_layers(cs._inchi_openbabel(cs.parse_smiles(inferred)))
    assert "?" in ob_layers.get("b", ""), "OpenBabel's /b layer is all unknowns"


# --- the layer parser, which everything above depends on ---


def test_inchi_layers_splits_the_layers_it_claims_to():
    layers = cs.inchi_layers(
        "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m1/s1"
    )

    assert layers["formula"] == "C3H7NO2"
    assert layers["c"] == "c1-2(4)3(5)6"
    assert layers["h"] == "h2H,4H2,1H3,(H,5,6)"
    assert layers["t"] == "t2-"
    assert layers["m"] == "m1"
    assert layers["s"] == "s1"


def test_a_single_atom_has_no_connectivity_layer_by_construction():
    """Not a failure. This is the distinction that
    `inchi_available` exists to preserve and that the monatomic defect noted in
    TODO.md gets wrong on the reporting side."""

    layers = cs.inchi_layers(cs.to_inchi(cs.parse_smiles("[Zn+2]")))

    assert layers["formula"] == "Zn"
    assert "c" not in layers
    assert cs.connectivity_of("[Zn+2]") == ""


@pytest.mark.parametrize("software", [cs.RDKIT, cs.OPENBABEL, ""])
def test_connectivity_of_can_be_pinned_to_one_toolkit(software):
    """compare() pins it mid-comparison so the fragment check does not straddle
    two InChI versions."""

    assert cs.connectivity_of("CCO", software) == "c1-2-3"


def test_largest_fragment_ranks_by_heavy_atoms_not_string_length():
    main, dropped = cs.largest_fragment("CCO.[HH]")

    assert main == "CCO"
    assert dropped == ["[HH]"]
