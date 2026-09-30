"""End-to-end ligand identity: the metadata names a file, and the molecule in
that file has to be the one the metadata declares.

WHY THIS EXISTS SEPARATELY from test_ligand_check.py. Every other ligand test
takes plain SMILES strings, and would pass unchanged even if
`structure_file_name` were ignored entirely. This is the only test that starts
from the TOML, follows the name to a real file, perceives the molecule from its
coordinates, and asks for a verdict. It is the seam, and nothing covered it.

The fixture is a real, redacted DDD bundle -- 2lk1, chosen because it is the
smallest complete post-import bundle in the corpus at 110 KB and because its
declared/inferred pair is the archive's central case: a sulfonic acid perceived
from coordinates as three single-bonded hydroxyls. Same skeleton, two hydrogens
apart, which under the connectivity bar publishes with a flag.

Run from `simulation-processing/python`:

    ./.venv/bin/python -m pytest tests/test_ligand_identity.py -q
"""

import json
import pathlib
import tomllib

import pytest

import ligand_check
import mol_id

BUNDLE = pathlib.Path(__file__).parent / "inputs/ligands/2lk1"


@pytest.fixture(scope="module")
def meta():
    return tomllib.loads((BUNDLE / "mdrepo-metadata.toml").read_text())


@pytest.fixture(scope="module")
def recorded():
    """What mol_id.py wrote for this bundle when it was actually imported."""

    return json.loads((BUNDLE / "processed/inferred_ligands.json").read_text())[0][
        "structure"
    ]


def test_structure_file_name_points_at_a_file_that_exists(meta):
    """The metadata's own promise, and the first thing that can be wrong."""

    assert meta["structure_file_name"] == "Pro_lig.pdb"
    assert (BUNDLE / meta["structure_file_name"]).is_file()


def test_the_molecule_perceived_from_the_named_file_is_the_one_it_holds(
    meta, recorded
):
    """Closes the loop: TOML -> file name -> coordinates -> molecule.

    The file holds anthraquinone-2-sulfonate: seven hydrogens, every one on a
    carbon, none on the sulfonate's oxygens. The recorded reading, from
    OpenBabel's bond angles at import time, made the three S-O bonds single
    and gave each oxygen an implicit hydrogen: C14H10O5S, three hydrogens the
    file does not have. The bond orders now come from the explicit hydrogens
    (mol_id._bond_orders_from_hydrogens), which leave only the anion.

    The recorded JSON is kept as imported -- get_inferred_ligands caches it and
    never regenerates it -- so it and a fresh reading now differ, by exactly
    those three hydrogens and the charge, over the same heavy atoms.
    """

    got = mol_id.structure_to_smiles(
        str(BUNDLE / meta["structure_file_name"]), resname="LIG"
    )
    got = got[0] if isinstance(got, list) else got

    assert got["smiles"] == "O=C1c2cc(ccc2C(=O)c2c1cccc2)S(=O)(=O)[O-]"
    assert got["formula"] == "C14H7O5S-"
    assert got["charge"] == -1
    assert got["num_heavy_atoms"] == recorded["num_heavy_atoms"]
    assert recorded["formula"] == "C14H10O5S"


def test_the_perceived_skeleton_is_the_declared_one(meta):
    """With the hydrogens deciding the bond orders, the perceived key's
    skeleton block is the declared molecule's, and only the protonation flag
    differs: the file simulates the sulfonate, the metadata declares the acid.
    The geometric reading shared neither block."""

    got = mol_id.structure_to_smiles(
        str(BUNDLE / meta["structure_file_name"]), resname="LIG"
    )
    got = got[0] if isinstance(got, list) else got
    declared = [lig["smiles"] for lig in meta["ligands"] if lig.get("smiles")]
    want = mol_id._inchikey_from_smiles(declared[0])

    assert got["inchikey"].split("-")[0] == want.split("-")[0]
    assert (got["inchikey"][-1], want[-1]) == ("M", "N")
    assert ligand_check.check(declared, [got["smiles"]]) == (
        ligand_check.FLAG,
        "ligand[0] protonation",
    )


def test_the_inchikey_survived_the_move_to_rdkit(recorded):
    """The recorded key was computed with OpenBabel's InChI 1.04 in August;
    InChIKeys now come from RDKit's 1.07.3.

    For this molecule the two agree exactly. Corpus-wide they do not: measured
    2026-09-09, 3,425 of 6,659 inferred keys change -- every one of them in the
    STEREO block only, none in the skeleton block, and none lost. So a stored
    key is still a reliable skeleton match and is no longer a reliable
    full-string match. This test is the canary for that boundary moving.

    It keys the recorded SMILES, so that it measures the InChI implementation
    alone and not how the molecule is perceived from its coordinates.
    """

    got = mol_id._inchikey_from_smiles(recorded["smiles"])

    assert got == recorded["inchikey"]
    assert got.split("-")[0] == recorded["inchikey"].split("-")[0]


def test_declared_against_perceived_gives_the_expected_verdict(meta, recorded):
    """The whole point, in one assertion."""

    declared = [lig["smiles"] for lig in meta["ligands"] if lig.get("smiles")]
    assert declared, "the fixture must carry a declared structure"

    verdict, detail = ligand_check.check(declared, [recorded["smiles"]])

    assert verdict == ligand_check.FLAG
    assert detail == (
        "ligand[0] formula (C14H8O5S vs C14H10O5S), hydrogen attachment"
    )


def test_a_ligand_declared_only_by_inchi_is_invisible_to_this_path(meta):
    """Property 2 fails on the python side too, by its own mechanism.

    Both callers of ligand_check build their declared list from
    `lig.get("smiles")` alone -- preflight_ligands.py and bulk_process_local.py
    -- so a ligand carrying only an `inchi` contributes nothing, the list comes
    back empty, and the bundle is BLOCKed for having no declaration at all.

    The Rust side fails the same property in a different place, at
    upsert_ligand. Fixing this needs both edits, which is why it is asserted
    here rather than assumed to follow from the Rust tests.
    """

    inchi_only = [{"name": "ethanol", "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"}]
    declared = [lig["smiles"] for lig in inchi_only if lig.get("smiles")]

    assert declared == []
    assert ligand_check.check(declared, ["CCO"]) == (
        ligand_check.BLOCK,
        "no declared ligand smiles; inference adopted unchecked",
    )


def test_the_redacted_fixture_still_carries_the_contributors_real_chemistry(meta):
    """Redaction removed the contributor's name, ORCID, email and institution.
    It must not have touched anything a test reads -- if it did, these tests
    would be checking a file we invented rather than one we received.
    """

    assert meta["integration_timestep_fs"] == 1
    assert meta["pdb_id"] == "2lk1"
    assert meta["ligands"][0]["name"] == (
        "9,10-dioxo-9,10-dihydroanthracene-2-sulfonic acid"
    )
    assert "@" not in (BUNDLE / "mdrepo-metadata.toml").read_text()
