"""minimal_drops.json: the one list of residue names minimal.* drops.

The conversion script strips by it and mdr-process's output check "solute"
(mdrepo-rs#108) reads it to tell a ligand dropped by mistake (MISATO's MOL,
2025-10-14..2026-04-29, simulation-processing#30) from water, ions and
lipids dropped on purpose.
"""

import json
import os
import shutil
import subprocess

import pytest

import cpptraj_gmx_traj_manipulation as c

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(os.path.dirname(HERE), "minimal_drops.json")

# What the script held as literals before the list moved to the file
OLD_WATER = [
    "WAT",
    "HOH",
    "TIP3",
    "TIP3P",
    "TIP4",
    "TIP4P",
    "TIP5",
    "OPC",
    "SPC",
    "SOL",
]
OLD_IONS = [
    "NA",
    "CL",
    "K",
    "SOD",
    "CLA",
    "NA+",
    "CL-",
    "K+",
    "Mg2+",
    "MG",
    "Na+",
    "Cl-",
]
OLD_LIPIDS = ["DPPC", "POPC", "POPE", "DOPC", "DLPC", "DMPC", "DSPC", "CHOL"]


def write(tmp_path, data):
    p = tmp_path / "drops.json"
    p.write_text(json.dumps(data))
    return str(p)


def good():
    with open(FILE) as fh:
        return json.load(fh)


def test_the_file_sits_beside_the_script():
    assert os.path.samefile(c.MINIMAL_DROPS_FILE, FILE)


def test_the_lists_are_what_the_script_held():
    assert c.KNOWN_WATER == OLD_WATER
    assert c.KNOWN_IONS == OLD_IONS
    assert c.KNOWN_LIPIDS == OLD_LIPIDS


def test_lists_are_copies_not_the_loaded_data():
    c.KNOWN_WATER.append("XXX")
    try:
        assert "XXX" not in c.load_minimal_drops()["water"]
    finally:
        c.KNOWN_WATER.pop()


@pytest.mark.parametrize("key", ["water", "ions", "lipids", "gromacs_water"])
def test_no_duplicates(key):
    names = good()[key]
    assert len(names) == len(set(names)), key


@pytest.mark.parametrize("name", ["MOL", "LIG", "UNL", "UNK", "DRG", "INH"])
def test_no_ligand_name_is_dropped(name):
    d = good()
    for key in ("water", "ions", "lipids", "gromacs_water"):
        assert name.lower() not in {n.lower() for n in d[key]}, (key, name)


@pytest.mark.parametrize(
    "name",
    [
        "CA",
        "Ca2+",
        "CAL",
        "ZN",
        "Zn2+",
        "ZN2",
        "CU",
        "MN",
        "NI",
        "Fe2+",
        "MG2+",
    ],
)
def test_bound_metals_stay(name):
    # "MG2+" differs from the stripped "Mg2+" only in case; the script
    # compares exactly, so it is kept (mdr-process compares without case)
    assert name not in good()["ions"]


@pytest.mark.parametrize(
    "data,key",
    [
        ({"ions": [], "lipids": [], "gromacs_water": []}, "water"),
        (
            {"water": "WAT", "ions": [], "lipids": [], "gromacs_water": []},
            "water",
        ),
        (
            {"water": [], "ions": [""], "lipids": [], "gromacs_water": []},
            "ions",
        ),
        (
            {"water": [], "ions": [], "lipids": [3], "gromacs_water": []},
            "lipids",
        ),
        ({"water": [], "ions": [], "lipids": []}, "gromacs_water"),
    ],
)
def test_a_bad_file_is_refused(tmp_path, data, key):
    with pytest.raises(ValueError, match=key):
        c.load_minimal_drops(write(tmp_path, data))


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        c.load_minimal_drops(str(tmp_path / "nope.json"))


def test_a_good_file_loads(tmp_path):
    d = {
        "water": ["W"],
        "ions": ["NA"],
        "lipids": [],
        "gromacs_water": ["SOL"],
        "_about": ["x"],
    }
    assert c.load_minimal_drops(write(tmp_path, d)) == d


def test_strip_commands_use_the_file():
    # the make_ndx exclusion and the cpptraj strip mask name every ion
    excluded = c.backbone_fit_mask({"CA", "C", "N", "O"})
    for name in good()["ions"] + good()["lipids"]:
        assert name in excluded


def gmx_waters():
    gmx = shutil.which("gmx")
    if not gmx:
        return None
    out = subprocess.run(
        [gmx, "-quiet", "--version"], capture_output=True, text=True
    )
    for line in out.stdout.splitlines():
        if line.strip().startswith("Data prefix:"):
            path = os.path.join(
                line.split(":", 1)[1].strip(),
                "share",
                "gromacs",
                "top",
                "residuetypes.dat",
            )
            if os.path.isfile(path):
                with open(path) as fh:
                    return {
                        l.split()[0]
                        for l in fh
                        if len(l.split()) == 2 and l.split()[1] == "Water"
                    }
    return None


def test_gromacs_water_matches_residuetypes():
    waters = gmx_waters()
    if waters is None:
        pytest.skip("no gmx / residuetypes.dat")
    assert set(good()["gromacs_water"]) == waters
