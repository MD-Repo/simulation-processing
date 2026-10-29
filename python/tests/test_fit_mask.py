"""The RMS fit excludes ions; the strip list does not grow to do it.

Amber's calcium is residue CA, atom CA, so `@CA,C,N` fitted the trajectory
on freely diffusing ions (review CONV-3, TQ-03). The fit mask names the ion
residues it leaves out; minimal.pdb keeps them, because a bound Zn2+ or
Ca2+ is part of the structure and stripping more ions is a policy decision
(review #8) this code does not make.
"""

import importlib.util
import os
import sys
import types


def _load_module():
    """Import the script with its conda-only dependencies stubbed out, only
    while its body runs: a stub left in sys.modules breaks MDAnalysis for
    every later test in the process (see test_dcd_sampling.py)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    planted = [
        name
        for name in ("pytraj", "parmed")
        if sys.modules.setdefault(name, types.ModuleType(name)).__spec__ is None
    ]
    try:
        spec = importlib.util.spec_from_file_location(
            "cpptraj_gmx_traj_manipulation",
            os.path.join(here, "cpptraj_gmx_traj_manipulation.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in planted:
            sys.modules.pop(name, None)


c = _load_module()

# The strip list as it stood before the review fixes.
STRIPPED_IONS = [
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


def _excluded_residues(mask):
    assert mask.startswith("@CA,C,N&!:")
    return mask[len("@CA,C,N&!:") :].split(",")


def test_the_strip_list_is_unchanged():
    assert c.KNOWN_IONS == STRIPPED_IONS


def test_bound_metals_are_not_stripped():
    """Zn2+ in a zinc finger, Ca2+ in an EF hand: kept in minimal.pdb"""
    for name in ("CA", "Ca2+", "CAL", "ZN", "Zn2+", "ZN2", "CU", "MN", "NI", "Fe2+"):
        assert name not in c.KNOWN_IONS


def test_the_fit_leaves_out_every_ion_and_lipid():
    excluded = _excluded_residues(c.backbone_fit_mask({"CA", "C", "N", "O"}))
    for name in STRIPPED_IONS + ["CA", "Ca2+", "CAL", "ZN", "ZN2", "POT", "IOD"]:
        assert name in excluded
    for name in c.KNOWN_LIPIDS:
        assert name in excluded


def test_the_fit_excluded_ions_include_the_stripped_ones():
    assert set(STRIPPED_IONS) <= set(c.FIT_EXCLUDED_IONS)
    assert len(c.FIT_EXCLUDED_IONS) == len(set(c.FIT_EXCLUDED_IONS))


def test_a_coarse_grained_model_fits_on_every_atom():
    assert c.backbone_fit_mask({"BB", "SC1"}) == "@*"
