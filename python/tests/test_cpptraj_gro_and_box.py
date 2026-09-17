"""Tests for the .gro and box handling in cpptraj_gmx_traj_manipulation.py

Two defects shipped malformed .gro files for about half the repository --
49,602 of 100,330 simulations -- and both are the kind that pass every check
because the file still looks complete.

THE LOST ATOM. parmed reads `structure[selection]` as a list of atom indices,
or, when the length of the selection happens to equal the atom count, as a
BOOLEAN MASK. That is documented in its `__getitem__` docstring and
implemented at parmed/structure.py:1281. The code passed indices, which is
correct right up until nothing is stripped -- then the length matches,
`[0, 1, 2, ...]` is read as a mask, and atom 0 is deselected because 0 is
false. Exactly the first atom, silently. `FakeStructure` below reproduces that
rule rather than mocking it away, because a fake that just returns the atoms
you asked for cannot show the bug.

THE MISSING BOX LINE. has_box() judged a frame by all six box values, so a run
with no periodic box -- zero lengths, default 90 degree angles -- read as
having one. That zero-length cell reached parmed's GRO writer, which raised
ZeroDivisionError building box vectors AFTER every atom line was already
written; the caller logged a warning and carried on, so a file that looks
complete and is not got pushed.

The numbers in test_bounding_box_matches_the_pipeline are from MDR00022148 and
are not invented: 5.95140 6.30240 5.96510 is what a real reprocess wrote, and
matching it byte-for-byte is what made a targeted repair defensible instead of
610 GB of reprocessing.
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The module imports pytraj and parmed at the top and neither is in this venv.
# Nothing under test here touches either -- these are pure functions over
# duck-typed objects -- so stub them rather than pull in the simproc env.
for _name in ("pytraj", "parmed"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

import cpptraj_gmx_traj_manipulation as c  # noqa: E402


# --------------------------------------------------
class FakeAtom:
    def __init__(self, resname, xyz=(0.0, 0.0, 0.0)):
        self.residue = types.SimpleNamespace(name=resname)
        self.xyz = xyz


class FakeStructure:
    """A structure that selects the way parmed really does.

    The length test is the whole point: an iterable as long as the atom list
    is read as a mask, anything shorter as a list of indices. Reproduced from
    parmed/structure.py:1281.
    """

    def __init__(self, resnames, coords=None, box=None):
        self.atoms = [FakeAtom(n) for n in resnames]
        self.coordinates = None if coords is None else np.asarray(coords)
        self.box = box
        self.saved_to = None

    def __getitem__(self, selection):
        selection = list(selection)
        if len(selection) == len(self.atoms):
            keep = [i for i, val in enumerate(selection) if val]
        else:
            keep = selection
        picked = FakeStructure([], box=self.box)
        picked.atoms = [self.atoms[i] for i in keep]
        if self.coordinates is not None:
            picked.coordinates = self.coordinates[keep]
        return picked

    def save(self, path, format=None):  # noqa: A002 - parmed's own signature
        self.saved_to = path
        with open(path, "wt") as out:
            print(f"{len(self.atoms)} atoms", file=out)


STRIP = {"HOH", "SOL", "NA", "CL"}


# --------------------------------------------------
def test_nothing_stripped_keeps_every_atom():
    """The bug, in the shape that caused it: no water, ions or lipids

    A plain index list here is [0, 1, 2, 3], whose length equals the atom
    count, so parmed reads it as a mask and drops MET. The mask keeps it.
    """
    structure = FakeStructure(["MET", "ALA", "ALA", "VAL"])

    index_list = [i for i, a in enumerate(structure.atoms)
                  if a.residue.name not in STRIP]
    assert len(structure[index_list].atoms) == 3, "the bug is not reproduced"
    assert structure[index_list].atoms[0].residue.name == "ALA"

    kept = structure[c.keep_mask(structure, STRIP)]
    assert len(kept.atoms) == 4
    assert kept.atoms[0].residue.name == "MET", "the first atom was dropped"


def test_mask_matches_the_index_list_whenever_anything_is_stripped():
    """The ordinary solvated case has to come out exactly as before"""
    structure = FakeStructure(["MET", "HOH", "ALA", "SOL", "VAL", "NA"])

    index_list = [i for i, a in enumerate(structure.atoms)
                  if a.residue.name not in STRIP]
    old = [a.residue.name for a in structure[index_list].atoms]
    new = [a.residue.name for a in structure[c.keep_mask(structure, STRIP)].atoms]

    assert old == new == ["MET", "ALA", "VAL"]


def test_mask_is_always_one_value_per_atom():
    """A mask cannot be mistaken for indices at any length, which is the point"""
    for resnames in (["MET"], ["MET", "HOH"], ["HOH"] * 5, ["ALA"] * 300):
        structure = FakeStructure(resnames)
        assert len(c.keep_mask(structure, STRIP)) == len(structure.atoms)


def test_everything_stripped_selects_nothing():
    structure = FakeStructure(["HOH", "SOL", "NA"])
    assert c.keep_mask(structure, STRIP) == [0, 0, 0]


# --------------------------------------------------
def test_a_zero_length_box_is_not_a_box():
    """The exact value a vacuum run reports, and the reason it slipped through

    Zero lengths with the default 90 degree angles is not all-zeros, so a test
    over all six values called it real.
    """
    assert c.usable_box([0, 0, 0, 90, 90, 90]) is False
    assert not all(abs(v) < 1e-6 for v in [0, 0, 0, 90, 90, 90]), \
        "the old all-six test would have accepted this"


def test_a_real_box_is_a_box():
    assert c.usable_box([60.0, 60.0, 60.0, 90.0, 90.0, 90.0]) is True
    assert c.usable_box([79.6, 79.6, 79.6, 109.47, 109.47, 109.47]) is True


def test_absent_and_all_zero_boxes_are_rejected():
    assert c.usable_box(None) is False
    assert c.usable_box([0, 0, 0, 0, 0, 0]) is False
    assert c.usable_box([0.0, 60.0, 60.0, 90, 90, 90]) is False, \
        "one zero length is still not a cell"


def test_has_box_reads_a_frame_the_same_way():
    frame = types.SimpleNamespace(box=[0, 0, 0, 90, 90, 90])
    assert c.has_box(frame) is False
    frame = types.SimpleNamespace(box=[60.0, 60.0, 60.0, 90, 90, 90])
    assert c.has_box(frame) is True
    assert c.has_box(types.SimpleNamespace(box=None)) is False


# --------------------------------------------------
def test_bounding_box_is_per_dimension_not_per_atom():
    """parmed's own fallback takes the extent with axis=1 and is unusable

    These coordinates are chosen so the two readings cannot be confused: the
    per-dimension extents are 10, 20 and 30, and nothing per-atom gives that.
    """
    structure = FakeStructure(
        ["ALA"] * 3,
        coords=[[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [5.0, 10.0, 15.0]],
    )
    box = c.bounding_box(structure)
    # Angstroms, with parmed's 5 angstrom clearance. The WRITER divides by 10
    # to get the nanometres a .gro stores, so this must not do it here -- doing
    # it twice is a cell a tenth the size of the molecule.
    assert box[:3] == pytest.approx([15.0, 25.0, 35.0])
    assert box[3:] == [90.0, 90.0, 90.0]


def test_bounding_box_matches_the_pipeline():
    """MDR00022148's real extents, against what a real reprocess wrote

    Angstroms in AND out; the .gro writer is what converts to nanometres. The
    reprocessed minimal.gro ends `5.95140 6.30240 5.96510`, so this function
    has to produce exactly ten times that.
    """
    lo, hi = np.array([-24.086, -26.163, -25.401]), np.array([30.428, 31.861, 29.250])
    structure = FakeStructure(["ALA"] * 2, coords=[lo, hi])
    box = c.bounding_box(structure)
    assert box[:3] == pytest.approx([59.5140, 63.0240, 59.6510], abs=1e-4)
    written = [v / 10 for v in box[:3]]
    assert written == pytest.approx([5.95140, 6.30240, 5.96510], abs=1e-5)


def test_bounding_box_declines_without_coordinates():
    assert c.bounding_box(FakeStructure(["ALA"], coords=None)) is None


# --------------------------------------------------
def test_save_gro_gives_a_boxless_structure_a_box(tmp_path):
    """A .gro without the box line is malformed, so one is always written"""
    structure = FakeStructure(
        ["ALA"] * 2, coords=[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], box=None)
    out = tmp_path / "minimal.gro"
    c.save_gro(structure, str(out))
    assert out.is_file()
    assert c.usable_box(structure.box), "no box was derived"


def test_save_gro_does_not_touch_a_real_box(tmp_path):
    real = [60.0, 60.0, 60.0, 90.0, 90.0, 90.0]
    structure = FakeStructure(["ALA"], coords=[[0.0, 0.0, 0.0]], box=list(real))
    c.save_gro(structure, str(tmp_path / "full.gro"))
    assert structure.box == real


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    """The defect this guards against wrote every atom and then died

    parmed writes the box last, so a write that fails there leaves a file
    holding every atom line and no box line -- one that looks complete, passes
    a size check, and is not a valid .gro. It was then pushed.
    """
    out = tmp_path / "minimal.gro"

    class Exploding(FakeStructure):
        def save(self, path, format=None):  # noqa: A002
            with open(path, "wt") as handle:
                print("every atom line, and then:", file=handle)
            raise ZeroDivisionError("division by zero")

    structure = Exploding(["ALA"], coords=[[0.0, 0.0, 0.0]], box=None)
    with pytest.raises(ZeroDivisionError):
        c.save_gro(structure, str(out))
    assert not out.exists(), "a partial .gro survived the failure"


def test_save_gro_replaces_an_existing_file(tmp_path):
    out = tmp_path / "minimal.gro"
    out.write_text("stale\n")
    structure = FakeStructure(["ALA"] * 3, coords=[[0.0, 0.0, 0.0]] * 3,
                              box=[60.0, 60.0, 60.0, 90.0, 90.0, 90.0])
    c.save_gro(structure, str(out))
    assert "stale" not in out.read_text()


# --------------------------------------------------
# The minimal_lipid path strips water and ions but KEEPS lipids, so it has its
# own strip set and its own two call sites (lines 964 and 1210). Same bug, same
# fix, and it is the one path the by-hand verification never exercised.
LIPID_STRIP = {"HOH", "SOL", "NA", "CL"}


def test_lipid_path_keeps_lipids_and_still_keeps_the_first_atom():
    """A membrane system where only water and ions go

    POPC survives, and MET at index 0 survives, which it would not if the
    selection were a plain index list of full length.
    """
    structure = FakeStructure(["MET", "POPC", "HOH", "POPC", "ALA", "NA"])
    kept = [a.residue.name
            for a in structure[c.keep_mask(structure, LIPID_STRIP)].atoms]
    assert kept == ["MET", "POPC", "POPC", "ALA"]


def test_lipid_path_with_nothing_to_strip_is_the_failing_shape():
    """A membrane system with no water or ions at all still keeps atom 0"""
    structure = FakeStructure(["MET", "POPC", "POPC"])
    index_list = [i for i, a in enumerate(structure.atoms)
                  if a.residue.name not in LIPID_STRIP]
    assert len(structure[index_list].atoms) == 2, "the bug is not reproduced"

    kept = structure[c.keep_mask(structure, LIPID_STRIP)]
    assert [a.residue.name for a in kept.atoms] == ["MET", "POPC", "POPC"]
