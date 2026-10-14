"""Tests for frames_to_sample() in sample_trajectory.py

sampled.xtc (the site's Mol* preview) is meant to hold at most 100 frames from
the first 1,000. The step used to round down, so 101-199 frames were all kept
and 999 kept 111; MDR00105114's preview had 184.
"""

import importlib.util
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "sample_trajectory.py"
_SPEC = importlib.util.spec_from_file_location("sample_trajectory", _PATH)
sample_trajectory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sample_trajectory)
frames_to_sample = sample_trajectory.frames_to_sample


@pytest.mark.parametrize("num_frames", [1, 2, 99, 100, 101, 150, 184, 199, 200,
                                        250, 299, 999, 1000, 1001, 50_000])
def test_never_more_than_100_frames(num_frames):
    assert 0 < len(frames_to_sample(num_frames)) <= 100


@pytest.mark.parametrize("num_frames, want", [
    (1, 1), (100, 100),          # short: every frame
    (101, 51), (184, 92),        # step 2
    (999, 100), (1000, 100),     # step 10
    (5000, 100),                 # only the first 1,000 are read
])
def test_counts(num_frames, want):
    assert len(frames_to_sample(num_frames)) == want


def test_starts_at_the_first_frame_and_stays_within_the_first_1000():
    got = frames_to_sample(50_000)
    assert got[0] == 0
    assert got[-1] < 1000
