"""create_preview.py: what the page is given (#42)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from MDAnalysis.lib.formats.libmdaxdr import XTCFile  # noqa: E402

import create_preview  # noqa: E402


def write_xtc(path, frames):
    box = np.eye(3, dtype=np.float32) * 50
    with XTCFile(str(path), "w") as f:
        for i, xyz in enumerate(frames):
            f.write(xyz, box, i, float(i * 10), 1000.0)


def test_only_the_first_frame_is_passed(tmp_path):
    rng = np.random.default_rng(1)
    frames = [rng.uniform(0, 50, (300, 3)).astype(np.float32) for _ in range(20)]
    traj = tmp_path / "sampled.xtc"
    write_xtc(traj, frames)

    data = create_preview.first_frame_xtc(str(traj))
    out = tmp_path / "out.xtc"
    out.write_bytes(data)
    with XTCFile(str(out)) as f:
        got = [f.read() for _ in range(len(f))]

    assert len(got) == 1
    np.testing.assert_allclose(got[0].x, frames[0], atol=1e-3)
    assert got[0].time == 0.0
    assert len(data) < traj.stat().st_size / 10


def test_no_offsets_cache_is_left_beside_the_trajectory(tmp_path):
    traj = tmp_path / "sampled.xtc"
    write_xtc(traj, [np.zeros((10, 3), dtype=np.float32)] * 3)
    create_preview.first_frame_xtc(str(traj))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sampled.xtc"]
