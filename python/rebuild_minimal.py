#!/usr/bin/env python3
"""
Purpose: Rebuild a simulation's minimal files from its full ones

From IN_DIR/full.pdb and IN_DIR/full.xtc, write to OUT_DIR:

  minimal.pdb, minimal.xtc, minimal.gro  the GROMACS path's own commands
                                         (write_minimal_commands): every
                                         atom but water, the stripped ions
                                         (and lipids), rotated to principal
                                         axes, the trajectory fitted on it
  sampled.xtc                            sample_trajectory.py, as processing
  full.xtc                               only with --dt: the same frames
                                         with the time axis t0 + i * dt
  full.pdb                               a copy, so OUT_DIR can be checked
                                         (mdr-process check-outputs)
  rebuild.json                           what was done and counted

Why: from 2025-10-14 to 2026-04-29 the GROMACS path stripped residue MOL
from minimal, so MISATO's 16,149 sims were released with no ligand in
minimal.* and sampled.xtc (#30). Their full files are intact, and their XTCs
have every frame at 0 ps; the database says 8 ns, a frame every 80 ps.

IN_DIR is never written. Exit status 1 if a step fails or the outputs do not
add up (minimal must keep every residue of full.pdb that is not dropped on
purpose, and every trajectory must keep full.xtc's frame count).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import List, NamedTuple, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cpptraj_gmx_traj_manipulation as conv  # noqa: E402

GMX = "/usr/local/gromacs/bin/gmx"


class Args(NamedTuple):
    in_dir: str
    out_dir: str
    gmx: str
    t0: float
    dt: Optional[float]
    keep_temp: bool


# --------------------------------------------------
def get_args(argv=None) -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Rebuild minimal.* and sampled.xtc from full.pdb + full.xtc",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--in-dir", required=True, help="Holds full.pdb, full.xtc"
    )
    parser.add_argument(
        "-o", "--out-dir", required=True, help="Where to write"
    )
    parser.add_argument(
        "-g", "--gmx", default=shutil.which("gmx") or GMX, help="gmx binary"
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Write a time axis into full.xtc first: ps between frames",
    )
    parser.add_argument(
        "--t0", type=float, default=0.0, help="First frame's time, ps"
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep the gmx intermediates"
    )
    args = parser.parse_args(argv)

    if os.path.abspath(args.in_dir) == os.path.abspath(args.out_dir):
        parser.error(
            "--out-dir must differ from --in-dir (inputs are never written)"
        )
    if args.dt is not None and not args.dt > 0:
        parser.error("--dt must be > 0")
    for name in ("full.pdb", "full.xtc"):
        if not os.path.isfile(os.path.join(args.in_dir, name)):
            parser.error(f"{args.in_dir}: no {name}")

    return Args(
        args.in_dir, args.out_dir, args.gmx, args.t0, args.dt, args.keep_temp
    )


# --------------------------------------------------
def run(cmd: List[str], stdin: str = "", cwd: Optional[str] = None) -> str:
    """Run a command; its output, or an error naming it"""

    p = subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, cwd=cwd
    )
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
        raise RuntimeError(
            f"{' '.join(cmd)} failed ({p.returncode}):\n{tail}"
        )
    return p.stdout + p.stderr


# --------------------------------------------------
def read_xtc(path: str):
    """(times in ps, coordinates in A as an array frames x atoms x 3)"""

    from MDAnalysis.coordinates.XTC import XTCReader

    r = XTCReader(path)
    times, xyz = [], []
    for ts in r:
        times.append(float(ts.time))
        xyz.append(ts.positions.copy())
    r.close()
    return np.array(times), np.array(xyz)


# --------------------------------------------------
def set_time_axis(gmx: str, src: str, dst: str, t0: float, dt: float) -> int:
    """Write `src` to `dst` with frame i at t0 + i * dt; the coordinates must
    come through unchanged (XTC stores them already rounded). Frames."""

    run(
        [
            gmx,
            "trjconv",
            "-f",
            src,
            "-o",
            dst,
            "-t0",
            str(t0),
            "-timestep",
            str(dt),
        ]
    )
    t_old, x_old = read_xtc(src)
    t_new, x_new = read_xtc(dst)
    if x_new.shape != x_old.shape:
        raise RuntimeError(
            f"{dst}: {x_new.shape} frames x atoms, {src} {x_old.shape}"
        )
    moved = float(np.abs(x_new - x_old).max()) if x_old.size else 0.0
    if moved > 1e-3:
        raise RuntimeError(f"{dst}: coordinates moved up to {moved:.4f} A")
    want = t0 + dt * np.arange(len(t_old))
    if not np.allclose(t_new, want, atol=1e-3):
        raise RuntimeError(
            f"{dst}: times {t_new[:3]}..., wanted {want[:3]}..."
        )
    return len(t_new)


# --------------------------------------------------
def residue_counts(pdb: str) -> dict:
    """{residue name: residues}, the first model of a .pdb, the 4-letter name
    read from column 17 as cpptraj writes it"""

    counts, last = {}, None
    with open(pdb, errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM  ", "HETATM")):
                key = (line[16:21].strip(), line[21], line[22:27])
                if key != last:
                    counts[key[0]] = counts.get(key[0], 0) + 1
                    last = key
    return counts


def dropped(resname: str) -> bool:
    """A residue minimal drops on purpose (the conversion script's lists)"""

    r = resname.upper()
    return r in {
        n.upper()
        for n in conv.KNOWN_WATER
        + conv.KNOWN_IONS
        + conv.KNOWN_LIPIDS
        + ["OHH", "TIP", "T3P", "T4P", "T5P", "T3H"]
    }


def missing_from_minimal(full_pdb: str, min_pdb: str) -> List[str]:
    """Residues of full.pdb, not dropped on purpose, that minimal lacks"""

    have = residue_counts(min_pdb)
    return [
        f"{name} ({have.get(name, 0)} of {n})"
        for name, n in sorted(residue_counts(full_pdb).items())
        if not dropped(name) and have.get(name, 0) < n
    ]


# --------------------------------------------------
def rebuild(args: Args) -> dict:
    """Do it; the report"""

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.abspath(args.out_dir)
    report = {"in_dir": os.path.abspath(args.in_dir), "out_dir": out}

    shutil.copy(
        os.path.join(args.in_dir, "full.pdb"), os.path.join(out, "full.pdb")
    )
    full_xtc = os.path.join(out, "full.xtc")
    if args.dt is not None:
        # Read a copy: MDAnalysis leaves an offsets file beside what it reads
        src = os.path.join(out, "full.in.xtc")
        shutil.copy(os.path.join(args.in_dir, "full.xtc"), src)
        report["time_axis"] = {"t0_ps": args.t0, "dt_ps": args.dt}
        report["full_frames"] = set_time_axis(
            args.gmx, src, full_xtc, args.t0, args.dt
        )
        os.remove(src)
    else:
        shutil.copy(os.path.join(args.in_dir, "full.xtc"), full_xtc)

    full_pdb = os.path.join(out, "full.pdb")
    lipid = any(
        r.upper() in conv.KNOWN_LIPIDS for r in residue_counts(full_pdb)
    )
    script = os.path.join(out, "rebuild_minimal.sh")
    with open(script, "w") as f:
        f.write("#!/usr/bin/env bash\nset -e\n")
        f.write(
            f"GMX={args.gmx}\nOUTDIR={out}\nSTRUCT_REF=$OUTDIR/full.pdb\n\n"
        )
        conv.write_minimal_commands(f, out, lipid)
    report["log"] = run(["bash", script], cwd=out)[-4000:]

    for name in ("minimal.pdb", "minimal.xtc", "minimal.gro"):
        if not os.path.isfile(os.path.join(out, name)):
            raise RuntimeError(f"{name} was not written")

    sampled = os.path.join(out, "sampled.xtc")
    if os.path.exists(sampled):
        os.remove(sampled)
    run(
        [
            sys.executable,
            os.path.join(HERE, "sample_trajectory.py"),
            "--trajectory",
            os.path.join(out, "minimal.xtc"),
            "--structure",
            os.path.join(out, "minimal.pdb"),
            "--outfile",
            sampled,
        ],
        cwd=HERE,
    )

    # Count and check
    t_full, x_full = read_xtc(full_xtc)
    t_min, x_min = read_xtc(os.path.join(out, "minimal.xtc"))
    t_smp, _ = read_xtc(sampled)
    report.update(
        lipids=lipid,
        full_atoms=int(x_full.shape[1]),
        minimal_atoms=int(x_min.shape[1]),
        frames={
            "full": len(t_full),
            "minimal": len(t_min),
            "sampled": len(t_smp),
        },
        minimal_residues=residue_counts(os.path.join(out, "minimal.pdb")),
    )
    problems = []
    if len(t_min) != len(t_full):
        problems.append(
            f"minimal.xtc has {len(t_min)} frames, full.xtc {len(t_full)}"
        )
    if not np.allclose(t_min, t_full, atol=1e-3):
        problems.append("minimal.xtc times differ from full.xtc's")
    missing = missing_from_minimal(full_pdb, os.path.join(out, "minimal.pdb"))
    if missing:
        problems.append("minimal.pdb lacks " + ", ".join(missing))
    report["problems"] = problems

    if not args.keep_temp:
        for name in (
            "minimal_temp.xtc",
            "minimal_temp.pdb",
            "min_princ.pdb",
            "index_minimal.ndx",
        ):
            p = os.path.join(out, name)
            if os.path.exists(p):
                os.remove(p)
    for name in os.listdir(out):
        # gmx backups, and MDAnalysis's offsets for files just read
        if (name.startswith("#") and name.endswith("#")) or (
            name.startswith(".") and name.endswith("_offsets.npz")
        ):
            os.remove(os.path.join(out, name))
    return report


# --------------------------------------------------
def main(argv=None) -> int:
    """Make a jazz noise here"""

    args = get_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        report = rebuild(args)
    except Exception as e:
        report = {"in_dir": args.in_dir, "error": str(e)}
    with open(os.path.join(args.out_dir, "rebuild.json"), "w") as fh:
        json.dump(
            {k: v for k, v in report.items() if k != "log"}, fh, indent=2
        )
    if report.get("error") or report.get("problems"):
        print(
            report.get("error") or "; ".join(report["problems"]),
            file=sys.stderr,
        )
        return 1
    print(
        f"{args.out_dir}: minimal {report['minimal_atoms']} of "
        f"{report['full_atoms']} atoms, frames {report['frames']}"
    )
    return 0


# --------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
