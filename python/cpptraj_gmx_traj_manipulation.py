#!/usr/bin/env python3

import os
import argparse
import shutil
import struct
import warnings
import subprocess
import pytraj as pt
import parmed as pmd
import sys
from subprocess import getstatusoutput
from typing import List

warnings.filterwarnings("ignore")


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Convert Amber/NAMD/CHARMM/GROMACS files to GROMACS format"
    )
    parser.add_argument("-t", "--top", required=True, help="Topology file")
    parser.add_argument("-c", "--coord", help="Coordinate file (PDB/CRD)")
    parser.add_argument("-f", "--traj", help="Trajectory file (NC/DCD/XTC)")
    parser.add_argument("-r", "--tpr", help="TPR file")
    parser.add_argument("-g", "--gmx", help="GMX exe")
    parser.add_argument("-o", "--outdir", default="processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    return args


# --------------------------------------------------
def generate_tpr_from_top(gro_file, top_file, outdir, gmx_path):
    """Generate a TPR file from GRO and TOP files.

    Args:
        gro_file: Path to .gro structure file
        top_file: Path to .top topology file (text format)
        outdir: Output directory for generated files
        gmx_path: Path to gmx executable

    Returns:
        Path to generated TPR file, or None if generation fails
    """
    tpr_path = os.path.join(outdir, "generated.tpr")
    mdp_path = os.path.join(outdir, "minimal.mdp")

    # Create minimal MDP file
    with open(mdp_path, "w") as f:
        f.write("; Minimal MDP for TPR generation (no simulation)\n")
        f.write("integrator = md\n")
        f.write("nsteps = 0\n")

    # Run grompp to generate TPR
    cmd = [
        gmx_path,
        "grompp",
        "-f",
        mdp_path,
        "-c",
        gro_file,
        "-p",
        top_file,
        "-o",
        tpr_path,
        "-maxwarn",
        "10",
    ]

    verbose(f"Generating TPR file from {os.path.basename(top_file)}...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=os.path.dirname(top_file) or "."
    )

    if result.returncode == 0 and os.path.isfile(tpr_path):
        verbose(f"Generated TPR file: {tpr_path}")
        return tpr_path
    else:
        warn(f"Failed to generate TPR: {result.stderr}")
        return None


# --------------------------------------------------
def find_matching_top_file(gro_file):
    """Find a .top file that matches the given .gro file.

    Searches for .top files with similar naming patterns in the same directory.

    Args:
        gro_file: Path to .gro file

    Returns:
        Path to matching .top file, or None if not found
    """
    gro_dir = os.path.dirname(gro_file) or "."
    gro_base = os.path.basename(gro_file)
    gro_stem = os.path.splitext(gro_base)[0]

    # Try exact match first (same name, different extension)
    exact_match = os.path.join(gro_dir, gro_stem + ".top")
    if os.path.isfile(exact_match):
        return exact_match

    # Look for .top files in the same directory
    top_files = [f for f in os.listdir(gro_dir) if f.endswith(".top")]

    if not top_files:
        return None

    # Prefer files with similar naming pattern
    # e.g., for "1hyo_gromacs_cleaned.gro", prefer "1hyo_gromacs_cleaned.top"
    for top_file in top_files:
        top_stem = os.path.splitext(top_file)[0]
        # Check if stems share a common prefix
        if gro_stem.startswith(top_stem) or top_stem.startswith(gro_stem):
            return os.path.join(gro_dir, top_file)

    # If only one .top file exists, use it
    if len(top_files) == 1:
        return os.path.join(gro_dir, top_files[0])

    # Return None if ambiguous
    return None


# --------------------------------------------------
def find_matching_struct_for_top(base_files: List[str], ext: str):
    """Find a structure file (.tpr or .gro) alongside a topology or trajectory file.

    Args:
        base_files: List of path to topology/trajectory files
        ext: Extension to search for (e.g. ".tpr" or ".gro")

    Returns:
        Path to matching file, or None if not found
    """

    for base_file in base_files:
        base_dir = os.path.dirname(base_file) or "."
        base_stem = os.path.splitext(os.path.basename(base_file))[0]

        # Try exact stem match first
        exact_match = os.path.join(base_dir, base_stem + ext)
        if os.path.isfile(exact_match):
            return exact_match

        # Fall back to any file with that extension in the same directory
        candidates = [f for f in os.listdir(base_dir) if f.endswith(ext)]
        if len(candidates) == 1:
            return os.path.join(base_dir, candidates[0])

    return None


# --------------------------------------------------
def main():
    """LFG"""

    args = get_args()
    fmt = detect_format(args.top, args.tpr)
    topology_file = args.top
    coordinate_file = args.coord
    trajectory_file = args.traj
    tpr_file = args.tpr

    if fmt == "gromacs":
        verbose("Detected GROMACS files. Running GROMACS workflow...")
        gmx_path = args.gmx or which("gmx")

        # Auto-generate TPR if not provided
        if not tpr_file and topology_file.endswith(".gro"):
            # topology is a .gro — find a matching .top and generate TPR
            top_file = find_matching_top_file(topology_file)
            if top_file:
                verbose(f"No TPR provided. Found topology: {top_file}")
                tpr_file = generate_tpr_from_top(
                    topology_file, top_file, args.outdir, gmx_path
                )
                if not tpr_file:
                    warn("Could not generate TPR file. Some operations may fail.")
            else:
                warn("No TPR provided and no matching .top file found.")
        elif not tpr_file and topology_file.endswith(".top"):
            # topology is a .top — look for a .tpr to use directly, or a .gro to generate one
            tpr_file = find_matching_struct_for_top(
                [topology_file, trajectory_file], ".tpr"
            )
            if tpr_file:
                verbose(f"Found TPR file alongside .top: {tpr_file}")
            else:
                gro_file = find_matching_struct_for_top(
                    [topology_file, trajectory_file], ".gro"
                )
                if gro_file:
                    verbose(
                        f"No TPR found. Generating from {os.path.basename(topology_file)} + {os.path.basename(gro_file)}..."
                    )
                    tpr_file = generate_tpr_from_top(
                        gro_file, topology_file, args.outdir, gmx_path
                    )
                    if not tpr_file:
                        warn("Could not generate TPR file. Some operations may fail.")
                else:
                    warn(
                        "No TPR or GRO found alongside .top file. Some operations may fail."
                    )

        lipid_present = detect_lipids_gromacs(
            topology_file, tpr_file, args.outdir, gmx_path
        )
        bash_script = write_gromacs_bash(
            topology_file,
            trajectory_file,
            tpr_file,
            coordinate_file,
            args.outdir,
            lipid_present,
            gmx_path,
        )

        verbose(f"Running GROMACS processing script: {bash_script}")
        rv, out = subprocess.getstatusoutput(bash_script)
        if rv == 0:
            verbose("GROMACS processing completed.")

            # Restore chain IDs from original coordinate file if provided
            if coordinate_file and coordinate_file.endswith(".pdb"):
                verbose("Restoring chain IDs from original PDB...")
                pdb_files = ["full.pdb", "minimal.pdb"]
                if lipid_present:
                    pdb_files.append("minimal_lipid.pdb")

                for pdb_file in pdb_files:
                    pdb_path = os.path.join(args.outdir, pdb_file)
                    if os.path.isfile(pdb_path):
                        # Create backup
                        backup_path = pdb_path.replace(".pdb", "_nochain.pdb")
                        shutil.copy(pdb_path, backup_path)
                        # Restore chain IDs
                        restore_chain_ids(coordinate_file, backup_path, pdb_path)
        else:
            sys.exit(f"Failed to run '{bash_script}':\n{out}")

    elif fmt == "namd":
        verbose("Detected NAMD/PSF topology. Processing trajectory...")
        process_namd_trajectory(
            topology_file, coordinate_file, trajectory_file, args.outdir
        )
        verbose("NAMD trajectory processing completed.")

    elif fmt == "charmm":
        verbose("CHARMM support will be added later.")

    elif fmt == "amber":
        verbose("Detected Amber topology. Processing trajectory...")
        process_amber_trajectory(
            topology_file, coordinate_file, trajectory_file, args.outdir
        )
        verbose("Amber trajectory processing completed.")
    else:
        sys.exit("Unable to detect format")

    # A cpptraj failure now fails the run. Until 2026-08-11 it did not: every
    # call site warned and carried on, and this function returned normally, so
    # the script exited 0 having printed both "cpptraj full failed with exit
    # code 256" and "trajectory processing completed" (ticket 2175). The only
    # thing that caught it was mdr-process separately checking that the
    # expected outputs exist -- a check any other caller would have to
    # duplicate. The "completed" lines above are left as they were, because
    # they describe reaching the end of a workflow, not its success; this is
    # the verdict.
    if CPPTRAJ_FAILURES:
        sys.exit(
            "cpptraj failed during processing: "
            + "; ".join(CPPTRAJ_FAILURES)
            + " -- see the cpptraj output above for the reason"
        )


# --------------------------------------------------
def verbose(msg):
    print(f"[*] {msg}")


# --------------------------------------------------
def warn(msg):
    print(f"[!] {msg}", file=sys.stderr)


# Every cpptraj failure seen during this run. main() exits non-zero if this is
# non-empty. Before 2026-08-11 a failing cpptraj only produced a warn() and the
# script still exited 0 -- ticket 2175's log shows "cpptraj full failed with
# exit code 256" followed by "trajectory processing completed", and the only
# reason it was caught at all is that mdr-process separately checks the
# expected output files exist ("Command succeeded but failed to create ...").
# Anything downstream that did not verify outputs would have read it as success.
CPPTRAJ_FAILURES = []

# One machine-readable line for mdr-process, emitted exactly once per run. See
# report_time_axis() for the values and why "unknown" is not treated as absent.
TIME_AXIS_MARKER = "[mdrepo] source_has_time_axis="

# The frame spacing recovered from the source trajectory's own header, in ps,
# or "unknown". See source_sampling_ps() for why this is worth more than either
# the metadata declaration or anything measurable after conversion.
SAMPLING_MARKER = "[mdrepo] source_sampling_ps="

# One AKMA time unit in ps. CHARMM and NAMD write the integration timestep into
# the DCD header in AKMA units; multiplying by NSAVC gives the spacing between
# saved frames. Same constant MDAnalysis uses (1 / 20.45482949774598).
AKMA_TO_PS = 4.888821e-2


# --------------------------------------------------
def run_cpptraj(cppin, label, capture=True, fatal=True):
    """Run cpptraj on an input file, recording any failure

    Returns (returncode, combined output). Replaces the os.system calls this
    script used to make, for two reasons beyond capturing the text:

    os.system returns a *wait status*, not an exit code, which is why the old
    log line read "failed with exit code 256" for what was really exit code 1.
    And a failure recorded here makes the whole run exit non-zero at the end,
    rather than warning into a log nobody reads.

    Output is captured by default so callers can parse cpptraj's own trajin
    report. It is echoed to our stdout regardless, because it is the only
    diagnosis of a conversion failure and mdr-process quotes our stdout into
    the error it raises.

    `fatal=False` for the steps that have a designed fallback, and this
    distinction matters: the frame0 and conf extractions feed a parmed load
    guarded by `if os.path.isfile(...)` inside a try/except that explicitly
    tolerates the file being absent, and the conf path prefers the submitter's
    own coordinate file anyway. Those two were the calls previously sent to
    /dev/null with their status ignored, so treating them as fatal now would
    start failing directories that recover today. A non-fatal failure is still
    warned about; it just does not condemn the run.
    """

    result = subprocess.run(
        ["cpptraj", "-i", cppin],
        capture_output=capture,
        text=True,
    )
    out = ""
    if capture:
        out = (result.stdout or "") + (result.stderr or "")
        if out.strip():
            print(out, end="" if out.endswith("\n") else "\n")

    if result.returncode != 0:
        warn(f"cpptraj {label} failed with exit code {result.returncode}")
        if fatal:
            CPPTRAJ_FAILURES.append(f"{label} (exit {result.returncode})")
        else:
            warn(f"cpptraj {label} has a fallback; continuing")

    return result.returncode, out


# --------------------------------------------------
def trajectory_has_time_axis(cpptraj_output):
    """Whether cpptraj reported a `time` variable in the trajectory it read

    cpptraj's trajin report enumerates exactly what a file holds, e.g.

        '..._0.nc' is a NetCDF AMBER trajectory with coordinates, box, ...

    for MDR00048669's trajectory -- no `time` -- against `with coordinates,
    time, box` for one that has it. That report is free: the command already
    runs, and this only reads what it printed.

    Returns True, False, or None when no such line was found (a format whose
    report we cannot parse, which must not be mistaken for "no time axis").
    """

    for line in cpptraj_output.splitlines():
        if " is a " in line and " with " in line:
            contents = line.split(" with ", 1)[1]
            # The real line continues past the contents list with the parm and
            # box, e.g. "... with coordinates, box, Parm foo.prmtop (Truncated
            # octahedron box) (reading 1 of 5614)". Cut at " Parm " so a file
            # or path that happens to contain "time" cannot be read as a time
            # variable -- a false positive there would silently restore the
            # fabricated spacing this whole change exists to stop.
            contents = contents.split(" Parm ", 1)[0]
            # Split on periods as well as commas: some formats end the list
            # with a period ("with coordinates, time."), which would otherwise
            # leave the last field as "time." and never match.
            fields = [f.strip() for f in contents.replace(".", ",").split(",")]
            return "time" in fields
    return None


# --------------------------------------------------
def report_time_axis(has_time):
    """Emit the one marker line mdr-process reads

    `false` is a positive finding: cpptraj read the file and said it carries no
    time. That is the case worth acting on -- conversion to XTC stamps
    cpptraj's default 1 ps/frame onto the output, so by the time anything
    measures the converted file the fabricated spacing is indistinguishable
    from a real one.

    `unknown` means we could not tell, and mdr-process treats it as today's
    behaviour rather than as absence. Reporting absence we did not establish
    would fail directories over a parsing gap.
    """

    value = {True: "true", False: "false", None: "unknown"}[has_time]
    print(f"{TIME_AXIS_MARKER}{value}")


# --------------------------------------------------
def dcd_sampling_ps(path):
    """Frame spacing in ps from a DCD header, or None if it cannot be read

    DCD stores the integration timestep (DELTA) and the number of steps between
    saved frames (NSAVC), which is everything needed -- but cpptraj does not
    expose either, and reports a DCD as holding only `coords`. So a DCD lands
    in trajectory_has_time_axis()'s "unknown" bucket, conversion stamps 1
    ps/frame onto the XTC, and the fabricated spacing is measured back as
    though it were real. Reading the header here is what breaks that chain.

    Parsed by hand rather than with MDAnalysis, which reads this correctly but
    is not installed in the simproc env this script runs under. The layout is
    fixed and ancient; offsets below are from readdcd.h, the reference
    implementation MDAnalysis itself vendors:

        0..3    block size, always 84 -- also the endianness tell
        4..7    "CORD"
        8..11   NSET    number of frames
        12..15  ISTART  starting step
        16..19  NSAVC   steps between saved frames
        44..47  DELTA   timestep: float32 if CHARMM, float64 if X-PLOR
        84..87  CHARMM version, nonzero for CHARMM/NAMD files

    Returns None rather than raising for anything unexpected: failing to
    recover the spacing is a normal outcome that falls back to the declaration,
    not an error worth killing a conversion over.
    """

    try:
        with open(path, "rb") as fh:
            head = fh.read(92)
        if len(head) < 92:
            return None

        # Endianness comes from the leading 84, not from the platform: DCDs get
        # copied between machines and the magic is the only thing that says.
        for endian in ("<", ">"):
            if struct.unpack(endian + "i", head[0:4])[0] == 84:
                break
        else:
            return None

        if head[4:8] != b"CORD":
            return None

        nsavc = struct.unpack(endian + "i", head[16:20])[0]
        is_charmm = struct.unpack(endian + "i", head[84:88])[0] != 0

        # X-PLOR widens DELTA to a double over the same offset CHARMM uses for
        # a float. Reading the wrong width gives garbage, not a small error.
        if is_charmm:
            delta = struct.unpack(endian + "f", head[44:48])[0]
        else:
            delta = struct.unpack(endian + "d", head[44:52])[0]

        sampling_ps = delta * AKMA_TO_PS * nsavc
        if nsavc <= 0 or not (0 < sampling_ps < float("inf")):
            return None
        return sampling_ps
    except Exception as e:
        warn(f"Could not read DCD header from {path}: {e}")
        return None


# --------------------------------------------------
def source_sampling_ps(trajectory_file):
    """Frame spacing recovered from the source trajectory, or None

    Only DCD is handled, because DCD is the only format we ingest that carries
    the spacing in a header cpptraj declines to expose. Formats with a real
    time axis (NetCDF, XTC) survive conversion with their timing intact and are
    measured correctly downstream; formats without one have nothing to recover.
    """

    if not trajectory_file or not os.path.isfile(trajectory_file):
        return None
    if trajectory_file.lower().endswith(".dcd"):
        return dcd_sampling_ps(trajectory_file)
    return None


# --------------------------------------------------
def report_source_sampling(sampling_ps):
    """Emit the frame-spacing marker mdr-process reads

    `unknown` is the honest answer for every format we cannot recover, and
    mdr-process falls back to the metadata declaration or the measurement. Only
    a positive number here overrides those.
    """

    value = "unknown" if sampling_ps is None else repr(float(sampling_ps))
    print(f"{SAMPLING_MARKER}{value}")


# --------------------------------------------------
def time_action(sampling_ps):
    """cpptraj `time` action stamping a real frame spacing, or "" if unknown

    Without this, converting a source that cpptraj reports as coords-only
    writes an XTC whose frames are 1 ps apart -- a spacing nothing in the
    simulation ever had. That number does not just mislead our own duration
    arithmetic: full.xtc and minimal.xtc are published, so every downstream
    consumer inherits the fabricated time axis too.

    Emitted only when the spacing was actually recovered. Guessing here would
    reproduce the original bug with a different constant.
    """

    if sampling_ps is None:
        return ""
    return f"time time0 0 dt {sampling_ps}\n"


# --------------------------------------------------
def detect_format(top_file, tpr_file):
    exts = set(
        map(
            lambda f: os.path.splitext(f)[1].lower(),
            filter(None, [top_file, tpr_file]),
        )
    )

    if exts.intersection(set([".prmtop", ".parm7"])):
        return "amber"

    if exts.intersection(set([".psf"])):
        return "namd"

    if exts.intersection(set([".rtf", ".prm"])):
        return "charmm"

    # NB: A ".top" file is not sufficient for the topology
    # but the meta check should ensure that either a ".tpr"
    # or ".gro" file is also available.
    if exts.intersection(set([".top", ".tpr", ".gro"])):
        return "gromacs"

    return None


# --------------------------------------------------
def usable_box(box):
    """Whether `box` describes a real unit cell.

    Judged on the three lengths alone. A simulation run without a periodic
    box reports zero lengths with the default 90 degree angles, which is not
    a box but is not all zeros either, so a test over all six values calls it
    real. Everything downstream then believes there is a cell to work in:
    cpptraj is told to autoimage, and parmed is handed a zero-length cell it
    cannot turn into box vectors.
    """
    if box is None:
        return False
    return all(abs(v) > 1e-6 for v in list(box)[:3])


def has_box(frame):
    return usable_box(getattr(frame, "box", None))


def bounding_box(structure):
    """A cell big enough to hold `structure`, with 5 angstroms of clearance.

    The GRO format always wants the box line, even for a simulation that had
    no periodic box. Parmed writes one itself in that case, but takes the
    extent along the wrong axis -- per atom rather than per dimension -- so
    the numbers it puts there do not describe the molecule at all.

    Returns None if the structure has no coordinates to measure.
    """
    coords = getattr(structure, "coordinates", None)
    if coords is None:
        return None
    extent = coords.max(axis=0) - coords.min(axis=0) + 5.0
    return [float(extent[0]), float(extent[1]), float(extent[2]), 90.0, 90.0, 90.0]


def keep_mask(structure, strip_residues):
    """A per-atom 0/1 mask over `structure`, 1 where the atom is kept.

    Parmed reads `structure[selection]` one of two ways: as a list of atom
    indices, or -- when the length of the selection happens to equal the atom
    count -- as a boolean mask. So a plain index list changes meaning the
    moment nothing is stripped: `[0, 1, 2, ...]` read as a mask deselects
    atom 0, because 0 is false, and the first atom of the structure silently
    disappears. A mask means the same thing at every length.
    """
    return [
        0 if atom.residue.name in strip_residues else 1
        for atom in structure.atoms
    ]


def save_gro(structure, path):
    """Write `structure` to `path` as a .gro, with a box line that is real.

    `combine="all"` is load-bearing, not a tuning knob. Without it parmed's
    GRO writer calls `struct.split()` to group atoms into moleculetypes, and
    that loop raises `RuntimeError: Could not find <Atom ...>` MID-WRITE for
    any structure whose bonded groups it cannot match -- after the count line
    and a handful of atom lines are already on disk. That is how 46,578
    released `minimal.gro` files came to declare thousands of atoms and hold
    one. `combine="all"` skips the split entirely (`gromacsgro.py:257`), and
    parmed's own docstring is explicit that only the default may reorder
    atoms: every other value leaves the order alone, which is what we want,
    because the order is the topology's.

    Verified against 40 released simulations: byte-identical to the default
    wherever the default succeeds (27 of 27), and succeeds on all 13 where
    the default raises. Reprocessing does NOT fix these files without it --
    proved on MDR00021551, where a clean reprocess still wrote no .gro at
    all.

    A failed write is not left behind. Parmed writes the box last, so a write
    that dies there leaves a file holding every atom line and no box line --
    one that looks complete, passes a size check, and is not a valid .gro.
    """
    if not usable_box(getattr(structure, "box", None)):
        structure.box = bounding_box(structure)
    if os.path.isfile(path):
        os.remove(path)
    try:
        structure.save(path, format="gro", combine="all")
    except Exception:
        if os.path.isfile(path):
            os.remove(path)
        raise


# --------------------------------------------------
def autoimage_action(box_present):
    """cpptraj `autoimage` line, or "" when there is no unit cell to image in

    The full trajectory has always had this; the stripped one never did, and
    the stripped one is what becomes minimal.xtc and then the sampled.xtc
    video. So a complex whose chains sit in different periodic images -- an
    FXR ligand-binding domain and its coactivator peptide, say -- was written
    to the video with the peptide flung to the far side of the box, jumping a
    whole box vector between frames.

    Nothing was wrong with the simulation or with full.xtc. Only the video.
    """

    if not box_present:
        return ""
    return "autoimage\n"


# --------------------------------------------------
def process_stripped_trajectory(
    topology_file,
    trajectory_file,
    outdir,
    strip_mask,
    prefix,
    fit_mask="@CA,C,N",
    sampling_ps=None,
    box_present=False,
):
    """Process a stripped trajectory with principal rotation workflow.

    Workflow:
    1. Extract first frame, strip atoms, rotate to principal components, rotate 90° Z
    2. Superimpose full trajectory to rotated reference

    Args:
        topology_file: Path to topology file (prmtop, psf, etc.)
        trajectory_file: Path to trajectory file (nc, xtc, dcd, etc.)
        outdir: Output directory
        strip_mask: cpptraj strip mask (e.g., ':WAT,HOH,NA,CL')
        prefix: Output file prefix (e.g., 'minimal' or 'minimal_lipid')
        sampling_ps: Frame spacing recovered from the source, stamped onto the
            output so the published trajectory carries a real time axis
        box_present: Whether the source carries a unit cell, enabling the
            autoimage that keeps a multi-chain complex whole (see below)

    Returns:
        tuple: (xtc_path, pdb_path, ref_path) or (None, None, None) on failure
    """

    output_xtc = os.path.join(outdir, f"{prefix}.xtc")
    output_pdb = os.path.join(outdir, f"{prefix}.pdb")
    ref_pdb = os.path.join(outdir, f"{prefix}_ref.pdb")
    base_cppin = os.path.join(outdir, f"cpptraj_{prefix}")

    # Step 1: Extract first frame, strip, rotate to principal components, then orient vertically
    # principal dorotation aligns longest axis with X, we want it along Y (vertical)
    # rotate z 90: X→Y (long axis now vertical)
    # rotate x -90: adjust if needed to get proper front-facing orientation
    verbose(f"  Creating rotated reference structure for {prefix}...")
    cppin_ref = base_cppin + "_ref.in"
    with open(cppin_ref, "w") as f:
        f.write(f"parm {topology_file}\n")
        f.write(f"trajin {trajectory_file} 1 1\n")
        # Before strip, so the imaging sees whole molecules and a real box.
        # The principal axes are measured from this frame; a chain sitting in
        # the wrong periodic image drags them off and tilts the whole video.
        f.write(autoimage_action(box_present))
        f.write(f"strip {strip_mask}\n")
        f.write(f"principal {fit_mask} dorotation\n")
        f.write("rotate z 90\n")
        f.write("rotate x -90\n")
        f.write(f"trajout {ref_pdb} pdb\n")
        f.write("run\n")
    rv, _ = run_cpptraj(cppin_ref, f"{prefix} reference")
    if rv != 0:
        return None, None, None
    fix_pdb_element_symbols(ref_pdb)

    # Step 2: Process full trajectory, RMS fit to rotated reference
    # Load reference with stripped topology, load traj with full topology and strip on-the-fly
    verbose(f"  Superimposing {prefix} trajectory to rotated reference...")
    cppin_traj = base_cppin + ".in"
    with open(cppin_traj, "w") as f:
        f.write(f"parm {topology_file} [full]\n")
        f.write(f"parm {topology_file} [stripped]\n")
        f.write(f"parmstrip {strip_mask} parmindex 1\n")
        f.write(f"reference {ref_pdb} parm [stripped] [rotref]\n")
        f.write(f"trajin {trajectory_file} parm [full]\n")
        # MUST precede both strip and rms. Imaging needs the unrotated box:
        # `rms` rewrites coordinates into the reference's frame, after which
        # the stored box vectors no longer describe the periodic lattice and
        # imaging silently does the wrong thing. Stripping first would also
        # remove the solvent that tells autoimage which molecules are mobile.
        f.write(autoimage_action(box_present))
        f.write(f"strip {strip_mask}\n")
        f.write(f"rms ref [rotref] {fit_mask}\n")
        f.write(time_action(sampling_ps))
        f.write(f"trajout {output_xtc} xtc\n")
        f.write(f"trajout {output_pdb} pdb onlyframes 1\n")
        f.write("run\n")
    rv, _ = run_cpptraj(cppin_traj, prefix)
    if rv != 0:
        return None, None, None
    fix_pdb_element_symbols(output_pdb)

    return output_xtc, output_pdb, ref_pdb


# --------------------------------------------------
def detect_lipids_gromacs(topfile, tpr_file, outdir, gmx_path):
    """Detect if Lipids are present using make_ndx output"""

    verbose("Detecting lipids in GROMACS system...")
    ndx_path = os.path.join(outdir, "index.ndx")

    if tpr_file:
        topfile = tpr_file

    try:
        subprocess.run(
            [gmx_path, "make_ndx", "-f", topfile, "-o", ndx_path],
            input="q\n",
            text=True,
            capture_output=True,
            check=True,
        )
        with open(ndx_path, "r") as f:
            content = f.read()
        if "Lipid" in content:
            verbose("Lipid group detected.")
            return True
        else:
            verbose("No Lipid group detected.")
            return False
    except subprocess.CalledProcessError as e:
        warn(f"Error detecting lipids: {e}")
        return False


# --------------------------------------------------
def generate_group_string(ions, lipids=None):
    """Create compact GROMACS trjconv group string"""

    group_str = "_".join(ions)
    if lipids:
        group_str += "_" + "_".join(lipids)
    return group_str


# --------------------------------------------------
def process_amber_trajectory(topology_file, coordinate_file, trajectory_file, outdir):
    """Process Amber trajectory files to generate full and minimal variants"""

    # Use iterload for memory efficiency with large trajectories
    traj = None
    if trajectory_file:
        traj = pt.iterload(trajectory_file, top=topology_file)
    elif coordinate_file:
        traj = pt.iterload(coordinate_file, top=topology_file)
    else:
        warn("No trajectory or coordinates loaded; exiting.")
        return

    first_frame = traj[0]
    # Hoisted out of the full-trajectory branch below: the stripped
    # trajectories need the same answer, and re-deriving it there would let
    # the two paths disagree about whether there is a cell to image in.
    box_present = has_box(first_frame)
    atom_names = {atom.name for atom in traj.top.atoms}
    fit_mask = "@CA,C,N" if atom_names & {"CA", "C", "N"} else "@*"
    if fit_mask == "@*":
        verbose(
            "No backbone atoms (CA/C/N) found; using all atoms for RMS fit and principal rotation."
        )

    # conf_cif = os.path.join(outdir, "conf.cif")
    # pt.write_traj(
    #    filename=conf_cif,
    #    traj=traj,
    #    frame_indices=[0],
    #    format="cif",
    #    overwrite=True,
    # )

    conf_pdb = os.path.join(outdir, "conf.pdb")
    pt.write_traj(
        filename=conf_pdb,
        traj=traj,
        frame_indices=[0],
        format="pdb",
        overwrite=True,
    )

    # Extract a PDB from trajectory that matches topology atom count (for parmed)
    # We trust the toplogy file instead of the uploaded pdb file because in amber MD simulation
    # the prmtop is built with LEaP/tleap which adds terminal atoms and ions.
    traj_pdb = os.path.join(outdir, "traj_frame0.pdb")
    cppin_frame0 = os.path.join(outdir, "cpptraj_frame0.in")
    with open(cppin_frame0, "w") as f:
        f.write(f"parm {topology_file}\n")
        f.write(f"trajin {trajectory_file} 1 1\n")
        f.write(f"trajout {traj_pdb} pdb\n")
        f.write("run\n")
    # This output used to go to /dev/null. It is the only place the source
    # trajectory is read before conversion rewrites its timing, so it is the
    # one chance to see whether it carries a time axis at all.
    _, frame0_out = run_cpptraj(cppin_frame0, "frame0 extraction", fatal=False)
    report_time_axis(trajectory_has_time_axis(frame0_out))

    # Recovered from the source header, not from cpptraj's report, because
    # cpptraj does not expose DCD timing at all. Everything below stamps this
    # onto the converted output so it carries a real time axis.
    sampling_ps = source_sampling_ps(trajectory_file)
    report_source_sampling(sampling_ps)
    if sampling_ps is not None:
        verbose(f"Recovered {sampling_ps} ps/frame from the source header")

    # Load structure with parmed for .gro file generation
    structure = None
    try:
        # Use the trajectory-extracted PDB which matches the topology
        if os.path.isfile(traj_pdb):
            structure = pmd.load_file(topology_file, xyz=traj_pdb)
        elif coordinate_file:
            structure = pmd.load_file(topology_file, xyz=coordinate_file)
        else:
            structure = pmd.load_file(topology_file)
        verbose("Loaded structure with parmed for .gro generation.")
    except (ValueError, Exception) as e:
        structure = None
        warn(f"Could not load structure with parmed: {e}")
        warn("Continuing without .gro file generation...")

    base_cppin = os.path.join(outdir, "cpptraj")
    full_xtc = os.path.join(outdir, "full.xtc")
    # full_cif = os.path.join(outdir, "full.cif")
    full_pdb = os.path.join(outdir, "full.pdb")
    full_gro = os.path.join(outdir, "full.gro")

    # if not all(map(file_exists, [full_xtc, full_cif, full_pdb])):
    if not all(map(file_exists, [full_xtc, full_pdb])):
        verbose("Generating full trajectory...")
        cppin_full = base_cppin + "_full.in"
        with open(cppin_full, "w") as f:
            f.write(f"parm {topology_file}\n")
            f.write(f"trajin {trajectory_file}\n")
            # Use autoimage only if box is present
            if box_present:
                verbose("Box detected. Using autoimage...")
                f.write("autoimage\n")
            else:
                verbose("No box detected. Skipping autoimage...")
            # RMS fit to first frame using backbone atoms (or all atoms for CG)
            f.write(f"rms first {fit_mask}\n")
            f.write(time_action(sampling_ps))
            f.write(f"trajout {full_xtc} xtc\n")
            # f.write(f"trajout {full_cif} cif onlyframes 1\n")
            f.write(f"trajout {full_pdb} pdb onlyframes 1\n")
            f.write("run\n")
        run_cpptraj(cppin_full, "full")
    else:
        verbose("Full trajectory files already exist, skipping...")

    # Generate full.gro from structure
    if structure is not None and not file_exists(full_gro):
        try:
            save_gro(structure, full_gro)
            verbose("Generated full.gro")
        except Exception as e:
            warn(f"Could not generate full.gro: {e}")

    # Minimal trajectory (strip water, ions, lipids)
    minimal_xtc = os.path.join(outdir, "minimal.xtc")
    minimal_pdb = os.path.join(outdir, "minimal.pdb")
    minimal_gro = os.path.join(outdir, "minimal.gro")
    strip_mask_minimal = ":" + ",".join(KNOWN_WATER + KNOWN_IONS + KNOWN_LIPIDS)

    if not all(map(file_exists, [minimal_xtc, minimal_pdb])):
        verbose("Generating minimal trajectory (strip water, ions, lipids)...")
        process_stripped_trajectory(
            topology_file,
            trajectory_file,
            outdir,
            strip_mask_minimal,
            "minimal",
            fit_mask=fit_mask,
            sampling_ps=sampling_ps,
            box_present=box_present,
        )

    # Generate minimal.gro from structure
    if structure is not None and not file_exists(minimal_gro):
        try:
            # Strip water, ions, and lipids from structure using efficient selection
            strip_residues = set(KNOWN_WATER + KNOWN_IONS + KNOWN_LIPIDS)
            # Select atoms to keep (not in strip list)
            minimal_struct = structure[keep_mask(structure, strip_residues)]
            save_gro(minimal_struct, minimal_gro)
            verbose("Generated minimal.gro")
        except Exception as e:
            warn(f"Could not generate minimal.gro: {e}")

    # Minimal lipid trajectory (strip water and ions only, keep lipids)
    has_lipid = any(res.name in KNOWN_LIPIDS for res in traj.top.residues)
    strip_mask_minlip = ":" + ",".join(KNOWN_WATER + KNOWN_IONS)

    if has_lipid:
        minlip_xtc = os.path.join(outdir, "minimal_lipid.xtc")
        minlip_pdb = os.path.join(outdir, "minimal_lipid.pdb")
        minlip_gro = os.path.join(outdir, "minimal_lipid.gro")
        if not all(map(file_exists, [minlip_xtc, minlip_pdb])):
            verbose("Lipids detected. Generating minimal_lipid trajectory...")
            process_stripped_trajectory(
                topology_file,
                trajectory_file,
                outdir,
                strip_mask_minlip,
                "minimal_lipid",
                fit_mask=fit_mask,
                sampling_ps=sampling_ps,
                box_present=box_present,
            )

        # Generate minimal_lipid.gro from structure
        if structure is not None and not file_exists(minlip_gro):
            try:
                # Strip water and ions only (keep lipids) using efficient selection
                strip_residues = set(KNOWN_WATER + KNOWN_IONS)
                # Select atoms to keep (not in strip list)
                minlip_struct = structure[keep_mask(structure, strip_residues)]
                save_gro(minlip_struct, minlip_gro)
                verbose("Generated minimal_lipid.gro")
            except Exception as e:
                warn(f"Could not generate minimal_lipid.gro: {e}")
    else:
        verbose("No lipids detected; skipping minimal_lipid trajectory.")


# --------------------------------------------------
def process_namd_trajectory(topology_file, coordinate_file, trajectory_file, outdir):
    """Process NAMD trajectory files (PSF topology) to generate full and minimal variants"""

    # For NAMD, we use cpptraj directly since pytraj may have issues with PSF
    # cpptraj can handle PSF files natively

    if not trajectory_file and not coordinate_file:
        warn("No trajectory or coordinates provided; exiting.")
        return

    # Determine input for trajectory processing
    traj_input = trajectory_file if trajectory_file else coordinate_file

    # Try to load trajectory to check for box and lipids
    # Use cpptraj to get topology info
    base_cppin = os.path.join(outdir, "cpptraj")

    # First, extract frame 0 to a temporary PDB to analyze the system
    traj_pdb = os.path.join(outdir, "traj_frame0.pdb")
    cppin_frame0 = os.path.join(outdir, "cpptraj_frame0.in")
    with open(cppin_frame0, "w") as f:
        f.write(f"parm {topology_file}\n")
        if trajectory_file:
            f.write(f"trajin {trajectory_file} 1 1\n")
        else:
            f.write(f"trajin {coordinate_file} 1 1\n")
        f.write(f"trajout {traj_pdb} pdb\n")
        f.write("run\n")
    _, frame0_out = run_cpptraj(cppin_frame0, "frame extraction", fatal=False)
    report_time_axis(trajectory_has_time_axis(frame0_out))

    # Recovered from the source header, not from cpptraj's report, because
    # cpptraj does not expose DCD timing at all. Everything below stamps this
    # onto the converted output so it carries a real time axis.
    sampling_ps = source_sampling_ps(trajectory_file)
    report_source_sampling(sampling_ps)
    if sampling_ps is not None:
        verbose(f"Recovered {sampling_ps} ps/frame from the source header")
    fix_pdb_element_symbols(traj_pdb)

    # Load with pytraj to check box and lipids
    # Load from trajectory file first (has box info), then check coordinate file
    box_info = None
    try:
        import pytraj as pt

        if trajectory_file and os.path.isfile(trajectory_file):
            traj = pt.iterload(trajectory_file, top=topology_file)
            first_frame = traj[0]
            box_present = has_box(first_frame)
            if box_present:
                box_info = list(first_frame.box)  # [a, b, c, alpha, beta, gamma]
            has_lipid = any(res.name in KNOWN_LIPIDS for res in traj.top.residues)
        elif os.path.isfile(traj_pdb):
            traj = pt.load(traj_pdb, top=topology_file)
            first_frame = traj[0]
            box_present = has_box(first_frame)
            if box_present:
                box_info = list(first_frame.box)
            has_lipid = any(res.name in KNOWN_LIPIDS for res in traj.top.residues)
        elif coordinate_file:
            traj = pt.load(coordinate_file, top=topology_file)
            first_frame = traj[0]
            box_present = has_box(first_frame)
            if box_present:
                box_info = list(first_frame.box)
            has_lipid = any(res.name in KNOWN_LIPIDS for res in traj.top.residues)
        else:
            box_present = False
            has_lipid = False
        atom_names = {atom.name for atom in traj.top.atoms}
        fit_mask = "@CA,C,N" if atom_names & {"CA", "C", "N"} else "@*"
        if fit_mask == "@*":
            verbose(
                "No backbone atoms (CA/C/N) found; using all atoms for RMS fit and principal rotation."
            )
    except Exception as e:
        warn(f"Could not load trajectory with pytraj: {e}")
        warn("Proceeding without box/lipid detection...")
        box_present = False
        has_lipid = False
        fit_mask = "@CA,C,N"

    # Write conf files
    conf_pdb = os.path.join(outdir, "conf.pdb")
    cppin_conf = os.path.join(outdir, "cpptraj_conf.in")
    with open(cppin_conf, "w") as f:
        f.write(f"parm {topology_file}\n")
        if trajectory_file:
            f.write(f"trajin {trajectory_file} 1 1\n")
        else:
            f.write(f"trajin {coordinate_file}\n")
        f.write(f"trajout {conf_pdb} pdb onlyframes 1\n")
        f.write("run\n")
    run_cpptraj(cppin_conf, "conf extraction", fatal=False)
    fix_pdb_element_symbols(conf_pdb)

    # Load full structure with parmed for .gro generation
    # For PSF files, we need to load topology and coordinates separately
    # Prefer original coordinate file over cpptraj-generated PDB (better formatting)
    full_structure = None
    try:
        full_structure = pmd.load_file(topology_file)
        # Prefer original coordinate file (has proper PDB formatting)
        coord_source = (
            coordinate_file
            if (coordinate_file and os.path.isfile(coordinate_file))
            else traj_pdb
        )
        if coord_source and os.path.isfile(coord_source):
            coords = pmd.load_file(coord_source)
            coord_natoms = (
                len(coords.atoms)
                if hasattr(coords, "atoms")
                else getattr(coords, "natom", None)
            )
            if coord_natoms == len(full_structure.atoms):
                full_structure.coordinates = coords.coordinates
                # Copy box information from coordinate file if available
                if hasattr(coords, "box") and coords.box is not None:
                    full_structure.box = coords.box
                # If no box in coord file but we have box from trajectory, use that
                elif box_info is not None:
                    full_structure.box = box_info
            else:
                warn(
                    f"Coordinate file atom count ({coord_natoms}) doesn't match topology ({len(full_structure.atoms)})"
                )
        # Set box from trajectory if structure still has no box
        if (
            full_structure.box is None or all(v == 0 for v in full_structure.box[:3])
        ) and box_info is not None:
            full_structure.box = box_info
    except (ValueError, Exception) as e:
        warn(f"Could not load structure with parmed: {e}")
        full_structure = None

    # Generate full trajectory
    full_xtc = os.path.join(outdir, "full.xtc")
    full_pdb = os.path.join(outdir, "full.pdb")
    full_gro = os.path.join(outdir, "full.gro")

    if trajectory_file and not all(map(file_exists, [full_xtc, full_pdb])):
        verbose("Generating full trajectory...")
        cppin_full = base_cppin + "_full.in"
        with open(cppin_full, "w") as f:
            f.write(f"parm {topology_file}\n")
            f.write(f"trajin {trajectory_file}\n")
            # Use autoimage only if box is present
            if box_present:
                verbose("Box detected. Using autoimage...")
                f.write("autoimage\n")
            else:
                verbose("No box detected. Skipping autoimage...")
            # RMS fit to first frame using backbone atoms (or all atoms for CG)
            f.write(f"rms first {fit_mask}\n")
            f.write(time_action(sampling_ps))
            f.write(f"trajout {full_xtc} xtc\n")
            f.write(f"trajout {full_pdb} pdb onlyframes 1\n")
            f.write("run\n")
        run_cpptraj(cppin_full, "full")
        fix_pdb_element_symbols(full_pdb)
    elif not trajectory_file:
        verbose("No trajectory file provided, skipping full trajectory generation...")
    else:
        verbose("Full trajectory files already exist, skipping...")

    # Generate full.gro
    if full_structure is not None and not file_exists(full_gro):
        try:
            save_gro(full_structure, full_gro)
            verbose("Generated full.gro")
        except Exception as e:
            warn(f"Could not generate full.gro: {e}")

    # Minimal trajectory (strip water, ions, lipids)
    minimal_xtc = os.path.join(outdir, "minimal.xtc")
    minimal_pdb = os.path.join(outdir, "minimal.pdb")
    minimal_gro = os.path.join(outdir, "minimal.gro")
    strip_mask_minimal = ":" + ",".join(KNOWN_WATER + KNOWN_IONS + KNOWN_LIPIDS)

    if trajectory_file and not all(map(file_exists, [minimal_xtc, minimal_pdb])):
        verbose("Generating minimal trajectory (strip water, ions, lipids)...")
        process_stripped_trajectory(
            topology_file,
            trajectory_file,
            outdir,
            strip_mask_minimal,
            "minimal",
            fit_mask=fit_mask,
            sampling_ps=sampling_ps,
            box_present=box_present,
        )
    elif not trajectory_file:
        verbose(
            "No trajectory file provided, skipping minimal trajectory generation..."
        )

    # Generate minimal.gro
    if full_structure is not None and not file_exists(minimal_gro):
        try:
            # Strip water, ions, and lipids from structure using efficient selection
            strip_residues = set(KNOWN_WATER + KNOWN_IONS + KNOWN_LIPIDS)
            # Select atoms to keep (not in strip list)
            minimal_struct = full_structure[keep_mask(full_structure, strip_residues)]
            save_gro(minimal_struct, minimal_gro)
            verbose("Generated minimal.gro")
        except Exception as e:
            warn(f"Could not generate minimal.gro: {e}")

    # Minimal lipid trajectory (strip water and ions only, keep lipids)
    minlip_gro = os.path.join(outdir, "minimal_lipid.gro")
    strip_mask_minlip = ":" + ",".join(KNOWN_WATER + KNOWN_IONS)

    if has_lipid and trajectory_file:
        minlip_xtc = os.path.join(outdir, "minimal_lipid.xtc")
        minlip_pdb = os.path.join(outdir, "minimal_lipid.pdb")
        if not all(map(file_exists, [minlip_xtc, minlip_pdb])):
            verbose("Lipids detected. Generating minimal_lipid trajectory...")
            process_stripped_trajectory(
                topology_file,
                trajectory_file,
                outdir,
                strip_mask_minlip,
                "minimal_lipid",
                fit_mask=fit_mask,
                sampling_ps=sampling_ps,
                box_present=box_present,
            )

        # Generate minimal_lipid.gro
        if full_structure is not None and not file_exists(minlip_gro):
            try:
                # Strip water and ions only (keep lipids) using efficient selection
                strip_residues = set(KNOWN_WATER + KNOWN_IONS)
                # Select atoms to keep (not in strip list)
                minlip_struct = full_structure[keep_mask(full_structure, strip_residues)]
                save_gro(minlip_struct, minlip_gro)
                verbose("Generated minimal_lipid.gro")
            except Exception as e:
                warn(f"Could not generate minimal_lipid.gro: {e}")
    elif has_lipid:
        verbose(
            "Lipids detected but no trajectory file provided; skipping minimal_lipid."
        )
    else:
        verbose("No lipids detected; skipping minimal_lipid trajectory.")


# --------------------------------------------------
def restore_chain_ids(reference_pdb, target_pdb, output_pdb):
    """Restore chain IDs from reference PDB to target PDB

    Args:
        reference_pdb: Original PDB file with chain information
        target_pdb: GROMACS-generated PDB without chain info
        output_pdb: Output PDB with restored chain IDs
    """

    if not reference_pdb or not os.path.isfile(reference_pdb):
        verbose(f"No reference PDB provided or file doesn't exist: {reference_pdb}")
        return False

    if not os.path.isfile(target_pdb):
        verbose(f"Target PDB doesn't exist: {target_pdb}")
        return False

    # Build chain mapping from reference PDB: resnum -> chain_id
    chain_map = {}
    with open(reference_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    resnum = int(line[22:26].strip())
                    chain_id = line[21:22].strip()
                    if chain_id:  # Only if chain ID exists
                        chain_map[resnum] = chain_id
                except (ValueError, IndexError):
                    continue

    if not chain_map:
        verbose("No chain information found in reference PDB")
        return False

    verbose(f"Found chain info for {len(chain_map)} residues in reference PDB")

    # Apply chain IDs to target PDB
    fixed_lines = []
    fixed_count = 0
    with open(target_pdb, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    resnum = int(line[22:26].strip())
                    if resnum in chain_map:
                        # Replace chain ID (column 21, 0-indexed position 21)
                        new_line = line[:21] + chain_map[resnum] + line[22:]
                        fixed_lines.append(new_line)
                        fixed_count += 1
                    else:
                        fixed_lines.append(line)
                except (ValueError, IndexError):
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

    # Write output
    with open(output_pdb, "w") as f:
        f.writelines(fixed_lines)

    verbose(f"Restored chain IDs for {fixed_count} atoms -> {output_pdb}")
    return True


# --------------------------------------------------
def write_gromacs_bash(
    topfile, trajfile, tpr_file, coord_file, outdir, lipid_present, gmx_path
):
    """Generate GROMACS bash script including PDB outputs"""

    script_path = os.path.join(outdir, "process_gmx.sh")

    if tpr_file:
        topfile = tpr_file

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Create full.gro for consistency across all MD software formats
    full_gro = os.path.join(outdir, "full.gro")
    if coord_file and coord_file.endswith(".gro"):
        shutil.copy(coord_file, full_gro)
    elif topfile.endswith(".gro"):
        shutil.copy(topfile, full_gro)
    elif topfile.endswith(".tpr"):
        cmd = f"{gmx_path} editconf -f {topfile} -o {full_gro}"
        rv, out = subprocess.getstatusoutput(cmd)
        if rv != 0:
            warn(f"Could not convert TPR to GRO: {cmd} failed '{out}'")

    with open(script_path, "w") as f:
        # f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        f.write("#!/usr/bin/env bash\nset -eu\n\n")
        f.write(
            "\n".join(
                [
                    f"GMX={gmx_path}",
                    f"TOP={topfile}",
                    f"TRAJ={trajfile}",
                    f"OUTDIR={outdir}",
                    "",
                ]
            )
        )
        f.write("[[ ! -d $OUTDIR ]] && mkdir -p $OUTDIR\n\n")

        if topfile.endswith(".tpr"):
            # TPR path: detect pre-stripped trajectories, wrap with -pbc mol, dump first frame.
            prestrip_exclude = " ".join(KNOWN_WATER + KNOWN_IONS + KNOWN_LIPIDS)
            f.write("echo '[*] Detecting trajectory atom count'\n")
            f.write(
                "TRAJ_NATOMS=$($GMX dump -f $TRAJ 2>/dev/null "
                "| awk -F'natoms=' '/natoms=/{print $2+0; exit}')\n"
            )
            f.write(
                "TOP_NATOMS=$($GMX dump -s $TOP 2>/dev/null "
                "| awk '/natoms/{print $3+0; exit}')\n"
            )
            f.write("PRESTRIPPED_GROUP=0\n\n")
            f.write('if [ "$TRAJ_NATOMS" -lt "$TOP_NATOMS" ]; then\n')
            f.write(
                '  echo "[*] Pre-stripped trajectory detected ($TRAJ_NATOMS atoms < $TOP_NATOMS in TPR)"\n'
            )
            f.write(
                "  BASE_DETECT=$(echo q | $GMX make_ndx -f $TOP 2>&1 "
                "| grep -E '^[[:space:]]*[0-9]+' | grep 'non-Water' "
                "| head -1 | awk '{print $1}')\n"
            )
            f.write('  [ -z "$BASE_DETECT" ] && BASE_DETECT=0\n')
            f.write(
                f"  $GMX make_ndx -f $TOP -o $OUTDIR/index_prestripped.ndx <<PRESTRIP_EOF\n"
                f"$BASE_DETECT & !r {prestrip_exclude}\n"
                f"q\n"
                f"PRESTRIP_EOF\n\n"
            )
            f.write(
                "  PRESTRIPPED_GROUP=$(grep -c '\\[ .* \\]' $OUTDIR/index_prestripped.ndx)\n"
            )
            f.write("  PRESTRIPPED_GROUP=$((PRESTRIPPED_GROUP - 1))\n")
            f.write("fi\n\n")

            f.write("echo '[*] Wrapping and centering trajectory'\n")
            f.write('if [ "$TRAJ_NATOMS" -lt "$TOP_NATOMS" ]; then\n')
            f.write(
                "  $GMX trjconv -s $TOP -f $TRAJ -o $OUTDIR/full.xtc "
                "-center -pbc mol -ur compact -n $OUTDIR/index_prestripped.ndx <<EOF\n"
                "1\n$PRESTRIPPED_GROUP\nEOF\n\n"
            )
            f.write("else\n")
            f.write(
                "  $GMX trjconv -s $TOP -f $TRAJ -o $OUTDIR/full.xtc "
                "-center -pbc mol -ur compact <<EOF\n"
                "1\n0\nEOF\n\n"
            )
            f.write("fi\n\n")

            f.write("echo '[*] Dumping first wrapped frame to PDB'\n")
            f.write('if [ "$TRAJ_NATOMS" -lt "$TOP_NATOMS" ]; then\n')
            f.write(
                "  $GMX trjconv -s $TOP -f $OUTDIR/full.xtc -o $OUTDIR/full.pdb "
                "-dump 0 -n $OUTDIR/index_prestripped.ndx <<EOF\n"
                "$PRESTRIPPED_GROUP\nEOF\n\n"
            )
            f.write("else\n")
            f.write(
                "  $GMX trjconv -s $TOP -f $OUTDIR/full.xtc -o $OUTDIR/full.pdb "
                "-dump 0 <<EOF\n"
                "0\nEOF\n\n"
            )
            f.write("fi\n\n")

            f.write('if [ "$TRAJ_NATOMS" -lt "$TOP_NATOMS" ]; then\n')
            f.write("  STRUCT_REF=$OUTDIR/full.pdb\n")
            f.write("else\n")
            f.write("  STRUCT_REF=$TOP\n")
            f.write("fi\n\n")

        else:
            # No-TPR path: use cpptraj autoimage (molecule-level PBC, same as -pbc mol)
            # with the coordinate PDB as topology, since cpptraj can't read GRO/TPR.
            if (
                coord_file
                and coord_file.endswith(".pdb")
                and os.path.isfile(coord_file)
            ):
                cpptraj_topo = coord_file
            else:
                cpptraj_topo = os.path.join(outdir, "topology.pdb")
                try:
                    orig_dir = os.getcwd()
                    os.chdir(os.path.dirname(os.path.abspath(topfile)))
                    try:
                        pmd.load_file(topfile).save(
                            cpptraj_topo, format="pdb", overwrite=True
                        )
                    finally:
                        os.chdir(orig_dir)
                    verbose("Generated PDB from GRO for cpptraj topology")
                except Exception as e:
                    warn(f"Could not generate PDB topology for cpptraj: {e}")
                    cpptraj_topo = None

            cppin_wrap = os.path.join(outdir, "cpptraj_wrap.in")
            full_xtc_path = os.path.join(outdir, "full.xtc")
            full_pdb_path = os.path.join(outdir, "full.pdb")
            with open(cppin_wrap, "w") as cf:
                cf.write(f"parm {cpptraj_topo}\n")
                cf.write(f"trajin {trajfile}\n")
                cf.write("autoimage\n")
                cf.write(f"trajout {full_xtc_path} xtc\n")
                cf.write(f"trajout {full_pdb_path} pdb onlyframes 1\n")
                cf.write("run\n")

            f.write("echo '[*] Wrapping trajectory with cpptraj autoimage'\n")
            f.write(f"cpptraj -i {cppin_wrap}\n\n")
            # cpptraj outputs full.pdb atom-for-atom with full.xtc, so use it as reference.
            f.write("STRUCT_REF=$OUTDIR/full.pdb\n\n")

        # Create index groups for minimal (protein-only)
        # First, check if "non-Water" group exists and get its number
        f.write("echo '[*] Detecting available groups'\n")
        f.write(
            "BASE_GROUP_NUM=$(echo q | $GMX make_ndx -f $STRUCT_REF -o /dev/null 2>&1 "
            "| grep -E '^[[:space:]]*[0-9]+' | grep 'non-Water' "
            "| head -1 | awk '{print $1}')\n"
        )
        f.write('if [[ -z "$BASE_GROUP_NUM" ]]; then\n')
        f.write("  BASE_GROUP_NUM=0  # Use System if non-Water not found\n")
        f.write("fi\n\n")

        if lipid_present:
            f.write("echo '[*] Creating index groups for minimal (protein-only)'\n")
            f.write("$GMX make_ndx -f $STRUCT_REF -o $OUTDIR/index_minimal.ndx <<EOF\n")
            f.write(
                f'$BASE_GROUP_NUM & !r {" ".join(KNOWN_IONS + KNOWN_LIPIDS)}\nq\nEOF\n\n'
            )
        else:
            f.write("echo '[*] Creating index groups (protein-only, no lipids)'\n")
            f.write("$GMX make_ndx -f $STRUCT_REF -o $OUTDIR/index_minimal.ndx <<EOF\n")
            f.write(f'$BASE_GROUP_NUM & !r {" ".join(KNOWN_IONS)}\nq\nEOF\n\n')

        # Extract protein-only trajectory (temp)
        # Get the last group number from the index file (the one we just created)
        f.write("echo '[*] Extracting protein-only trajectory (minimal_temp.xtc)'\n")
        f.write("MINIMAL_GROUP_NUM=$(grep -c '\\[ .* \\]' $OUTDIR/index_minimal.ndx)\n")
        f.write(
            "MINIMAL_GROUP_NUM=$((MINIMAL_GROUP_NUM - 1))  # Groups are 0-indexed\n"
        )
        f.write(
            "$GMX trjconv -s $STRUCT_REF -f $OUTDIR/full.xtc -n $OUTDIR/index_minimal.ndx "
            "-o $OUTDIR/minimal_temp.xtc <<EOF\n"
        )
        f.write("$MINIMAL_GROUP_NUM\nEOF\n\n")

        # Extract protein-only structure (temp)
        f.write("echo '[*] Writing temp minimal.pdb from full.pdb'\n")
        f.write(
            "$GMX trjconv -s $STRUCT_REF -f $OUTDIR/full.pdb -n $OUTDIR/index_minimal.ndx "
            "-o $OUTDIR/minimal_temp.pdb -dump 0 <<EOF\n"
        )
        f.write("$MINIMAL_GROUP_NUM\nEOF\n\n")

        # Rotate minimal structure to principal components and then rotate 90 degrees
        f.write("echo '[*] Rotating minimal structure to principal components'\n")
        f.write(
            "echo 1 | $GMX editconf -f $OUTDIR/minimal_temp.pdb "
            "-o $OUTDIR/min_princ.pdb -princ >/dev/null 2>&1\n"
        )
        f.write(
            "$GMX editconf -f $OUTDIR/min_princ.pdb "
            "-o $OUTDIR/minimal.pdb -rotate 0 0 90 >/dev/null 2>&1\n\n"
        )

        # Superimpose minimal trajectory using rotated structure
        f.write("echo '[*] Writing superimposed minimal.xtc'\n")
        f.write(
            "$GMX trjconv -s $OUTDIR/minimal.pdb -f $OUTDIR/minimal_temp.xtc "
            "-o $OUTDIR/minimal.xtc -fit rot+trans <<EOF\n0\n0\nEOF\n\n"
        )

        if os.path.isfile(os.path.join(outdir, "full.tpr")):
            f.write("echo '[*] Writing minimal.tpr'\n")
            f.write(
                "$GMX convert-tpr -s $OUTDIR/full.tpr -o $OUTDIR/minimal.tpr "
                "-n $OUTDIR/index_minimal.ndx <<EOF\n$MINIMAL_GROUP_NUM\nEOF\n\n"
            )
            f.write(
                "$GMX trjconv -s $OUTDIR/minimal.tpr -f $OUTDIR/minimal.xtc "
                "-o $OUTDIR/minimal.gro -dump 0 <<EOF\n0\nEOF\n\n"
            )
        else:
            # Dump frame 0 from the *fitted* minimal trajectory using the fitted ref
            f.write("echo '[*] Writing minimal.gro'\n")
            f.write(
                "$GMX trjconv -s $OUTDIR/minimal.pdb -f $OUTDIR/minimal.xtc "
                "-o $OUTDIR/minimal.gro -dump 0 <<EOF\n0\nEOF\n\n"
            )

        # Clean up temporary files
        # f.write(
        #    "rm -f $OUTDIR/minimal_temp.xtc $OUTDIR/minimal_temp.pdb $OUTDIR/min_princ.pdb\n\n"
        # )

        # Process lipid trajectory if lipids are present
        if lipid_present:
            # Extract protein+lipid trajectory (temp)
            f.write(
                "echo '[*] Extracting protein+lipid trajectory (temp minimal_lipid.xtc)'\n"
            )
            f.write(
                "LIPID_GROUP=$(grep -E '\\[ .* \\]' $OUTDIR/index.ndx | grep -v 'Water' | grep '!' | tail -1 | sed 's/\\[ //;s/ \\]//')\n"
            )
            f.write(
                "$GMX trjconv -s $STRUCT_REF -f $OUTDIR/full.xtc -n $OUTDIR/index.ndx "
                "-o $OUTDIR/minimal_lipid_temp.xtc <<EOF\n"
            )
            f.write("$LIPID_GROUP\nEOF\n\n")

            # Extract protein+lipid structure (temp)
            f.write("echo '[*] Writing temp minimal_lipid.pdb from full.pdb'\n")
            f.write(
                "$GMX trjconv -s $STRUCT_REF -f $OUTDIR/full.pdb -n $OUTDIR/index.ndx "
                "-o $OUTDIR/minimal_lipid_temp.pdb -dump 0 <<EOF\n"
            )
            f.write("$LIPID_GROUP\nEOF\n\n")

            # Rotate minimal_lipid structure to principal components and then rotate 90 degrees
            f.write(
                "echo '[*] Rotating minimal_lipid structure to principal components'\n"
            )
            f.write(
                "echo 1 | $GMX editconf -f $OUTDIR/minimal_lipid_temp.pdb "
                "-o $OUTDIR/minlip_princ.pdb -princ >/dev/null 2>&1\n"
            )
            f.write(
                "$GMX editconf -f $OUTDIR/minlip_princ.pdb "
                "-o $OUTDIR/minimal_lipid.pdb -rotate 0 0 90 >/dev/null 2>&1\n\n"
            )

            # Superimpose minimal_lipid trajectory using rotated structure
            f.write("echo '[*] Writing superimposed minimal_lipid.xtc'\n")
            f.write(
                "$GMX trjconv -s $OUTDIR/minimal_lipid.pdb -f $OUTDIR/minimal_lipid_temp.xtc "
                "-o $OUTDIR/minimal_lipid.xtc -fit rot+trans <<EOF\n0\n0\nEOF\n\n"
            )

            # Clean up temporary files
            f.write(
                "rm -f $OUTDIR/minimal_lipid_temp.xtc $OUTDIR/minimal_lipid_temp.pdb $OUTDIR/minlip_princ.pdb\n\n"
            )

        f.write("echo '[*] GROMACS trajectory processing completed.'\n")

    os.chmod(script_path, 0o755)
    return script_path


# --------------------------------------------------
# Residues
KNOWN_WATER = [
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
KNOWN_IONS = [
    "NA",
    "CL",
    "K",
    "SOD",
    "CLA",
    "NA+",
    "CL-",
    "K+",
    "Mg2+",
    "MG",  # Amber-style ion names
    "Na+",
    "Cl-",  # Case variants
]
KNOWN_LIPIDS = [
    "DPPC",
    "POPC",
    "POPE",
    "DOPC",
    "DLPC",
    "DMPC",
    "DSPC",
    "CHOL",
]


# --------------------------------------------------
def which(exe_name):
    """Find path of executable"""

    cmd = f"which {exe_name}"
    rv, out = getstatusoutput(cmd)
    if rv != 0:
        sys.exit(f"Failed to execute '{cmd}': {out}")

    return out


# --------------------------------------------------
def file_exists(path):
    """File exists and is nonzero size"""

    return os.path.isfile(path) and os.path.getsize(path) > 0


# --------------------------------------------------
def _element_from_atom_name(name):
    """Infer PDB element symbol from atom name; defaults to carbon."""
    if name and name[0] in "CNOSP":
        return name[0]
    if name and name[0] == "H":
        return "H"
    return "C"


# --------------------------------------------------
def fix_pdb_element_symbols(pdb_path):
    """Replace '??' element symbols in a cpptraj-written PDB with inferred elements.

    Occurs when cpptraj cannot determine element from atom name or mass (e.g.
    coarse-grained PSF topologies where all atoms are named 'A').
    """
    if not os.path.isfile(pdb_path):
        return
    lines = []
    changed = 0
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("ATOM") or line.startswith("HETATM")) and len(
                line
            ) > 78:
                if line[76:78] == "??":
                    element = _element_from_atom_name(line[12:16].strip())
                    line = line[:76] + element.rjust(2) + line[78:]
                    changed += 1
            lines.append(line)
    if changed:
        with open(pdb_path, "w") as f:
            f.writelines(lines)
        verbose(
            f"Fixed {changed} unknown element symbols in {os.path.basename(pdb_path)}"
        )


# --------------------------------------------------
if __name__ == "__main__":
    main()
