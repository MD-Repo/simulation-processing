#!/usr/bin/env python3

import os
import argparse
import shutil
import warnings
import subprocess
import pytraj as pt
import parmed as pmd
import sys
from subprocess import getstatusoutput

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
def find_matching_struct_for_top(top_file, ext):
    """Find a structure file (.tpr or .gro) alongside a .top file.

    Args:
        top_file: Path to .top topology file
        ext: Extension to search for (e.g. ".tpr" or ".gro")

    Returns:
        Path to matching file, or None if not found
    """
    top_dir = os.path.dirname(top_file) or "."
    top_stem = os.path.splitext(os.path.basename(top_file))[0]

    # Try exact stem match first
    exact_match = os.path.join(top_dir, top_stem + ext)
    if os.path.isfile(exact_match):
        return exact_match

    # Fall back to any file with that extension in the same directory
    candidates = [f for f in os.listdir(top_dir) if f.endswith(ext)]
    if len(candidates) == 1:
        return os.path.join(top_dir, candidates[0])

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
            tpr_file = find_matching_struct_for_top(topology_file, ".tpr")
            if tpr_file:
                verbose(f"Found TPR file alongside .top: {tpr_file}")
            else:
                gro_file = find_matching_struct_for_top(topology_file, ".gro")
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


# --------------------------------------------------
def verbose(msg):
    print(f"[*] {msg}")


# --------------------------------------------------
def warn(msg):
    print(f"[!] {msg}", file=sys.stderr)


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
def has_box(frame):
    if getattr(frame, "box", None) is None:
        return False
    vals = list(frame.box)
    if all(abs(v) < 1e-6 for v in vals):
        return False
    return True


# --------------------------------------------------
def process_stripped_trajectory(
    topology_file, trajectory_file, outdir, strip_mask, prefix, fit_mask="@CA,C,N"
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
        f.write(f"strip {strip_mask}\n")
        f.write(f"principal {fit_mask} dorotation\n")
        f.write("rotate z 90\n")
        f.write("rotate x -90\n")
        f.write(f"trajout {ref_pdb} pdb\n")
        f.write("run\n")
    rv = os.system(f"cpptraj -i {cppin_ref}")
    if rv != 0:
        warn(f"cpptraj {prefix} reference failed with exit code {rv}")
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
        f.write(f"strip {strip_mask}\n")
        f.write(f"rms ref [rotref] {fit_mask}\n")
        f.write(f"trajout {output_xtc} xtc\n")
        f.write(f"trajout {output_pdb} pdb onlyframes 1\n")
        f.write("run\n")
    rv = os.system(f"cpptraj -i {cppin_traj}")
    if rv != 0:
        warn(f"cpptraj {prefix} failed with exit code {rv}")
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
    os.system(f"cpptraj -i {cppin_frame0} > /dev/null 2>&1")

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
            if has_box(first_frame):
                verbose("Box detected. Using autoimage...")
                f.write("autoimage\n")
            else:
                verbose("No box detected. Skipping autoimage...")
            # RMS fit to first frame using backbone atoms (or all atoms for CG)
            f.write(f"rms first {fit_mask}\n")
            f.write(f"trajout {full_xtc} xtc\n")
            # f.write(f"trajout {full_cif} cif onlyframes 1\n")
            f.write(f"trajout {full_pdb} pdb onlyframes 1\n")
            f.write("run\n")
        rv = os.system(f"cpptraj -i {cppin_full}")
        if rv != 0:
            warn(f"cpptraj full failed with exit code {rv}")
    else:
        verbose("Full trajectory files already exist, skipping...")

    # Generate full.gro from structure
    if structure is not None and not file_exists(full_gro):
        try:
            if os.path.isfile(full_gro):
                os.remove(full_gro)
            structure.save(full_gro, format="gro")
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
        )

    # Generate minimal.gro from structure
    if structure is not None and not file_exists(minimal_gro):
        try:
            # Strip water, ions, and lipids from structure using efficient selection
            strip_residues = set(KNOWN_WATER + KNOWN_IONS + KNOWN_LIPIDS)
            # Select atoms to keep (not in strip list)
            keep_indices = [
                i
                for i, atom in enumerate(structure.atoms)
                if atom.residue.name not in strip_residues
            ]
            minimal_struct = structure[keep_indices]
            if os.path.isfile(minimal_gro):
                os.remove(minimal_gro)
            minimal_struct.save(minimal_gro, format="gro")
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
            )

        # Generate minimal_lipid.gro from structure
        if structure is not None and not file_exists(minlip_gro):
            try:
                # Strip water and ions only (keep lipids) using efficient selection
                strip_residues = set(KNOWN_WATER + KNOWN_IONS)
                # Select atoms to keep (not in strip list)
                keep_indices = [
                    i
                    for i, atom in enumerate(structure.atoms)
                    if atom.residue.name not in strip_residues
                ]
                minlip_struct = structure[keep_indices]
                if os.path.isfile(minlip_gro):
                    os.remove(minlip_gro)
                minlip_struct.save(minlip_gro, format="gro")
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
    rv = os.system(f"cpptraj -i {cppin_frame0}")
    if rv != 0:
        warn(f"cpptraj frame extraction failed with exit code {rv}")
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
    os.system(f"cpptraj -i {cppin_conf} > /dev/null 2>&1")
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
            f.write(f"trajout {full_xtc} xtc\n")
            f.write(f"trajout {full_pdb} pdb onlyframes 1\n")
            f.write("run\n")
        rv = os.system(f"cpptraj -i {cppin_full}")
        if rv != 0:
            warn(f"cpptraj full failed with exit code {rv}")
        fix_pdb_element_symbols(full_pdb)
    elif not trajectory_file:
        verbose("No trajectory file provided, skipping full trajectory generation...")
    else:
        verbose("Full trajectory files already exist, skipping...")

    # Generate full.gro
    if full_structure is not None and not file_exists(full_gro):
        try:
            if os.path.isfile(full_gro):
                os.remove(full_gro)
            full_structure.save(full_gro, format="gro")
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
            keep_indices = [
                i
                for i, atom in enumerate(full_structure.atoms)
                if atom.residue.name not in strip_residues
            ]
            minimal_struct = full_structure[keep_indices]
            if os.path.isfile(minimal_gro):
                os.remove(minimal_gro)
            minimal_struct.save(minimal_gro, format="gro")
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
            )

        # Generate minimal_lipid.gro
        if full_structure is not None and not file_exists(minlip_gro):
            try:
                # Strip water and ions only (keep lipids) using efficient selection
                strip_residues = set(KNOWN_WATER + KNOWN_IONS)
                # Select atoms to keep (not in strip list)
                keep_indices = [
                    i
                    for i, atom in enumerate(full_structure.atoms)
                    if atom.residue.name not in strip_residues
                ]
                minlip_struct = full_structure[keep_indices]
                if os.path.isfile(minlip_gro):
                    os.remove(minlip_gro)
                minlip_struct.save(minlip_gro, format="gro")
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
