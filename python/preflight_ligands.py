#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-28
Purpose: Vet every DDD bundle's ligand BEFORE the prod wave imports it.

         Phase B's full --dry-run pass costs ~45 min/bundle and ~5.3 GB of
         disk, and mdr-process only produces the ligand inference on its
         way through. That made the stricter ligand check look inherently
         post-import: you could not know a bundle was bad until you had
         already processed (and, in --no-dry-run, imported) it.

         It is not. mdr-process infers ligands by running "mol_id.py both"
         on the PROCESSED minimal.pdb, but the ligand is the same molecule
         in the bundle's own raw structure file. Verified on 1nju
         (2026-08-28): inference from raw Pro_lig.pdb and from processed
         minimal.pdb give the identical InChIKey
         (QNBSWGBYWJWXKM-GBFVQWPGSA-N), differing only in canonical-SMILES
         atom ordering, which is why the comparison goes through InChI
         layers rather than string equality.

         So the whole corpus can be vetted in ~2-3 hours instead of ~77,
         and nothing suspect ever has to reach prod. Output is a TSV of
         bundle -> verdict (see ligand_check: block / flag / pass), which
         feeds --go-list on the real run.

         Reads nothing but ddd/data/<name>.tgz, which is never modified.
         Touches no database on any server, and never writes into a bundle:
         the element-corrected structure it infers from is a scratch copy
         that is deleted with the temp dir. See
         ligand_check.write_element_corrected for why that correction exists
         and what it deliberately does NOT do.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import toml

import ligand_check
from common import stamp
from process_bundles2 import append_record, load_record

DATA_DIR_DEFAULT = "/media/volume/mdrepo_bd/ddd/data"
WORK_DIR_DEFAULT = "/media/volume/mdrepo_bd/ddd/work"
FIX_SMILES_DEFAULT = "/media/volume/mdrepo_bd/utils/python/fix_ligand_smiles.py"
TABLE_DEFAULT = os.path.expanduser("~/pdbbind_ligand_smiles.tsv")
SCRIPT_DIR_DEFAULT = "/media/volume/mdrepo_bd/simulation-processing/python"
UV_DEFAULT = "/home/user13/.local/bin/uv"
GO_CLASSES_DEFAULT = ("fixed", "already")
METADATA_NAME = "mdrepo-metadata.toml"

# One bundle's inference ran 48s at the slow end; well past that and
# something is wrong with the structure file rather than merely large.
MOL_ID_TIMEOUT = 900


class Args(NamedTuple):
    data_dir: str
    work_dir: str
    survey_tsv: str
    go_classes: Tuple[str, ...]
    record_file: str
    fix_smiles: str
    smiles_table: str
    script_dir: str
    uv: str
    parallel: int
    limit: Optional[int]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Vet DDD ligands from raw structure files, before the "
        "prod wave -- no processing, no DB, no push.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--data-dir", default=DATA_DIR_DEFAULT,
                        metavar="DIR", help="Local <name>.tgz source")
    parser.add_argument("-w", "--work-dir", default=WORK_DIR_DEFAULT,
                        metavar="DIR",
                        help="Parent for the per-bundle scratch dirs this "
                        "creates and removes; bundles unpacked here are NOT "
                        "reused, since their inference predates the "
                        "element-column correction")
    parser.add_argument("-s", "--survey-tsv", required=True, metavar="TSV",
                        help="survey_bundles.py output")
    parser.add_argument("--go-classes", nargs="+",
                        default=list(GO_CLASSES_DEFAULT), metavar="CLASS")
    parser.add_argument("-r", "--record", default=None, metavar="PATH",
                        help="Append-only verdict TSV (default: "
                        "<work-dir>/../logs/preflight/ligands.tsv)")
    parser.add_argument("--fix-smiles", default=FIX_SMILES_DEFAULT,
                        metavar="PATH")
    parser.add_argument("--smiles-table", default=TABLE_DEFAULT,
                        metavar="TSV")
    parser.add_argument("--script-dir", default=SCRIPT_DIR_DEFAULT,
                        metavar="DIR", help="Where mol_id.py lives")
    parser.add_argument("--uv", default=UV_DEFAULT, metavar="PATH")
    parser.add_argument("-p", "--parallel", type=int, default=12,
                        metavar="INT",
                        help="This touches no shared resource, so it is "
                        "bounded by local CPU only")
    parser.add_argument("--limit", type=int, default=None, metavar="INT")
    args = parser.parse_args()

    record = args.record or os.path.join(
        os.path.dirname(os.path.abspath(args.work_dir)),
        "logs", "preflight", "ligands.tsv",
    )
    return Args(
        args.data_dir, args.work_dir, args.survey_tsv,
        tuple(args.go_classes), record, args.fix_smiles, args.smiles_table,
        args.script_dir, args.uv, args.parallel, args.limit,
    )


# --------------------------------------------------
def eligible_bundles(args: Args) -> List[str]:
    """Survey bundles in the go-classes, minus anything already recorded."""

    record = load_record(args.record_file)
    names = []
    with open(args.survey_tsv) as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            if parts[1] in args.go_classes and parts[0] not in record:
                names.append(parts[0])
    return sorted(names)


# --------------------------------------------------
def extract_small(tgz_path: str, dest: str) -> Optional[str]:
    """Pull just the metadata and the structure PDB out of a bundle tarball
    into `dest`, without unpacking its multi-GB trajectories. Returns the
    directory holding them, or None if the metadata was not found.

    Streamed ("r|*") and stopped as soon as both are in hand, so a bundle
    whose small files sit early costs a fraction of reading the archive.
    """

    found_meta = False
    found_pdb = False

    with tarfile.open(tgz_path, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            wanted = base == METADATA_NAME or (
                base.endswith(".pdb") and not base.endswith("_nochain.pdb")
            )
            if not wanted:
                continue

            # Flatten: the bundle descends one directory, and the caller
            # only ever looks for files by basename.
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with open(os.path.join(dest, base), "wb") as out:
                shutil.copyfileobj(handle, out)

            found_meta = found_meta or base == METADATA_NAME
            found_pdb = found_pdb or base.endswith(".pdb")
            if found_meta and found_pdb:
                break

    return dest if found_meta else None


# --------------------------------------------------
def fill_smiles(local_dir: str, args: Args) -> None:
    """Run the same SMILES fill the real driver does, so the declared value
    compared here is the one that would actually be imported."""

    subprocess.run(
        [sys.executable, args.fix_smiles, local_dir,
         "--table", args.smiles_table],
        capture_output=True, text=True, timeout=300,
    )


# --------------------------------------------------
def infer_from_structure(
    pdb_path: str, out_json: str, args: Args
) -> List[str]:
    """mol_id.py's inferred ligand SMILES for one structure file.

    Same invocation mdr-process uses (process.rs get_inferred_ligands),
    pointed at the raw structure rather than the processed minimal.pdb.
    """

    proc = subprocess.run(
        [args.uv, "run", os.path.join(args.script_dir, "mol_id.py"),
         "both", pdb_path, "--outfile", out_json],
        capture_output=True, text=True, cwd=args.script_dir,
        timeout=MOL_ID_TIMEOUT,
    )
    if not os.path.isfile(out_json):
        # An APO structure legitimately has no ligand; mol_id.py raises.
        raise RuntimeError(
            " ".join((proc.stderr or "no ligand inferred").split())[:200]
        )

    import json
    with open(out_json, encoding="utf-8") as fh:
        inferred = json.load(fh)
    return [
        lig.get("structure", {}).get("smiles") for lig in inferred
        if lig.get("structure", {}).get("smiles")
    ]


# --------------------------------------------------
def declared_smiles(meta_path: str) -> Tuple[Sequence[str], Optional[str]]:
    """The TOML's declared ligand SMILES and its structure file name."""

    meta = toml.load(meta_path)
    declared = [
        lig.get("smiles") for lig in meta.get("ligands", [])
        if lig.get("smiles")
    ]
    return declared, meta.get("structure_file_name")


# --------------------------------------------------
def check_one(args: Args, name: str) -> Tuple[str, str, str]:
    """Vet one bundle. Returns (name, verdict, detail)."""

    # No cached fast path. processed/inferred_ligands.json on disk was
    # computed by mdr-process from the UNCORRECTED structure file, so
    # reusing it would reproduce the very bug this pass exists to remove.
    # Every bundle is re-inferred from an element-corrected scratch copy.
    tgz = os.path.join(args.data_dir, f"{name}.tgz")
    if not os.path.isfile(tgz):
        return name, "error", "no tarball"

    tmp = tempfile.mkdtemp(prefix=f"preflight-{name}-", dir=args.work_dir)
    try:
        try:
            local = extract_small(tgz, tmp)
        except (tarfile.TarError, EOFError, OSError) as e:
            return name, "error", " ".join(f"corrupt tarball: {e}".split())[:300]
        if local is None:
            return name, "error", "no mdrepo-metadata.toml in tarball"

        fill_smiles(local, args)
        try:
            declared, structure = declared_smiles(
                os.path.join(local, METADATA_NAME)
            )
        except (toml.TomlDecodeError, OSError) as e:
            return name, "error", f"unreadable toml: {e}"[:300]

        pdb = os.path.join(local, structure or "Pro_lig.pdb")
        if not os.path.isfile(pdb):
            return name, "error", f"structure file missing: {structure}"

        # Infer from an element-corrected SCRATCH copy: this corpus records
        # ligand chlorines as element C, and OpenBabel trusts that column, so
        # inferring from the file as-written silently loses every halogen.
        # `pdb` itself is untouched -- see ligand_check.write_element_corrected.
        corrected = os.path.join(tmp, "corrected.pdb")
        fixes = ligand_check.write_element_corrected(pdb, corrected)

        try:
            candidates = infer_from_structure(
                corrected if fixes else pdb,
                os.path.join(tmp, "inferred.json"), args,
            )
        except subprocess.TimeoutExpired:
            return name, "error", "mol_id.py timed out"
        except (RuntimeError, ValueError, OSError) as e:
            return name, "error", " ".join(f"inference failed: {e}".split())[:300]

        verdict, detail = ligand_check.check(declared, candidates, pdb)
        if fixes:
            detail = (f"[element-corrected: {' '.join(fixes[:3])}] "
                      f"{detail}").strip()
        return name, verdict, detail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------
def main() -> None:
    args = get_args()
    os.makedirs(os.path.dirname(args.record_file), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)

    bundles = eligible_bundles(args)
    if args.limit:
        bundles = bundles[: args.limit]

    print(f"{stamp()} vetting {len(bundles):,} bundle(s), "
          f"{args.parallel} at a time -- no DB, no push", flush=True)

    tally: Dict[str, int] = {}
    done = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.parallel
    ) as pool:
        futures = {pool.submit(check_one, args, b): b for b in bundles}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                name, verdict, detail = future.result()
            except Exception as e:
                verdict, detail = "error", f"{type(e).__name__}: {e}"
            append_record(args.record_file, name, verdict, detail)
            tally[verdict] = tally.get(verdict, 0) + 1
            done += 1
            if done % 25 == 0 or done == len(bundles):
                print(f"{stamp()} {done:,}/{len(bundles):,} "
                      f"({', '.join(f'{k}={v}' for k, v in sorted(tally.items()))})",
                      flush=True)

    print(f"{stamp()} finished: " +
          ", ".join(f"{k}={v}" for k, v in sorted(tally.items())), flush=True)


if __name__ == "__main__":
    main()
