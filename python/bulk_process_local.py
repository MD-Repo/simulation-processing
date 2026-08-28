#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-27
Purpose: Phase B/D of the ddd/HANDOFF survey-first plan (see
         ddd/HANDOFF/README.md and the plan this implements). Where
         process_bundles2.py claims ONE bundle at a time from IRODS,
         processes it, and deletes the local copy to fit a disk-constrained
         box, this drives MANY bundles in parallel against tarballs already
         local at ddd/data/ -- this host has 26 TB free, so there's no more
         reason to fetch-and-delete one at a time.

         Two modes, controlled by --dry-run / --no-dry-run:

           --dry-run (default): runs "mdr-process process --dry-run" per
           bundle. Confirmed by reading mdr-process/src/process.rs: this
           executes the FULL pipeline (trajectory conversion, tarring,
           BLAST, ligand inference/checking) and only skips the DB import
           and file push. This is Phase B -- "process everything, find
           every problem, touch nothing" -- not a cheap stub.

           --no-dry-run: the real push. This is Phase D, run ONCE, only
           after ddd/HANDOFF's triage (Phase C) has settled which bundles
           are going. Unlike Phase B, this writes to the production DB the
           live ticket queue also writes to, so keep --parallel modest here
           even though the box has cores to spare -- see the plan's Phase D
           note.

         No drain_process_queue.py flock and no md_process_job pending-work
         check: this host doesn't share CPU/IRODS load with the
         ticket-processing box, so there's nothing here to yield to. (This
         was a decision, not an oversight -- see the plan.)

         Disk accounting, corrected 2026-08-28: the "keep everything, disk
         is no longer the constraint" design this script launched with was
         wrong. A finished bundle costs ~5.3 GB, not the ~2.2 GB estimated,
         so the full 6,784-bundle corpus wants ~35 TB against 23.8 TB free
         -- it would have hit ENOSPC at ~79% of the way through. So a
         SUCCEEDED bundle is now reaped down to its small artifacts (see
         KEEP_TOP / KEEP_PROCESSED): the multi-GB rep_*/ trees, the
         processed tars and the unpacked source trajectories go, and the
         metadata, the JSON/TSV evidence and Pro_lig.pdb stay. That is
         ~2.6 MB/bundle, ~18 GB for the whole corpus.

         FAILED bundles are handled by kind. A PUSH failure is kept: it is
         retryable for the price of a re-push, since push_sim_files.py skips
         any file whose md5 already matches, and reaping it forces a ~45
         minute reprocess instead. A DATA failure is reaped like a success --
         it needs the data fixed before a retry means anything. Keeping the
         push failures is affordable only because the circuit breaker bounds
         how many can pile up.

         Nothing is lost by reaping: ddd/data/<name>.tgz is untouched, so
         unpack_local() rebuilds any bundle from source, and Phase D
         re-processes from scratch regardless (in this tool processing IS
         the push -- there is no separate push step to feed).

         The IRODS-mirrored source tarball at ddd/data/<name>.tgz is NEVER
         touched, matching process_bundles2.py's own invariant for the
         original IRODS collection.

         Reuses process_bundles2.py's run_mdr_process() (timeout / kill /
         log-tail handling) and load_record()/append_record() (the
         append-only TSV retry mechanism) rather than reimplementing them.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import toml

import ligand_check
from common import stamp
from process_bundles2 import append_record, load_record, run_mdr_process

DATA_DIR_DEFAULT = "/media/volume/mdrepo_bd/ddd/data"
WORK_DIR_DEFAULT = "/media/volume/mdrepo_bd/ddd/work"
FIX_SMILES_DEFAULT = "/media/volume/mdrepo_bd/utils/python/fix_ligand_smiles.py"
TABLE_DEFAULT = os.path.expanduser("~/pdbbind_ligand_smiles.tsv")
GO_CLASSES_DEFAULT = ("fixed", "already")

METADATA_NAME = "mdrepo-metadata.toml"
SCRIPT_DIR_DEFAULT = "/media/volume/mdrepo_bd/simulation-processing/python"
UV_DEFAULT = "/home/user13/.local/bin/uv"
MOL_ID_TIMEOUT = 900

# mdr-process/src/import.rs:46-47. lead_contributor_id() short-circuits on
# this exact ORCID and resolves the owner by USERNAME instead of by ORCID,
# so this is the supported way to attribute a bulk import to the admin user
# rather than a fake person. Verified on prod 2026-08-28: mdrepo_admin is
# md_user id 1 and already owns 68,184 of 90,290 simulations.
ADMIN_ORCID = "0000-0000-0000-0000"
COLLECTION_DEFAULT = "Dissociation Dynamic Database"

# libmdrepo/src/metadata.rs validates short_description at 300 chars. DDD's
# own generator templates the full IUPAC ligand name into it, which blows
# the cap for ~3.5% of the corpus (30 of the first 886, every one of them
# this same cause). The full text is already in `description`, which is
# uncapped, so truncating here loses nothing.
MAX_SHORT_DESCRIPTION = 300

# What survives the reap of a SUCCEEDED bundle. Everything else under the
# bundle directory goes. Keep the cheap evidence -- metadata, provenance,
# the ligand inference Phase C reads, the BLAST hits, the thumbnail -- and
# Pro_lig.pdb, which the MDR-40 element-column check needs.
KEEP_TOP = frozenset({
    METADATA_NAME,
    f"{METADATA_NAME}.orig",
    "Pro_lig.pdb",
    "Pro_lig.pdb.md5",
})
KEEP_PROCESSED = frozenset({
    "inferred_ligands.json",
    "import.json",
    "duration.json",
    "rmsd_rmsf.json",
    "sequence.fa",
    "thumbnail.png",
    "blast.isoform.tsv",
    "blast.swissprot.tsv",
    "blast.trembl.tsv",
})



class Args(NamedTuple):
    data_dir: str
    survey_tsv: str
    go_classes: Tuple[str, ...]
    work_dir: str
    log_dir: str
    record_file: str
    fix_smiles: str
    smiles_table: str
    server: str
    dry_run: bool
    num_threads: int
    parallel: int
    limit: Optional[int]
    collection: Optional[str]
    orcid: Optional[str]
    reap: bool
    verify: bool
    keep_unverified: bool
    blast_num_threads: Optional[int]
    transfer_threads: Optional[int]
    script_dir: str
    uv: str
    keep_failed: bool
    max_consecutive_faults: int
    fault_wait: int
    fault_window: int
    max_window_faults: int


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Bulk-drive mdr-process over local ddd/data/ bundles, "
        "many at once, instead of process_bundles2.py's one-at-a-time "
        "IRODS fetch-and-delete.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--data-dir", default=DATA_DIR_DEFAULT,
                        metavar="DIR", help="Local <name>.tgz source")
    parser.add_argument(
        "-s", "--survey-tsv", required=True, metavar="TSV",
        help="survey_bundles.py output: bundle, classification, detail",
    )
    parser.add_argument(
        "--go-classes", nargs="+", default=list(GO_CLASSES_DEFAULT),
        metavar="CLASS",
        help="Only process bundles whose survey classification is one of "
        "these (everything else -- no_smiles, name_mismatch, etc -- is "
        "Phase C's contributor-report track, not this driver's job)",
    )
    parser.add_argument("--work-dir", default=WORK_DIR_DEFAULT, metavar="DIR",
                        help="Scratch/kept unpacked bundles")
    parser.add_argument("--log-dir", default=None, metavar="DIR",
                        help="Per-bundle mdr-process debug logs "
                        "(default: <work-dir>/../logs/<dry-run|prod>)")
    parser.add_argument("--record", default=None, metavar="PATH",
                        help="Append-only outcome TSV "
                        "(default: <log-dir>/processed.tsv)")
    parser.add_argument("--fix-smiles", default=FIX_SMILES_DEFAULT,
                        metavar="PATH")
    parser.add_argument("--smiles-table", default=TABLE_DEFAULT,
                        metavar="TSV")
    parser.add_argument("--server", choices=["staging", "prod"],
                        default="prod",
                        help="Passed to mdr-process. Only meaningful for "
                        "--no-dry-run; a dry run touches no DB either way")
    dry = parser.add_mutually_exclusive_group()
    dry.add_argument("--dry-run", dest="dry_run", action="store_true",
                     default=True, help="Phase B: validate only (default)")
    dry.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                     help="Phase D: the real push. Use only on the "
                     "triaged, vetted set -- see ddd/HANDOFF")
    parser.add_argument("--num-threads", type=int, default=4, metavar="INT",
                        help="mdr-process's GLOBAL -t: the size of its rayon "
                        "pool, which is what parallelises the per-replicate "
                        "trajectory work (process.rs:113). Bundles run up to "
                        "50 replicates, so this is the main intra-bundle "
                        "knob. Load-bearing -- see RUNBOOK.md's MDR-33 note")
    parser.add_argument("--blast-num-threads", type=int, default=None,
                        metavar="INT",
                        help="blastp -num_threads. mdr-process defaults to "
                        "2. BLAST runs SERIALLY within a bundle (swissprot, "
                        "then isoform/trembl only if needed), so the peak "
                        "cost is --parallel x this, not more. Omit to leave "
                        "mdr-process's default alone")
    parser.add_argument(
        "--transfer-threads", type=int, default=0, metavar="INT",
        help="IRODS streams for a single put. 0 means 'let "
        "python-irodsclient decide' (3 above 32 MB, fewer below), which is "
        "push_sim_files.py's own considered default -- see e4363b9. It has "
        "to be passed EXPLICITLY because mdr-process always sends its own "
        "default of 3, which otherwise silently overrides push's 0 and "
        "forces three streams onto every small file. Concurrent connections "
        "are roughly --parallel x --threads x this, against a CyVerse "
        "ceiling of ~500 shared with every other user. NOTE: this was NOT "
        "what lost 7 of 8 bundles on 2026-08-28 -- that was this host being "
        "absent from CyVerse's VIP list and throttled to ~1 MB/s, which no "
        "thread setting can help. Changed anyway because the override is "
        "real and defeats push's deliberate choice",
    )
    parser.add_argument("--parallel", type=int, default=8, metavar="INT",
                        help="Bundles run concurrently. Phase B can push "
                        "this high (local CPU is the only constraint); "
                        "Phase D should stay modest -- it shares the prod "
                        "DB with the live ticket queue")
    parser.add_argument("--limit", type=int, default=None, metavar="INT",
                        help="Process only the first N eligible bundles")
    parser.add_argument(
        "--keep-failed", action="store_true",
        help="Keep a FAILED bundle's bulk files. Off by default: a failed "
        "bundle costs the same ~5.3 GB as a good one, and an outage fails "
        "every bundle, so keeping them all fills the volume long before the "
        "run ends. The metadata, structure and any small artifacts are kept "
        "either way, and the tarball re-extracts",
    )
    # Consecutive, not total, for merge_replicate_groups.py's reason: a
    # steady trickle of unrelated data failures is not an outage, but five
    # in a row is the server being gone.
    parser.add_argument(
        "--max-consecutive-faults", type=int, default=5, metavar="INT",
        help="Consecutive IRODS/push faults that mean the server is gone "
        "rather than flaky, triggering a wait (0 = never react)",
    )
    # The consecutive rule alone is not enough. On 2026-08-28 batch 1 lost
    # 7 of 8 bundles to IRODS and never tripped it: four faults, then one
    # success reset the count, then three more. With several workers in
    # flight a single stale success lands between failures and hides an
    # outage indefinitely. So also trip on a RATE over a recent window,
    # which no interleaving can mask.
    parser.add_argument(
        "--fault-window", type=int, default=10, metavar="INT",
        help="How many recent bundles the rate rule looks at (0 = off)",
    )
    parser.add_argument(
        "--max-window-faults", type=int, default=5, metavar="INT",
        help="IRODS faults within --fault-window that trip the breaker. A "
        "healthy run fails ~3.5%% of bundles, so 5 in 10 is far outside "
        "normal and cannot be produced by interleaving",
    )
    parser.add_argument(
        "--fault-wait", type=int, default=3600, metavar="SEC",
        help="How long to wait for IRODS to come back before ending the run "
        "(0 = stop as soon as the fault limit is hit)",
    )
    parser.add_argument("--script-dir", default=SCRIPT_DIR_DEFAULT,
                        metavar="DIR", help="Where mol_id.py lives")
    parser.add_argument("--uv", default=UV_DEFAULT, metavar="PATH")
    parser.add_argument(
        "--collection", default=COLLECTION_DEFAULT, metavar="NAME",
        help="Put every bundle in this collection via the TOML's "
        "'collections' field. Auto-created once under the owning user "
        "(md_collection is unique on user_id+name); '' to disable",
    )
    parser.add_argument(
        "--orcid", default=ADMIN_ORCID, metavar="ORCID",
        help="Rewrite lead_contributor_orcid to this. The default is "
        "mdr-process's ADMIN_ORCID sentinel, which resolves to the "
        "mdrepo_admin user; '' to leave the submitter's value alone",
    )
    reap = parser.add_mutually_exclusive_group()
    reap.add_argument("--reap", dest="reap", action="store_true", default=True,
                      help="Delete a SUCCEEDED bundle's bulk files, keeping "
                      "its small artifacts (default). Failures are always "
                      "kept for triage")
    reap.add_argument("--no-reap", dest="reap", action="store_false",
                      help="Keep everything. Needs ~35 TB for the full "
                      "corpus -- see the module docstring")
    ver = parser.add_mutually_exclusive_group()
    ver.add_argument("--verify", dest="verify", action="store_true",
                     default=True,
                     help="Phase C's stricter ligand check before reaping "
                     "(default). A bundle that fails is recorded as "
                     "'done-unverified' and is NOT reaped")
    ver.add_argument("--no-verify", dest="verify", action="store_false",
                     help="Reap on mdr-process's own exit code alone")
    parser.add_argument(
        "--keep-unverified", action="store_true",
        help="Also skip the reap for a bundle that failed the Phase C "
        "check. Off by default ON PURPOSE: the check's own evidence "
        "(inferred_ligands.json and the TOML) SURVIVES the reap, so the "
        "verdict stays reproducible without it -- and at the measured "
        "85%% unverified rate, keeping those would want ~30 TB and put "
        "the ENOSPC problem straight back",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.work_dir)), "logs",
        "dryrun" if args.dry_run else args.server,
    )
    record_file = args.record or os.path.join(log_dir, "processed.tsv")

    return Args(
        args.data_dir, args.survey_tsv, tuple(args.go_classes),
        args.work_dir, log_dir, record_file, args.fix_smiles,
        args.smiles_table, args.server, args.dry_run, args.num_threads,
        args.parallel, args.limit, args.collection or None,
        args.orcid or None, args.reap, args.verify, args.keep_unverified,
        args.blast_num_threads, args.transfer_threads,
        args.script_dir, args.uv, args.keep_failed,
        args.max_consecutive_faults, args.fault_wait,
        args.fault_window, args.max_window_faults,
    )


# --------------------------------------------------
def eligible_bundles(args: Args) -> list:
    """Bundle names from the survey whose classification is in go_classes,
    minus anything already in the record file."""

    record = load_record(args.record_file)
    names = []
    with open(args.survey_tsv) as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            bundle, classification = parts[0], parts[1]
            if classification in args.go_classes and bundle not in record:
                names.append(bundle)
    return sorted(names)


# --------------------------------------------------
def unpack_local(name: str, data_dir: str, work_dir: str) -> str:
    """Unpack ddd/data/<name>.tgz into work_dir, return the directory to
    process. The source tarball is left untouched -- see module docstring.

    Same "well-formed bundle descends one directory" handling as
    process_bundles2.py's fetch_bundle(), and the same filter="data" for
    extracting a contributor-supplied archive.
    """

    os.makedirs(work_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, f"{name}.tgz")
    dest = os.path.join(work_dir, name)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    with tarfile.open(tgz_path, mode="r:*") as tar:
        tar.extractall(dest, filter="data")

    entries = [e for e in os.listdir(dest) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest, entries[0])):
        return os.path.join(dest, entries[0])
    return dest


# --------------------------------------------------
def fix_smiles(local_dir: str, args: Args) -> Tuple[bool, str]:
    """Real (non-dry-run) SMILES fill. Returns (ok, detail)."""

    proc = subprocess.run(
        [sys.executable, args.fix_smiles, local_dir,
         "--table", args.smiles_table],
        capture_output=True, text=True,
    )
    detail = " ".join((proc.stdout or "").split())[:300]
    return proc.returncode == 0, detail


# --------------------------------------------------
def truncate_short_description(text: str, limit: int) -> str:
    """Shorten to `limit` characters on a word boundary.

    The result is a prefix of a string that was already valid inside TOML
    double quotes, so it needs no re-escaping -- except that cutting must
    not leave a dangling backslash, which would escape the closing quote.
    """

    if len(text) <= limit:
        return text

    cut = text[: limit - 3]
    if " " in cut[limit // 2:]:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip("\\").rstrip(" ,;:-") + "..."


# --------------------------------------------------
def retarget_metadata(local_dir: str, args: Args) -> List[str]:
    """Point the bundle at the admin user, put it in the target collection,
    and bring an over-long short_description inside the 300-char cap.
    Returns a list of what changed.

    Line-based, not a TOML round trip, for fix_ligand_smiles.py's stated
    reason: re-serialising reorders keys and drops the provenance comments
    that script leaves behind, so every line reads as changed when one key
    did. fix_smiles() has already taken the .orig backup by the time we get
    here and it never overwrites an existing one, so .orig still holds the
    submitter's true original rather than our edit.
    """

    path = os.path.join(local_dir, METADATA_NAME)
    if not os.path.isfile(path):
        return []

    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    changed: List[str] = []
    orcid_at = None

    for num, line in enumerate(lines):
        if args.orcid and re.match(r"\s*lead_contributor_orcid\s*=", line):
            orcid_at = num
            if f'"{args.orcid}"' not in line:
                lines[num] = f'lead_contributor_orcid = "{args.orcid}"'
                changed.append("orcid")
            continue

        match = re.match(r'\s*short_description\s*=\s*"(.*)"\s*$', line)
        if match and len(match.group(1)) > MAX_SHORT_DESCRIPTION:
            short = truncate_short_description(
                match.group(1), MAX_SHORT_DESCRIPTION
            )
            lines[num] = f'short_description = "{short}"'
            changed.append("short_description")

    # Top-level keys have to precede the first [table], so anchor the
    # insert to a key we know sits at the top rather than appending.
    has_collections = any(
        re.match(r"\s*collections\s*=", line) for line in lines
    )
    if args.collection and not has_collections and orcid_at is not None:
        lines.insert(
            orcid_at + 1,
            f"# collections added by bulk_process_local.py for the DDD "
            f"prod wave; not supplied by the submitter\n"
            f'collections = ["{args.collection}"]',
        )
        changed.append("collections")

    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    return changed



# --------------------------------------------------
def verify_ligands(local_dir: str) -> Tuple[str, str]:
    """Re-check a processed bundle against ligand_check's bar. Returns
    (verdict, detail) -- see ligand_check for why the bar is connectivity
    and not exact_match.

    This reads processed/inferred_ligands.json, which mdr-process wrote on
    its way through, so it is nearly free. preflight_ligands.py has already
    reached the same verdict from the raw structure file before the run
    started; this is the belt-and-braces re-check on what was actually
    processed, and the two disagreeing is itself worth knowing about.
    """

    inferred_path = os.path.join(
        local_dir, "processed", "inferred_ligands.json"
    )
    if not os.path.isfile(inferred_path):
        return ligand_check.BLOCK, "no inferred_ligands.json"

    try:
        with open(inferred_path, encoding="utf-8") as fh:
            inferred = json.load(fh)
        meta = toml.load(os.path.join(local_dir, METADATA_NAME))
    except (OSError, ValueError, toml.TomlDecodeError) as e:
        return ligand_check.BLOCK, f"unreadable: {type(e).__name__}: {e}"

    declared = [
        lig.get("smiles") for lig in meta.get("ligands", [])
        if lig.get("smiles")
    ]
    candidates = [
        lig.get("structure", {}).get("smiles") for lig in inferred
        if lig.get("structure", {}).get("smiles")
    ]
    structure = meta.get("structure_file_name")

    return ligand_check.check(
        declared, candidates,
        os.path.join(local_dir, structure) if structure else None,
    )


# --------------------------------------------------
def reap_bundle(local_dir: str) -> int:
    """Delete a succeeded bundle's bulk files, keep the small evidence.
    Returns bytes freed.

    Only ever called on success, and only for files under the bundle
    directory -- ddd/data/<name>.tgz is untouched, so unpack_local() can
    rebuild the whole thing from source if a rerun is ever needed.
    """

    freed = 0
    processed = os.path.join(local_dir, "processed")

    for entry in sorted(os.listdir(local_dir)):
        full = os.path.join(local_dir, entry)
        if full == processed:
            continue
        if entry in KEEP_TOP:
            continue
        freed += path_size(full)
        remove(full)

    if os.path.isdir(processed):
        for entry in sorted(os.listdir(processed)):
            if entry in KEEP_PROCESSED:
                continue
            full = os.path.join(processed, entry)
            freed += path_size(full)
            remove(full)

    return freed


# --------------------------------------------------
def path_size(path: str) -> int:
    """Bytes under a file or directory, ignoring anything unreadable."""

    if os.path.islink(path) or os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


# --------------------------------------------------
def remove(path: str) -> None:
    """rm -rf one path, tolerating a race with anything else looking."""

    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    except OSError:
        pass


# --------------------------------------------------
def seed_inferred_ligands(local_dir: str, args: Args) -> List[str]:
    """Pre-compute processed/inferred_ligands.json from an element-corrected
    SCRATCH copy of the structure file, so mdr-process's ligand check runs
    against the right molecule. Returns the corrections made.

    Why this works: get_inferred_ligands() (process.rs) reuses an existing
    inferred_ligands.json rather than recomputing, and it only wipes
    processed/ when --force is passed, which this driver does not pass. So
    seeding the file makes mdr-process adopt our inference.

    What it does NOT do: change anything that gets imported or pushed. The
    bundle's own structure file is untouched and goes to IRODS byte for
    byte; resolve_ligands() stores the DECLARED ligand when one is present
    and uses the inference only to verify it. This corrects how their data
    is READ, not their data.
    """

    meta_path = os.path.join(local_dir, METADATA_NAME)
    if not os.path.isfile(meta_path):
        return []
    try:
        structure = toml.load(meta_path).get("structure_file_name")
    except (toml.TomlDecodeError, OSError):
        return []
    if not structure:
        return []

    pdb = os.path.join(local_dir, structure)
    if not os.path.isfile(pdb):
        return []

    processed = os.path.join(local_dir, "processed")
    os.makedirs(processed, exist_ok=True)
    out_json = os.path.join(processed, "inferred_ligands.json")
    if os.path.isfile(out_json):
        return []

    with tempfile.TemporaryDirectory(prefix="elem-") as tmp:
        corrected = os.path.join(tmp, "corrected.pdb")
        fixes = ligand_check.write_element_corrected(pdb, corrected)
        if not fixes:
            return []
        proc = subprocess.run(
            [args.uv, "run", os.path.join(args.script_dir, "mol_id.py"),
             "both", corrected, "--outfile", out_json],
            capture_output=True, text=True, cwd=args.script_dir,
            timeout=MOL_ID_TIMEOUT,
        )
        if proc.returncode != 0 and not os.path.isfile(out_json):
            return []

    return fixes


# --------------------------------------------------
def is_irods_fault(result: str, detail: str) -> bool:
    """Does this failure look like the storage being gone, rather than this
    bundle being bad?

    The distinction is the whole point of a consecutive-fault count. A run
    of short_description overflows or corrupt tarballs is a data problem and
    must NOT trip the breaker; a run of failures in push_sim_files.py is
    CyVerse, and grinding through 6,600 bundles against a dead server would
    mint one hidden placeholder row each and reap nothing.
    """

    if result not in ("failed", "error"):
        return False
    lowered = detail.lower()
    return any(marker in lowered for marker in (
        "push_sim_files", "irods", "cyverse", "networkexception",
        "unix_file", "hierarchy_error", "sys_", "timed out", "timeout",
    ))


# --------------------------------------------------
def irods_healthy(args: Args) -> bool:
    """Ask irods_write_canary.py whether IRODS will accept a real write.

    A read probe is not enough: what fails here is the push. The canary
    writes, verifies twice and removes, which is exactly the operation the
    pipeline needs and cannot fake.
    """

    canary = os.path.join(args.script_dir, "irods_write_canary.py")
    if not os.path.isfile(canary):
        return False
    try:
        proc = subprocess.run(
            [sys.executable, canary, "-s", args.server, "--size-mb", "1",
             "--timeout", "120"],
            capture_output=True, text=True, timeout=180,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


# --------------------------------------------------
def wait_for_irods(args: Args) -> bool:
    """Sit out an IRODS outage, or report that it outlasted us.

    Waiting rather than exiting is deliberate, and taken from
    merge_replicate_groups.py's wait_for_irods(): this run takes days and is
    unattended, so a ten minute blip at hour twenty should not need a human
    to restart it. What it must not do is what the merge did on 2026-08-25 --
    carry on through 87 groups in 90 minutes without merging anything, and
    exit looking like a completed run.
    """

    waited = 0
    delay = 60

    while waited < args.fault_wait:
        nap = min(delay, args.fault_wait - waited)
        time.sleep(nap)
        waited += nap
        if irods_healthy(args):
            print(f"{stamp()} .. IRODS answered again after {waited}s, "
                  f"resuming", flush=True)
            return True
        print(f"{stamp()} .. still no IRODS after {waited}s of "
              f"{args.fault_wait}s", flush=True)

    return False


# --------------------------------------------------
def process_one(args: Args, name: str) -> Tuple[str, str, str]:
    """Unpack, fix SMILES, run mdr-process. Returns (name, result, detail)."""

    def status(msg: str) -> None:
        print(f"{stamp()} [{name}] {msg}", flush=True)

    try:
        local_dir = unpack_local(name, args.data_dir, args.work_dir)
    except (tarfile.TarError, EOFError, OSError) as e:
        # EOFError/OSError, not just tarfile.TarError: a truncated gzip
        # stream surfaces as EOFError from the gzip layer underneath
        # tarfile -- see survey_bundles.py's identical fix, 2026-08-27.
        # tarfile's own ReadError can itself be multi-line; flatten it so
        # append_record's TSV stays one row per line.
        detail = " ".join(f"corrupt or truncated tarball: {e}".split())
        return name, "fetch-failed", detail[:300]

    ok, detail = fix_smiles(local_dir, args)
    if not ok:
        status(f"fix_ligand_smiles.py non-zero exit: {detail}")
        # Not fatal here -- mdr-process fails on its own terms below, same
        # as process_bundles2.py's fix_missing_smiles() behavior.

    try:
        changed = retarget_metadata(local_dir, args)
    except OSError as e:
        return name, "failed", f"metadata rewrite failed: {e}"
    if changed:
        status(f"metadata: {', '.join(changed)}")

    try:
        fixes = seed_inferred_ligands(local_dir, args)
    except (subprocess.SubprocessError, OSError) as e:
        fixes = []
        status(f"element-correction skipped: {type(e).__name__}: {e}")
    if fixes:
        status(f"element column read as: {' '.join(fixes[:3])}")

    log_file = os.path.join(args.log_dir, f"{name}.log")
    returncode, run_detail = run_mdr_process(
        local_dir, args.server, log_file, args.dry_run, status,
        args.num_threads, args.blast_num_threads, args.transfer_threads,
    )

    if returncode != 0:
        status(f"FAILED: {run_detail[:200]}")
        # A PUSH failure is the cheap kind to retry: push_sim_files.py skips
        # any file whose md5 already matches, so a re-push moves only what is
        # missing, in minutes. Reaping it throws that away and forces a ~45
        # minute reprocess -- which is what happened to all 7 failures on
        # 2026-08-28. So keep those and reap only data failures, which need
        # the data fixed before a retry means anything either way.
        #
        # This is affordable ONLY because the circuit breaker bounds how many
        # push failures can accumulate before the run stops; without it, an
        # outage would keep every bundle and fill the volume.
        if is_irods_fault("failed", run_detail):
            status("kept for re-push (push failure, not a data failure)")
        elif args.reap and not args.keep_failed:
            freed = reap_bundle(local_dir)
            status(f"reaped {freed / 1e9:.1f} GB of the failed bundle")
        return name, "failed", run_detail

    prefix = "dry-run" if args.dry_run else "done"
    result = "dry-run-ok" if args.dry_run else "done"
    detail = ""

    # Phase C, inline. In --no-dry-run this is deliberately POST-import:
    # mdr-process does the import and the push in one step, and the ligand
    # inference this reads does not exist until processing has run, so
    # there is no point before it at which the stricter bar could have
    # been applied. A bundle flagged here is already in the DB -- but as
    # is_placeholder = true (import.rs hard-codes it), so it is not
    # curated yet, and this is what tells you which ones must not be.
    if args.verify:
        verdict, why = verify_ligands(local_dir)
        if verdict != ligand_check.PASS:
            result, detail = f"{prefix}-{verdict}", why
            status(f"{verdict.upper()}: {why[:200]}")

    # A hard failure always keeps its files -- that is Phase C's triage
    # input, and failures are a small enough minority to hold to the end.
    # An UNVERIFIED bundle is reaped anyway unless asked otherwise: the
    # evidence the verdict rests on survives the reap, and at the measured
    # rate keeping them all would reinstate the disk problem.
    #
    # This applies to --dry-run too. Phase B was launched believing a kept
    # bundle cost 2.2 GB; at the real 5.3 GB a resumed dry run fills the
    # disk just as surely as the prod run would.
    reapable = not result.endswith(f"-{ligand_check.BLOCK}") or (
        not args.keep_unverified
    )
    if args.reap and reapable:
        freed = reap_bundle(local_dir)
        status(f"OK, reaped {freed / 1e9:.1f} GB")
    else:
        status("OK")

    return name, result, detail


# --------------------------------------------------
def main() -> None:
    args = get_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if not args.dry_run:
        confirm = input(
            f"--no-dry-run: this pushes to '{args.server}' for real. "
            f"Type the server name to confirm: "
        )
        if confirm.strip() != args.server:
            sys.exit("Confirmation did not match, aborting")

    bundles = eligible_bundles(args)
    if args.limit:
        bundles = bundles[: args.limit]
    print(f"{stamp()} {len(bundles):,} bundle(s) to process "
          f"({'dry-run' if args.dry_run else 'REAL, server=' + args.server}, "
          f"{args.parallel} at a time)", flush=True)

    tally: Dict[str, int] = {}
    done = 0
    faults = 0
    recent: collections.deque = collections.deque(maxlen=args.fault_window
                                                  or 1)
    outage = False
    pending = list(bundles)

    # Submitted a window at a time rather than all at once: with every
    # bundle queued up front there is no way to stop early, which is what
    # turns an outage into 6,600 hidden placeholder rows and a full disk.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.parallel
    ) as pool:
        futures = {}
        while pending and len(futures) < args.parallel * 2:
            b = pending.pop(0)
            futures[pool.submit(process_one, args, b)] = b

        while futures:
            for future in concurrent.futures.as_completed(list(futures)):
                name = futures.pop(future)
                try:
                    name, result, detail = future.result()
                except Exception as e:
                    # One bundle's unexpected failure must never take the
                    # rest of the run down -- see survey_bundles.py's note.
                    result, detail = "error", f"{type(e).__name__}: {e}"
                append_record(args.record_file, name, result, detail)
                tally[result] = tally.get(result, 0) + 1
                done += 1
                print(f"{stamp()} {done:,}/{len(bundles):,} done "
                      f"({', '.join(f'{k}={v}' for k, v in tally.items())})",
                      flush=True)

                fault = is_irods_fault(result, detail)
                faults = faults + 1 if fault else 0
                recent.append(fault)

                consecutive_trip = (
                    args.max_consecutive_faults
                    and faults >= args.max_consecutive_faults
                )
                window_trip = (
                    args.fault_window and args.max_window_faults
                    and sum(recent) >= args.max_window_faults
                )

                if (consecutive_trip or window_trip) and not outage:
                    why = (f"{faults} consecutive"
                           if consecutive_trip
                           else f"{sum(recent)} of the last {len(recent)}")
                    print(f"{stamp()} !! {why} IRODS/push faults -- the "
                          f"server looks gone, not flaky", flush=True)
                    if args.fault_wait and wait_for_irods(args):
                        faults = 0
                        recent.clear()
                    else:
                        outage = True
                        pending = []
                        print(f"{stamp()} !! stopping: {why} faults" +
                              (f" and no IRODS in {args.fault_wait}s"
                               if args.fault_wait else ""), flush=True)

                if pending and not outage:
                    b = pending.pop(0)
                    futures[pool.submit(process_one, args, b)] = b
                break  # re-enter as_completed over the updated set

    verb = "Stopped" if outage else "finished"
    print(f"{stamp()} {verb}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(tally.items())) +
          f"; {len(bundles) - done:,} not attempted", flush=True)
    if outage:
        sys.exit(1)


if __name__ == "__main__":
    main()
