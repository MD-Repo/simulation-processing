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

         FAILED bundles are never reaped -- they are Phase C's triage
         input, they are a small minority, and at the observed rate they
         total ~115 GB even if every one is kept to the end.

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
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
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
                        help="Passed to mdr-process per bundle. Load-"
                        "bearing -- see RUNBOOK.md's MDR-33 note")
    parser.add_argument("--parallel", type=int, default=8, metavar="INT",
                        help="Bundles run concurrently. Phase B can push "
                        "this high (local CPU is the only constraint); "
                        "Phase D should stay modest -- it shares the prod "
                        "DB with the live ticket queue")
    parser.add_argument("--limit", type=int, default=None, metavar="INT",
                        help="Process only the first N eligible bundles")
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

    log_file = os.path.join(args.log_dir, f"{name}.log")
    returncode, run_detail = run_mdr_process(
        local_dir, args.server, log_file, args.dry_run, status,
        args.num_threads,
    )

    if returncode != 0:
        status(f"FAILED: {run_detail[:200]}")
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
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.parallel
    ) as pool:
        futures = {pool.submit(process_one, args, b): b for b in bundles}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                name, result, detail = future.result()
            except Exception as e:
                # One bundle's unexpected failure must never take the rest
                # of the run down with it -- see survey_bundles.py's note.
                result, detail = "error", f"{type(e).__name__}: {e}"
            append_record(args.record_file, name, result, detail)
            tally[result] = tally.get(result, 0) + 1
            done += 1
            print(f"{stamp()} {done:,}/{len(bundles):,} done "
                  f"({', '.join(f'{k}={v}' for k, v in tally.items())})",
                  flush=True)

    print(f"{stamp()} finished: " +
          ", ".join(f"{k}={v}" for k, v in tally.items()))


if __name__ == "__main__":
    main()
