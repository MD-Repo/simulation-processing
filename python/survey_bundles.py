#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-27
Purpose: Survey the bundles2/ backlog for the SMILES-fill problem, across the
         WHOLE local corpus rather than the 20-bundle alphabetical sample
         MDR-51 was scoped from (see ddd/HANDOFF/README.md and FINDINGS.md
         §2 -- that sample happened to hit a cluster of peptide ligands and
         should not be planned against).

         HANDOFF/FINDINGS.md §3 found there's no cheap survey against IRODS
         directly: the TOML sits an average of ~60% of the way into each
         compressed tarball, so reading it still costs most of a download.
         That math changes once the tarballs are local (ddd/data/, this
         host): there's no network cost left, only CPU to decompress up to
         the TOML, which this host has to spare.

         For each local bundle with a ready (.md5-sidecar) tarball, this
         pulls ONLY the mdrepo-metadata.toml member out of the tar stream --
         no full unpack -- and classifies it by running
         fix_ligand_smiles.py's own dry-run logic against it, so the
         classification is exactly the tool that will actually run in
         Phase B, not a reimplementation that could drift from it.

         Writes one row per bundle to a TSV: bundle, classification, detail.
         Classifications mirror fix_ligand_smiles.py's tally keys: "fixed"
         (SMILES successfully filled), "already" (TOML already had one),
         "no_smiles" (PDBbind row exists but is blank -- unfixable from this
         table), "name_mismatch" (TOML's ligand name disagrees with
         PDBbind's -- needs a human), "no_row" (pdb_id not in the table),
         "no_toml"/"no_ligands_table"/"error" (malformed bundle).
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tarfile
import tempfile
from typing import NamedTuple, Optional

from common import stamp

METADATA_NAME = "mdrepo-metadata.toml"
DATA_DIR_DEFAULT = "/media/volume/mdrepo_bd/ddd/data"
FIX_SMILES_DEFAULT = "/media/volume/mdrepo_bd/utils/python/fix_ligand_smiles.py"
TABLE_DEFAULT = os.path.expanduser("~/pdbbind_ligand_smiles.tsv")

TALLY_KEYS = (
    "fixed", "already", "no_toml", "no_row", "no_smiles",
    "name_mismatch", "ambiguous", "error",
)


class Args(NamedTuple):
    data_dir: str
    fix_smiles: str
    table: str
    out: str
    workers: int
    limit: Optional[int]


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Survey ddd/data/ bundles for the SMILES-fill problem, "
        "using fix_ligand_smiles.py's own dry-run classification against "
        "just the TOML member of each tarball (no full unpack).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data-dir", metavar="DIR", default=DATA_DIR_DEFAULT,
        help="Directory of local <name>.tgz / <name>.tgz.md5 pairs",
    )
    parser.add_argument(
        "--fix-smiles", metavar="PATH", default=FIX_SMILES_DEFAULT,
        help="Path to fix_ligand_smiles.py",
    )
    parser.add_argument(
        "-t", "--table", metavar="TSV", default=TABLE_DEFAULT,
        help="pdb_id/ligand_name/canonical_smiles table",
    )
    parser.add_argument(
        "-o", "--out", metavar="TSV", required=True,
        help="Output TSV: bundle, classification, detail",
    )
    parser.add_argument(
        "-w", "--workers", metavar="INT", type=int,
        default=min(32, os.cpu_count() or 4),
        help="Parallel bundles to scan at once",
    )
    parser.add_argument(
        "--limit", metavar="INT", type=int, default=None,
        help="Survey only the first N ready bundles (testing)",
    )
    args = parser.parse_args()
    return Args(args.data_dir, args.fix_smiles, args.table, args.out,
                args.workers, args.limit)


# --------------------------------------------------
def ready_bundles(data_dir: str) -> list:
    """Names (without .tgz) of bundles with a non-empty .md5 sidecar.

    Mirrors process_bundles2.py's readiness signal: a .md5 sidecar that
    exists and is non-empty. Sorted by name for a deterministic, resumable
    survey order -- there's no catalog create_time to sort by here, this is
    a local directory listing, not the IRODS collection.
    """

    names = []
    for entry in os.scandir(data_dir):
        if not entry.name.endswith(".tgz"):
            continue
        md5_path = entry.path + ".md5"
        if os.path.isfile(md5_path) and os.path.getsize(md5_path) > 0:
            names.append(entry.name[: -len(".tgz")])
    return sorted(names)


# --------------------------------------------------
def extract_toml(tgz_path: str) -> Optional[bytes]:
    """Pull just mdrepo-metadata.toml's bytes out of a tar stream.

    Iterates rather than calling getmembers(), so it stops decompressing as
    soon as the member is found instead of indexing the whole archive.
    """

    with tarfile.open(tgz_path, mode="r:*") as tar:
        for member in tar:
            if os.path.basename(member.name) == METADATA_NAME:
                fh = tar.extractfile(member)
                return fh.read() if fh else None
    return None


# --------------------------------------------------
def classify_one(args: Args, bundle: str) -> tuple:
    """Returns (bundle, classification, detail)"""

    tgz_path = os.path.join(args.data_dir, f"{bundle}.tgz")
    try:
        toml_bytes = extract_toml(tgz_path)
    except (tarfile.TarError, EOFError, OSError) as e:
        # EOFError/OSError, not just tarfile.TarError: a truncated gzip
        # stream surfaces as EOFError from the gzip layer underneath
        # tarfile, not as a TarError -- caught the hard way, see the
        # 2026-08-27 survey run that died on exactly this.
        # tarfile's own ReadError can itself be multi-line (one line per
        # compression method it tried), which broke the TSV's one-row-per-
        # line shape on 4bfd/4c1m in that same run -- flatten it.
        detail = " ".join(f"corrupt or truncated tarball: {e}".split())
        return bundle, "error", detail[:300]

    if toml_bytes is None:
        return bundle, "no_toml", f"no {METADATA_NAME} member in tarball"

    fd, tmp_path = tempfile.mkstemp(suffix=".toml")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(toml_bytes)

        proc = subprocess.run(
            [sys.executable, args.fix_smiles, tmp_path,
             "--table", args.table, "-n"],
            capture_output=True, text=True,
        )
    finally:
        os.remove(tmp_path)

    stdout = proc.stdout or ""
    tally_line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    classification = "error"
    for key in TALLY_KEYS:
        if f"{key}=" in tally_line:
            classification = key
            break

    detail_lines = [
        ln.strip().replace(tmp_path, bundle) for ln in stdout.splitlines()
        if ln.strip().startswith(("!!", "~~", "ok "))
    ]
    detail = " | ".join(detail_lines) or (proc.stderr or "").strip()[:200]
    return bundle, classification, " ".join(detail.split())[:300]


# --------------------------------------------------
def main() -> None:
    args = get_args()

    if not os.path.isfile(args.table):
        sys.exit(f"No such table: {args.table}")
    if not os.path.isfile(args.fix_smiles):
        sys.exit(f"No such script: {args.fix_smiles}")

    already = set()
    if os.path.isfile(args.out):
        with open(args.out) as fh:
            next(fh, None)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0]:
                    already.add(parts[0])
        print(f"{stamp()} resuming: {len(already):,} bundle(s) already "
              f"in {args.out}", flush=True)

    bundles = [b for b in ready_bundles(args.data_dir) if b not in already]
    if args.limit:
        bundles = bundles[: args.limit]
    print(f"{stamp()} {len(bundles):,} bundle(s) left to survey", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tally = {k: 0 for k in TALLY_KEYS}
    done = 0

    with open(args.out, "a") as out_fh:
        if not already:
            out_fh.write("bundle\tclassification\tdetail\n")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {
                pool.submit(classify_one, args, b): b for b in bundles
            }
            for future in concurrent.futures.as_completed(futures):
                bundle = futures[future]
                try:
                    bundle, classification, detail = future.result()
                except Exception as e:
                    # One bundle's unexpected failure must never take the
                    # rest of an 8,000-bundle run down with it -- see the
                    # 2026-08-27 run that died on an uncaught EOFError.
                    classification, detail = "error", f"{type(e).__name__}: {e}"
                tally[classification] = tally.get(classification, 0) + 1
                out_fh.write(f"{bundle}\t{classification}\t{detail}\n")
                out_fh.flush()
                done += 1
                if done % 200 == 0:
                    print(f"{stamp()} {done:,}/{len(bundles):,}", flush=True)

    print(f"{stamp()} done: " +
          ", ".join(f"{k}={v}" for k, v in tally.items() if v))


if __name__ == "__main__":
    main()
