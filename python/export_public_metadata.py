#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@gmail.com>
Date   : 2026-08-20
Purpose: Build the nightly public metadata tarball for static.mdrepo.org

Publishes one JSON record per publicly visible simulation, so consumers can
download the whole corpus instead of crawling /simulation_list page by page.

The tarball layout is:

    mdrepo-metadata-YYYY-MM-DD.tar.gz
    |-- index.json            generated-at, count, and the sorted ID list
    `-- simulations/
        |-- MDR00000002.json
        `-- ...
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import fabric
import psycopg2
from dotenv import dotenv_values
from typing import List, NamedTuple, Set

# Where the tarballs land on the media host. Sibling of the "mapping"
# directory that export_mapping_file.py has been writing since 2026-02-13.
REMOTE_DIR = "/home/web/mdrepo/static/metadata"

# Stable filename so a script can always fetch the newest export without
# having to guess or scrape a date. It is a copy, not a symlink: the remote
# is served over plain HTTP and a symlink adds a way for the two to disagree.
LATEST_NAME = "mdrepo-metadata-latest.tar.gz"

# What "publicly visible" means, as a SQL predicate on an md_simulation
# aliased "s". This is the same rule the Rust exporter applies via
# ops::get_visible_simulation_ids, and both mirror visible_simulations_q() in
# the Django app (md_repo_app/models/simulation/simulation_filters.py), which
# is the source of truth. Duplicated here on purpose: this copy is not how
# rows get selected, it is the independent audit that catches it if the Rust
# copy and the Django rule ever drift apart. Kept identical to VISIBLE_SIM in
# export_mapping_file.py.
VISIBLE_SIM = """
    s.is_public
    and not s.is_deprecated
    and not s.is_placeholder
    and not s.is_embargoed
"""


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    mdr_export: str
    keep: int
    work_dir: str
    dry_run: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Build the nightly public metadata tarball",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--server",
        help="Database to export",
        metavar="STR",
        choices=["prod", "staging"],
        default="prod",
    )

    parser.add_argument(
        "-m",
        "--mdr-export",
        help="Path to the mdr-export binary",
        metavar="PROG",
        default="mdr-export",
    )

    parser.add_argument(
        "-k",
        "--keep",
        help="Dated tarballs to retain on the media host",
        metavar="INT",
        type=int,
        default=7,
    )

    parser.add_argument(
        "-w",
        "--work-dir",
        help="Parent for the build directory (default: system temp)",
        metavar="DIR",
        default=None,
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        help="Build the tarball but do not push it",
        action="store_true",
    )

    args = parser.parse_args()

    if args.keep < 1:
        parser.error(f'--keep "{args.keep}" must be a positive integer')

    return Args(
        server=args.server,
        mdr_export=args.mdr_export,
        keep=args.keep,
        work_dir=args.work_dir,
        dry_run=args.dry_run,
    )


# --------------------------------------------------
def visible_ids(dsn: str) -> Set[int]:
    """Simulation IDs the public may see, straight from the database"""

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"select s.id from md_simulation s where {VISIBLE_SIM}")
        return {row[0] for row in cur.fetchall()}


# --------------------------------------------------
def run_export(args: Args, dsn: str, sim_dir: str) -> None:
    """Shell out to mdr-export to write one JSON file per visible simulation"""

    # Always a directory that did not exist a moment ago. mdr-export skips any
    # output file that is already present (absent --overwrite), so a reused
    # directory would serve last night's copy of an edited simulation, and --
    # much worse -- would keep serving a simulation that has since been
    # embargoed or deprecated. A fresh build directory makes deletions
    # propagate for free.
    os.makedirs(sim_dir)

    # mdr-export reads PRODUCTION_DSN/STAGING_DSN from the environment (see
    # mdr-export/src/main.rs). Passing it explicitly is what lets this script
    # own the credential lookup -- note that /opt/mdrepo/mdrepo-rs/.env names
    # the same value PRODUCTION_DB_URL, which mdr-export does NOT read, so
    # relying on the binary to find its own .env would fail.
    env = dict(os.environ)
    env["PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"] = dsn

    # No visibility flag: "--format public-json" selects the publicly visible
    # rows by definition, so there is no way to ask this for a public export
    # and not get the filter.
    #
    # stdout is one progress line per simulation -- ~55,000 of them, which is
    # noise in a cron log. stderr carries the per-simulation failures and is
    # left attached so they land in the log.
    proc = subprocess.run(
        [
            args.mdr_export,
            "--server",
            args.server,
            "--format",
            "public-json",
            "--out-dir",
            sim_dir,
        ],
        env=env,
        stdout=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        sys.exit(f"mdr-export failed with exit code {proc.returncode}")


# --------------------------------------------------
def prune_hidden(sim_dir: str, allowed: Set[int]) -> List[int]:
    """Delete exported records for simulations that are no longer visible"""

    # mdr-export snapshots the ID list once and then spends several minutes
    # writing files, so its output reflects visibility at the START of the
    # run. A simulation embargoed or deprecated while it worked is still on
    # disk. Re-checking against a fresh read of the database closes that
    # window in the direction that matters: something newly hidden must not
    # ship. (The opposite case -- newly visible -- simply waits for tomorrow.)
    #
    # This is not hypothetical. A 2026-08-20 full run exported 54,599 records
    # while the visible count moved 54,601 -> 54,598 underneath it; two
    # simulations were deprecated mid-export.
    removed = []
    for name in sorted(os.listdir(sim_dir)):
        if not name.endswith(".json"):
            continue
        sim_id = int(name[len("MDR") : -len(".json")])
        if sim_id not in allowed:
            os.remove(os.path.join(sim_dir, name))
            removed.append(sim_id)

    return removed


# --------------------------------------------------
def write_index(build_dir: str, sim_dir: str, server: str) -> int:
    """Write index.json describing the export, and return the record count"""

    ids = sorted(
        int(name[len("MDR") : -len(".json")])
        for name in os.listdir(sim_dir)
        if name.endswith(".json")
    )

    index = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "server": server,
        "count": len(ids),
        # The visibility rule spelled out, so someone holding only the tarball
        # can tell what it does and does not contain without reading our code.
        "visibility": (
            "is_public and not is_deprecated and not is_placeholder "
            "and not is_embargoed"
        ),
        "simulations": [f"MDR{i:08}" for i in ids],
    }

    with open(os.path.join(build_dir, "index.json"), "wt", encoding="UTF-8") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")

    return len(ids)


# --------------------------------------------------
def make_tarball(build_dir: str, tar_path: str) -> None:
    """Pack the build directory into a gzipped tarball"""

    # Sorted, so two runs over unchanged data pack their members in the same
    # order. Deterministic ordering is what makes it meaningful to compare
    # one night's tarball against the previous one.
    with tarfile.open(tar_path, "w:gz") as tar:
        for name in sorted(os.listdir(build_dir)):
            tar.add(os.path.join(build_dir, name), arcname=name)


# --------------------------------------------------
def sha256(path: str) -> str:
    """Hex digest of a file, read in chunks so a large tarball stays cheap"""

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------------------
def push(conn: fabric.Connection, local: str, remote: str) -> None:
    """Upload a file and move it into place only once it is complete"""

    # put() straight onto the served name would expose a partial file for the
    # length of the transfer -- tens of seconds for a tarball this size, every
    # night. Uploading beside it and renaming makes the swap atomic, since
    # mv within a directory is a rename(2). The mapping export gets away with
    # a bare put() only because its files are small.
    staging = f"{remote}.tmp"
    conn.put(local, remote=staging)
    conn.run(f"mv -f {staging} {remote}", hide=True)


# --------------------------------------------------
def prune_remote(conn: fabric.Connection, keep: int) -> None:
    """Delete all but the newest "keep" dated tarballs on the media host"""

    # Matches only the dated files: LATEST_NAME and the .sha256 sidecars are
    # excluded by the glob so retention can never remove the stable download.
    listing = conn.run(
        f"ls -1 {REMOTE_DIR}/mdrepo-metadata-????-??-??.tar.gz 2>/dev/null || true",
        hide=True,
    )
    dated = sorted(line.strip() for line in listing.stdout.splitlines() if line.strip())

    # Names sort chronologically because the date is ISO-8601, so the tail of
    # the sorted list is the newest.
    for stale in dated[:-keep] if len(dated) > keep else []:
        conn.run(f"rm -f {stale} {stale}.sha256", hide=True)
        print(f"Pruned '{os.path.basename(stale)}'")


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    # Name the .env beside this file rather than leaning on a bare
    # dotenv_values(), matching export_mapping_file.py and the queue scripts.
    # Every cron job on this box loads the .env next to the script it runs.
    env = dotenv_values(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    )

    def get_env(key):
        if val := env.get(key, os.environ.get(key, None)):
            return val
        else:
            sys.exit(f"Missing env '{key}'")

    dsn = get_env("PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN")

    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    tar_name = f"mdrepo-metadata-{date}.tar.gz"

    # mkdtemp, not a fixed path: two overlapping runs must not write into each
    # other's export. The whole tree is removed in the finally below.
    build_root = tempfile.mkdtemp(prefix="mdrepo-metadata-", dir=args.work_dir)
    try:
        build_dir = os.path.join(build_root, "build")
        sim_dir = os.path.join(build_dir, "simulations")
        os.makedirs(build_dir)

        run_export(args, dsn, sim_dir)
        exported = len([n for n in os.listdir(sim_dir) if n.endswith(".json")])

        removed = prune_hidden(sim_dir, visible_ids(dsn))
        if removed:
            ids = ", ".join(f"MDR{i:08}" for i in removed[:10])
            more = f" (and {len(removed) - 10} more)" if len(removed) > 10 else ""
            print(f"Dropped {len(removed)} no longer visible: {ids}{more}")

        count = write_index(build_dir, sim_dir, args.server)
        if not count:
            # An empty export is never legitimate here and would replace a
            # good tarball with a useless one, so refuse to publish it.
            sys.exit("Refusing to publish: export produced no records")

        tar_path = os.path.join(build_root, tar_name)
        make_tarball(build_dir, tar_path)
        digest = sha256(tar_path)
        size_mb = os.path.getsize(tar_path) / 1024 / 1024

        # One sidecar per published name. The digest is identical -- it is the
        # same bytes -- but the filename inside each must match the file it
        # sits beside, or "sha256sum -c" looks for a name that is not there.
        sum_path = f"{tar_path}.sha256"
        with open(sum_path, "wt", encoding="UTF-8") as fh:
            fh.write(f"{digest}  {tar_name}\n")

        latest_sum_path = os.path.join(build_root, f"{LATEST_NAME}.sha256")
        with open(latest_sum_path, "wt", encoding="UTF-8") as fh:
            fh.write(f"{digest}  {LATEST_NAME}\n")

        print(
            f"Exported {exported}, published {count} records "
            f"in {tar_name} ({size_mb:.1f} MB, sha256 {digest[:12]})"
        )

        if args.dry_run:
            kept = os.path.join(os.getcwd(), tar_name)
            shutil.copy(tar_path, kept)
            print(f"Dry run, tarball left at '{kept}', nothing pushed")
            return

        media_host = get_env("MEDIA_HOST")
        conn = fabric.Connection(
            media_host,
            port=get_env("MEDIA_PORT"),
            user=get_env("MEDIA_USER"),
            connect_kwargs={"password": get_env("MEDIA_PASSWORD")},
        )

        conn.run(f"mkdir -p {REMOTE_DIR}", hide=True)
        push(conn, tar_path, f"{REMOTE_DIR}/{tar_name}")
        push(conn, sum_path, f"{REMOTE_DIR}/{tar_name}.sha256")

        # The stable name is published only after the dated one is safely in
        # place, so "latest" never points at an export that failed midway.
        push(conn, tar_path, f"{REMOTE_DIR}/{LATEST_NAME}")
        push(conn, latest_sum_path, f"{REMOTE_DIR}/{LATEST_NAME}.sha256")

        prune_remote(conn, args.keep)
        print(f"Done, pushed metadata to '{media_host}:{REMOTE_DIR}/{tar_name}'")
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


# --------------------------------------------------
if __name__ == "__main__":
    main()
