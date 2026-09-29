#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-28
Purpose: Confirm that every file row a simulation has actually exists in
         IRODS, at the right size, with the right checksum.

         WHY THIS EXISTS. A bulk import's own record is not evidence that
         the files landed. On 2026-08-28 the ddd/bundles2 prod wave was
         killed mid-flight and left four simulations behind: two that its
         `processed.tsv` recorded as `failed` had in fact pushed NOTHING
         (clean), while the damage sat in the two the record said nothing
         about -- one complete, and one with four good files plus a
         **0-byte trajectories.tar** whose row claimed 177,111,040 bytes.
         That object existed, carried a registered checksum, and would pass
         any presence check. The record could not have told you.

         So this asks IRODS, row by row, and reports four ways a row can be
         wrong:

           missing    -- no object at the path the row names
           zero       -- the object is 0 bytes and the row says otherwise
           size       -- sizes disagree
           checksum   -- registered checksum disagrees with md5_hash

         Read-only. It changes nothing and has no --apply; the repair half
         is replace_and_record.py, which takes a manifest in the shape this
         writes with --manifest-out.

         NOT THE TOOL FOR A WHOLE-ARCHIVE PASS, since 2026-09-04. It asks
         IRODS about one row at a time, so all of prod is about 1.5 million
         round trips and roughly 11 hours. The catalog returns size and
         checksum in bulk, so reconcile_release_files.py gets the same
         answers for the whole archive in one walk, and finds objects with
         no row at the same time, which this cannot do at all.

         What is still worth using this for:

           --deep         reads every byte back and hashes it instead of
                          trusting the registered checksum. The catalog
                          cannot answer that, so this is the only tool that
                          catches an object whose registered checksum is
                          itself wrong. Expensive: use it on a sample.
           narrow scopes  -c a collection, --record a driver's processed.tsv,
                          -i a handful of ids, --placeholder-only. Point
                          lookups win when the target is a small set rather
                          than an id range.

         Checksums are COMPARED, never forced. Calling obj.chksum() to
         create a missing one is the RPC that returned HIERARCHY_ERROR on 46
         replicate-merge groups, so a row whose object has no registered
         checksum is reported as unverifiable rather than made to have one.

         Sizes and checksums only -- it does not read the bytes back.
         Reading a whole prod wave would be ~6.6 TB. Use --deep on a sample
         when you want proof rather than a claim.

         RUNNING THIS WHILE A WAVE IS IN FLIGHT will report the bundle
         currently being pushed as broken, and it looks exactly like the
         real thing: a half-written tar is 0 bytes, and files not yet
         reached are missing. Those simulations are still `is_placeholder`,
         because mdr-process only clears that flag once the push verifies.
         So **--exclude-placeholder is the meaningful scope**: it asks the
         question worth asking, which is whether anything the pipeline
         believes it FINISHED is actually incomplete. Run it that way
         during a wave, or unfiltered only after one ends.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, NamedTuple, Optional, Tuple

import psycopg2
from dotenv import dotenv_values
from irods.session import iRODSSession

ENV_CANDIDATES = (
    "/media/volume/mdrepo_bd/utils/python/.env",
    "/opt/mdrepo/utils/python/.env",
    "/opt/mdrepo/simulation-processing/python/.env",
)
TABLES = ("md_uploaded_file", "md_processed_file")
CHUNK = 1024 * 1024


class Args(NamedTuple):
    """Command-line arguments"""

    server: str
    simulation_ids: List[int]
    id_range: Optional[Tuple[int, int]]
    collection: Optional[str]
    record: Optional[str]
    placeholder_only: bool
    exclude_placeholder: bool
    exclude_filenames: Tuple[str, ...]
    tables: Tuple[str, ...]
    threads: int
    deep: bool
    out: Optional[str]
    manifest_out: Optional[str]


class Row(NamedTuple):
    """One file row as the database has it"""

    table: str
    id: int
    simulation_id: int
    filename: str
    local_file_path: str
    size: Optional[int]
    md5: Optional[str]


class Finding(NamedTuple):
    """What IRODS said about one row"""

    row: Row
    verdict: str
    detail: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Verify a simulation's file rows against IRODS. "
        "Read-only; pair with replace_and_record.py to repair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--server", choices=["prod", "staging"],
                        required=True)
    parser.add_argument("-i", "--simulation-id", type=int, nargs="+",
                        default=[], metavar="INT", dest="simulation_ids")
    parser.add_argument("-r", "--id-range", nargs=2, type=int, default=None,
                        metavar=("LOW", "HIGH"),
                        help="Inclusive simulation id range")
    parser.add_argument("-c", "--collection", default=None, metavar="NAME",
                        help="Every simulation in this collection -- the "
                        "natural scope after a bulk wave")
    parser.add_argument(
        "--record", default=None, metavar="TSV",
        help="A driver's processed.tsv; verifies the simulations its "
        "successful bundles produced. Complements the record rather than "
        "trusting it -- see the module docstring",
    )
    parser.add_argument(
        "--placeholder-only", action="store_true",
        help="Only simulations still flagged is_placeholder. These are the "
        "ones whose push did not verify, so they are where incomplete "
        "pushes are already known to be",
    )
    parser.add_argument(
        "--exclude-placeholder", action="store_true",
        help="Skip simulations still flagged is_placeholder. USE THIS WHILE "
        "A WAVE IS RUNNING: an in-flight push shows up as 0-byte and "
        "missing files, indistinguishable from real damage. What is left is "
        "the question that matters -- is anything the pipeline called "
        "finished actually incomplete?",
    )
    parser.add_argument(
        "--exclude-filename", nargs="+", default=[], metavar="NAME",
        dest="exclude_filenames",
        help="Skip rows with this filename. For a filename whose rows are "
        "known to disagree with IRODS for a diagnosed, benign reason -- "
        "otherwise one such class buries every real finding. The standing "
        "case is 'mdrepo-metadata.toml': a 2026-05 migration rewrote every "
        "released copy and kept the original as 'mdrepo-metadata.v1.toml', "
        "and md_uploaded_file was never updated, so ~21,300 rows in the "
        "id <= 21,607 band describe the v1 file. Excluding a name here also "
        "blinds the sweep to genuine damage to it, so exclude deliberately "
        "and sweep the name separately once its rows are repaired",
    )
    parser.add_argument("-t", "--tables", nargs="+", default=list(TABLES),
                        choices=list(TABLES))
    parser.add_argument("--threads", type=int, default=8, metavar="INT",
                        help="Concurrent IRODS lookups. Point lookups are "
                        "cheap, but this shares CyVerse's connection ceiling "
                        "with everything else running")
    parser.add_argument(
        "--deep", action="store_true",
        help="Read every object back and hash it instead of trusting the "
        "registered checksum. Correct but expensive -- a whole wave is "
        "~6.6 TB. Use on a sample",
    )
    parser.add_argument("-o", "--out", default=None, metavar="PATH",
                        help="TSV of every finding, including the good ones")
    parser.add_argument(
        "--manifest-out", default=None, metavar="PATH",
        help="Write the bad rows in replace_and_record.py --manifest shape "
        "(simulation_id, filename, local_file). The local path is left "
        "empty: the bytes have to come from somewhere, and only a human "
        "knows whether that is a work dir or a re-fetch",
    )
    args = parser.parse_args()

    scopes = [bool(args.simulation_ids), bool(args.id_range),
              bool(args.collection), bool(args.record)]
    if sum(scopes) != 1:
        parser.error("give exactly one of --simulation-id, --id-range, "
                     "--collection, --record")
    if args.placeholder_only and args.exclude_placeholder:
        parser.error("--placeholder-only and --exclude-placeholder are "
                     "opposites")

    return Args(
        args.server, args.simulation_ids,
        tuple(args.id_range) if args.id_range else None,
        args.collection, args.record, args.placeholder_only,
        args.exclude_placeholder, tuple(args.exclude_filenames),
        tuple(args.tables), args.threads,
        args.deep, args.out, args.manifest_out,
    )


# --------------------------------------------------
def find_dsn(server: str) -> str:
    """The DSN for this deployment, from whichever .env this host has"""

    key = "PRODUCTION_DSN" if server == "prod" else "STAGING_DSN"
    for path in ENV_CANDIDATES:
        if os.path.isfile(path):
            dsn = dotenv_values(path).get(key)
            if dsn:
                return dsn
    dsn = os.environ.get(key)
    if not dsn:
        sys.exit(f"No '{key}' in {ENV_CANDIDATES} or the environment")
    return dsn


# --------------------------------------------------
def bundles_from_record(path: str) -> List[str]:
    """Successful bundle names from a driver's append-only processed.tsv.

    Later line wins, matching load_record() in process_bundles2.py, so a
    bundle that failed and was rerun reads as done.
    """

    latest = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] != "timestamp":
                latest[parts[2]] = parts[1]
    return [b for b, r in latest.items() if r.startswith("done")]


# --------------------------------------------------
def target_ids(cur, args: Args) -> List[int]:
    """The simulation ids this run covers"""

    if args.simulation_ids:
        ids = list(args.simulation_ids)
    elif args.id_range:
        cur.execute(
            "select id from md_simulation where id between %s and %s",
            args.id_range,
        )
        ids = [r[0] for r in cur.fetchall()]
    elif args.collection:
        cur.execute(
            """select sc.simulation_id from md_simulation_collection sc
               join md_collection c on c.id = sc.collection_id
               where c.name = %s""",
            (args.collection,),
        )
        ids = [r[0] for r in cur.fetchall()]
    else:
        # A record names bundles, not simulations. The link is the alias-free
        # path each bundle's files were pushed to, so go via the filename.
        bundles = bundles_from_record(args.record)
        if not bundles:
            return []
        cur.execute(
            """select distinct simulation_id from md_uploaded_file
               where local_file_path like any(%s)""",
            ([f"%/{b}/%" for b in bundles],),
        )
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            print("NOTE: no simulations matched the record's bundle names by "
                  "path; fall back to --collection or --id-range",
                  file=sys.stderr)

    if args.placeholder_only and ids:
        cur.execute(
            "select id from md_simulation where id = any(%s) "
            "and is_placeholder = true",
            (ids,),
        )
        ids = [r[0] for r in cur.fetchall()]
    elif args.exclude_placeholder and ids:
        cur.execute(
            "select id from md_simulation where id = any(%s) "
            "and is_placeholder = false",
            (ids,),
        )
        ids = [r[0] for r in cur.fetchall()]

    return sorted(set(ids))


# --------------------------------------------------
def load_rows(cur, args: Args, ids: List[int]) -> List[Row]:
    """Every file row for these simulations, across the chosen tables"""

    rows: List[Row] = []
    # Filtered in SQL rather than after the fetch: at prod scale the excluded
    # class is tens of thousands of rows and there is no reason to carry them
    # into Python only to drop them.
    exclude = list(args.exclude_filenames)
    clause = "and filename <> all(%s)" if exclude else ""
    params = (ids, exclude) if exclude else (ids,)

    for table in args.tables:
        cur.execute(
            f"""select id, simulation_id, filename, local_file_path,
                       file_size_bytes, md5_hash
                from   {table}
                where  simulation_id = any(%s)
                {clause}
                order  by simulation_id, filename""",
            params,
        )
        rows += [Row(table, *r) for r in cur.fetchall()]
    return rows


# --------------------------------------------------
def md5_irods(session, path: str) -> str:
    """Hash what actually reads back, not what the catalog claims"""

    import hashlib

    digest = hashlib.md5()
    with session.data_objects.open(path, "r") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------
def inspect(session, root: str, row: Row, deep: bool) -> Finding:
    """Ask IRODS about one row"""

    path = f"{root}/{row.local_file_path}"
    try:
        obj = session.data_objects.get(path)
    except Exception as err:
        return Finding(row, "missing", f"{path} ({type(err).__name__})")

    if obj.size == 0 and (row.size or 0) > 0:
        return Finding(row, "zero",
                       f"{path} is 0 bytes, row says {row.size:,}")

    if row.size is not None and obj.size != row.size:
        return Finding(row, "size",
                       f"{path} irods={obj.size:,} row={row.size:,}")

    if deep:
        try:
            got = md5_irods(session, path).lower()
        except Exception as err:
            return Finding(row, "unreadable",
                           f"{path} ({type(err).__name__})")
        if row.md5 and got != row.md5.lower():
            return Finding(row, "checksum",
                           f"{path} read={got} row={row.md5}")
        return Finding(row, "ok", "bytes read back and hashed")

    registered = (obj.checksum or "").replace("sha2:", "").strip().lower()
    if not registered:
        # Never forced -- see the module docstring on HIERARCHY_ERROR.
        return Finding(row, "no-checksum", f"{path} has none registered")
    if row.md5 and registered != row.md5.lower():
        return Finding(row, "checksum",
                       f"{path} registered={registered} row={row.md5}")

    return Finding(row, "ok", "")


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    conn = psycopg2.connect(find_dsn(args.server))
    # Read-only is not lock-free -- see reconcile_release_files.py. This
    # sweep holds its connection open across every IRODS lookup, so a snapshot
    # would outlive any migration someone tries to apply while it runs.
    conn.autocommit = True
    cur = conn.cursor()

    ids = target_ids(cur, args)
    if not ids:
        sys.exit("No simulations in scope")
    rows = load_rows(cur, args, ids)
    if not rows:
        sys.exit(f"{len(ids):,} simulation(s) in scope but no file rows")

    print(f"{len(rows):,} file row(s) across {len(ids):,} simulation(s), "
          f"server {args.server}{'  [DEEP]' if args.deep else ''}", flush=True)

    root = f"/iplant/home/shared/mdrepo/{args.server}/release"
    irods_env = os.environ.get(
        "IRODS_ENVIRONMENT_FILE",
        os.path.expanduser("~/.irods/irods_environment.json"),
    )

    findings: List[Finding] = []
    with iRODSSession(irods_env_file=irods_env,
                      connection_timeout=300) as session:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for num, finding in enumerate(
                pool.map(lambda r: inspect(session, root, r, args.deep), rows),
                start=1,
            ):
                findings.append(finding)
                if finding.verdict != "ok":
                    print(f"  {finding.verdict.upper():<12} "
                          f"{finding.row.simulation_id} "
                          f"{finding.row.filename}: {finding.detail}",
                          flush=True)
                if num % 500 == 0:
                    print(f"  .. {num:,}/{len(rows):,}", flush=True)

    tally: dict = {}
    for f in findings:
        tally[f.verdict] = tally.get(f.verdict, 0) + 1
    print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    bad = [f for f in findings if f.verdict != "ok"]

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("table\tsimulation_id\tfilename\tverdict\tdetail\n")
            for f in findings:
                fh.write(f"{f.row.table}\t{f.row.simulation_id}\t"
                         f"{f.row.filename}\t{f.verdict}\t"
                         f"{' '.join(f.detail.split())}\n")
        print(f"Report: {args.out}")

    if args.manifest_out and bad:
        with open(args.manifest_out, "w") as fh:
            fh.write("simulation_id\tfilename\tlocal_file\n")
            for f in bad:
                fh.write(f"{f.row.simulation_id}\t{f.row.filename}\t\n")
        print(f"Manifest (local paths left blank, fill before use): "
              f"{args.manifest_out}")

    # Non-zero so this can gate a wave's completion in a script rather than
    # needing someone to read the tally.
    sys.exit(1 if bad else 0)


# --------------------------------------------------
if __name__ == "__main__":
    main()
