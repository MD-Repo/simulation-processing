#!/usr/bin/env python3
"""
Author : exouser <exouser@localhost>
Date   : 2026-02-13
Purpose: Create Uniprot/PDB/InChIKey mapping file

WHERE THESE FILES END UP. Written locally to --out-file, pushed over SSH to
MEDIA_HOST at /home/web/mdrepo/static/mapping/<basename>, and served from
**https://static.mdrepo.org/mapping/<basename>** -- a different host and a
different path from the one they are written to, and not guessable from
either. Recorded here because it was not written down anywhere and probing
mdrepo.org for it only produces 404s.

  uniprot.txt   https://static.mdrepo.org/mapping/uniprot.txt
  pdb.txt       https://static.mdrepo.org/mapping/pdb.txt
  inchikey.txt  https://static.mdrepo.org/mapping/inchikey.txt

The consumers are external: PDB, UniProt, and as of 2026-09-09 MDDB, which is
what inchikey.txt was added for. That is also why VISIBLE_SIM below is not
negotiable -- a mistake in it is a disclosure to a third party, not an
internal display bug.
"""

import argparse
import csv
import fabric
import os
import psycopg2
import psycopg2.extras
import sys
from dotenv import dotenv_values
from typing import NamedTuple


# What "publicly visible" means, as a SQL predicate on an md_simulation
# aliased "s". Mirrors visible_simulations_q() in the Django app
# (md_repo_app/models/simulation/simulation_filters.py), which is the source of
# truth. All four conditions are required: is_public alone is not enough,
# because embargo outlives approval. Without this the simulations column would
# publish embargoed and unapproved simulations to PDB, UniProt and MDDB -- the
# same leak the API serializers had, see the comment on
# PdbDetailSerializer.get_simulations.
VISIBLE_SIM = """
    s.is_public
    and not s.is_deprecated
    and not s.is_placeholder
    and not s.is_embargoed
"""


class Args(NamedTuple):
    """Command-line arguments"""

    mapping: str
    out_file: str
    delimiter: str


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Create Uniprot/PDB/InChIKey mapping file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "mapping",
        help="Mapping file",
        metavar="STR",
        choices=["uniprot", "pdb", "inchikey"],
    )

    parser.add_argument(
        "-o", "--out-file", help="Output file", metavar="FILE"
    )

    parser.add_argument(
        "-d",
        "--delimiter",
        help="Field delimiter",
        metavar="STR",
        default="\t",
    )

    args = parser.parse_args()

    return Args(
        mapping=args.mapping,
        out_file=args.out_file or f"{args.mapping}.txt",
        delimiter=args.delimiter,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    # Name the .env beside this file rather than leaning on bare
    # dotenv_values(). Not because the cwd would break it -- it routes through
    # find_dotenv(), which defaults to usecwd=False and walks up from the
    # *calling file*, so a bare call already resolves here from any cwd (the
    # cwd is used only in a REPL, under a debugger, or in a frozen build).
    # Being explicit just keeps that subtlety from being load-bearing when the
    # cron line drops its "cd".
    env = dotenv_values(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    )

    def get_env(key):
        if val := env.get(key, os.environ.get(key, None)):
            return val
        else:
            sys.exit(f"Missing env '{key}'")

    dsn = get_env("PRODUCTION_DSN")
    media_host = get_env("MEDIA_HOST")
    media_port = get_env("MEDIA_PORT")
    media_user = get_env("MEDIA_USER")
    media_pass = get_env("MEDIA_PASSWORD")
    media_server = fabric.Connection(
        media_host,
        port=media_port,
        user=media_user,
        connect_kwargs={"password": media_pass},
    )

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    data = []
    fieldnames = []

    # The "simulations" column is a scalar subquery rather than a join with a
    # GROUP BY so that the row set stays exactly what it was before the column
    # existed -- one row per md_pdb/md_uniprot row, IDs with no visible
    # simulation included with an empty cell (7,027 of 28,903 for pdb, 4,173 of
    # 12,462 for uniprot as of 2026-08-06). PDB and UniProt have been ingesting
    # those rows all along; adding a column is a change they can absorb,
    # dropping rows is not.
    #
    # The slug is built here to match Simulation.slug ("MDR%08d") in the Django
    # app; it is what /explore/<slug> takes and what MDDB will link against.
    # Ordered by s.id so a given day's file is byte-stable when nothing changed.
    if args.mapping == "uniprot":
        cur.execute(f"""
            select   u.uniprot_id,
                     'https://mdrepo.org/uniprot/view/' || u.uniprot_id as url,
                     coalesce((
                         select string_agg('MDR' || lpad(s.id::text, 8, '0'),
                                           ',' order by s.id)
                         from   md_simulation s
                                join md_simulation_uniprot su
                                  on su.simulation_id = s.id
                         where  su.uniprot_id = u.id
                           and  {VISIBLE_SIM}
                     ), '') as simulations
            from     md_uniprot u
            order by 1
            """)
        data = cur.fetchall()
        fieldnames = ["uniprot_id", "url", "simulations"]
    elif args.mapping == "inchikey":
        # THE ONLY ONE OF THE THREE WITH NO DEDUP TABLE BEHIND IT, and that
        # changes the shape of the query rather than just its columns.
        # `md_pdb` and `md_uniprot` are one row per identifier with a join
        # table to simulations, which is what lets the two queries above stay
        # one-row-per-identifier and keep an empty cell for an ID no visible
        # simulation uses. `md_ligand` carries `simulation_id` directly, so a
        # key exists only where a ligand row does: grouping is the only way to
        # get one row per InChIKey, and "identifiers with no visible
        # simulation" stops being a category, because there is no standalone
        # InChIKey table for one to live in. Nothing is dropped that the other
        # two would have kept -- such a row could not be constructed.
        #
        # DISTINCT because one simulation can carry the same molecule twice
        # (two ligand rows, one InChIKey), which would otherwise repeat its
        # slug in the cell. The slug is zero-padded, so ordering it as text
        # orders it numerically, and the ORDER BY has to repeat the whole
        # expression: Postgres requires the sort key to appear in the
        # aggregate's argument list when DISTINCT is used.
        #
        # `inchikey is not null` skips the ligands nothing could compute a key
        # for -- 32 rows as of 2026-09-09, all of them molecules RDKit refuses,
        # about half malformed SMILES and half impossible valences. They have
        # no identifier to publish, which is the honest reason to omit them.
        #
        # The URL carries a "view" segment, matching the other two.
        #
        # VERIFY THIS AGAINST THE ROUTER, NOT AGAINST AN HTTP STATUS. The site
        # is a single-page app: every path answers 200, including nonsense, so
        # a wrong URL here publishes thousands of links that satisfy any
        # checker and render Not Found to a person. The authority is the Elm
        # page layout -- `Pages/Inchikey/View/Id_.elm` -- and, for what is
        # actually live, the deployed bundle, which encodes the route as
        # `"inchikey","view"` exactly as it does `"uniprot","view"`.
        #
        # Worth re-checking rather than assuming: this route shipped on
        # 2026-09-09 as `/inchikey/:id` and was renamed to `/inchikey/view/:id`
        # the same day (elm-mdrepo 88bf2c6), while this exporter was being
        # written against the first spelling.
        cur.execute(f"""
            select   l.inchikey,
                     'https://mdrepo.org/inchikey/view/' || l.inchikey as url,
                     string_agg(
                         distinct 'MDR' || lpad(s.id::text, 8, '0'),
                         ',' order by 'MDR' || lpad(s.id::text, 8, '0')
                     ) as simulations
            from     md_ligand l
                     join md_simulation s on s.id = l.simulation_id
            where    l.inchikey is not null
              and    {VISIBLE_SIM}
            group by l.inchikey
            order by 1
            """)
        data = cur.fetchall()
        fieldnames = ["inchikey", "url", "simulations"]
    else:
        cur.execute(f"""
            select   p.pdb_id,
                     'https://mdrepo.org/pdb/view/' || p.pdb_id as url,
                     coalesce((
                         select string_agg('MDR' || lpad(s.id::text, 8, '0'),
                                           ',' order by s.id)
                         from   md_simulation s
                         where  s.pdb_id = p.id
                           and  {VISIBLE_SIM}
                     ), '') as simulations
            from     md_pdb p
            order by 1
            """)
        data = cur.fetchall()
        fieldnames = ["pdb_id", "url", "simulations"]

    # Close before pushing. Written open, the last partial buffer is still in
    # memory when put() reads the file, so what lands on the media host is
    # truncated at a block boundary -- 370,109 of 484,413 bytes for uniprot,
    # 23.6% of the mapping silently missing, every time this has ever run.
    # newline="" as the csv module documents: it stops the file object doing
    # its own newline translation, which makes lineterminator below the only
    # thing deciding the line ending.
    #
    # lineterminator="\n" because csv defaults to "\r\n". Every version of
    # these files from 2026-02-13 to 2026-08-06 went out CRLF; they are Unix
    # text now. The visible consequence for a consumer splitting on "\n"
    # without stripping is that the last field no longer carries a trailing
    # "\r" -- which used to be the url, and since 2026-08-06 was the
    # simulations column.
    with open(args.out_file, "wt", encoding="UTF-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames, delimiter=args.delimiter, lineterminator="\n"
        )
        writer.writeheader()
        for row in data:
            writer.writerow(dict(row))

    # basename, because os.path.join discards the directory when the second
    # argument is absolute: an absolute --out-file would silently retarget the
    # REMOTE path to it. The local file may live anywhere; the remote name is
    # always the bare file in the mapping directory.
    media_path = os.path.join(
        "/home/web/mdrepo/static/mapping", os.path.basename(args.out_file)
    )
    media_server.put(args.out_file, remote=media_path)
    os.remove(args.out_file)
    print(f"Done, pushed mapping to '{media_host}:{media_path}'")


# --------------------------------------------------
if __name__ == "__main__":
    main()
