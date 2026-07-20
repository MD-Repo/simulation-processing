#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-01-22
Purpose: Import preprocessed JSON
"""

import argparse
import json
import os
import psycopg2
import psycopg2.extras
import sys
from datetime import datetime, timezone
from dotenv import dotenv_values
from typing import List, NamedTuple, TextIO, Optional

ADMIN_ORCID = "0000-0000-0000-0000"


class Args(NamedTuple):
    """Command-line arguments"""

    file: str
    server: str
    out_file: TextIO
    data_dir: str
    simulation_id: Optional[int]
    replace_original_files: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Import preprocessed JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-f", "--file", help="Input JSON file", metavar="FILE")

    parser.add_argument(
        "-s",
        "--server",
        help="Target server",
        metavar="STR",
        choices=["staging", "prod"],
        default="staging",
    )

    parser.add_argument(
        "-o",
        "--out-file",
        help="Output file",
        metavar="FILE",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--data-dir",
        help="Data directory",
        metavar="DIR",
        required=True,
        # default="/mdrepotmp/kinase/upload/upload_dir",
        # default="/opt/mdrepo/uploads",
    )

    parser.add_argument(
        "--simulation-id",
        type=int,
        help="Simulation ID (if reprocessing, will remove existing processed files)",
    )

    parser.add_argument(
        "--replace-original-files",
        action="store_true",
        help="Delete original/uploaded files",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        parser.error(f"Invalid --data-dir '{args.data_dir}'")

    return Args(
        file=args.file,
        server=args.server,
        out_file=open(args.out_file, "wt"),
        data_dir=args.data_dir,
        simulation_id=args.simulation_id,
        replace_original_files=args.replace_original_files,
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()

    env_key = "PRODUCTION_DSN" if args.server == "prod" else "STAGING_DSN"
    dot_env = dotenv_values()
    dsn = dot_env.get(env_key, os.environ.get(env_key, ""))
    if not dsn:
        sys.exit(f"Cannot find environment '{env_key}'")

    print(f"Connecting to '{args.server}'")
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    print(f"Importing '{args.file}'")
    # root = os.path.splitext(os.path.basename(args.file))[0]
    data = json.loads(open(args.file).read())
    sim = data["simulation"]
    sim_id = get_simulation(cur, sim, args.simulation_id)
    mdrepo_id = f"MDR{sim_id:08d}"
    print(f" => {mdrepo_id}")

    if args.simulation_id:
        print("Removing previous processed files")
        cur.execute(
            """
            select id
            from   md_processed_file
            where  simulation_id=%s
            """,
            (sim_id,),
        )

        for res in cur.fetchall():
            cur.execute(
                """
                delete
                from   md_frontend_download_instance_processed_files
                where  simulationprocessedfile_id=%s
                """,
                (res[0],),
            )

        cur.execute(
            """
            delete
            from   md_processed_file
            where  simulation_id=%s
            """,
            (sim_id,),
        )

        if args.replace_original_files:
            print("Removing previous uploaded files")
            cur.execute(
                """
                select id
                from   md_uploaded_file
                where  simulation_id=%s
                """,
                (sim_id,),
            )

            for res in cur.fetchall():
                cur.execute(
                    """
                    delete
                    from   md_frontend_download_instance_uploaded_files
                    where  simulationuploadedfile_id=%s
                    """,
                    (res[0],),
                )

            cur.execute(
                """
                delete
                from   md_uploaded_file
                where  simulation_id=%s
                """,
                (sim_id,),
            )

    for file in sim.get("original_files", []):
        file_id = create_uploaded_file(cur, sim_id, mdrepo_id, "uploaded", file)
        print(f"Original file {file['file_type']} => {file_id}")

    for file in sim["processed_files"]:
        file_id = create_processed_file(cur, sim_id, mdrepo_id, "processed", file)
        print(f"Processed file {file['file_type']} => {file_id}")

    for trajectory_file_name in sim.get("replicates", []):
        replicate_id = create_replicate(cur, sim_id, trajectory_file_name)
        print(f"Replicate '{trajectory_file_name}' => {replicate_id}")

    for rank, contributor in enumerate(sim.get("contributors", []), start=1):
        c_id = create_contributor(cur, sim_id, contributor, rank)
        print(f"Contributor {contributor['name']} => {c_id}")

    for ligand in sim.get("ligands", []):
        ligand_id = create_ligand(cur, sim_id, ligand)
        print(f"Ligand {ligand['name']} => {ligand_id}")

    for solute in sim.get("solutes", []):
        if solute_id := create_solute(cur, sim_id, solute):
            print(f"Solute {solute['name']} => {solute_id}")

    for link in sim.get("external_links", []):
        if link_id := create_external_link(cur, sim_id, link):
            print(f"External link {link['url']} => {link_id}")

    for paper in sim.get("papers", []):
        paper_id = create_paper(cur, sim_id, paper)
        print(f"Paper {paper['title']} => {paper_id}")

    for uniprot in sim.get("uniprots", []):
        uniprot_id = create_uniprot(cur, sim_id, uniprot)
        print(f"Uniprot {uniprot['name']} => {uniprot_id}")

    if pdb := sim.get("pdb"):
        pdb_id = create_pdb(cur, sim_id, pdb)
        print(f"PDB {pdb['pdb_id']} => {pdb_id}")

    output = {
        "server": args.server,
        "filename": args.file,
        "simulation_id": sim_id,
        "data_dir": args.data_dir,
    }
    print(json.dumps(output, indent=4), file=args.out_file)
    print(f'Done, see "{args.out_file.name}".')


# --------------------------------------------------
def get_simulation(cur, sim, sim_id) -> int:
    """Find or create simulation"""

    user_id = get_user(cur, sim["lead_contributor_orcid"])
    software_id = create_software(
        cur, sim["software_name"], sim.get("software_version")
    )
    is_placeholder = True
    is_deprecated = False

    if not sim_id and "simulation_id" in sim:
        cur.execute(
            """
            select count(*)
            from   md_simulation
            where  id=%s
            """,
            (sim["simulation_id"],),
        )
        count = cur.fetchone()[0]
        if count == 1:
            sim_id = sim["simulation_id"]
        else:
            sys.exit(f'Invalid simulation ID: sim["simulation_id"]')

    alias = sim.get("alias")
    if alias and not sim_id:
        cur.execute(
            """
            select id
            from   md_simulation
            where  alias=%s
            and    created_by_id=%s
            """,
            (alias, user_id),
        )
        if res := cur.fetchone():
            sim_id = res[0]

    if not sim_id:
        print(f"Searching unique_file_hash_string")
        cur.execute(
            """
            select id
            from   md_simulation
            where  unique_file_hash_string=%s
            """,
            (sim["unique_file_hash_string"],),
        )

        if res := cur.fetchone():
            sim_id = res[0]

    if sim_id:
        # Update existing simulation
        cur.execute(
            """
            update md_simulation
            set    software_id=%s,
                   created_by_id=%s,
                   unique_file_hash_string=%s,
                   alias=%s,
                   description=%s,
                   short_description=%s,
                   run_commands=%s,
                   duration=%s,
                   sampling_frequency=%s,
                   integration_timestep_fs=%s,
                   water_type=%s,
                   water_density=%s,
                   rmsd_values=%s,
                   rmsf_values=%s,
                   forcefield=%s,
                   forcefield_comments=%s,
                   fasta_sequence=%s,
                   num_replicates=%s,
                   temperature=%s,
                   protonation_method=%s,
                   is_placeholder=%s,
                   is_deprecated=%s,
                   is_embargoed=%s,
                   is_coarse_grained=%s
            where  id=%s
            """,
            (
                software_id,
                user_id,
                sim["unique_file_hash_string"],
                sim.get("alias"),
                sim.get("description"),
                sim["short_description"],
                sim.get("run_commands"),
                sim["duration"],
                sim["sampling_frequency"],
                sim["integration_timestep_fs"],
                sim.get("water_type", ""),
                sim.get("water_density", None),
                sim["rmsd_values"],
                sim["rmsf_values"],
                sim.get("forcefield", ""),
                sim.get("forcefield_comments", ""),
                sim["fasta_sequence"],
                sim["num_replicates"],
                sim["temperature_kelvin"],
                sim.get("protonation_method"),
                is_placeholder,
                is_deprecated,
                sim.get("is_embargoed", False),
                sim.get("is_coarse_graing", False),
                sim_id,
            ),
        )
    else:
        # Creat new simulation
        is_public = False
        cur.execute(
            """
            insert
            into  md_simulation
                  (unique_file_hash_string,
                   alias,
                   software_id,
                   created_by_id,
                   run_commands,
                   description,
                   short_description,
                   duration,
                   sampling_frequency,
                   integration_timestep_fs,
                   water_type,
                   water_density,
                   rmsd_values,
                   rmsf_values,
                   forcefield,
                   forcefield_comments,
                   fasta_sequence,
                   num_replicates,
                   temperature,
                   protonation_method,
                   is_placeholder,
                   is_deprecated,
                   is_public,
                   is_embargoed,
                   is_coarse_grained,
                   creation_date)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning id;
            """,
            (
                sim["unique_file_hash_string"],
                sim.get("alias"),
                software_id,
                user_id,
                sim.get("run_commands"),
                sim.get("description"),
                sim["short_description"],
                sim["duration"],
                sim["sampling_frequency"],
                sim["integration_timestep_fs"],
                sim.get("water_type", ""),
                sim.get("water_density", None),
                sim["rmsd_values"],
                sim["rmsf_values"],
                sim.get("forcefield", ""),
                sim.get("forcefield_comments", ""),
                sim["fasta_sequence"],
                sim["num_replicates"],
                sim["temperature_kelvin"],
                sim.get("protonation_method"),
                is_placeholder,
                is_deprecated,
                is_public,
                sim.get("is_embargoed", False),
                sim.get("is_coarse_grained", False),
                datetime.now(timezone.utc),
            ),
        )
        sim_id = cur.fetchone()[0]

    return sim_id


# --------------------------------------------------
def create_software(cur, name, version) -> int:
    """Get or create software"""

    cur.execute(
        """
        select id
        from   md_software
        where  name=%s
        and    version=%s
        """,
        (name, version),
    )

    if res := cur.fetchone():
        return res[0]

    cur.execute(
        """
        insert
        into   md_software (name, version)
        values (%s, %s)
        returning id;
        """,
        (name, version),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def get_user(cur, orcid) -> Optional[int]:
    if orcid == ADMIN_ORCID:
        return

    cur.execute(
        """
        select u.id, u.username
        from   socialaccount_socialaccount s, md_user u
        where  s.provider='orcid'
        and    s.uid=%s
        and    s.user_id=u.id
        """,
        (orcid,),
    )

    if res := cur.fetchone():
        return res[0]

    sys.exit(f"Failed to find ORCID '{orcid}'")


# --------------------------------------------------
def create_contributor(cur, sim_id, contributor, rank) -> int:
    name = contributor.get("name", "")
    email = contributor.get("email", "")
    orcid = contributor.get("orcid", "")
    institution = contributor.get("institution", "")
    rank = contributor.get("rank", rank)

    qry = ()
    if orcid:
        qry = ("orcid", orcid)
    elif email:
        qry = ("email", email)
    elif name:
        qry = ("name", name)
    else:
        sys.exit(f"Bad contributor {contributor}")

    cur.execute(
        f"""
        select id
        from   md_contribution
        where  simulation_id=%s
        and    {qry[0]}=%s
        """,
        (sim_id, qry[1]),
    )

    if res := cur.fetchone():
        contributor_id = res[0]
        cur.execute(
            """
            update md_contribution
            set    orcid=%s, name=%s, email=%s, institution=%s, rank=%s
            where  id=%s
            """,
            (orcid, name, email, institution, rank, contributor_id),
        )
        return contributor_id

    cur.execute(
        """
        insert
        into   md_contribution
               (name, email, institution, orcid, simulation_id, rank)
        values (%s, %s, %s, %s, %s, %s)
        returning id
        """,
        (name, email, institution, orcid, sim_id, rank),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_processed_file(cur, sim_id, mdrepo_id, table_name, file) -> int:
    """Get or create processed file"""

    filename = os.path.basename(file["name"])
    file_type = file["file_type"]
    file_size_bytes = file["size"]
    md5_hash = file["md5_sum"]
    description = file.get("description")
    local_file_path = os.path.join(
        mdrepo_id,
        "original" if table_name == "uploaded" else "processed",
        filename,
    )

    cur.execute(
        """
        select id
        from   md_processed_file
        where  simulation_id=%s
        and    filename=%s
        """,
        (sim_id, filename),
    )

    if res := cur.fetchone():
        file_id = res[0]
        cur.execute(
            """
            update md_processed_file
            set    file_type=%s, description=%s, local_file_path=%s,
                   md5_hash=%s, file_size_bytes=%s
            where  id=%s
            """,
            (
                file_type,
                description,
                local_file_path,
                md5_hash,
                file_size_bytes,
                file_id,
            ),
        )

        return file_id

    cur.execute(
        """
        insert
        into   md_processed_file
               (simulation_id,
                filename,
                file_type,
                description,
                local_file_path,
                file_size_bytes,
                md5_hash)
        values (%s, %s, %s, %s, %s, %s, %s)
        returning id
        """,
        (
            sim_id,
            filename,
            file_type,
            description,
            local_file_path,
            file_size_bytes,
            md5_hash,
        ),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_uploaded_file(cur, sim_id, mdrepo_id, table_name, file) -> int:
    """Get or create uploaded file"""

    filename = file["name"]
    file_type = file["file_type"]
    file_size_bytes = file["size"]
    md5_hash = file["md5_sum"]
    description = file.get("description")
    is_primary = file.get("is_primary", False)
    local_file_path = os.path.join(
        mdrepo_id,
        "original" if table_name == "uploaded" else "processed",
        filename,
    )

    cur.execute(
        """
        select id
        from   md_uploaded_file
        where  simulation_id=%s
        and    filename=%s
        """,
        (sim_id, filename),
    )

    if res := cur.fetchone():
        file_id = res[0]
        cur.execute(
            """
            update md_uploaded_file
            set    file_type=%s, description=%s, local_file_path=%s,
                   md5_hash=%s, file_size_bytes=%s, is_primary=%s
            where  id=%s
            """,
            (
                file_type,
                description,
                local_file_path,
                md5_hash,
                file_size_bytes,
                is_primary,
                file_id,
            ),
        )

        return file_id

    cur.execute(
        """
        insert
        into   md_uploaded_file
               (simulation_id,
                filename,
                file_type,
                description,
                local_file_path,
                file_size_bytes,
                md5_hash,
                is_primary)
        values (%s, %s, %s, %s, %s, %s, %s, %s)
        returning id
        """,
        (
            sim_id,
            filename,
            file_type,
            description,
            local_file_path,
            file_size_bytes,
            md5_hash,
            is_primary,
        ),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_uniprot(cur, sim_id, uniprot) -> int:
    uniprot_id = uniprot["uniprot_id"]
    name = uniprot["name"]
    sequence = uniprot["sequence"]

    cur.execute(
        f"""
        select id
        from   md_uniprot
        where  uniprot_id=%s
        """,
        (uniprot_id,),
    )

    uniprot_pk = None
    if res := cur.fetchone():
        uniprot_pk = res[0]
        cur.execute(
            """
            update md_uniprot
            set    name=%s, amino_length=%s, sequence=%s
            where  id=%s
            """,
            (name, len(sequence), sequence, uniprot_pk),
        )
    else:
        cur.execute(
            """
            insert
            into   md_uniprot (uniprot_id, name, amino_length, sequence)
            values (%s, %s, %s, %s)
            returning id;
            """,
            (uniprot_id, name, len(sequence), sequence),
        )
        uniprot_pk = cur.fetchone()[0]

    cur.execute(
        """
        select id
        from   md_simulation_uniprot
        where  uniprot_id=%s
        and    simulation_id=%s
        """,
        (uniprot_pk, sim_id),
    )

    if res := cur.fetchone():
        return res[0]

    cur.execute(
        """
        insert
        into   md_simulation_uniprot (uniprot_id, simulation_id)
        values (%s, %s)
        returning id;
        """,
        (uniprot_pk, sim_id),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_pdb(cur, sim_id, pdb) -> int:
    pdb_id = pdb["pdb_id"].lower()
    title = pdb["title"]
    classification = pdb["classification"]

    cur.execute(
        """
        select id
        from   md_pdb
        where  pdb_id=%s
        """,
        (pdb_id,),
    )

    pdb_pk = None
    if res := cur.fetchone():
        pdb_pk = res[0]
        cur.execute(
            """
            update md_pdb
            set    title=%s, classification=%s
            where  id=%s
            """,
            (title, classification, pdb_pk),
        )
    else:
        cur.execute(
            """
            insert
            into    md_pdb (pdb_id, title, classification)
            values  (%s, %s, %s)
            returning id
            """,
            (pdb_id, title, classification),
        )
        pdb_pk = cur.fetchone()[0]

    if not pdb_pk:
        sys.exit(f"Failed to get PDB '{pdb_id}'")

    cur.execute(
        """
        update md_simulation
        set    pdb_id=%s
        where  id=%s
        """,
        (pdb_pk, sim_id),
    )

    return pdb_pk


# --------------------------------------------------
def create_ligand(cur, sim_id, ligand) -> int:
    name = ligand["name"]
    smiles = ligand["smiles"]

    cur.execute(
        f"""
        select id
        from   md_ligand
        where  name=%s
        and    simulation_id=%s
        """,
        (name, sim_id),
    )

    if res := cur.fetchone():
        ligand_id = res[0]
        cur.execute(
            """
            update md_ligand
            set    smiles_string=%s
            where  id=%s
            """,
            (smiles, ligand_id),
        )
        return ligand_id

    cur.execute(
        """
        insert
        into   md_ligand (name, smiles_string, simulation_id)
        values (%s, %s, %s)
        returning id;
        """,
        (name, smiles, sim_id),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_external_link(cur, sim_id, link) -> Optional[int]:
    url = link["url"]

    cur.execute(
        f"""
        select id
        from   md_external_link
        where  url=%s
        and    simulation_id=%s
        """,
        (url, sim_id),
    )

    label = link.get("label")
    if res := cur.fetchone():
        link_id = res[0]
        if label:
            cur.execute(
                """
                update md_external_url
                set    label=%s
                where  id=%s
                """,
                (label, link_id),
            )
        return link_id

    cur.execute(
        """
        insert
        into   md_external_link
               (url, label, simulation_id)
        values (%s, %s, %s)
        returning id
        """,
        (url, label, sim_id),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_solute(cur, sim_id, solute) -> Optional[int]:
    rename = {"Na": "Na+", "Cl": "Cl+", "K": "K+"}
    name = solute["name"]
    name = rename.get(name, name)
    concentration = solute["concentration_mol_liter"]

    cur.execute(
        f"""
        select id
        from   md_solute
        where  name=%s
        and    simulation_id=%s
        """,
        (name, sim_id),
    )

    if res := cur.fetchone():
        solute_id = res[0]
        cur.execute(
            """
            update md_solute
            set    concentration=%s
            where  id=%s
            """,
            (concentration, solute_id),
        )
        return solute_id

    cur.execute(
        """
        insert
        into   md_solute
               (name, concentration, simulation_id)
        values (%s, %s, %s)
        returning id
        """,
        (name, concentration, sim_id),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_paper(cur, sim_id, paper) -> int:
    title = paper["title"]
    authors = paper["authors"]
    journal = paper["journal"]
    volume = paper["volume"]
    year = paper["year"]
    doi = paper["doi"]

    if doi:
        cur.execute(
            """
            select id
            from   md_pub
            where  doi=%s
            """,
            (doi,),
        )
    else:
        cur.execute(
            """
            select id
            from   md_pub
            where  title=%s
            and    authors=%s
            and    journal=%s
            and    volume=%s
            and    year=%s
            """,
            (title, authors, journal, volume, year),
        )

    pub_id = ""
    if res := cur.fetchone():
        pub_id = res[0]
    else:
        cur.execute(
            """
            insert
            into   md_pub
                   (title, authors, journal, volume, year, doi)
            values (%s, %s, %s, %s, %s, %s)
            returning id;
            """,
            (title, authors, journal, volume, year, doi),
        )
        pub_id = cur.fetchone()[0]

    if not pub_id:
        sys.exit(f"Failed to get pub_id for {paper}")

    cur.execute(
        """
        select id
        from   md_simulation_pub
        where  pub_id=%s
        and    simulation_id=%s
        """,
        (pub_id, sim_id),
    )

    if res := cur.fetchone():
        return res[0]

    cur.execute(
        """
        insert
        into   md_simulation_pub (pub_id, simulation_id)
        values (%s, %s)
        returning id
        """,
        (pub_id, sim_id),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
def create_replicate(cur, sim_id, trajectory_file_name) -> int:
    cur.execute(
        """
        select id
        from   md_replicate
        where  simulation_id=%s
        and    trajectory_file_name=%s
        """,
        (sim_id, trajectory_file_name),
    )

    if res := cur.fetchone():
        return res[0]

    cur.execute(
        """
        insert
        into   md_replicate (simulation_id, trajectory_file_name)
        values (%s, %s)
        returning id
        """,
        (sim_id, trajectory_file_name),
    )

    return cur.fetchone()[0]


# --------------------------------------------------
if __name__ == "__main__":
    main()
