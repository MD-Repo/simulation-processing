#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-14
Purpose: Fill in a missing ligand "smiles" in mdrepo-metadata.toml from a
         pdb_id -> SMILES table.

         The DDD/PDBbind bundles in IRODS bundles2/ declare a ligand by name
         and no SMILES, so mdr-process refuses them before it does any work:

             Error: Failed to parse input: TOML parse error at line 25
             [[ligands]]  missing field `smiles`

         The table is keyed on pdb_id, which the TOML also carries, so the
         join is exact and needs nothing inferred from the structure files.

         --smiles overrides the table lookup entirely, for the case the
         table cannot serve: a value taken from a SIBLING SIMULATION already in
         our own database. The PDBbind rows are keyed on pdb_id and carry no
         stereochemistry at all -- 2g97 gives "...C=Cc3cccnc3" and 3g8e gives
         "...C2C(C(C(O2)CO)O)O" -- so filling from the table would discard the
         E-alkene and the ribose configuration that the sibling records and the
         coordinate files both support. --source is mandatory alongside it, so
         the release TOML says where the string came from.

         Edits the TOML TEXTUALLY -- one inserted line -- rather than
         re-serialising it. A round trip through a TOML writer reorders keys,
         drops comments and reformats the long arrays these files carry,
         which turns a one-line fix into an unreviewable diff.

         Output is Unix LF. The DDD TOMLs arrive CRLF, so the written file
         differs from the original on every line even though one key changed
         -- accepted deliberately (Ken, 2026-08-14). What makes that safe is
         the backup: the original bytes are copied to "<name>.orig" before
         anything is written, and an existing .orig is never overwritten, so
         re-running keeps the true original rather than a previous edit.
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import date
from typing import Dict, List, NamedTuple, Optional, Tuple

import toml

METADATA_NAME = "mdrepo-metadata.toml"
DEFAULT_TABLE = os.path.expanduser("~/pdbbind_ligand_smiles.tsv")

# A submitter template whose field name was never replaced with a value:
# `smiles = "smiles_string"`. It is present as far as TOML is concerned, so
# the tool used to report "already has smiles" and refuse to act, while
# mdr-process died on it at validation -- OpenBabel rejects it with "SMILES
# string contains a character 'm' which is invalid". Found 2026-08-17 on
# MDR00004396, which could not be reprocessed for that reason; 4 rows in
# md_ligand carry it, on MDR00004394-4397, all one ligand under pdb 2g97.
#
# Only this exact literal counts. A SMILES that is merely WRONG must not be
# swept in: MDR00004406-4409 carry a real string with a one-character
# corruption (`c3` written as `s`), and it holds stereochemistry the PDBbind
# row does not, so overwriting it from the table would lose information the
# submitter got right. That one is a targeted repair, not a fill.
PLACEHOLDER = "smiles_string"


# --------------------------------------------------
def is_unset(value: Optional[str]) -> bool:
    """Is this ligand's SMILES absent, blank, or the template placeholder?"""

    return not value or value.strip() == PLACEHOLDER


class Entry(NamedTuple):
    ligand_name: str
    smiles: str


class Args(NamedTuple):
    targets: List[str]
    table: str
    dry_run: bool
    allow_name_mismatch: bool
    smiles: Optional[str]
    source: Optional[str]
    replace: bool


# --------------------------------------------------
def get_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Add a missing ligand SMILES to mdrepo-metadata.toml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", metavar="PATH", nargs="+",
                        help="Bundle directories, or TOML files directly")
    parser.add_argument("-t", "--table", metavar="TSV", default=DEFAULT_TABLE,
                        help="pdb_id/ligand_name/canonical_smiles table")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Show the line that would be added, change nothing")
    parser.add_argument("--allow-name-mismatch", action="store_true",
                        help="Take the table's SMILES even when its ligand "
                             "name disagrees with the TOML's. Off by default: "
                             "a disagreement means one of the two is about a "
                             "different molecule, and PDBbind has been wrong "
                             "at least once (see 186l in TODO MDR-40)")
    parser.add_argument("--smiles", metavar="SMILES",
                        help="Write THIS SMILES instead of looking one up in "
                             "the table. For a value taken from a sibling "
                             "simulation already in our own database, which "
                             "the pdb_id-keyed table cannot express -- the "
                             "PDBbind rows carry no stereochemistry. Pair it "
                             "with --source to say where it came from.")
    parser.add_argument("--source", metavar="TEXT",
                        help="Provenance note written above the key. Required "
                             "with --smiles: an unattributed third-party "
                             "string in a release TOML is exactly what the "
                             "comment exists to prevent.")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite a smiles that is PRESENT and is not "
                             "the placeholder. Only valid with --smiles. This "
                             "is how a corrupted string gets repaired: the "
                             "3g8e ligand on MDR00004406-4409 carries `c3` "
                             "written as `s`, which is a real value as far as "
                             "TOML is concerned, so the ordinary fill path "
                             "refuses it on purpose.")
    args = parser.parse_args()

    if args.replace and not args.smiles:
        parser.error("--replace is only valid with --smiles; there is no "
                     "table path that should ever overwrite a real value")
    if args.smiles and not args.source:
        parser.error("--smiles requires --source to record where it came from")
    if args.smiles:
        # The name check compares the TOML against the TABLE's name, and with
        # --smiles there is no table row in play. Silently leaving it "on"
        # would suggest a check is happening that is not.
        if args.allow_name_mismatch:
            parser.error("--allow-name-mismatch is meaningless with --smiles; "
                         "there is no table name to disagree with")
        try:
            toml_literal(args.smiles)
        except ValueError as err:
            parser.error(str(err))

    return Args(args.target, args.table, args.dry_run,
                args.allow_name_mismatch, args.smiles, args.source,
                args.replace)


# --------------------------------------------------
def load_table(path: str) -> Dict[str, Entry]:
    """pdb_id -> (ligand_name, smiles), lowercased ids"""

    table: Dict[str, Entry] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            pdb = (row.get("pdb_id") or "").strip().lower()
            if pdb:
                table[pdb] = Entry(
                    (row.get("ligand_name") or "").strip(),
                    (row.get("canonical_smiles") or "").strip(),
                )
    return table


# --------------------------------------------------
def toml_literal(value: str) -> str:
    """Quote a SMILES for TOML.

    Literal (single-quoted) strings, which have no escape processing at all.
    Measured on the 16,839 non-empty SMILES in the PDBbind table: 8 contain a
    backslash -- e.g. 1c5p's "[H]/N=C(\\N)c1ccccc1" -- and in a basic
    double-quoted string "\\N" is an invalid escape. None contain a single
    quote, so literal strings are lossless here; refuse rather than mangle if
    that ever stops being true.
    """

    if "'" in value or "\n" in value:
        raise ValueError(f"cannot express as a TOML literal string: {value!r}")
    return f"'{value}'"


# --------------------------------------------------
def toml_basic(value: str) -> str:
    """Quote a ligand NAME for TOML.

    Basic (double-quoted) strings, not literal ones as for SMILES: chemical
    names carry primes -- "GUANOSINE-5'-DIPHOSPHATE" -- so a single-quoted
    literal cannot hold them. Backslash and double quote are the only
    characters that need escaping in a single-line basic string.
    """

    if "\n" in value:
        raise ValueError(f"cannot express as a one-line TOML string: {value!r}")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --------------------------------------------------
def find_toml(target: str) -> Optional[str]:
    """The metadata file for a bundle directory, or the path if it is one"""

    if os.path.isfile(target):
        return target
    direct = os.path.join(target, METADATA_NAME)
    if os.path.isfile(direct):
        return direct
    for root, _dirs, files in os.walk(target):
        if METADATA_NAME in files:
            return os.path.join(root, METADATA_NAME)
    return None


# --------------------------------------------------
def insert_smiles(text: str, smiles: str, source: str,
                  replace: bool = False,
                  name: str = "", name_source: str = "") -> Tuple[str, int]:
    """Add "smiles = ..." to the one [[ligands]] table that lacks it.

    Returns (new_text, line_number). Walks the file as lines and tracks which
    table it is inside, because "name =" appears under [[contributors]] and
    [[solutes]] too -- a naive search for the first "name =" after
    "[[ligands]]" would land in the wrong table on a reordered file.

    Line endings are normalised to LF on write, so a CRLF original comes out
    wholly rewritten. That is why write_toml() takes a backup first.

    A provenance comment goes in above the key, and it is the point of the
    whole exercise rather than a nicety. This file is pushed verbatim to
    release/<MDR>/original/, so without it the archive would present a
    third-party value as the submitter's own declaration. Nothing downstream
    rewrites the TOML, so the comment survives into permanent storage.

    With `name`, a BLANK name in the same table is filled too, under its own
    provenance comment. A name the submitter did give is never touched, even
    if it disagrees with `name` -- that disagreement is the name check's
    business, not this function's.
    """

    lines = text.split("\n")
    in_ligands = False
    name_line = None
    smiles_line = None       # index of an existing "smiles =" line, if any
    placeholder = False      # ... and whether its value is PLACEHOLDER
    blocks = []

    def flush():
        blocks.append((name_line, smiles_line, placeholder))

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[["):
            if in_ligands:
                flush()
            in_ligands = stripped == "[[ligands]]"
            name_line, smiles_line, placeholder = None, None, False
            continue
        if stripped.startswith("[") and not stripped.startswith("[["):
            if in_ligands:
                flush()
            in_ligands = False
            continue
        if in_ligands:
            if stripped.startswith("name"):
                name_line = i
            elif stripped.startswith("smiles"):
                smiles_line = i
                value = stripped.split("=", 1)[-1].strip().strip("\"'")
                placeholder = value == PLACEHOLDER
    if in_ligands:
        flush()

    # "Missing" is either no key at all or the template placeholder. Both need
    # the same value written; they differ only in whether there is a line to
    # replace or a line to add. Under --replace the net is wider still: a
    # present, valid-looking value is a candidate too, because the caller has
    # named both the file and the string and is repairing a known-bad one.
    if replace:
        # Any smiles line is a candidate, placeholder or not.
        missing = [b for b in blocks if b[1] is not None]
        if not missing:
            raise ValueError("no [[ligands]] table has a smiles to replace")
    else:
        missing = [b for b in blocks if b[1] is None or b[2]]
        if not missing:
            raise ValueError("no [[ligands]] table is missing smiles")
    if len(missing) > 1:
        raise ValueError(
            f"{len(missing)} [[ligands]] tables are missing smiles; the table "
            f"is keyed on pdb_id alone and cannot say which is which"
        )

    name_at, smiles_at, is_placeholder = missing[0]
    overwrite = is_placeholder or replace
    anchor = smiles_at if overwrite else name_at
    if anchor is None:
        raise ValueError("the [[ligands]] table has no name to anchor to")

    indent = lines[anchor][: len(lines[anchor]) - len(lines[anchor].lstrip())]
    new_line = f"{indent}smiles = {toml_literal(smiles)}"

    if name and name_at is not None:
        value = lines[name_at].split("=", 1)[-1].strip()
        if value in ('""', "''"):
            lines[name_at] = f"{indent}name = {toml_basic(name)}"
            lines.insert(name_at, f"{indent}# {name_source}")
            if smiles_at is not None and smiles_at > name_at:
                smiles_at += 1
            name_at += 1

    if overwrite:
        # Replace the existing value in place rather than inserting beside it,
        # so the file never carries two `smiles` keys in one table -- which
        # is invalid TOML and would fail the re-parse check in main().
        lines[smiles_at] = new_line
        lines.insert(smiles_at, f"{indent}# {source}")
        return "\n".join(lines), smiles_at + 2

    lines.insert(name_at + 1, new_line)
    lines.insert(name_at + 1, f"{indent}# {source}")
    return "\n".join(lines), name_at + 3


# --------------------------------------------------
def write_toml(path: str, new_text: str) -> Optional[str]:
    """Back the original up, then write. Returns the backup path if made.

    The backup is copy2 (bytes and mtime) BEFORE the write, and an existing
    .orig is left alone: run this twice and the .orig still holds what the
    contributor sent, not the output of the first run. Losing that is the
    only way this script can destroy something -- the source tarball in
    IRODS is untouched, but re-extracting it to recover one line is a
    quarter-gigabyte download.
    """

    backup = f"{path}.orig"
    made = None
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        made = backup

    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        fh.write(new_text)
    os.replace(tmp, path)
    return made


# --------------------------------------------------
def main() -> None:
    args = get_args()

    table = {}
    if args.smiles:
        print(f"Using the SMILES given on the command line: {args.smiles}")
    else:
        if not os.path.isfile(args.table):
            sys.exit(f"No such table: {args.table}")
        table = load_table(args.table)
        print(f"{len(table):,} pdb_ids in {args.table}")
    if args.dry_run:
        print("DRY RUN: nothing will be written")

    tally = {k: 0 for k in
             ("fixed", "already", "no_toml", "no_row", "no_smiles",
              "name_mismatch", "ambiguous", "error", "named")}

    for target in args.targets:
        path = find_toml(target)
        if path is None:
            print(f"  !! {target}: no {METADATA_NAME} found")
            tally["no_toml"] += 1
            continue

        text = open(path).read()
        try:
            meta = toml.loads(text)
        except Exception as err:
            print(f"  !! {path}: unreadable TOML: {err}")
            tally["error"] += 1
            continue

        ligands = meta.get("ligands") or []
        if ligands and not args.replace and \
                not any(is_unset(l.get("smiles")) for l in ligands):
            print(f"  -- {path}: already has smiles")
            tally["already"] += 1
            continue

        pdb_id = (meta.get("pdb_id") or "").strip().lower()

        if args.smiles:
            # No lookup, so no pdb_id join and no table name to check against.
            # The caller has asserted this molecule for THESE targets, which is
            # why --smiles takes explicit paths rather than globbing a tree.
            entry = Entry("", args.smiles)
            source = f"{args.source} (applied {date.today().isoformat()})"
        else:
            entry = table.get(pdb_id)
        if not args.smiles and (not pdb_id or entry is None):
            print(f"  !! {path}: pdb_id {pdb_id or '(none)'} not in the table")
            tally["no_row"] += 1
            continue
        if not args.smiles and not entry.smiles:
            # 2,604 of 19,443 rows are like this -- PDBbind lists a peptide or
            # similar with no SMILES at all. Nothing to take; the submitter
            # has to supply it.
            print(f"  !! {path}: {pdb_id} is in the table with an EMPTY "
                  f"SMILES ({entry.ligand_name or 'unnamed'}) -- cannot fix")
            tally["no_smiles"] += 1
            continue

        declared = [(l.get("name") or "").strip() for l in ligands
                    if args.replace or is_unset(l.get("smiles"))]
        if not args.smiles and declared and entry.ligand_name and \
                declared[0].lower() != entry.ligand_name.lower():
            if not args.allow_name_mismatch:
                print(f"  !! {path}: name mismatch, skipping\n"
                      f"       TOML  : {declared[0]}\n"
                      f"       table : {entry.ligand_name}")
                tally["name_mismatch"] += 1
                continue
            print(f"  ~~ {path}: name mismatch overridden")

        if not args.smiles:
            source = (
                f"smiles added by fix_ligand_smiles.py on "
                f"{date.today().isoformat()} from "
                f"{os.path.basename(args.table)} "
                f"(pdb_id {pdb_id}); not supplied by the submitter"
            )
        name_source = (
            f"name added by fix_ligand_smiles.py on "
            f"{date.today().isoformat()} from "
            f"{os.path.basename(args.table)} "
            f"(pdb_id {pdb_id}); the submitter left it blank"
        )
        try:
            new_text, line_no = insert_smiles(
                text, entry.smiles, source, args.replace,
                "" if args.smiles else entry.ligand_name, name_source)
        except ValueError as err:
            print(f"  !! {path}: {err}")
            tally["ambiguous"] += 1
            continue

        # Never write a file this cannot read back. The inserted value has to
        # survive a parse as the exact bytes from the table -- that is what
        # catches a quoting mistake, which is the only way this can silently
        # corrupt a molecule.
        try:
            check = toml.loads(new_text)
            got = [l.get("smiles") for l in check.get("ligands", [])
                   if l.get("smiles")]
            if entry.smiles not in got:
                raise ValueError("value did not survive the round trip")
            named = new_text.count(name_source) == 1
            if named and entry.ligand_name not in [
                    l.get("name") for l in check.get("ligands", [])]:
                raise ValueError("name did not survive the round trip")
        except Exception as err:
            print(f"  !! {path}: refusing to write, {err}")
            tally["error"] += 1
            continue

        print(f"  ok {path}\n"
              f"       + line {line_no - 1}: # {source}\n"
              f"       + line {line_no}: smiles = {toml_literal(entry.smiles)}")
        if named:
            print(f"       + name = {toml_basic(entry.ligand_name)}")
            tally["named"] += 1
        if args.dry_run and not os.path.exists(f"{path}.orig"):
            print(f"       would save the original to {path}.orig")
        if not args.dry_run:
            backup = write_toml(path, new_text)
            if backup:
                print(f"       original saved to {backup}")
        tally["fixed"] += 1

    print("\n" + ", ".join(f"{k}={v}" for k, v in tally.items() if v))


if __name__ == "__main__":
    main()
