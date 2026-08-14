# bundles2/ backlog

How the ~15,000 simulation directories a contributor is pushing directly into
IRODS get processed, and how to operate `process_bundles2.py`.

## Overview

A contributor is uploading simulations straight into IRODS at
`/iplant/home/shared/mdrepo/bundles2/`, bypassing the ticket-upload flow
entirely — so nothing in `md_ticket` / `md_process_job` (see
[processing-queue.md](processing-queue.md)) knows these directories exist.
`process_bundles2.py` is the equivalent path for this source: cron-driven,
one bundle claimed and processed per invocation, no long-lived daemon.

```
                    (cron, every 5 min)
  bundles2/*.md5 ──▶ process_bundles2.py ──▶ mdr-process process <dir>
   contributor        claims oldest ready      same "process a bare
   uploads here        bundle, downloads it      directory" entry point
                        via gocmd                the ticket flow uses
```

**Readiness signal:** the contributor writes a sibling `bundles2/<name>.md5`
once `bundles2/<name>/` is fully uploaded (agreed with them directly, not
inferred). A 0-byte `.md5` — a truncated write caught mid-upload — is not
treated as ready, mirroring `check_new_simulations.py`'s
`upload_complete()` marker check. The `.md5`'s contents are not otherwise
verified; it is a completion signal only.

**Coordinating with the live ticket queue:** this backlog must never delay a
real contributor's ticket submission, and must never run `mdr-process`
concurrently with `drain_process_queue.py`. Two mechanisms, both in
`process_bundles2.py`'s `main()`:
- It takes the *same* flock `drain_process_queue.py` uses
  (`/tmp/drain_process_queue-<server>.lock`), so the two never run
  `mdr-process` at once.
- Before claiming a bundle it checks `md_process_job` for any
  `pending`/`running` row on this server (`ticket_queue_busy`) and exits
  without claiming one if so — a waiting ticket gets the next opening rather
  than sitting behind however long the current backlog item takes.

**Success/failure:** `mdr-process process` returning exit 0 already means
verified-and-permanent — `process.rs` only clears `is_placeholder` after
`push_sim_files.py` confirms every file by MD5. On success,
`process_bundles2.py` deletes both the local scratch copy and the
`bundles2/<name>/` + `.md5` source (the same bar
[`purge_processed_landings.py`](../python/purge_processed_landings.py) uses
before deleting a landing collection). On failure, only the local copy is
deleted — disk is shared with live ticket processing, and there is no DB
row or ticket tracking these directories, so a local copy adds nothing a
human couldn't get by re-running `gocmd get`. The `bundles2/` source and its
`.md5` are left in place, and a Slack message is sent
(`mdrepo-alerts`). **Nothing here retries automatically** — same
"failures need a human" model as `md_process_job`.

## Components

| Piece | Path | Role |
|-------|------|------|
| Worker | `python/process_bundles2.py` | Claim the oldest ready bundle, fetch, run `mdr-process process`, clean up, record state. |
| State | `/opt/mdrepo/state/process_bundles2-<server>.json` | `{bundle_name: {status, timestamp, error?}}`, written atomically. Not a DB table — this is a one-off backlog, not a permanent queue. |
| Logs | `/opt/mdrepo/logs/bundles2/<server>/<name>.log` | Per-bundle `mdr-process` debug log, same convention as `TICKET_LOG_ROOT`. |
| Scratch | `/opt/mdrepo/landing/bundles2/<name>` | Local download destination, alongside `fetch_uploads.py`'s `/opt/mdrepo/landing/<server>`. Deleted after every attempt, success or failure. |

## Before installing the cron job

**Do not install cron until these have been run by hand and inspected:**

1. `--dry-run` against 2-3 real `bundles2/` directories. This downloads the
   bundle and runs `mdr-process process --dry-run` on it, but touches neither
   the IRODS source nor the state file — safe to repeat. Confirm the output
   looks right and the state file stays empty.
2. One real (non-dry-run) run against a single small bundle. Confirm
   afterward: the local scratch dir is gone, the IRODS source + `.md5` are
   gone from `bundles2/`, the simulation shows up on the target server with
   `is_placeholder = false`, and the state file has one `done` entry.
3. One deliberately-broken bundle (e.g. delete a required file from a copy
   before uploading it) to confirm the failure path: local scratch dir gone,
   IRODS source **untouched**, state file has a `failed` entry with a useful
   error, Slack message received in `mdrepo-alerts`.
4. Queue a real ticket (or insert a manual `md_process_job` row) and confirm
   a `process_bundles2.py` run exits without claiming a bundle while it is
   `pending`/`running` — this is the priority check that protects real
   uploads from the backlog.

Only after all four pass, add the cron line below.

## Cron setup

One line, added to the **same** `exouser` crontab as
[processing-queue.md](processing-queue.md)'s jobs, reusing its `PATH=` block
(`mdr-process` shells out to `uv`, `blastp`, `gmx` — none are on cron's
default PATH). Add it as a separate line, not folded into the scan/drain
pair; a 5-minute interval is plenty since a single `mdr-process` run can take
a long time and this only ever has one bundle in flight:

```cron
# Process the bundles2/ direct-upload backlog (one bundle per tick, shares
# drain_process_queue.py's lock and steps aside for pending/running tickets).
*/5 * * * * cd /opt/mdrepo/simulation-processing/python && .venv/bin/python process_bundles2.py --server prod >> logs/process_bundles2-prod.log 2>&1
```

No external `flock -n` wrapper needed on this line — `process_bundles2.py`
takes `drain_process_queue.py`'s lock internally via `acquire_lock`, same as
the drain does for itself.

## Operations

### Logs to watch

```bash
# What this tick did (claimed / stepped aside / succeeded / failed)
tail -n 20 /opt/mdrepo/simulation-processing/python/logs/process_bundles2-prod.log

# Per-bundle mdr-process debug log
tail -n 50 /opt/mdrepo/logs/bundles2/prod/<name>.log
```

### Inspecting progress

```bash
# Counts by outcome
jq -r '[.[].status] | group_by(.) | map("\(.[0]): \(length)") | .[]' \
  /opt/mdrepo/state/process_bundles2-prod.json

# Everything a human needs to deal with
jq -r 'to_entries[] | select(.value.status == "failed") | .key' \
  /opt/mdrepo/state/process_bundles2-prod.json
```

A failed bundle's `bundles2/<name>/` and `.md5` are still on IRODS exactly
where the contributor put them — nothing else in this pipeline touches them.
To retry after fixing the cause, delete that bundle's entry from the state
JSON (or just accept the wait: the state file, not IRODS, is what marks it
"already tried") and let the next cron tick pick it up again, or run
`process_bundles2.py --dry-run` by hand against it first.

## Future work

- No stuck-`running` monitor — a bundle stuck mid-`mdr-process` past
  `PROCESS_TIMEOUT` (12h, imported from `drain_process_queue.py`) is killed
  by the same timeout handling the drain uses, so this is lower-risk than it
  was for `md_process_job`, but nothing pages if a whole tick silently stops
  firing (e.g. cron itself is down).
- No log rotation for `logs/bundles2/<server>/*.log` yet.
