# Simulation-processing queue

How completed simulation uploads get turned into processed simulations, and how
to operate the queue that does it.

## Overview

Two cron jobs and a Postgres table decouple *finding* a completed upload from
*processing* it:

```
                         (cron, every 5 min)
  check_new_simulations.py ──▶ md_process_job (Postgres)  ──▶ drain_process_queue.py
   "scanner": find complete       one 'pending' row            "worker": run mdr-process,
   uploads, enqueue a job         per queued ticket            report to Slack
                                                               (cron, every 1 min)
```

- The **scanner** looks for `md_ticket` rows whose iRODS upload is complete,
  marks each one, and enqueues a job. It no longer runs `mdr-process` itself, so
  it returns in seconds.
- The **worker** drains the queue: for each `pending` job it runs
  `mdr-process ticket`, records the outcome on the job row, and posts a Slack
  notice. Jobs run **one at a time**.

This replaced an earlier design where the scanner ran `mdr-process`
synchronously in its own process (blocking the scan for up to 12h per run).

## Components

| Piece | Path | Role |
|-------|------|------|
| Scanner | `python/check_new_simulations.py` | Find complete uploads; mark + enqueue (atomically). Also deletes incomplete tickets older than 7 days. |
| Worker | `python/drain_process_queue.py` | Claim `pending` jobs, run `mdr-process`, mark terminal, Slack the result. |
| Shared | `python/common.py` | `send_slack_message`, `FRONTEND_BASE_URLS`. |
| Table | `md_process_job` | The queue. Schema owned by Django. |
| Model | `django/.../models/process_job/process_job.py` (`MDRepoProcessJob`) | Owns the DDL only — the scripts talk to the table via raw SQL and never import Django. |

## Design decisions

- **No retry; failures need a human.** `mdr-process` is not safe to blindly
  re-run (it wants `--force`), so a failed job is **terminal**. The worker marks
  it `failed` and posts a Slack notice — that notice *is* the human handoff.
  Nothing re-enqueues it. (This is also why we did not use Celery/RQ: their
  headline feature, retry-with-backoff, is unwanted, and it isn't worth running
  Redis + a worker daemon for it.)
- **Serial processing (head-of-line blocking accepted).** The worker holds a
  single lock and runs jobs one at a time in `created_at` order. A large
  trajectory therefore delays smaller jobs queued behind it. That is fine given
  infrequent uploads and the cost of running parallel `mdr-process`. See
  *Future work* for the bounded-concurrency escape hatch.
- **Atomic mark-then-enqueue.** The scanner marks the ticket
  (`upload_notification_sent`, `used_for_upload`) and inserts the job in a
  **single CTE statement**, so a crash can't leave a ticket marked "notified" but
  never queued (which no later scan would re-find). Safe under the existing
  `autocommit`.
- **Django owns the schema; the scripts own the runtime SQL.** The `md_*` tables
  are Django models; the standalone scripts already query them with psycopg2.
  `md_process_job` follows suit. `created_at` uses `db_default=Now()` so the
  worker's raw-SQL inserts still get a server-side timestamp.
- **Raw SQL gets no `on_delete` behaviour — the ORM emulates it, the database
  does not.** Django's `on_delete` (`SET_NULL` on `Simulation.md_repo_ticket` and
  `SimulationUploadInstance.ticket`, `CASCADE` on `ProcessJob.ticket`) is applied
  by the ORM in Python; the FK constraints carry no `ON DELETE` clause. So
  `ticket.delete()` nulls the referencing rows first, while
  `delete from md_ticket ...` just raises `ForeignKeyViolation`. This bit the
  scanner when it was ported off the Django management command: the reap below
  crashed on the first referenced ticket it met. Any raw delete of a `md_*` row
  must handle the referencing rows itself — check the models, not the DB, for
  what points at it.
- **The reap only touches genuinely abandoned tickets.** A ticket that is
  incomplete and older than `MAX_DAYS_OLD` is deleted along with its IRODS
  collections. But "incomplete" only means the upload never got its
  `mdrepo-submission.completed.json` marker — some old tickets produced real,
  public simulations anyway. `ticket_dependents()` therefore skips any ticket
  with rows referencing it (`TICKET_REFERENCES`, kept in sync with the models),
  so the reap can never unlink a simulation from its provenance. Keep that guard
  in front of the delete.
- **Two independent locks, same goal.** The scanner has no internal guard, so the
  cron line wraps it in `flock -n`. The worker locks *itself* (`fcntl.flock` in
  `acquire_lock`), so its cron line needs no external `flock`.

## Data model — `md_process_job`

| Column | Notes |
|--------|-------|
| `id` | PK |
| `ticket_id` | FK → `md_ticket(id)`, `ON DELETE CASCADE` |
| `server` | `staging` or `prod`; workers filter on it |
| `status` | `pending` → `running` → `succeeded` \| `failed` |
| `exit_code` | `mdr-process` return code (null until it runs) |
| `last_error` | stderr / debug-log tail on failure |
| `log_file` | reserved; the worker writes logs to `logs/ticket-<id>-<server>.log` |
| `created_at` | `db_default=Now()` |
| `started_at` | set when claimed |
| `finished_at` | set on terminal state |

State machine (only these transitions):

```
pending ──▶ running ──▶ succeeded
                   └──▶ failed        (terminal; needs a human)
```

- The scanner inserts `pending`.
- The worker claims with `UPDATE ... FOR UPDATE SKIP LOCKED` → `running` (the
  `SKIP LOCKED` makes it safe to run multiple workers later).
- On exit 0 → `succeeded`; on non-zero / timeout / missing binary → `failed`.

## Cron setup

Installed under the `exouser` crontab, for **staging and prod**:

```cron
# --- simulation-processing queue (staging) ---
# PATH set here so the worker can find every binary mdr-process needs, none of
# which are on cron's default PATH:
#   ~/.cargo/bin           mdr-process itself
#   ~/.local/bin           uv (mdr-process shells out to `uv run` for its python
#                          helpers: fetch_uploads.py, canonicalize_smiles.py, ...)
#   /usr/local/blast/bin   blastp (the sequence-search step)
#   /usr/local/gromacs/bin gmx (the trajectory-manipulation step)
PATH=/home/exouser/.cargo/bin:/home/exouser/.local/bin:/usr/local/blast/bin:/usr/local/gromacs/bin:/usr/local/bin:/usr/bin:/bin

# Scan for completed uploads and enqueue mdr-process jobs (every 5 min).
# flock -n guards against overlapping scans (the scanner has no internal lock).
# --verbose: the scanner DELETES abandoned tickets and their IRODS collections;
# without it those removals leave no record at all (it only logs on a crash).
*/5 * * * * cd /opt/mdrepo/simulation-processing/python && flock -n /tmp/check_new_simulations-staging.lock .venv/bin/python check_new_simulations.py --server staging --verbose >> logs/check_new_simulations-staging.log 2>&1

# Drain the queue: run pending jobs serially, one mdr-process at a time (every min).
# The worker self-locks via fcntl.flock, so a tick fired while a job is running exits immediately.
* * * * * cd /opt/mdrepo/simulation-processing/python && .venv/bin/python drain_process_queue.py --server staging >> logs/drain_process_queue-staging.log 2>&1

# --- simulation-processing queue (prod) ---
# Mirrors staging above; shares the PATH= line. Distinct -prod lock/log names so
# the two servers never collide (the worker also self-locks per server).
*/5 * * * * cd /opt/mdrepo/simulation-processing/python && flock -n /tmp/check_new_simulations-prod.lock .venv/bin/python check_new_simulations.py --server prod --verbose >> logs/check_new_simulations-prod.log 2>&1
* * * * * cd /opt/mdrepo/simulation-processing/python && .venv/bin/python drain_process_queue.py --server prod >> logs/drain_process_queue-prod.log 2>&1
```

Notes:

- **`cd` into `python/` first** so `load_dotenv()` finds `.env` (DSNs,
  `SLACK_TOKEN`) and the relative `logs/` dir resolves.
- **`.venv/bin/python`, not `uv run`.** Cron runs exactly what's installed — no
  surprise dependency syncs mid-run. The tradeoff: after pulling changes that
  touch dependencies, run `uv sync` once as part of deploy (nothing in cron does
  it for you).
- **`mdr-process`'s own binaries must be on the cron PATH — not just the
  launcher's.** We launch the scripts with `.venv/bin/python` (which needs
  nothing extra), but `mdr-process` then shells out to other tools, and cron's
  default PATH finds none of them. Known ones so far:
  - **`uv`** (`which("uv")` in `ticket.rs`) — missing it fails every job at the
    fetch step with *"Failed to find uv (cannot find binary path)"* plus a
    downstream *"ticket-<id>/ticket.json: No such file or directory"*.
  - **`blastp`** — the sequence-search step; missing it fails with a
    *blastp not found* error.
  - **`gmx`** (GROMACS) — the trajectory-manipulation step
    (`cpptraj_gmx_traj_manipulation.py`); missing it fails with
    *"Failed to execute 'which gmx'"*. PATH alone is enough: `gmx` locates its
    own data prefix, so cron does **not** need to source `GMXRC`. (An
    interactive shell picks `gmx` up because `~/.bashrc` sources `GMXRC`, which
    is why this breaks only under cron.)

  All three are covered by the `PATH=` line above. If a future job fails with a
  "not found" / "cannot find binary" error, the fix is almost always adding that
  tool's directory to this `PATH=`.
- **Staging and prod share one `PATH=` line.** Cron applies an environment
  setting to every job *below* it in the file, so the single `PATH=` covers both
  server blocks — and the two `export_mapping_file.py` jobs above it keep cron's
  default environment. Add new queue jobs below the `PATH=` line, not above it.
- **Staging and prod are kept on distinct lock / log names** (`-staging` vs
  `-prod`) so the two never collide; the worker also self-locks per server.

## Operations

### Logs to watch

```bash
# Worker: what it ran and each job's outcome
tail -n 20 /opt/mdrepo/simulation-processing/python/logs/drain_process_queue-staging.log

# Scanner: tickets found / enqueued
tail -n 20 /opt/mdrepo/simulation-processing/python/logs/check_new_simulations-staging.log

# Per-ticket mdr-process debug log (written by the worker for each job)
tail -n 50 /opt/mdrepo/simulation-processing/python/logs/ticket-<TICKET_ID>-<server>.log
```

The **scanner** runs with `--verbose` (see the cron block above): it deletes
abandoned tickets and their IRODS collections, and those removals must not be
silent. The **worker** does not, so an idle drain tick writes nothing — an empty
`drain_process_queue-*.log` is normal, not a sign of failure.

Note what this means for the scanner's history: before `--verbose` was added, a
successful run printed nothing at all, so `check_new_simulations-*.log` contains
**only crash tracebacks** up to 2026-07-16. A long quiet stretch in the old log
is not an outage — it is the scan working.

Cron stdout/stderr → the `*-staging.log` files above; `mdr-process`'s own debug
output → `logs/ticket-<id>-<server>.log`.

### Inspecting the queue

```sql
-- Anything a human needs to deal with (failed = terminal, no auto-retry)
select id, ticket_id, exit_code, finished_at, left(last_error, 200) as err
from   md_process_job
where  status = 'failed'
order  by finished_at desc;

-- Backlog and in-flight work
select status, count(*) from md_process_job group by status;

-- Possibly-stuck: 'running' with no live worker (see below)
select id, ticket_id, started_at
from   md_process_job
where  status = 'running' and started_at < now() - interval '12 hours';
```

A failed job's full context: `last_error` on the row, plus the referenced
`logs/ticket-<id>-<server>.log`. There is no automatic recovery — clearing a failed job is
a manual decision (fix the cause, then re-run `mdr-process` by hand, typically
with `--force`).

### Stuck `running` jobs

If a worker is killed mid-run, its job stays `running` with no live process.
Unlike a vanished background process this is *queryable* (the `running` query
above). Reconcile it against `mdr-process`'s own state (`md_upload_instance` /
`md_upload_instance_message`, `md_ticket.processing_complete`) to decide whether
it actually succeeded or needs a rerun. There is no automatic monitor for this
yet — see *Future work*.

### Historical tickets excluded from the scan (prod, 2026-07-16)

96 prod tickets (ids 7 … 1006, 190–884 days old) had simulations and upload
instances attached but `upload_notification_sent = false` — a fossil of the old
Django path, which set the flag *after* processing rather than atomically with
the enqueue. They were incomplete and old, so the reap tried to delete them every
run and crashed on the FK (see the `on_delete` note above), which blocked the
prod scan entirely.

They were resolved by backfilling the scan's own gate rather than deleting them,
which preserved the ticket rows and the **4,273** simulation links hanging off
them:

```sql
update md_ticket t
set    upload_notification_sent = true
where  t.ticket_type = 'u'
  and  t.upload_notification_sent = false
  and (exists (select 1 from md_simulation s where s.md_repo_ticket_id = t.id)
    or exists (select 1 from md_upload_instance u where u.ticket_id = t.id));
-- 96 rows; reverse by setting the flag back to false.
```

Current code cannot recreate this state: the enqueue marks the ticket atomically
*before* any simulation exists, so a ticket can't end up with dependents and the
flag still false. `ticket_dependents()` guards the case regardless.

## Future work

- **Bounded concurrency** if head-of-line blocking becomes a problem: give the
  worker a **slot pool** — try `…-slot0.lock`, `…-slot1.lock`, … up to N, take
  the first free slot or exit if all N are held. Cron then runs up to N workers
  concurrently; `FOR UPDATE SKIP LOCKED` already guarantees they claim distinct
  jobs. Cap N small (2–3). ~15 lines in `acquire_lock`.
- **Stuck-job monitor:** a periodic query + Slack alert for jobs `running` longer
  than `PROCESS_TIMEOUT`.
- **Log rotation:** the `*-staging.log` files append unbounded; add a `logrotate`
  rule.
