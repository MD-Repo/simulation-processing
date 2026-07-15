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

Installed under the `exouser` crontab (currently **staging** only):

```cron
# --- simulation-processing queue (staging) ---
# PATH set here so the worker can find every binary mdr-process needs, none of
# which are on cron's default PATH:
#   ~/.cargo/bin         mdr-process itself
#   ~/.local/bin         uv (mdr-process shells out to `uv run` for its python
#                        helpers: fetch_uploads.py, canonicalize_smiles.py, ...)
#   /usr/local/blast/bin blastp (the sequence-search step)
PATH=/home/exouser/.cargo/bin:/home/exouser/.local/bin:/usr/local/blast/bin:/usr/local/bin:/usr/bin:/bin

# Scan for completed uploads and enqueue mdr-process jobs (every 5 min).
# flock -n guards against overlapping scans (the scanner has no internal lock).
*/5 * * * * cd /opt/mdrepo/simulation-processing/python && flock -n /tmp/check_new_simulations-staging.lock .venv/bin/python check_new_simulations.py --server staging >> logs/check_new_simulations-staging.log 2>&1

# Drain the queue: run pending jobs serially, one mdr-process at a time (every min).
# The worker self-locks via fcntl.flock, so a tick fired while a job is running exits immediately.
* * * * * cd /opt/mdrepo/simulation-processing/python && .venv/bin/python drain_process_queue.py --server staging >> logs/drain_process_queue-staging.log 2>&1
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

  Both are covered by the `PATH=` line above. If a future job fails with a
  "not found" / "cannot find binary" error, the fix is almost always adding that
  tool's directory to this `PATH=`.
- **Adding prod later:** copy both lines with `--server prod` and `-prod` lock /
  log names. The `PATH=` line already covers both. Keep staging and prod on
  distinct lock/log names so they never collide.

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

These scripts run without `--verbose`, so an **idle tick writes nothing** — an
empty log is normal, not a sign of failure. Add `--verbose` to a cron line if you
want a "found N tickets / no pending jobs" heartbeat.

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
