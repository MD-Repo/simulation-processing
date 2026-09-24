# simulation-processing

The cron-driven processing queue, the iRODS transfer scripts, and the scientific
helpers `mdr-process` shells out to. Runs on the processing VM under the
`exouser` crontab. Nothing here imports Django; the `md_*` tables are reached
with raw psycopg2 SQL.

## Commands

```console
cd python
uv sync
source .venv/bin/activate && playwright install   # for create_preview.py
uv run <program>
uv run pytest tests/
```

Cron uses `.venv/bin/python` directly, never `uv run` — cron must execute exactly
what is installed, with no surprise dependency sync mid-run. After pulling changes
that touch dependencies, run `uv sync` as a deploy step; nothing in cron does it.

## The queue

Two cron jobs and a Postgres table decouple *finding* a completed upload from
*processing* it. `docs/processing-queue.md` is accurate and worth reading before
changing either script.

```
check_new_simulations.py  ──▶  md_process_job  ──▶  drain_process_queue.py
  scanner, every 5 min          the queue          worker, every minute
```

- The **scanner** has no internal guard, so its cron line wraps it in `flock -n`.
- The **worker** self-locks with `fcntl.flock`, so its cron line needs no `flock`.
- `--verbose` on the scanner is not optional in practice: without it, deletions
  leave no record at all.

## Things to know

- **Most failures are terminal by design.** `mdr-process` is not safe to blindly
  re-run once it has started mutating, so a failed job stays failed and the Slack
  notice *is* the human handoff. The sole exception is a transient iRODS
  connection failure during the fetch step, which runs before any mutation.
- **The retry budget is 7.75 hours across six executions** — waits of 5, 20, 80,
  180, 180 minutes. A comment in `drain_process_queue.py` says ~4.75 h; it counts
  only four waits and is wrong.
- **A requeue deliberately stops the drain tick**, so the retry waits for the next
  cron minute instead of being reclaimed immediately.
- **Nothing may stop the drain draining.** `python-irodsclient` is imported lazily
  inside the canary function specifically so a broken client library cannot prevent
  the worker from starting. Preserve that property.
- **Alerts stay silent on success.** An idle tick writes nothing. `cron_notify.py`
  exists because cron on this box cannot report anything — no MAILTO, postfix
  masked — and it carries the alert-threshold and repeat-suppression state machine.
  Its state advances only on a *delivered* Slack message, never an attempted one.
- **`send_slack_message` returns False on a rejected post.** Slack answers HTTP 200
  with `{"ok": false}` for an archived channel or rotated token, so status alone
  would silently disable every alert.
- **The purge needs two agreeing signals** before deleting a landing, and reports
  abandoned-landing deletions separately because those skip verification.
- **Shared constants live in `common.py`** — `TICKET_LOG_ROOT`, `CRON_STATE_ROOT`,
  `EX_TEMPFAIL`, `FRONTEND_BASE_URLS`. They are shared rather than duplicated
  because a drifted second definition would delete from the wrong directory or
  disagree about an exit code.
