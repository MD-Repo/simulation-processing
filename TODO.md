# simulation-processing — planned work

## Queue the `mdr-process` runs from `check_new_simulations.py` — DONE

Implemented. The scanner now enqueues jobs into the `md_process_job` Postgres
table and a separate worker (`drain_process_queue.py`) drains it. Full design and
operations notes: [`docs/processing-queue.md`](docs/processing-queue.md).

Remaining follow-ups (tracked under *Future work* in that doc): bounded-concurrency
slot pool if head-of-line blocking bites, a stuck-`running`-job monitor, and log
rotation.

## Simplify the success/failure bookkeeping in `mdr-process/src/ticket.rs`

**Mostly done.** The overlapping "did it work?" signals came from a non-fatal
`errors` channel: `process::process` could return `Ok` while carrying a
non-empty `ProcessResult.errors`, so callers had to re-derive the verdict from
three sources (the `Result`, `errors`, and a per-landing `bool`). That channel
has been removed:

- `process_trajectory` now `bail!`s on missing full/minimal outputs instead of
  recording a non-fatal error and continuing.
- `ProcessedTrajectory.errors` and `ProcessResult.errors` are gone, along with
  their now-dead consumers in `process.rs`, `main.rs`, and `reprocess.rs`
  (`reprocess` now returns `Result<()>`).
- `process_landing`'s `Ok` arm is unconditional success. So `process::process`
  returning `Ok` unambiguously means success; `Ok` ⇒ success, `Err` ⇒ failure.

**Still open (low priority):** `ticket::process` signals a failed ticket two
ways at once — it leaves `md_ticket.processing_complete` unset *and* `bail!`s for
a non-zero exit. The non-zero exit is load-bearing: the queue worker keys off it
(see [`docs/processing-queue.md`](docs/processing-queue.md)), so any change must
preserve it. The redundancy is just the DB flag being set separately from the
exit signal.
