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

## Cleanups in the `python/push_sim_files.py` upload path

Deferred nits from the review of the python-irodsclient upload path. None are
correctness bugs today; each is a latent trap or a small inconsistency.

- **`PRINT_LOCK` only guards one side.** `put_file` takes it for the retry
  message, but the main thread's per-file prints in the `as_completed` loop do
  not, so nothing is actually serialized. Either take the lock in both places or
  drop it.
- **Retries reuse a possibly-poisoned session.** When an upload dies
  mid-protocol the connection goes back into that clone's pool and the next
  attempt can pick it up again. A `session.cleanup()` before retrying forces
  fresh connections — relevant given the `SYS_HEADER_READ_LEN_ERR` failures that
  motivated the clone-per-thread design.
- **Backoff holds the borrowed session.** `ABORT.wait(2**attempt)` blocks while
  the worker still owns its clone. Harmless only because the number of clones
  equals `max_workers`; if those ever diverge, a sleeping worker starves a live
  one.
- **Clones are built outside the `try`.** A failure partway through the
  `session.clone()` loop leaks the earlier clones (python-irodsclient's atexit
  hook eventually collects them). They are also rebuilt per `sub_dir`, so we
  authenticate twice per run instead of once.
- **Per-file elapsed time includes retry backoff.** `start` is set before the
  retry loop, so the reported "took" is wall clock, not transfer time.
- **Partial state on failure.** `sys.exit` on upload errors skips the
  `is_placeholder` update and the `--out-file` write while some files have
  already landed in IRODS. Pre-existing behavior, but per-file failures make it
  more reachable than the old single `gocmd put` did.
