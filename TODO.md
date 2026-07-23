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

## Cleanups in the `python/push_sim_files.py` upload path — DONE

Nits from the review of the python-irodsclient upload path, all addressed:
`PRINT_LOCK` now covers the main thread's per-file prints as well as the upload
threads' retry messages; `put_file` borrows a clone per attempt (so the backoff
no longer holds one) and calls `session.cleanup()` after a failure rather than
retrying over a connection that may be stuck mid-protocol; the clone pool is
built once outside the `sub_dir` loop, grown only as needed, and drained in a
`finally`, so we authenticate once and leak nothing on a partial failure; and
the reported per-file time now covers only the attempt that succeeded.

Deliberately left alone: `sys.exit` on upload errors skips the `is_placeholder`
update and the `--out-file` write even though some files have already landed in
IRODS. A rerun skips those files via the size check anyway, and keeping the
out-file to successful runs only means its presence stays unambiguous.
