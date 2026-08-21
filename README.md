# simulation-processing
Code for processing simulations

## Docs

- [Simulation-processing queue](docs/processing-queue.md) — how completed uploads
  are found, queued (`md_process_job`), and processed by `mdr-process`; cron
  setup and logs to watch.
- The **bundles2/ backlog** — tarballs a contributor pushes directly into
  IRODS, outside the ticket flow, processed by `python/process_bundles2.py`.
  `docs/bundles2-backlog.md` was removed on 2026-08-21: it documented a design
  the tooling no longer has (it claimed the IRODS source is deleted on success
  and that state lives in a JSON file under `/opt/mdrepo/state/` — neither is
  true). The module docstring in `process_bundles2.py` is the source of truth
  for how the worker behaves; the operational runbook, the pre-cron checklist
  and the state of the backlog live with the DDD handoff notes, and the
  project item is MDR-51.
