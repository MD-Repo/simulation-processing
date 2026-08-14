# simulation-processing
Code for processing simulations

## Docs

- [Simulation-processing queue](docs/processing-queue.md) — how completed uploads
  are found, queued (`md_process_job`), and processed by `mdr-process`; cron
  setup and logs to watch.
- [bundles2/ backlog](docs/bundles2-backlog.md) — how directories a contributor
  pushes directly into IRODS (outside the ticket flow) are found and processed
  by `process_bundles2.py`; verification checklist, cron setup, and operations.
