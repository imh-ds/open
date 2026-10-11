# Sample-size feasibility pilot

This pilot prepares a larger study of sample-size requirements. It runs twelve
synthetic datasets with the production recursive layer-consensus engine. It
does **not** establish a minimum sample size or recommend new defaults.

The [protocol](sample_size_pilot_protocol.md) defines all datasets and settings.
The committed `study_manifests/sample-size-pilot.json` freezes identities,
configuration, source hashes, protocol and compute limits. Text hashes normalize
CRLF to LF; other source changes invalidate the manifest. Do not regenerate the
manifest after looking at results to make altered sources appear compatible.

The Actions workflow uses three concurrent case jobs, four assessment workers
per case, a 90-minute soft work budget and a 110-minute hard job limit. Ordinary
pushes register the workflow without starting fitting. Explicit dispatch or a
commit containing `[run-sample-size-pilot]` starts the bounded pilot. No automatic
reruns or larger validation study are authorized by this workflow.

Each local assessment is saved atomically before aggregation. Saved objects
retain their original sampled rows, splits and run IDs. Recursion still finalizes
each layer before testing its upstream targets. Failed assessments and incomplete
cases remain visible. A soft-budget stop preserves checkpoints; a hard runner
termination may leave incomplete artifacts, which cannot become finished results.

Before dispatch, copy the committed manifest to
`examples/results/2026-10-10-sample-size-pilot/manifest.json`. Run
`python examples/sample_size_pilot.py freeze` to validate it, then `matrix` to
list the twelve cases. `case --case chain-n300-p5` runs one fixed case. `audit`
collects all planned identities and writes `audit.json` and `REPORT.md`, including
missing cases. Audit errors fail the command; valid incomplete cases remain
explicitly incomplete in the report even when the audit command succeeds.

The audit regenerates inputs and splits, validates checkpoint evidence, recounts
production network aggregation and verifies actual fitting-row diagnostics.
It checks frozen network configuration and validates saved incomplete networks.
For finished cases it checks prediction arithmetic and independently refits both
final models with saved grid-valid parameters. It does not repeat every local fit
or independently repeat hyperparameter selection.

Per-worker metrics record OS process-lifetime peak working-set memory, including
imports. These peaks cannot be summed to obtain concurrent total job memory.
Network provenance records the main process's traced Python allocations, which
also do not represent total resident memory across workers.

For a small orchestration check, run `python examples/sample_size_pilot.py smoke
--output .superpowers/sample-size-smoke`. This uses B1 and fifty test rows in a
separate namespace, checks saved evidence, and cannot support reliability claims.
It is deliberately different from the actual B100 pilot.

Keep prediction, direct-link recovery and recursive-path recovery separate when
interpreting results. One dataset per condition measures feasibility and exposes
possible failures; guidance needs a separate study with many independent datasets
and uncertainty bounds. Historical studies remain unchanged.
