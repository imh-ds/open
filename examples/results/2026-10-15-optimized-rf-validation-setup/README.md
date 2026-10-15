# Fresh optimized RF validation launch

Run: [38816428068](https://github.com/imh-ds/open/actions/runs/38816428068)

This is a fresh timing and implementation-equivalence check of the optimized
random-forest engine, using the shipped recursive settings. It is not a broader
reliability or sample-size study. Original N=80 and N=200 chain fixtures are
regenerated under new case IDs; no earlier checkpoint is reused.

The owner approved **up to 300 additional runner-minutes**, specifically for
these two cases. The reservation is 286 minutes: registration 1, preparation 15,
two case jobs at 120 each, and a 30-minute independent audit. Fourteen minutes
remain unallocated. The case jobs can run concurrently; this sum is a compute
ceiling, not the expected elapsed wait. No automatic fit retry is authorized.

The new versioned audit independently checks the two-candidate singleton and
four-candidate multivariable RF search. Historical audit policy and archives
remain unchanged. The final audit checks fresh losses, predictions, partitions,
fitted parameters, recursive decisions, fit counts, and provenance, then compares
the scientific results against the hash-pinned original RF archives. Only the
independently checked tuning-fit count is omitted from that result comparison.

Before launch, 33 focused audit/workflow/compatibility tests passed, including
the explicit local no-fit guard and saved-checkpoint arithmetic replay. The
saved checkpoint copy with adjusted counters is a software fixture, not a new
scientific result. The pinned installed package passed the frozen-manifest
check in both development and public checkouts. All 90 required remote Git blobs
matched the public checkout; original ZIP bytes matched their reference hashes.
An independent read-only review found two reporting issues, both corrected and
regression tested before dispatch. No new scientific model was fitted locally.

Evidence:

- `prelaunch-verification.json`: source, reference-file and test verification.
- `no-fit-tests.xml`: the 33-test invocation, with zero failures.
- `authorization/`: the actual authorization and exact two-case matrix.
- `launch-verification.json`: verified run, source, first attempt, authorization,
  successful preparation and case compute starts.
- `run.json` and `jobs-start.json`: GitHub API launch snapshots.
- Frozen protocol: `examples/optimized_rf_validation_protocol.md`.
- Frozen manifest: `examples/study_manifests/optimized-rf-validation.json`.
- Dispatch source: `c9b512459be657d0e11c33fb808e4616708a6de9`.
- Manifest digest: `840948cb5d67b0f5b28f784209b474835c6851b7426c23c074134e54048cebc7`.

Preparation succeeded and the downloaded authorization matches the frozen
manifest, dispatch source, run, first attempt and exact two-case matrix. Both
recorded production-recursion steps started: N=200 at 22:22:38 UTC and N=80 at
22:22:49 UTC on 2026-10-15.

Results are pending. A successful launch does not imply prediction equivalence,
chain recovery, runtime improvement, or qualification. Preserve all raw outputs
and the final audit after completion, including failures and partial cases.
