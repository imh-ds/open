# Multi-engine recursive null validation launch packet

**Status:** Frozen for review; no scientific run has been dispatched.

This packet prepares a null-only validation of the recursive RF, GBM, and
LightGBM engines. It contains the full case manifest, frozen protocol, and
GitHub Actions workflow. Preparation and verification performed zero model
fits. The scientific results do not exist yet.

## Frozen run

- 330 cases: 55 repetitions for each engine at N=80 and N=200.
- Two paired waves in one Actions run, with at most ten concurrent jobs per
  wave (20 total requested).
- 100 subsamples, 80% sampling, repeated 5-fold CV (2 repeats), three tuning
  folds, 100 trees, four workers per case, and one numeric thread per worker.
- Case caps total 18,425 runner-minutes; preparation, audit, report, and release
  reserve 256 more. The 20,000 runner-minute ceiling leaves 1,319 unallocated.
- Missing or incomplete cases stay unknown. No automatic retries or replacement
  cases are allowed. A cell passes only with all 55 cases known and an exact
  95% Clopper–Pearson upper bound no greater than .10.

The manifest is `examples/study_manifests/tree-recursive-null.json`; its
canonical-content SHA256 is recorded below. The protocol is
[`tree_recursive_null_validation_protocol.md`](../../tree_recursive_null_validation_protocol.md).

## Review and launch boundary

The workflow is configured so a push to `codex/multiengine-preflight` only
registers the workflow and performs no fits. A manual `workflow_dispatch` runs
the frozen 330-case study. Before dispatch, the exact source commit and this
manifest must be checked together. The workflow binds both waves to that one
GitHub source SHA and refuses retries.

No fit, Actions dispatch, artifact upload, or release publication was performed
while preparing this packet. Dispatch remains a separate approval step.

## Local verification

- Focused recursive qualification and transport suite: 32 passed.
- Python compile checks, workflow YAML parse, manifest regeneration, and
  manifest self-validation passed.
- Local no-fit preparation produced both exact 165-case wave matrices: 330
  authorized cases total and zero model fits.
- Full repository suite: 853 passed, 2 skipped, 45 failed. Failures were
  outside this packet's focused suite; several frozen studies reject the local
  Python 3.14 environment, and chain evidence currently fails its existing
  null-report consistency check. The local full-suite run does not qualify the
  frozen GitHub Python 3.11.9 runtime.

**Frozen manifest SHA256:** `169c83ebb14568e7711d3f184362e23e9118c6772e44642ec09de6a66c4eb57e`
