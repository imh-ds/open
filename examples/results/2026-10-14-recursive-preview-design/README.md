# Reference-validation design checks

Status: **design prepared for review, not an executed simulation**. The owner approved up to 10,000 Actions runner-minutes, not a release publication. The concrete [protocol](../../recursive_preview_validation_protocol.md) remains subject to design review before generation/new fitting/launch.

Proposed study: 480 independent datasets, noise/chain/correlated-direct scenarios, N80/N200, 80 repetitions per cell, four eligible predictors including two nuisance variables, unchanged default elastic net/B100/CV5x2/tuning3/retention.70. No interactions or engine expansion.

## Verified without generating data or fitting models

- [Reproducible planning checker](../../check_recursive_preview_design.py) passed: analytic population variances, CP integer boundaries by beta quantiles and independent binomial-tail checks, 480 identities/48,480 distinct reserved generator/resample seeds, and the 9,871-minute allocation within the approved10,000 ceiling.
- [Design check](design-check.json) inventories all historical study manifests and their exact hashes. None of the proposed reserved seeds matches any of 145,663 distinct explicit historical manifest seed values. Formula-derived namespace and runtime-receipt checks remain required before final freeze; this is not claimed as a complete historical collision audit.
- [Two existing adapter regression tests](adapter-tests.txt) passed. They use scripted local analyses to check restoration of sample identities, matching three-depth production recursion/layers, and zero-additional-calculation checkpoint reuse. They do not establish real-fit numerical equivalence, multiprocessing performance, or scientific accuracy.

The initial proposed .05 upper-error-bound calculation is retained in [superseded planning evidence](superseded-initial-five-percent-calculation.json). At n80 it requires zero errors per cell and has only .00803 probability of passing all six error gates if each true event rate is1%. The revised proposal uses a .10 upper bound, requiring at most2wrong-finding datasets per cell, and a recovery lower bound of.80, requiring at least72/80exact recoveries per signal cell. This change occurred during design, before data generation or design approval. It is an explicit preview trade-off, not a5%error guarantee or field standard.

The hypothetical all-six false-gate assurance is .95461 at a true.5%rate, .75124 at1%, and .23296 at2%. These calculations explain the study's precision and are not observed method performance. Per-cell intervals are marginal, not simultaneous confidence guarantees.

## Remaining gates

Review the exact design/guard; implement and test the study-only runner/auditor/workflow; freeze source/runtime/identities; pass full-default saved-fixture equivalence and representative N80/N200 timing on Actions; audit and preserve Stage A before any Stage B continuation. Neither the scientific release gate nor runtime feasibility has passed. No new dataset, model fit or Actions dispatch occurred in these design checks.

Reproduce in a new output location:

```powershell
python examples/check_recursive_preview_design.py --out NEW_DIRECTORY/design-check.json
python -m pytest tests/test_sample_size_pilot.py::test_worker_returns_pickle_safe_receipt_and_restores_original_identity tests/test_sample_size_pilot.py::test_checkpoint_adapter_matches_native_three_depth_recursion -q
```
