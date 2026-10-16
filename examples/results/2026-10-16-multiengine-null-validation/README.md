# Multi-engine recursive null validation: checked results

Source: `279c380c637ad7b0420df9d93f1e680480f2cf3e`.
[Completed run](https://github.com/imh-ds/open/actions/runs/38846677048).
[Durable reports and all 330 raw case archives](https://github.com/imh-ds/open/releases/tag/tree-recursive-null-2026-10-15-38846677048).

## Simple interpretation

We generated datasets with no real links. The correct answer was an empty network.

| Engine | N=80 | N=200 | Interpretation |
|---|---|---|---|
| GBM | 55/55 audited; zero false networks | 55/55 audited; zero false networks | Passed this false-link check |
| LightGBM | 55/55 audited; zero false networks | 55/55 audited; zero false networks | Passed this false-link check |
| Random Forest | 55 results unverified | 55 results unverified | Audit-code mismatch blocks interpretation |

For applied researchers, GBM and LightGBM showed restraint: they did not turn random noise into relationships in any of these 220 audited analyses. This supports their false-link control for the tested independent Gaussian population at N=80 and N=200. It does not establish that they can detect real direct/indirect links, correlated predictors, or interactions. An engine that always returns an empty network could also pass this test; signal recovery is the necessary next scientific check.

For each passing 55-case cell, the exact two-sided 95% upper confidence bound on the false-network rate is 6.49%, below the frozen 10% gate. This is a per-cell statistical bound, not a guarantee for arbitrary populations or an estimate that the error rate is 6.49%.

## RF audit issue

All 110 RF case jobs completed successfully. Every RF audit row was rejected with `ValueError: explicit grid policy required`, before independent verification of the results. The new tree-null wrapper passes a manifest labeled `tree-null-v1` to the historical RF auditor, which explicitly accepts only `rf-singleton-v1`. These are different labels for the wrapper and RF-specific audit policy. The appropriate repair must validate the wrapper's engine/grid contract and invoke the existing RF tuning audit without weakening historical policy guards.

RF's apparent zero in the machine report is the number of verified false networks, with all 110 cases unknown. It must not be presented as 0/110 evidence of success. A green Actions run means execution and publication succeeded; it does not mean all scientific gates passed.

Recommended next step: repair and regression-test the RF audit adapter, then independently re-audit the saved RF archives without refitting models. Preserve this original report and its unknown rows. If that repair validates the cases, produce a separately identified corrected audit report. GBM and LightGBM can advance to a separately designed and approved signal-recovery study; this check alone does not establish release readiness.

## Evidence checks and cost

- 334 workflow jobs: all 330 case jobs, preparation, aggregation and publication succeeded; registration was skipped as intended.
- Frozen manifest digest matches the approved corrected launch; artifact transport reports verified with no errors.
- Six downloaded report/control files match GitHub release SHA256 digests.
- The six raw release bundle digests and sizes match the published archive inventory; each bundle lists 55 case archives. Raw bundles were durably published and remotely downloaded/hash-verified by the successful publication job. This local check did not redownload the approximately 2 GB raw bundles.
- Observed summed job durations: approximately 5,759 runner-minutes, below the 20,000-minute ceiling. This is elapsed runner time from API timestamps, not billing-adjusted usage. The prior failed preparation added about 71 seconds, and registration overhead is separate.
- Workflow wall time: approximately 6 hours 30 minutes across parallel jobs, not the duration of a single case.

The original published `study-report.json` and `interpretation.md` are retained unchanged. No model fits, scientific reruns, or new Actions dispatches were performed while checking these results.
