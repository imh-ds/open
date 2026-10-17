# Matched Elastic Net and OLS reference launch

Owner approved4,000 additional runner-minutes for the N200 comparison, without
waiting for or altering the existing LightGBM N300 wave. Sixty analyses: Elastic
Net and OLS on the same30 frozen LightGBM N200 datasets (ten each at r=.20/.30/.40).
Identical data,analysis seeds,subsamples,CV and nonlearner settings;70% retention.
Original LightGBM manifest digest:
`d5ce001bf45ec0641fabdcda61673cd2222236deee807494ce3c7ffa1be0920f`.

OLS needs no coefficient penalty to participate. The learner fits every supplied
predictor; network links still require held-out predictive gain, unique/shared
relevance and repeatable selection across subsamples. No OLS p-value cutoff is
added. Elastic Net adds coefficient regularization inside each fit. Both use
identical external network-selection rules. OLS has zero tuning fits; Elastic Net
retains its shipped25-setting grid. The default learner stays Elastic Net.

The [protocol](../../../linear_effect_pilot_protocol.md) fixes all scientific
and compute choices. Reserve3,871 minutes:30x90 EN,30x30 OLS,prepare30,audit120,
publish120,registration1;129 minutes unallocated. Up to20 case jobs in parallel;
actual runner availability can queue jobs alongside the ongoing LightGBM study.
No scientific retries or extra allocation for N300 references.

The frozen manifest is `manifest.json`.56 targeted learner/risk/config/effect/pilot
checks passed; final13 OLS/pilot checks passed after source freeze. Independent
review found no launch blockers. A small installed-package implementation probe
also completed recursive OLS assessments and full independent checkpoint replay:
four assessments,28 outer fits,zero tuning fits. Its recorded verification is
explicitly a deterministic unit probe, not scientific pilot findings. No full
scientific reference analysis was run locally.

Preserve results and all available raw receipts/checkpoints,including interrupted
cases,in a dedicated prerelease;download/hash-verify published assets. Record
unknown cases in planned denominators. These are development comparisons on
already-seen data,not independent scientific qualification. Current snapshot and
launch controls stay separate from future completed results. Exact evidence bytes
are preserved across Git checkouts.

## Verified dispatch

[Run39004942944](https://github.com/imh-ds/open/actions/runs/39004942944), first
manual attempt at public source2e0da27ec17745af37085b02f2627df7756507e4.
Preparation succeeded; the downloaded live authorization equals the frozen
manifest and all60 expected case-job identities match. `dispatch-verification.json`,
the live producer authorization and startup run/job API records preserve these
checks. Scientific outcomes are pending. The separate LightGBM N300 run remains
at its original source and was not cancelled or changed.
