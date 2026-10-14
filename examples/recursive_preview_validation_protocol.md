# Recursive-network reference validation

Version 1, 2026-10-14. **Design, implementation and staged launch approved by the owner; execution evidence pending.** The owner approved this concrete protocol and requested launch after choosing a ceiling of 10,000 GitHub Actions runner-minutes. Source/runtime freeze, numerical-equivalence and timed preflight gates apply before the full study. It replaces neither historical protocols nor their conclusions.

## Question and claim boundary

Across new independently generated datasets, how often does the unchanged default elastic-net procedure miss a stipulated predictive relation, add an incorrect relation, or assign an incorrect depth? Evaluate N=80 and N=200 separately. This is a narrow continuous, Gaussian, independent-participant reference benchmark, not certification of arbitrary networks, engines, interactions, causal direction, or a universal minimum sample size.

There are **480 independent datasets**: three scenarios x two sample sizes x 80 repetitions. The 100 stability resamples within each dataset are not additional independent replications. No engine comparison, product terms, interaction screen, threshold search, or predictive-effect calibration is added. Near-duplicate group handling retains its existing fixed-fixture evidence; this study does not estimate its population error rate.

## Frozen data-generating design

All innovations below are mutually independent standard normal variables, all rows independent, and all columns continuous and fully observed. Each dataset has columns `(Y, X1, X2, Z1, Z2)`, where Z1/Z2 are independent nuisance predictors. They remain eligible at every depth until promoted. The root is Y. Larger datasets are generated independently, not as extensions of smaller datasets.

|Scenario|Construction|Stipulated retained relations and depths|
|---|---|---|
|Noise|Y, X1, X2, Z1, Z2 independent standard normal|No relations; only Y at depth 0|
|Chain|X1 standard normal; X2=.6 X1 + .8 E2; Y=.8 X2 + .6 EY|X2 -> Y and X1 -> X2; depths Y=0, X2=1, X1=2; Z1/Z2 unassigned|
|Correlated direct predictors|X1 standard normal; X2=.4 X1 + sqrt(.84) E2; Y=.6 X1 + .6 X2 + sqrt(.672) EY|Separate unique X1 -> Y and X2 -> Y; depths Y=0, X1=X2=1; Z1/Z2 unassigned|

For the chain the population root R-squared is .64 and the X1 -> X2 local R-squared is .36. The extra direct X1 -> Y association has no additional population conditional-mean contribution once X2 is known. For correlated direct predictors the predictor correlation is .4, signal variance is 1.008, noise variance is .672, and root R-squared is .60; each predictor has distinct conditional information. These targets are declared predictive comparisons under this traversal, not a claim that observational data identify causal paths.

Order scenarios as noise, chain, correlated; within each, N=80 then N=200; within each sample size, repetitions 0..79. Index i is 0..479 in that order. Generator seed is 1,600,000,000 + 1,000 i. Analysis seed is 1,700,000,000 + 1,000 i. The latter reserves seeds base..base+99 for all default resamples. Before freezing, compare every reserved seed to historical manifest seed fields, recorded namespace formulas, and prior runtime receipts. Any collision must be resolved by a recorded protocol revision before generation, never by silently replacing a dataset. No independent test sample is required: the endpoints here are network construction, not another prediction-performance study.

## Unchanged analysis

Load `configs/default.yaml` and override only root name Y and the per-dataset analysis seed. Preserve default elastic net, its grid/solver, max_depth=3, MSE, fold-local preprocessing, ablation parameter reuse, 5-fold performance CV repeated twice, 3-fold tuning CV, 80% subsampling without replacement, 100 resamples per evaluated target, and retention .70. All relevance/dependence thresholds remain unchanged. Do not replace 100 with `final_resamples=500` or with a smaller demonstration budget.

Reuse the production `make_layer_sample_plans`, `assess_layer_target`, consensus aggregation, and `fit_layer_consensus` traversal. A study-only checkpoint adapter may distribute independent local assessments to four single-threaded processes. Finalize every complete layer before scheduling the next. Preserve original sample indices, seeds, sampled rows and split plans; restore singleton worker indices before production aggregation. No replacement selection or recursion implementation is permitted.

The existing study adapter runs local fits in child processes, so the parent's Python allocation tracer does not measure their allocations. Record child OS process-lifetime peak working set, identify its scope, and do not present the parent's traced peak as total model memory. This changes execution/telemetry, not statistical settings. Numeric/structural equivalence with the saved sequential default fixture is a prerequisite, not an assumption. Do not modify production tracing or other package behavior for this study.

## Endpoints and denominators

Keep 80 planned independent datasets in each scenario/N cell. Preserve every planned identity, timeout, failure, partial assessment, warning, and gate-skipped status. A complete empty network is a valid result. An execution failure is not an empty network or evidence against a link.

For each completed dataset, independently reconstruct from saved evidence:

- Actual unique relations and joint-group identities, promoted members and assigned depths.
- Missing expected unique relations, extra/incorrect relations, and any unexpected joint group. In these scenarios the expected joint-group set is empty.
- Expected non-root node depth matches; unassigned required nodes are mismatches. Keep nuisance promotion and root recovery separately visible.
- Exact structure recovery: all and only stipulated relations and depths, no joint groups and no nuisance promotions.
- Any incorrect retained relation, plus counts of misses/extras and their actual targets/depths. Downstream relations prevented by an upstream miss still count as missing in whole-network recovery.
- Scheduled targets/candidates, local prediction gains, observed/train/tuning/validation row counts, estimator-fit counts, warnings, stop reasons, elapsed time and telemetry scope. Conditional local selection frequencies remain separate from independent-dataset recovery rates.

Report missing/extra relations per expected/possible relation descriptively as well as dataset-level events. Inferential intervals below use independent datasets as units, not pooled edges or resamples. The correlated scenario tests two direct links, not an attempt to orient the association between same-depth predictors.

## Prespecified qualification and interpretation

Use two-sided 95% Clopper-Pearson intervals per cell, without rounding for decisions. Qualification requires all planned cases in the relevant stage to complete and pass the integrity audit:

1. In each of the two noise cells, the upper bound for datasets with any retained relation must be <= .10.
2. In each of the four signal cells, the upper bound for datasets with any incorrect retained relation must be <= .10, and the lower bound for exact structure recovery must be >= .80.

With n=80 these mean at most two incorrect-finding datasets per cell (2/80 upper bound .0874071559; 3/80 upper bound .1057019850) and at least 72/80 exact recoveries in each signal cell (72/80 lower bound .8124348925; 71/80 lower bound .7971534081). Save the separate missed-link counts even if a cell qualifies. This is the owner-approved research-preview guard, not an established acceptable error standard for psychology or a 5% error guarantee.

Design assurance matters: a .05 upper-bound criterion at n80 would require zero errors in every cell; if the true event rate were .01 in all six independent cells, the chance of passing that part of the overall gate would be only .00803. The proposed .10 bound permits up to two per cell; at assumed true rates .005, .01 and .02, its per-cell pass probabilities are .99229, .95345 and .78442, and all-six probabilities .95461, .75124 and .23296. These are planning calculations, not observed method performance. They show that even this bounded study can be inconclusive despite useful performance. Do not escalate the budget or loosen the guard after observing data.

These are marginal per-cell intervals, not simultaneous confidence bounds or general false-discovery control. Passing all cells permits only the scoped statement that these reference conditions met these prespecified criteria. It is not a probability that an individual empirical link is true. Failure remains a failure; no cutoff adjustment, seed replacement, new scenario exclusion, or post-hoc pooling can rescue qualification. Report results by N even if the larger sample does not improve recovery.

If incomplete, publish observed counts with unknowns and inclusive best/worst-case bounds over all 80 planned identities; withhold positive qualification. Gate-skipped signal cases are not observed misses or false findings. The final report must explain simply which links were missed, which appeared spuriously, and whether depth mistakes were caused by earlier selection changes.

## Staged execution and hard compute ceiling

Use ordinary Windows2022 runners, Python3.11.9, and the same exact scientific dependency versions as the completed default fixture pilot. No paid-capacity purchase, larger runners, automatic model retries, numerical fallback or new replication after a timeout. Set OMP/BLAS/MKL threads to one. Each shard runs two consecutive datasets with four assessment processes per dataset. Permit at most 20 shard jobs concurrently; this is an execution cap, not a promise of account capacity or wall-clock completion time. Stage A has 80 shards and Stage B has 160, each below GitHub's [256-job matrix limit](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstrategymatrix). Assign adjacent indices to each shard; preserve dataset-specific receipts and denominators.

|Allocation|Hard cap in runner-minutes|
|---|---:|
|No-fit workflow registration|1|
|Preflight/equivalence job|120|
|Stage A preparation and aggregation|15 + 60 = 75|
|Stage A: 160 noise datasets in 80 two-dataset shards, each shard at most 40 minutes|3,200|
|Stage B preparation and aggregation|15 + 60 = 75|
|Stage B: 320 signal datasets in 160 two-dataset shards, each shard at most 40 minutes|6,400|
|Total allocated ceiling|**9,871**|
|Unallocated within the owner-approved ceiling|**129**|

Each shard reserves at most 3 minutes for setup, 30 for computation (15 per dataset) and 7 for preservation inside its 40-minute hard cap. A soft deadline stops scheduling new assessment batches before the computation allocation expires; an in-progress batch can overrun the soft deadline but not the job cap. Upload available checkpoints on failure where Actions permits. A hard kill may leave unknown in-progress work; retain started-attempt records and never claim its fit cost was zero. The 129 minutes are not automatic permission for reruns or timeout increases.

Preflight must first reconstruct and compare all assessment payloads/layers from the existing saved sequential full-default chain fixture against a checkpointed parallel execution. Compare exact numerical risks, fitted parameters, selections, depths and sample/split identities; distinguish expected provenance/timing/warning-location differences. Next execute planned chain IDs i=160 (N80) and i=240 (N200) once, retaining their outputs and computation costs. Each planned case must finish within its 15-minute computation allocation on the intended runner. These two datasets remain in the 480 planned identities; Stage B must reuse their verified outputs with zero new fits, not count them as extra replications. Preflight fitting of the old fixture is additional computational work, not scientific replication. Do not use the signal results to change settings or the launch rule.

Require measured storage/upload/aggregation feasibility within the allocated caps. If equivalence, timing, memory or evidence preservation fails, stop before Stage A; propose a separately reviewed execution change or narrower study. Do not reduce resamples/CV/settings to make timing pass. No expensive native preflight outside this Actions allocation is authorized by this protocol.

Launch Stage A once only after the protocol/implementation/runtime/source/seed/equivalence checks are frozen and approved. Audit both 80-dataset noise cells independently and preserve their raw results before Stage B. Launch Stage B once only if all 160 noise cases complete with no integrity errors and both null bounds pass; otherwise preserve the negative result and skip Stage B. Verify the committed Stage A audit/selection fingerprints before continuation. No production policy is changed by either stage.

## Evidence and release decision

Before launch, freeze the exact implementation revision, normalized source hashes, original config hash, full case manifest, seed-collision evidence, protocol hash, runtime lock, endpoints, job caps and adapter equivalence/preflight receipts. Preserve initial/completed/failed attempt records, all generated rows and full assessment payloads, warnings, checkpoint digests, completed networks and independent arithmetic/structure audits. Content/identity mismatches stop qualification; matching filenames alone are insufficient.

Aggregate without new model fits. Recompute losses and relevance contrasts from saved observation-level evidence, verify sampled/split identities and layer finalization, then independently score relations/depths and binomial bounds. Preserve every unknown/error row. Distinguish evidence re-audit/transport from new modelling; no retry of scientific fits is allowed automatically.

After each stage, download and preserve exact artifacts with SHA256 hashes alongside the reports in the repository; Actions retention is temporary storage. Keep large raw archives losslessly reconstructible and record exact public-branch provenance. Save a table and exportable graphs for recovery, any incorrect relation and depth errors, with N-specific uncertainty and a plain-language interpretation. No preview release tag, main merge, version change or package upload is included in this study authorization. Review the scoped findings before selecting the release artifact.

## Approved execution amendment: 2026-10-14

The original preflight run38615814397 remains failed. Its200completed fixture assessments are retained with the original manifest and archive fingerprints. Independent no-fit diagnosis found only derived floating-point roundoff (maximum2.220446049250313e-16), with exact saved decisions and140761passing observation-loss checks. Neither planned case160 nor240 ran. The [saved diagnosis](results/2026-10-14-recursive-preview-validation/preflight-failure-38615814397/interpretation.md) records the evidence and limits.

The owner approved the [no-refit continuation](recursive_preview_continuation_proposal.md). Only the equivalence checker changes: derived floating-point fields permit rtol1e-12/atol1e-14; data, fitted-parameter fields, splits, selections, config and graph decisions remain exact. The continuation must verify original archive/receipt/producer hashes, identical production sources and unchanged scientific settings before reusing the200assessments. Its control receipt explicitly records zero new fixture fits. The two previously unstarted study cases run once each under their original15-minute case caps. All original downstream scientific, audit, preservation and stage gates remain binding.

The new job has a45-minute hard cap, plus one1-minute no-fit workflow registration. Retaining every original allocation yields9871+45+1=**9917runner-minutes**, with83unallocated under the10000ceiling. The earlier120-minute preflight allocation is retained in this accounting, not reused. A failed continuation stops again; no automatic retry, new scientific identities or relaxed scientific cutoffs are authorized.
