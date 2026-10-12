# Fresh matched interaction-comparison development

This research-only revision separates overall model improvement from the
increment attributable to a joint spline block at matched marginal settings.
Production defaults and all earlier frozen studies stay unchanged.

For each training-selected joint model, fit an additive counterpart on the
same training rows using exactly its knots and ridge alpha. Standardized
marginal representation and regularization match; coefficients are refitted
when adding the joint block. Assess both on disjoint observations.

Overall gain = (shared tuned additive MSE - joint MSE)/reference MSE.
Matched gain = (matched additive MSE - joint MSE)/reference MSE.
Marginal-change gain = (shared additive MSE - matched additive MSE)/reference.
Overall = marginal-change + matched. Reference is assessment risk of the
training-outcome mean. These are prediction skill increments, not percentages
of variance explained, causal effects, significance tests or interaction
certificates. A positive matched gain can still occur by sampling noise.

Compare production v2, minimum-CV weighting, and the previous one-SE-style
penalty preference as an unqualified competitor. Selection stays based on
training-only joint-model inner-CV prediction, not assessment gains. Keep
weights1/10/100, knots3/5, alpha.1/1/10/100; three outer and three inner folds.
No new grid, clipping, difficult-row exclusion or learned cutoff.

Freeze240 fresh datasets: 48cells, five each; noise/additive/pure/mixed,
rho0/.7/.9, (N,p)=(80,5),(300,5),(600,5),(300,15). Seeds208/218/228million,
IDs matched--. Original Gaussian generator/reference unchanged. Fixed x1:x2
and x1:x3 pairs,1000independent assessment rows. The control pair's population
joint reference is zero; correlated inputs may nevertheless make a control
variable a predictive proxy. This is not causal graph recovery.

Retain raw selected settings, matched additive/joint predictions, all candidate
inner losses, outer splits, original and matched gains, marginal-change gains,
errors/warnings/costs/source/runtime receipts. Pool risks by assessment count
before ratios. Report descriptive cutoffs0/.01/.02/.05 separately for overall
and matched scores; all missing/error cases stay in planned denominators.

Interpretation fixed before fresh results: compare positive gains in noise and
additive-only cases, retention of pure/mixed gains, and negative small-N tails.
Matching is meant to prevent marginal-fit changes being credited to the joint
block, not eliminate every false gain or repair tail extrapolation. Five
replications per cell cannot calibrate all-pair false-positive control, define
minimum sample sizes or establish reliable detection. No automatic adoption.
Earlier saved diagnostic cases are development inputs, not fresh evidence.

Cost:223old fitting/tuning fits plus6matched fixed-setting fits =229/comparison,
cap232. Four comparisons/dataset =916fits;219840planned. Saved-setting
verification reproduces13models/comparison =52/dataset,12480total, not retuning.
Check matched settings, prediction risks/decomposition, both score summaries,
checkpoint digests/context, partial costs and unknown interrupted costs.
Top-level verified-task costs overlap retained checkpoint totals; do not add.

Dedicated pinned source/runtime/task manifest, runner, setup and workflow.
Native two-case preflight uses two planned identities, not extra replications.
120two-task shards, two single-thread workers, maxparallel256 subject to account
limits. Ordinary windows-2022; no paid runners/automatic retries. Job30minutes,
setup4/study22, prepare15/aggregate60:3675cappedrunner minutes. Review and
native preflight precede one dispatch. Retain raw completion evidence in repo.
Production adoption or disjoint confirmatory evaluation requires a later
scientific decision. No RF/GBM/network-recovery claim follows from this study.
