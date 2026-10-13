# Interaction confidence diagnostic development probe

User approved proceeding with the recommended targeted investigation. This research probe diagnoses D-040 assurance precision and correlated-signal misses. It makes no change to the completed confirmation outcome, frozen thresholds, production engines or accepted scientific claims.

## Questions and scope

1. How much assurance power does the existing simultaneous false-alarm gate have at hypothetical fixed low rates? Use exact binomial calculations, not additional model fits. Distinguish unconditional rank guarantees from conditional calibration uncertainty. Calculate feasible calibration replication counts and confirmation sample sizes as prospective illustrations; they do not revise D-040.
2. At comparable ideal population additive-versus-joint information, do correlated cases still lose predictive information? Distinguish finite-sample fitted-model loss, candidate eligibility, fixed-cutoff attrition and descriptive curve support. Population references are diagnostic truth, never decision inputs or finite-sample CV bounds.

## Fresh development design

Mixed generator from frozen D-040, overall population R-squared .6. N300/600, p5, rho0/.7, target distinct-joint reference q=.01/.04, six independent replications per cell: 48 datasets. For D=1.36+1.2rho, A=1+rho squared, C=(1-rho squared) squared/(1+rho squared), set b squared=qD/(.6C-qA). Reject unattainable q; this changes coefficients to hold distinguishable information constant. The target is a population reference, not an effect-size label.

Twelve fresh null diagnostics: six N300/p5/rho0/noise, six N300/p15/rho.7/nonlinear-additive cases. They investigate fitting/selection behavior, and cannot estimate or certify rare false-alarm rates.

Three outer and three inner folds; same minimum-CV weighted spline grid, two fixed pairs x1:x2 and x1:x3; 1000 independently generated assessment rows. Retain four comparisons/task, 692 fit/tune calls plus20 saved-setting verification refits. Apply the historical strict thresholds for diagnostic attrition only, with all raw gains and exclusions retained. Evaluate ideal additive and joint predictions separately on the same independent rows; do not feed known truth into model selection, eligibility, thresholds or visualization support.

Fresh seeds508000000/518000000/528000000 plus manifest indices. Check against the completed calibration/confirmation manifests. No D-040 assessment identities are reused. Missing or errored cases remain unknown; no reliability labels, confidence bounds from these six-replication cells or new threshold artifact.

## Assurance calculations

Historical simultaneous tail .05/20, target upper false-alarm limit .05. Compute exact allowable count at n200 and prospective n400/600/1000/2000; illustrative fixed true rates .001/.005/.01/.02. For independent equal-rate strata, joint gate-passing probability is single-stratum passing probability to power20; clearly hypothetical, actual rates differ and calibration thresholds are random. Distribution-free continuous-null maximum-of-m reference: expected future exceedance1/(m+1), conditional probability exceedance rate exceeds p is (1-p) to power m; per-class bound for historical max of two classes is conservative. Do not claim arbitrary-population interaction significance or posterior confidence.

## Execution and provenance

Six ten-task shards, two single-thread workers each, ordinary windows-2022; max-parallel256 subject to account limit. Job15min, setup4/computation8; prepare15/aggregate20: cap125runner minutes, no paid runners or automatic retries. Total41,520 fit/tune and1,200 verification refits. Native preflight planned development cases may be replayed on Actions, counted as the same identities; record extra computation separately. Freeze source/runtime, all60identities, fixed thresholds checksum and protocol before dispatch. Sparse checkout includes all required frozen source/protocol/manifests but excludes historical result archives. Pinned checkout action documentation: https://github.com/actions/checkout/blob/11d5960a326750d5838078e36cf38b85af677262/README.md .

Research scripts are dedicated diagnostic utilities, not production method changes. Retain checkpoint/source/verification/timing evidence, failure history and all unique raw vectors/curves. Independently audit arithmetic and generator references; preserve stage budgets and missing denominators.

## Stop and interpretation

Report this small probe as diagnostic evidence, not confirmation. If matched-information cases improve, prior correlation misses partly reflect information differences; if they remain poor, inspect model/comparator selection. If independent gains exist but cutoff rejects cases, describe decision conservatism rather than disappearance of interaction. If patterns lack support, withhold curve interpretation. Assurance design limitations do not retroactively pass D-040. Any method revision or revised confidence claim needs a separate frozen confirmation protocol.
