# Prospective, condition-specific interaction validation proposal

Status: design approved by the user on 2026-10-13; implementation plan prepared and awaiting review. No new model fits, threshold learning, confidence adoption or Actions dispatch. This is the prospective response to the completed confidence diagnostic probe, not a revision of the failed historical confirmation.

## Purpose and alternatives

Users need to know when an interaction finding and its displayed curve can be trusted. The previous global assurance gate failed; a small diagnostic also showed detection losses at low distinct interaction information and correlated predictors. Neither outcome establishes that interactions cannot be assessed.

Recommend a new, disjoint validation with predeclared condition-specific qualification and adequate replication. Increasing only the old replication count would leave the interpretation of global failure unchanged. Changing cutoffs or tuning now would introduce a different candidate requiring fresh development and recalibration; this proposal keeps the candidate fixed. It permits a supported condition to be reported while another remains unsupported, with simultaneous uncertainty across the entire planned family. Historical D-040 remains failed under its original global gate.

## Frozen candidate and scope

Use the existing minimum-CV weighted spline, three outer and three inner folds, the same spline grid, separately tuned additive baseline and matched-settings additive counterpart. Two fixed pairs only: x1:x2 and x1:x3. A candidate is eligible when pooled overall CV gain is positive and matched gain is positive in at least two of three outer folds. Its score is the nonnegative pooled matched gain if eligible, otherwise zero. Require strict exceedance of the historical cell-specific cutoff. Preserve every raw score, fold result and reason for exclusion.

The existing threshold artifact is `study_manifests/pair-confidence-thresholds.json`, normalized-LF SHA256 `755a6fe65f908c8b3f61a6ca4fa2d5aa5622a912519b53238b470773dbecbf17`. Freeze a copy/reference and all source/runtime identities before running. Do not recalibrate, select a new grid or change cutoffs after seeing this study. This validates a particular already-developed candidate, conditional on its frozen calibration artifact, rather than the performance of repeated recalibration.

Claims concern these Gaussian generators, this spline comparator and two fixed pairs. They do not establish causal moderation, an arbitrary-population significance test, RF/GBM reliability, screened/all-pair error control, recursive network recovery, ordinal/Likert performance or a universal minimum sample size. Do not put these new labels into the production package automatically.

## Conditions and fresh identities

| Dataset N | Predictors p | Correlation rho | Matched-information targets q |
|---:|---:|---:|---|
| 300 | 5 | 0 | .01, .04 |
| 300 | 5 | .7 | .01, .04 |
| 600 | 5 | 0 | .01, .04 |
| 600 | 5 | .7 | .01, .04 |
| 300 | 15 | .7 | .01, .04 |
| 300 | 15 | .9 | .002, .005 |

Each of six conditions has noise-only and nonlinear-additive null classes (12 strata), and four alternatives (24 strata): two mixed matched-q alternatives, realistic mixed b=.8, and pure interaction b=.8. Overall population R-squared remains .6. At rho=.9, q=.01/.04 are unattainable under this generator; the smaller targets are explicit and are not equal-strength comparisons with other rows. The pure b=.2 case is omitted because scaling b at fixed R-squared does not weaken standardized signal. Omitted N80/150 conditions receive no new qualification or inference.

For mixed scenarios, D=1.36+1.2rho, A=1+rho squared, C=(1-rho squared) squared/(1+rho squared), and b squared=qD/(.6C-qA). Require a positive denominator and independently check ideal population references. q describes population joint information beyond an ideal additive projection, not a causal effect size or a lower bound on fitted CV performance.

Proposed seed bases are 608000000 (training), 618000000 (independent assessment), 628000000 (splits), plus unique prospective manifest indices. Before freezing, audit uniqueness against every prior study manifest and actual seed field; resolve any collision before dispatch, not by dropping a case later. Independent assessment remains 1,000 rows per task, descriptive only and never used in candidate selection or qualification.

## Endpoints and simultaneous uncertainty

All intervals are exact one-sided Clopper-Pearson binomial bounds over independent simulated datasets. Allocate total error probability .05 across three endpoint families. The union-bound coverage does not require independence between endpoints. It describes frequentist coverage of the planned simulated-generator rates, not the probability a particular reported finding is true.

1. **False findings:** alpha=.02 across 36 rates, hence .02/36 per bound. Twelve rates are probability of any selected pair in each null stratum. Twenty-four are probability of selecting the incorrect control pair x1:x3 in each alternative stratum. Require upper bound <=.05. Familywise here means the two predeclared pairs per null dataset; it is not all possible pairs.
2. **True-pair detection:** alpha=.02 across 24 rates, hence .02/24 per bound. Require lower bound >=.80 for the probability the known true x1:x2 passes the frozen candidate rule.
3. **Conditional curve fidelity:** alpha=.01 across 24 rates, hence .01/24 per bound. Among true-pair findings passing the rule, common-support check and descriptive fold-consistency check, count a success when minimum fold-to-known-truth cosine is >=.80. Require lower bound >=.90. Known truth is used only to score benchmark fidelity, never in model selection, support or display eligibility. Zero eligible curves is unresolved.

Retain the existing five-by-five anchored contrast grid, training-only four-anchor support check, common support across three folds, and minimum pairwise cosine >=.80 for descriptive consistency. Verify exact definitions against frozen D-040 sources rather than implementing an approximate substitute. Conditional binomial inference is justified by independent, identically generated datasets within each stratum and a fixed, truth-blind eligibility rule; the number selected is random. This curve rate is explicitly conditional on being displayed, not accuracy of every interaction or every point of an arbitrary surface. Cosine checks relative direction/pattern over common supported locations; it does not certify magnitude, exact pointwise values, extrapolated areas or the full surface. Quantitative effect-size accuracy remains a separate unvalidated endpoint.

An alternative receives **predictive support** only if both null-class bounds in its design condition, its incorrect-control bound and its true-pair detection bound pass. **Curve-pattern support** additionally requires the conditional curve bound. No curve label follows just from predictive support. Report displayed-curve count/total planned signal cases separately, so high conditional accuracy cannot conceal rare supported displays. Broad support across the entire grid requires all applicable gates to pass; do not pool strata to override a failed gate.

Require every planned case in a stratum complete and audited before any label for that stratum. Missing/errored cases remain unknown, not negatives or excluded denominators. Point estimates and counts may be shown with explicit incompleteness. A skipped alternative is unassessed, not a failed detection result. Nonpassing gates mean insufficient support under this rule/condition; distinguish an interval that clearly misses a requirement from an interval spanning it. Do not interpret either as proof of no interaction.

## Replication, assurance and staged stop rule

Stage A: 1,200 new datasets per null stratum, 14,400 total. At alpha=.02/36, upper bound <=.05 permits at most 36 false-alarm datasets out of 1,200. Hypothetical true rate 2% gives approximately .99237 single-stratum probability of passing; all 12 independent equal-rate null strata pass with probability approximately .91222. These are planning assumptions, not empirical estimates or guarantees. Actual rates and cutoffs differ by condition.

After a complete independent Stage A audit, proceed to Stage B only for design conditions with both null classes passing. Freeze all 10,800 prospective alternative identities before Stage A and retain skipped identities and reasons. Do not change the 36/24/24 multiplicity allocations because fewer conditions pass. If none pass, stop: the candidate requires new development rather than more signal simulations.

Stage B: 450 datasets per alternative stratum, up to 10,800 total. At alpha=.02/24, detection lower bound >=.80 requires at least 387 detections out of 450; hypothetical true sensitivity .90 gives approximately .99724 probability of passing. For shape, the denominator is eligible displayed curves, not 450: at n_selected=200, at least 194 correct curves pass the .90 lower-bound requirement. At hypothetical conditional fidelity .99 this gate passes with probability approximately .99570. For the wrong-control gate, at most 8/450 false findings pass. At hypothetical control false-positive rate 1%, single-stratum passing probability is approximately .96056; at 2%, it is only .45432. The 450 count is chosen to exceed 95% single-stratum assurance at a 1% control rate; null counts target a different hypothetical 2% rate. These rates are assumptions, not estimates. Detection and control outcomes within a stratum are dependent; do not multiply their powers. These examples are not a power promise at weak signals; weak conditions intentionally locate limits.

The assurance calculation file and reproducible script beside this protocol give exact gate counts and hypothetical probabilities. Only the data-independent calculations have been run. No inference should be drawn about unrun alternatives from these calculations.

## Actions and durable evidence budget

At the existing 692 fit/tune calls plus20 saved-setting verification refits per dataset, maximum planned computation is 25,200 datasets, 17,438,400 fits/tuning calls and504,000 verification refits. Native preflight repeats, failed attempts and uncertain interrupted costs are recorded separately. This cap is new; unused old-study budget does not transfer.

Propose 80 tasks/shard and two single-thread workers, ordinary windows-2022 runners: 180 Stage A and at most135 Stage B worker jobs. Each worker timeout20min, setup allowance4min, computation12min, remaining4min for finalization; prepare15min and aggregate60min each stage. Maximum timeout-based cap=180*20+135*20+2*(15+60)=6,450 runner minutes. This is not expected runtime or a billing quote. Set max-parallel256, with actual concurrency constrained by account capacity; do not claim all jobs will run simultaneously. Both stage matrices are below GitHub's [256-job matrix limit](https://docs.github.com/en/actions/reference/limits).

Before dispatch, verify an 80-task shard's feasibility from a representative native timing preflight, process isolation/resume checks, sparse-checkout source guard, pinned runtime, independently reviewed frozen manifests and exact source export. If timing fails, stop and revise shard sizing/budget before launch. No automatic retries, enlarged timeouts or paid runners. Record one dispatch receipt per stage. Stage B remains blocked until Stage A audited evidence and the selected condition list are committed and frozen.

Retain source/runtime/API metadata, all scores/predictions/support/curves, failed attempts, checkpoint cost records, complete inventories, archive hashes and standalone arithmetic verifiers. Deduplicate only byte-identical copies. Archive and verify evidence before Actions artifacts expire, including skipped/error identities. Export public source/protocol/results without internal outline files and preserve unrelated public edits. Partition evidence and pushes below GitHub's [2 GiB per-push limit](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit); propose <=1 GiB added payload per evidence push, sequential archival commits, and estimate storage from preflight before launching. No silent history rewrite or substitute for raw evidence.

## Approval and next development gate

This document is the reviewable specification. Approving it permits preparing the written implementation plan; no runner change or dispatch is bundled into this design status. The implementation must preserve production and old frozen studies, add condition-specific rule/manifest/aggregation tests, run a representative preflight, receive independent review, and freeze/export exact source before the once-only Stage A launch. Material changes to candidate, endpoints, counts, skip rule or budget require a revised prospective specification.

Simple interpretation of the eventual report: which tested conditions reliably find an interaction, which also reliably show its relative supported pattern, and which remain unsupported. Any empirical N guidance is restricted to the tested N/p/rho/signal conditions, with effective outer-training size about two-thirds N and inner-fit size about four-ninths N. No interpolation to a universal sample-size rule.
