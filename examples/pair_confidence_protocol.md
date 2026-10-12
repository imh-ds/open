# Interaction confidence calibration and confirmation protocol

Status: approved research protocol; implementation and construction checks precede dispatch.
Purpose: give applied researchers evidence-based guidance about when predictive interaction findings are dependable, rather than promise a universal minimum N.

## Intent and scope

The approved next step is to calibrate the matched predictive score, quantify false alarms and detection under different sample sizes, and examine whether displayed interaction patterns are trustworthy. This design operationalizes that request. Numerical operating targets and the larger study budget below are prespecified research choices, not established methodological facts.

Use the research minimum-CV weighted spline comparator that retained useful interactions in D-039. Hold its knots (3,5), ridge alpha (.1,1,10,100), joint weights (1,10,100), solver and preprocessing unchanged. Compare only the two prespecified pairs x1:x2 and x1:x3. Do not select among engines or selectors on confirmation results. Production defaults, previous studies, RF/GBM screeners and recursive network criteria remain unchanged.

A benchmark-calibrated rule is valid only within its tested simulation conditions. It is not a general p-value for an arbitrary user dataset. Unmeasured heteroskedasticity, clustered observations, missingness, ordinal scales and other interaction shapes remain outside this first study. Broader application requires additional null models and full-pipeline screening calibration; ordinary outcome permutation would remove main effects as well as interactions and is not the proposed null test.

## Study conditions and generated signals

Ten predictor-design cells:
- N in {80,150,300,600}, p=5, rho in {0,.7}: eight cells.
- N=300, p=15, rho in {.7,.9}: two stress cells.

Retain the earlier Gaussian generator, with x2=rho*x1+sqrt(1-rho^2)*z2, all other covariates independent Gaussian noise. Null classes are noise-only and nonlinear additive (x1+.6*(x2^2-1)). These test correlated, curved individual-variable relationships, not just a straight-line null.

Alternatives are pure and mixed signals with interaction coefficient b in {.2,.8}:
- pure: b*(x1*x2-rho);
- mixed: x1+.6*x2+b*(x1*x2-rho).
For alternatives set residual variance to 2/3 of population signal variance, retaining population predictive R-squared .6. Signal variance is b^2*(1+rho^2) for pure and 1.36+1.2*rho+b^2*(1+rho^2) for mixed. The smaller b in the pure class scales the whole outcome rather than reducing its standardized signal; both pure levels are deliberate scale-invariance checks. In mixed cases it changes the interaction's share of the signal. Reports must not call both pure levels weak versus strong standardized effects.

The Gaussian population incremental joint reference is b^2*(1-rho^2)^2/(1+rho^2) divided by total outcome variance. This is context, not a fitted-model performance bound or decision threshold.

Each dataset uses three outer and three inner folds, independently generated 1000-row assessment data, and four fitted comparisons (three outer plus one full training/independent). Independent assessment never determines the rule or display eligibility. It is a secondary diagnostic of generalization and selection optimism.

## Candidate prediction rule

For each fixed pair:
1. Fit a separately tuned additive baseline and the weighted joint model using training rows only.
2. Refit the additive counterpart at the chosen joint model's knots and ridge alpha. Retain observed values and all three prediction vectors.
3. Pool held-out MSE contrasts before division by pooled training-mean baseline risk. Retain overall gain and matched gain separately.
4. Pair is eligible only if pooled overall CV gain>0 and matched gain>0 in at least two of three outer folds.
5. Candidate score is pooled matched CV gain when eligible, otherwise zero. Keep all raw values and failed eligibility reasons. A nonpositive score cannot qualify.

For a dataset, the null reference statistic is the maximum score over its two fixed pairs. Calibration against this maximum accounts for these two comparisons; it does not control error across arbitrary all-pair searches or training-selected candidates. The fold agreement filter is a stability heuristic, not three independent replications or a confidence interval.

## Two disjoint stages

Stage A: 100 new noise-only and 100 new additive-only datasets per predictor-design cell: 2000 datasets. For each null class, sort its 100 maximum scores and use the maximum (the finite-sample 99th-percentile order statistic, rank ceil(101*.99)=100). For a cell, threshold is max(0, noise maximum, additive maximum). Require strict score>threshold; ties do not pass. Under exchangeability with one specified null generator, exceeding its 100-sample maximum has probability at most 1/101; this rank argument is not a distribution-free interaction test for arbitrary populations.

Stage A aggregation writes thresholds, contributing raw evidence hashes, full source/runtime/protocol identity and planned/complete counts. Freeze that JSON in a separate commit before Stage B. Stage B must consume that exact checksum. No threshold changes after inspecting Stage B; revisions require a new disjoint study.

Stage B: 200 new datasets per null class per cell (4000), plus 30 per alternative scenario/coefficient/cell (1200): 5200 datasets. Seeds and identities are disjoint from Stage A and all previous studies. Reserve seed families 308000000/318000000/328000000 for calibration data/assessment/splits and 408000000/418000000/428000000 for confirmation, using unique manifest indices. Freeze generated identities and all protocol/source hashes before each stage. Stage B identities may be preregistered before Stage A, but cannot be fitted before the threshold freeze.

Native preflight uses planned identities, not additional independent replications. Do not infer rates from it. Do not inspect Stage B development cases for code debugging; use separate disposable smoke identities and never include them in evidence denominators.

## Error-control and guidance gates

Primary null endpoint: any qualifying fixed pair in a dataset (family false alarm), assessed independently for each of 20 cell/null-class strata. Use exact one-sided binomial upper bounds with tail probability .05/20, giving a simultaneous 95% upper bound across the planned 20 strata. Require all null strata complete and upper bound<=.05 before calling the candidate rule null-qualified in this panel. With 200 datasets, 0/1/2 false alarms give upper bounds .02951/.04033/.04961; 3 or more fail. This is a prespecified scientific gate, not an implemented or accepted production rule.

For alternatives, report detection of the true pair, incorrect control-pair inclusion and missed/unknown results. Compute exact binomial intervals per scenario/coefficient/cell. A cell may receive provisional 'good detection in this simulation setting' guidance only after the entire null gate passes and its pointwise 95% lower detection bound is at least .8. At 30 replications, 28/30 gives lower .80467. Mark that sensitivity qualification as pointwise, not simultaneous across every signal cell; also display multiplicity-adjusted bounds. Never average favorable strong-signal cells with weak ones to hide failures. Never convert these conditional results into a universal N recommendation.

No individual finding is labeled '95% probability of being a real interaction': a repeated-sampling false-alarm bound is not the probability a particular discovery is true. The research vocabulary is 'rule qualified under tested conditions', 'predictive evidence', 'uncertain', and 'outside tested scope'. Final user-facing labels require a later production decision.

## Display-pattern reliability: separate endpoint

Store each outer-fold selected model's pair prediction slice on a fixed population reference grid [-1.2815516,-.5244005,0,.5244005,1.2815516] on each axis; set nuisance Gaussian covariates to zero. State explicitly that this is a simulation reference slice, not observed conditional means, a causal effect or an ALE curve.

Compute the anchored interaction contrast f(a,b)-f(a,0)-f(0,b)+f(0,0), divided by that fold's training outcome SD. This cancels additive fitted components. Predict all grid rows in one batch without additional fitting. Retain full slice predictions, contrast arrays, grid and support masks.

A grid location is supported only if each of its four anchor combinations has at least five training observations within .5 on both displayed axes. Exclude center-row/column zero contrasts from pattern similarity. Across folds use only common supported locations; require at least four nonzero-reference locations. Missing support or negligible contrast norm (<1e-8) yields 'pattern unresolved', never zero evidence of interaction.

Report minimum pairwise cosine similarity among the three contrast arrays on common support. A descriptive 'consistent plotted pattern' flag requires minimum similarity>=.8; this is not an inferential guarantee. Evaluate similarity to the known population contrast b*a*b_axis separately in simulations, and count stable-looking but wrong surfaces. Do not use the known truth to qualify a finding. Report predictive decisions both with and without the descriptive pattern flag; flagging the plot never changes the primary prediction rule or calibrated null endpoint.

## Costs, workflow and evidence

Per comparison: separately tuned additive 25 fits; two joint searches 2*73 fits; two matched additive refits 2 fits =173. Four comparisons/dataset =692 fitting/tuning fits; saved-setting replay reproduces five models per comparison =20 extra refits/dataset. Total 7200 datasets, 4,982,400 fit/tune calls and 144,000 saved-setting verification refits. Grid predictions add no fits. Partial/repeated/interrupted costs are recorded separately; never count worker replay as additional independent replication.

Two separate workflow dispatches with ordinary windows-2022 runners, two single-thread workers per shard, 40 datasets/shard, maximum parallel256 subject to account limits. Stage A50 shards; Stage B130 shards. Each shard job30min (setup4, computation22, remainder artifact upload), each prepare15min and aggregate60min: combined hard cap5550 runner minutes. Do not automatically launch Stage B before threshold/code/evidence verification. No paid runners or automatic retries. Preflight timing may require reducing tasks per shard; any increase to the total cap requires an explicit protocol revision before launch.

Dedicated protocol, runner, rule/interval helper, tests, two immutable manifests, setup and workflow. Do not modify old frozen runners or src/open production interfaces. Public method protocol documents the research scope; internal planning remains separate. Preserve unrelated public-checkout EOL changes and export exact commits with internal outline excluded.

Checkpoint each comparison with exact task/source/runtime fingerprint and digest. Audit generator reconstruction, split keys, training-only tuning candidate losses, same-setting marginal parameters, score arithmetic, pooled ratios, curve contrast arithmetic, support masks, rule outcomes, verification stamp and costs. Missing/error identities remain in planned denominators and produce lower/upper outcome bounds; no qualification when required coverage is incomplete. Reject stale thresholds, foreign identities, altered raw evidence and reused assessment data. Preserve all unique raw predictions, selection traces, fitted settings, surface arrays, logs, receipts and manifest hashes in durable partitioned archives.

## Implementation and review acceptance

Tests must witness failure before implementation, then cover independent stage identities; signal scaling and analytic references; null-class/two-pair maximum calibration with strict ties; ineligible/unknown/missing/nonfinite scores; exact simultaneous intervals; threshold provenance and changed-rule rejection; assessment-outcome invariance of tuning/predictions/masks; pooled risk arithmetic; supported anchored contrast and unknown zero-norm patterns; prediction/surface/parameter tampering; partial checkpoint cost accounting; resumed task error history; complete-versus-partial qualification and workflow budgets.

Native preflight must cover one nonlinear null and one alternative with correlated inputs. Raw arithmetic is independently recomputed without runner aggregation helpers. Verify old production/frozen-source diff is empty. Review model selection and multiple-comparison scope before frozen dispatch. Commit protocol/decision, tested helpers and runner, workflows/manifests, native evidence and launch receipts in independently reviewable steps.

## Stop conditions and disposition

If null qualification fails, do not publish supported labels or rescue it by raising thresholds on confirmation data. Diagnose on separate development identities and propose a new frozen protocol. If null qualification passes but detection is poor, report that the method is conservative or underpowered in those settings; do not infer no interaction. If prediction is supported but plot is unresolved, report added predictive information while withholding interpretation of an unstable shape.

Completion means auditable results for this fixed-pair Gaussian spline panel, not production acceptance. Public adoption, additional null families, RF/GBM candidates, screened/many-pair control, ordinal psychology data and recursive network recovery require separate validation decisions.

## Design self-review

Checked: dataset/shard/fit totals; two independent freezes; no assessment tuning; explicit null maximum controls only the fixed family; calibrated thresholds do not imply posterior confidence; pure-coefficient scale invariance is disclosed; plotted contrast support includes all anchors; uncertainty counts are preserved; pointwise and simultaneous sensitivity bounds differ; no new production method rule is implied. The gate targets, larger budget and study scope remain approved for research implementation.

## Method references

- Bates, Hastie and Tibshirani: CV dependence and uncertainty, https://arxiv.org/abs/2104.00673 . Fold agreement here is a heuristic, not a CV confidence interval.
- SciPy exact binomial interval API, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html . One-sided bounds must use the stated alternative and adjusted confidence level.
- Simonsohn, Interacting With Curves, https://journals.sagepub.com/doi/10.1177/25152459231207787 . Nonlinear correlated main effects belong in null investigations. This design does not implement or claim equivalence to its bootstrap procedure.
