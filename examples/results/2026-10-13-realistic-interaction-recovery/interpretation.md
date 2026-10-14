# What the completed study means for applied researchers

The method can detect and display genuine interactions. Its main limitation in this study was missing interactions when samples were small, signals were weaker, or predictors were measured with error.

All 10,800 planned development datasets are complete. The repaired aggregate reused original evidence; no new models were fitted. The earlier failure was a diagnostic floating-point equality problem, not evidence about interaction detectability. Calibration separately completed all 3,600 datasets and learned 18 fixed cutoffs.

## Detection in the middle tested scenarios

These combine product and saturating interactions at coefficient 0.4, three underlying correlations (0.1, 0.3, 0.5), and 80 independent datasets per shape/correlation. Each table cell contains 480 datasets. The coefficient is a study-specific setting, not a universal small/medium/large effect-size definition. Values are descriptive averages, not qualification gates.

|People per dataset|Error-free predictor measurements|Noisy predictor measurements|
|---:|---:|---:|
|150|73/480 (15%)|10/480 (2%)|
|300|247/480 (51%)|90/480 (19%)|
|600|395/480 (82%)|191/480 (40%)|

Noisy measurements used a classical independent-error model with reliability 0.8; this does not equate the simulation setting automatically with Cronbach's alpha or omega. Observed correlations in the noisy conditions were 0.08, 0.24 and 0.40, rather than the error-free 0.1, 0.3 and 0.5.

Larger samples helped, but N600 was not a guarantee. Weaker coefficient-0.2 interactions remained hard to detect. The stronger product reference at N600 was detected in 240/240 error-free and 232/240 noisy cases across the tested correlations. Ordinary predictor correlation therefore did not make detection impossible; performance depended on the full condition, including signal strength, measurement error and calibrated cutoff. The relationship with correlation was not uniformly monotonic.

Among 1,986 displayed genuine-interaction cases, 1,920 (96.7%) passed the broad pattern-agreement criterion. This percentage is conditional on a plot being displayed, not the probability of finding every real interaction. It supports broad shape interpretation, not precise magnitude accuracy or causal moderation. There were 19/2,880 (0.66%) datasets with at least one selected pair across the observed-input noise/additive null scenarios. This pooled descriptive proportion does not certify false-positive control in each separate condition. The noisy latent-additive diagnostic is a separate target and is not included in that null denominator.

For researchers: report an absent interaction as **not detected**; measurement quality matters alongside sample size. A stable retained plot offers useful predictive evidence under the studied conditions. These results do not establish a universal minimum N. The design used continuous Gaussian scores, five predictors, two prechosen pairs, one spline candidate and 3-by-3 nested CV. Ordinal items, missingness, clustering, many searched pairs and other engines remain outside this evidence.

## Possible linear-regression companion: discussion, not implementation

`Y ~ X1 + X2 + X1:X2 + other covariates` is moderated regression. Although called a linear model, the product term lets the effect of X1 change with X2. It tests a specific form: that change progresses at a constant rate. It can therefore provide useful supporting evidence for product-like interactions, while nonlinear interactions may be missed or poorly summarized.

Recommendation for later investigation: use regression as an optional companion or candidate-prioritization signal, not a mandatory significance gate and not an automatic override of OPEN's predictive evidence. An LM-positive/OPEN-negative result should be reported as a discrepancy between the two approaches. A non-significant product coefficient should not prevent a theoretically motivated or nonlinear candidate from being examined. No LM/GAM comparison was run in this study, so an improvement in sensitivity has not been established.

Allow curved main effects before interpreting interaction evidence. Otherwise the product term may partly represent omitted curvature, particularly when predictors are correlated. A generalized additive model can represent separate curved effects plus a flexible joint interaction; see the primary sources below. Such a companion still requires its own calibration, covariate policy, multiplicity handling and uncertainty assessment.

If a screen is used to choose what OPEN fits, perform it inside training folds and evaluate the complete selection-and-fitting procedure on untouched validation data. Keep main effects and relevant covariates, handle multiple tested pairs, and preserve disagreement rather than merging p-value significance and out-of-sample predictive support into one unvalidated decision rule. These are prospective design considerations, not changes to the frozen completed study.

Primary methodological sources: [Ganzach (1997), Misleading Interaction and Curvilinear Terms](https://www.tau.ac.il/~yoavgn/files/misleading_interactions.pdf); [mgcv tensor-product interaction documentation](https://www.stat.ethz.ch/R-manual/R-devel/library/mgcv/html/te.html).

## Saved evidence

[Full report and graphs](recovered/report.md), [all dataset metrics](recovered/dataset_metrics.csv), [all condition metrics](recovered/stratum_metrics.csv), [recovery verification](recovery-verification.json), and [calibration report](../2026-10-13-realistic-interaction-calibration/README.md). The surrounding folder also preserves original failed reports, job/API receipts, logs, raw archive indexes and verification scripts. [All-stage preservation check](all-stage-preservation-verification.json) covers both original raw collections. All 180 raw ZIPs and receipts are committed locally; large raw publication to GitHub remains pending. Public reports and proofs are already independent of Actions artifact expiry.
