# Evaluating OPEN and interpreting the researcher report

Use this workflow when the question is whether the complete discovery procedure
produces predictors that work on new independent participants:

```python
from open.reporting import analyze_researcher, write_researcher_outputs
from open.metadata import validate_metadata

# data is a dataframe containing numeric scale scores; config is an OpenConfig.
variables = tuple(data.columns)  # exclude IDs, leakage and ineligible scores first
metadata = validate_metadata(variables, [
    {"name": name, "type": "continuous"} for name in variables
], config.analysis.root)
analysis = analyze_researcher(
    data, config.analysis.root, variables, config, outer_folds=5,
)
write_researcher_outputs(analysis, "results/my-study", metadata=metadata, plots=True)
```

Install the visualization extra for `plots=True`. The writer refuses to overwrite
an existing destination and publishes the complete report atomically. The current
workflow accepts only layer-consensus configuration, independent cross-sectional
participants, numeric scale predictors and a fully observed continuous root.
Metadata validation and substantive exclusions remain the researcher's responsibility.
An ordinal predictor is treated as a numeric score, not an ordinal likelihood.

## What happens inside an outer fold?

1. Put some participants aside.
2. Discover the complete network using the remaining participants only.
3. Collect retained direct root predictors, including all members of joint groups.
4. Tune and refit a root prediction model with those predictors on training data.
5. Independently tune/refit a model with all original eligible candidates.
6. Predict the participants put aside with both models and the training mean.

Every participant is tested once. Discovery's own CV and resampling occur inside
the outer training fold. Imputation, scaling and tuning never see outer test rows.
This evaluates the discovery-and-refit algorithm, not a single final network.
Indirect nodes are not substituted for measured direct predictors. No path
products or mediated predictions are calculated.

The configuration and engine must be specified before evaluating. Trying many
engines/cutoffs and publishing the best outer score turns that evaluation into
selection data; use fresh confirmation data or another outer selection layer.

## Reading the scores

- **RMSE:** typical prediction-error magnitude in the outcome's original units.
  Smaller is better; it is the square root of mean squared error, not mean absolute error.
- **R²:** `1 − pooled squared prediction errors / pooled held-out outcome variation`.
  Negative values are preserved. Constant outcomes leave it unavailable.
- **Prediction skill:** `1 − squared prediction errors / training-mean baseline errors`.
  Zero means equal baseline performance; 0.30 means 30% less squared error; negative
  means worse. A perfect zero-error baseline leaves it unavailable.

All scores use the same outer test participants. Pool individual losses, rather
than averaging fold R² values. Comparing retained predictors with all candidates
shows whether the simpler predictor set loses or preserves prediction. Neither
good prediction nor similarity to the full model proves the network structure.
Do not interpret a performance difference as significant without a suitable
uncertainty analysis. This workflow supplies no confidence intervals or p-values.

A completed discovery retaining no predictors legitimately predicts the training
mean. Incomplete discovery anywhere in a fold, or a prediction-model failure,
marks the fold failed. Every planned fold remains in the evidence; combined scores
are withheld if any fails. Valid depth caps and ordinary stops remain complete.

## Why the report also contains a reference network and effects

The reference network is separately discovered from all supplied participants.
Its conditional selection counts help describe repeatability under that dataset
and configuration. It is not independent structural confirmation. A link near
the cutoff deserves scrutiny, but no arbitrary proximity boundary is calibrated.

Optional root effects use the fixed engine and full original candidate set in a
separate descriptive local-CV supplement. They describe a different adjusted
model from the retained-predictor outer evaluation. Read unique/joint ΔCV R² with
the signed percentile changes, overall curve magnitude and support records.
Root effect metrics are integrated in the report; effects for indirect targets
require separate explicit local supplements. First-order ALE can average away
interactions. Standardized units facilitate comparison but do not remove differences
in reliability, populations, adjustment sets or curve resolution.

Saved evidence contains configuration, data fingerprint, original row positions,
fold seeds, parameters, predictions, warnings and failure reasons. Each fold's
training-only network and the full-data reference have their normal auditable
bundles. The input fingerprint excludes dataframe index labels; row identities
are positional. The exported JSON is schema `open.procedure_evaluation`, version 1.

## What this workflow does not validate

It supplies no SEM global fit, causal orientation, mediation, false-discovery
guarantee, sampling uncertainty or minimum sample-size rule. Threshold sensitivity
must reconstruct the network using `fit_layer_sensitivity`; the report marks it
unassessed rather than implying it ran. Same-depth comparisons remain a separate
exploratory supplement. Dependent observations require grouped or temporal
evaluation, which is not supported yet.

Scientific validation must test known development and confirmation scenarios,
including pure noise, redundant substitutes, chains, nonlinearities, interactions,
colliders/common causes and measurement error. Record structural/class/depth
errors, prediction, incomplete rates and computational cost. Freeze final seeds,
sample/noise/reliability grids, thresholds, methods and success rules before final
execution. A small example batch establishes integration only; it cannot calibrate
the 70% default or establish “small/medium/large” effect labels.

## Recorded development evidence

The [predefined pilot and survey report](../examples/results/2026-10-09-procedure-evaluation/REPORT.md)
preserves successful and unfavorable results. In the interaction-only pilot,
GBM predicted well when supplied both candidates, but the selection procedure
often discarded them. Its retained-predictor model performed worse than the
training-mean baseline. The pilot used 80 synthetic participants and only three
assessments per target; two out of three selections fall below 70% retention.
This illustrates why the all-candidate comparator and scientific calibration
are necessary. Tree capacity to learn interactions does not guarantee that OPEN
retains the required inputs. No threshold was changed in response to the result.

The [interaction investigation](../examples/results/2026-10-10-interaction-investigation/REPORT.md)
reproduces the original root decisions and separates assessment count from local
CV training size. Conditional ablation can retain an interaction input even when
its standalone model predicts poorly; there is no standalone-gain requirement
for a unique link. A local full model that fails the predictive-gain gate selects
no inputs, however, and unstable local fits can lose useful combinations.

The quick pilot used two local folds and three assessments. Research defaults
are five local folds with two repeats and 100 assessments. Nested subsampling
and CV can leave each fitted model with far fewer people than the dataset's
headline sample size. Inspect those sizes and local low-gain stops. More
assessments refine repeatability estimates; they do not give each fit more data.
Neither a larger budget nor five-fold CV guarantees interaction recovery.

Keep settings fixed before outer evaluation. If retained-input prediction falls
behind all-candidate prediction, acknowledge possible loss of useful inputs;
do not repair the network by selecting a threshold from those test scores.
No acceptable-gap cutoff, interaction-group rescue or automatic prediction
safeguard has been calibrated. Separate development and confirmation studies
are needed before changing the selection policy.

The survey's root effect plot uses predictor percentiles on its horizontal axis.
Equal percentile steps can cover unequal changes in the original score, so even
a linear score-based association can look curved on that axis. Use the actual
predictor values in the exported percentile table, or plot the ALE against those
values, before interpreting bending as a nonlinear score-based relationship.
