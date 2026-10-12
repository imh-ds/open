# Development revision: pair validation and supported plots

The previous pair-screen study correctly prioritized known pairs but produced
incidental candidates under additive and noise outcomes. Its detailed grids also
failed systematically with small samples and correlated inputs. This revision
investigates those two limits; no production interaction-retention rule is adopted.

## Prediction check

Compare a model allowing a separate curved relationship for every predictor with
the same model plus a flexible joint surface for one shortlisted pair. Use cubic
splines and ridge regularization for this diagnostic. This is a separate
validation model family, not coefficients extracted from RF/GBM and not an
interaction term supplied to the original tree learner.

Select pairs using only the outer training data and its fitted RF. Fit imputation,
knots, scaling and tune both comparison models inside that training partition.
Evaluate paired predictions on the untouched outer assessment observations.
Report the improvement relative to the training-mean prediction baseline and the
two model risks, without a significance claim. Hold the candidate identities
fixed within that outer fold; absence from a fold is unexamined, not a negative.

The comparison is a diagnostic, not proof of population moderation. It depends
on the additive model being sufficiently flexible, and joint surfaces can proxy
other omitted relationships. Correlated inputs and higher-order interactions can
make interpretation ambiguous. A positive gain is provisional; searched pairs,
shared CV training sets and a small number of folds do not yield a p-value.

Training-only preprocessing and nested tuning follow the rationale in
[scikit-learn's nested-CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Separate splines plus joint spline features follow the construction illustrated
in its [nonlinear feature-engineering example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html).
These sources support implementation patterns, not scientific validation of this
particular candidate-screening method.

## Supported detail plots

An opt-in adaptive detail mode first tries the requested grid, then coarser grids
down to two bins per axis. Prefer a complete supported grid. If none is complete,
use the coarsest grid and examine only cells with the minimum observed support.
Mask the other cells, preserve their counts, and record partial coverage.
The coarse screen remains unchanged. Adaptive grids do not make off-manifold
corner predictions trustworthy, and their scores are not comparable across
resolutions as if they were a single effect-size definition.

## Development gate

First regression-test analytic separate/joint functions, split isolation, bounded
fits, unknown handling, adaptive support, masking and serialization. Then run
fresh development simulations using the prior six generating scenarios and four
N,p conditions; report null gains, true-pair gains, selection coverage and support.
Prespecified gain cutoffs are sensitivity summaries, never selected after results.
No default change, minimum sample rule or calibrated detection claim follows.
Any proposed acceptance rule needs a separately frozen disjoint confirmation.

## Experimental usage

For supported detail plots, pass `PairScreenConfig(adaptive_details=True)` to
the existing interaction screen. The default remains fixed grids.

For a prediction check, call `open.pair_validation.validate_pairs` with the full
numeric feature frame, outcome array, disjoint training/assessment index tuples,
and a tuple of candidate pairs selected using training data only. For example:

```python
from open.pair_validation import PairValidationConfig, validate_pairs

check = validate_pairs(
    X, y, training_indices, assessment_indices,
    (("Social Support", "Perceived Stress"),),
    config=PairValidationConfig(max_pairs=2, max_fits=100, seed=615),
)
```

Each returned pair retains predictions, both MSEs, the reference MSE and its
held-out gain. A gain of .03 means three percentage points of improvement in
predictive skill relative to the training-mean reference, on those assessment
observations. Negative gain means the joint model predicted worse. It is not
variance explained solely by the interaction, an RF coefficient or a significance
test. Missing results remain unavailable. This helper is separate from network
construction; it does not automatically retain candidate junctions.

The frozen [development protocol](../examples/pair_validation_protocol.md) uses
RF for shortlisting and these comparators for evaluation. Replay earlier frozen
studies from their recorded source commits; their manifests are intentionally
unchanged by this revision.
