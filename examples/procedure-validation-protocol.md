# Complete-procedure development pilot: protocol fixed before execution

This is a software integration pilot, not the final scientific evaluation.
Do not choose thresholds or establish general error rates from these results.

- Five deterministic scenarios: noise, direct linear, redundant substitutes,
  U-shaped signal, and independent-predictor interaction.
- One generated dataset per scenario, 80 independent participants; generator
  seeds 9301–9305 in the listed order. Two independent standard-normal predictors.
- Noise outcome: independent standard normal.
- Direct: `y = 2*a + error`; error SD .3.
- Redundant: `b = a + noise(SD .03)`; `y = 2*a + error(SD .3)`.
- U shape: `y = 2*a**2 + error(SD .3)`.
- Interaction: `y = 2*a*b + error(SD .3)`.
- Engines evaluated separately: elastic net and GBM; no winner selection or pooling.
- Outer folds 3, seed 8401. Discovery seed 613, B=3 at every target, 80% subsamples,
  local performance 2 folds/1 repeat, tuning 2 folds, maximum depth 2.
- Shipped relevance/dependence/retention defaults, including 70% retention.
- Preserve every planned fold, complete discovery evidence, predictions, selected
  sets, scores and failures. No successful-fold-only aggregate. No confidence intervals.
- Survey demonstration uses the existing saved 180-participant Well-being fixture,
  elastic net, outer folds 3/seed 8401, discovery seed 613, B=10, max depth 3;
  the same two-fold local/tuning configuration and shipped thresholds. Full-data
  reference and root effects (maximum 10 adaptive nominal bins) are descriptive.
- Report pooled RMSE, R², baseline-relative prediction skill, fold completeness and
  retained sets; include prediction/reference/curve graphs for the survey.
- Acceptance: evidence can be independently recounted, held-out identities match
  planned folds, and all planned attempts/failures are saved. Unexpected scientific
  patterns are findings, not reasons to tune this protocol or discard attempts.

Final P6 remains separate: freeze larger sample/predictor/correlation/noise/reliability
grids, seeds and confirmation scenarios; include chains, latent causes, colliders,
outcome consequences, measurement error and incomplete/missing-data cases; define
predictive/classification/depth targets and success criteria. Measure false selections,
missed predictive objects, wrong shared classes and placements, incomplete rates,
runtime and fitting cost. Causal-generating arrows are not automatically the OPEN
predictive ground truth; define the intended predictive object for each scenario.
