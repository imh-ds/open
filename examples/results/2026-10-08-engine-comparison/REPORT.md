# Matched engine comparison: synthetic demonstrations

New fixed-target nested-CV comparisons; these are not the earlier 20-resample network analysis. No engine is chosen as a winner and no consensus network is pooled across engines.

Survey: the same 180 saved rows and all five Well-being candidates. Shared three outer folds, two inner folds, seed 613. ALE uses four quantile bins (including both contrast endpoints), held-out reference rows, and common reference outcome SD. Nonlinear check: 120 deterministic rows, Y = X², seed 13, same CV dimensions.

## Prediction performance

| Engine | Survey CV R² | Nonlinear CV R² |
| --- | ---: | ---: |
| elastic_net | 0.5282 | -0.0023 |
| random_forest | 0.4600 | 0.9915 |
| gbm | 0.4268 | 0.9941 |
| lightgbm | 0.4172 | 0.9702 |

![Performance](prediction_performance.png)

## Comparable effects on Well-being

Signed standardized ALE contrast: change along the estimated prediction function from predictor Q25 to Q75, expressed in Well-being SDs. ΔCV R² is the unique loss in predictive R² after removing that predictor; it is a different quantity. Values reflect fixed candidate sets, not causal effects or coefficients.

| Predictor | Engine | Q25→Q75 contrast (Y SD) | Unique ΔCV R² | Status |
| --- | --- | ---: | ---: | --- |
| Sleep Quality | elastic_net | +0.2868 | +0.0357 | available |
| Sleep Quality | random_forest | +0.3435 | +0.0353 | available |
| Sleep Quality | gbm | +0.6134 | +0.0510 | available |
| Sleep Quality | lightgbm | +0.4081 | +0.0497 | available |
| Perceived Stress | elastic_net | -0.3415 | +0.0267 | available |
| Perceived Stress | random_forest | -0.4161 | +0.0650 | available |
| Perceived Stress | gbm | -0.3188 | +0.0405 | available |
| Perceived Stress | lightgbm | -0.3767 | +0.0345 | available |
| Social Support | elastic_net | +0.4885 | +0.1027 | available |
| Social Support | random_forest | +0.4745 | +0.1350 | available |
| Social Support | gbm | +0.5242 | +0.1241 | available |
| Social Support | lightgbm | +0.6114 | +0.1560 | available |
| Physical Activity | elastic_net | +0.2446 | +0.0319 | available |
| Physical Activity | random_forest | +0.2398 | +0.0202 | available |
| Physical Activity | gbm | +0.2602 | +0.0327 | available |
| Physical Activity | lightgbm | +0.4189 | +0.0316 | available |
| Work Demands | elastic_net | -0.0552 | +0.0009 | available |
| Work Demands | random_forest | -0.0257 | +0.0022 | available |
| Work Demands | gbm | -0.1252 | +0.0034 | available |
| Work Demands | lightgbm | -0.1370 | -0.0101 | available |

![ALE curves](ale_curves.png)

## Joint removal diagnostic

Social Support and Sleep Quality are removed together on the same held-out observations. This is the total predictive contribution of the pair, including unique and overlapping information; it does not establish that this pair is redundant or promote a shared group into the network.

| Engine | Joint ΔCV R² |
| --- | ---: |
| elastic_net | +0.1316 |
| random_forest | +0.1763 |
| gbm | +0.2034 |
| lightgbm | +0.1996 |

## Interpretation and limits

In the nonlinear fixture, elastic net can only fit a straight line; the tree engines can learn the U-shaped relationship. The Q25→Q75 contrast can be near zero despite that strong curved relationship. Inspect the curve together with predictive R².

Every engine shares the data, folds and effect-size reference. Elastic net searches 25 tuning candidates; each tree engine searches four. This is not an equal-budget benchmark, an engine-selection test, or a scientific calibration. No confidence intervals are inferred from folds. Different recursive networks require separately labeled candidate sets.

[Survey JSON](survey/comparison.json), [survey effect table](survey/effects.csv), [nonlinear JSON](nonlinear/comparison.json), [nonlinear effect table](nonlinear/effects.csv), [audit](audit.json).

[Method and usage](../../../docs/engines-and-effect-sizes.md).

## Software verification

Python 3.11 full suite: **467 passed**, 29 existing warnings (25 constant-outcome optimizer and 4 missing-CJK-font warnings). Python 3.14 focused: **43 passed**, one optional-LightGBM skip and four deselected recursive checks. The real four-engine demonstrations captured zero warnings. Fresh independent reviews of both core code and final artifacts found no actionable issues.

[Verification record](verification.json) records offline scoring/aggregation/reference/table/link checks, identical numerical rerun, manual plot inspection and existing-output protection. These are software checks, not scientific validation or calibrated effect-size inference.
