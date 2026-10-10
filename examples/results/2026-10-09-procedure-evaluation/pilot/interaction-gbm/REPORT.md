# OPEN researcher report: y

Complete-procedure evaluation: **COMPLETE**; 3/3 folds completed.

This evaluates root prediction, not SEM global fit or the correctness of indirect pathways.

Engine: `gbm`. Retention threshold: 70%. Independent participant rows: 80.

## Does the selected predictor set predict new participants?

Each fold discovers its own complete network using training participants only. Retained direct predictors (including joint members) are then separately tuned/refitted. The all-candidate comparator is separately tuned on the same training participants. No held-out participants enter discovery, preprocessing or tuning.

| Predictor | RMSE (outcome units) | R² | Prediction skill |
| --- | ---: | ---: | ---: |
| Training-mean baseline | 2.067 | -0.035 | 0.000 |
| OPEN retained predictors | 2.240 | -0.216 | -0.174 |
| All original candidates | 1.423 | 0.509 | 0.526 |

Lower RMSE means smaller prediction errors. R² compares squared errors with the variation of the pooled held-out outcomes. Prediction skill compares errors with the training-mean baseline: 0 means equal performance; negative means worse; 0.30 means 30% less squared prediction error. It is not a path effect size.

| Fold | Status | Retained direct predictors | Failure |
| --- | --- | --- | --- |
| 0 | completed | none; training mean | — |
| 1 | completed | none; training mean | — |
| 2 | completed | b | — |

Fold predictor sets can differ. These are training-fold discoveries, not one consensus graph assembled from their arrows. Settings must be fixed before this evaluation; selecting settings after inspecting these scores needs new confirmation data.

## What does the descriptive reference network show?

The reference network is discovered separately using all supplied participants. Its relation frequencies are conditional on finalized earlier layers. This is descriptive evidence, not independent confirmation of its structure.

Reference network unavailable.

## How large and what shape are the root predictive associations?

Root effect supplement: **not assessed or unavailable**.

## What still needs checking?

- Threshold sensitivity: **not assessed by this workflow**; reconstruct the full network at other cutoffs.
- Sampling uncertainty and calibrated false-selection rates: **not assessed**.
- Same-depth peer relations and interaction-specific effect curves: **not assessed**.
- Scope: independent cross-sectional scale-level data with a continuous observed root. Grouped, longitudinal and categorical-outcome support is unavailable.
- Scientific validation: development examples do not establish reliable defaults or minimum sample sizes.

Evidence: [evaluation.json](evaluation.json), [predictions.csv](predictions.csv). Training-fold network bundles are saved in `fold_N/`; the full-data reference is in `reference/`.
