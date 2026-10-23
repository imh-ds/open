# Engines

Decision date: 2026-10-19 (owner decision).

## Supported engines

| Engine | Exploratory mode | Confirmatory p-values (Phase B) |
|---|---|---|
| OLS | Validated, N 300–1,000 | Calibrated from N ≥ 275 (OLS + HC4, Phase A′) |
| Elastic-Net | Validated, N 300–1,000 | Not tested |
| LightGBM | Validated; finds real effects reliably from about N = 1,000 and is the only validated engine that detects U-shaped effects | Calibrated from N ≥ 300 with the K = 10 cluster correction (G2) |

`open.explore` accepts only these engines (`SUPPORTED_ENGINES`).

Elastic-Net tuning fits median imputation and scaling on each inner training
fold once, then reuses those transformed training and validation inputs for all
25 parameter combinations. It retains the same fold scores, grid order and
first-candidate tie rule. Preprocessing is never shared across folds or tuning
calls, and outer model fits still create fresh pipelines.

## Dropped engines

Random forest, scikit-learn gradient boosting, PLS regression and support
vector regression were evaluated and dropped. The evidence trail is in
[dropped-engines.md](dropped-engines.md). PLS and SVR are still in the code
(`engine="pls"`, `engine="svr"`), accepted by `explore` with a warning, but are
not offered.

## Adding engines later

Dropped engines, XGBoost and others are added only after passing the same
validation study (`examples/exploratory_validation.py`, run for that engine
alone). They also need a Phase B–style calibration check if they are to be
offered for confirmatory p-values.
