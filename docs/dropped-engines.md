# Engines considered and dropped

Decision dates: 2026-10-19 (random forest, gradient boosting), 2026-10-22 (PLS,
SVR). Owner decisions. The supported engines are OLS, Elastic-Net and LightGBM
([engines.md](engines.md)). Code for the four engines below stays in the
repository or its archive ([provenance.md](provenance.md)); none is offered.

| Engine | Dropped because | Evidence |
|---|---|---|
| Random forest | Redundant with LightGBM; confirmatory test calibrated only at N = 5,000 | Phase B, run 39144406136 |
| scikit-learn gradient boosting | Redundant with LightGBM; confirmatory test failed at N = 5,000 | Phase B, run 39144406136 |
| PLS regression | Behaved like OLS where tested; the case where it should differ was not tested | Pilot, run 39754861983 |
| Support vector regression (RBF) | Not better than an existing engine in any cell of the validation study | Pilot, run 39754861983; study, run 39822336635 |

## Random forest and scikit-learn gradient boosting

- **Redundant with LightGBM.** All three are tree ensembles and capture the
  same effects: curves, thresholds and interactions. LightGBM is itself a
  gradient-boosting implementation, and it is faster.
- **Not validated for exploratory mode.** Neither was part of the exploratory
  validation study (run 39184696821); both were too slow for it.
- **Confirmatory p-values (Phase B, run 39144406136;
  [write-up](../examples/results/2026-10-18-phase-b-local-detector/interpretation.md)).**
  - *Random forest:* calibrated only at N = 5,000, the next ladder step after
    2,000. At N = 1,000 and 2,000 its false-positive rates were slightly over
    the limit (about 6.3–6.6% against about 6.4%) in cells with curved or
    interaction effects.
  - *Gradient boosting (depth-2 trees):* passed from N = 300 to 2,000, then
    failed at N = 5,000 in the interaction cell, where null predictors were
    rejected 25–31% of the time. It has no supported range. The likely cause
    is that shallow trees under-fit a strong interaction.
- **Bad defaults.** An engine that needs 5,000 or more rows, or fails at large
  N, would mislead a typical user with N = 300–1,000.

## PLS regression

Added as an experimental engine on 2026-10-22, then dropped the same day.

- **Pilot** ([write-up](../examples/results/2026-10-22-pilot-pls-svr/interpretation.md);
  run 39754861983): 10 simulated scenarios, 3 datasets each, up to 40
  candidates. PLS gave the same links as OLS in 7 of 10 scenarios, often
  identical to three decimals, because tuning chooses enough components to
  reproduce OLS. It differed only with many weak effects (4 of 12 true links
  kept against 0 of 12 for OLS), where Elastic-Net did better (8 of 12). It
  missed U-shapes and interactions like every linear engine.
- **What was not tested.** PLS is expected to differ from OLS when predictors
  are as many as cases or more, or heavily redundant. The pilot never went past
  40 candidates at N = 300. The drop rests on the lack of any observed benefit
  and on Elastic-Net already covering wide data, not on a failure at p ≥ n.
- **Also for the drop.** OPEN judges a link by what the model loses when its
  column is left out. PLS spreads weight over correlated columns through shared
  components, which tends to shrink each column's unique loss.

## Support vector regression (RBF kernel)

Added as an experimental engine on 2026-10-22, then dropped the same day.

- **Pilot** (run 39754861983): promising. It kept the U-shaped predictor in all
  3 datasets at N = 500 with the highest network R² (.32 against .24 for
  LightGBM) and kept no noise links. The effect there was about r² .24 (the study's
  is .09), and there were only 3 datasets per cell.
- **Validation study** (run 39822336635, timing probe 39820966028; tables in
  [examples/results/2026-10-22-svr-validation](../examples/results/2026-10-22-svr-validation/)).
  The design is the 2026-10-19 study's (10 candidates, true effects of r² .01
  to .09, shapes linear, saturating, threshold and U, N = 300, 500, 1,000, 200
  datasets per cell, defaults, depth 2), with SVR alone. The SVR cells used the
  same datasets as the earlier engines, so the engines compare directly.

| Result | SVR | OLS | Elastic-Net | LightGBM |
|---|---|---|---|---|
| False links kept per dataset | 0.012 | 0.018 | 0.162 | 0.014 |
| False links flagged per dataset | 1.4 | 2.3 | 1.6 | 1.2 |
| Mean seconds per dataset (runner) | 106 | 55 | 61 | 83 |
| A → Y kept, linear, N = 300 | 48% | 36% | 71% | 14% |
| A → Y kept, linear, N = 1,000 | 96% | 98% | not listed here | 78% |
| A → Y kept, U-shape, N = 500 | 8% | 0% | not listed here | 24% |
| A → Y kept, U-shape, N = 1,000 | 57% | 1% | not listed here | 81% |
| A → Y flagged or kept, U-shape, N = 500 | 55% | 17% | not listed here | 95% |

The Elastic-Net figures for A → Y at N = 300 come from the 2026-10-19 study
([interpretation](../examples/results/2026-10-19-exploratory-validation/interpretation.md));
the other Elastic-Net rows are per-dataset averages over all cells.

- **False links:** SVR is as safe as OLS and LightGBM. That is the minimum bar.
- **Linear, saturating and threshold effects:** about the same as OLS;
  Elastic-Net keeps more at small N.
- **U-shaped effects:** far better than the linear engines, but behind
  LightGBM at both N = 500 and N = 1,000.
- **Small effects and interactions:** no better than OLS. The substitute pair
  was flagged or kept less often than with OLS at N ≤ 500 (31–90% against
  52–96%).
- **Pilot versus study:** the pilot's advantage did not hold at the study's
  effect size.
- **Cost:** about 106 s per dataset on average, up to 240–270 s at N = 1,000
  for linear and saturating shapes.
- **Known failure from the pilot:** 4 weak effects among 40 candidates at
  N = 300; it kept 1 of 12 true links where Elastic-Net kept 8 of 12.
- **Confirmation:** SVR has no calibrated test of its own.

The one niche left is curved effects at N ≈ 500–1,000 without LightGBM. The
study shows SVR beats the linear engines there, not that it beats LightGBM.

## Re-adding an engine

An engine is added again only after it passes the same exploratory validation
study (`examples/exploratory_validation.py`, run for that engine alone), and
after a Phase B-style calibration check if it is to be offered for confirmatory
p-values. For PLS, a test at p ≥ n would also be needed.
