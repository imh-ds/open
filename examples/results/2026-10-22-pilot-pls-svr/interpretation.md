# Pilot: PLS and SVR engines (2026-10-22)

A pilot, not a validation study. Ten simulated scenarios, three datasets each,
explored with all five engines at depth 2 without interactions (the
interaction step does not depend on the engine). Three datasets per cell show
tendencies, not rates.

- **Script:** `examples/pilot_pls_svr.py` (scenario definitions, scoring).
- **Run:** GitHub Actions run 39754861983, workflow `pilot-pls-svr.yml`,
  pinned packages (numpy 1.26.4, scipy 1.11.2, pandas 2.2.3,
  scikit-learn 1.3.0, lightgbm 4.6.0), Python 3.11.9.
- **Files:** `summary.csv` (one row per scenario and engine), `runs.csv` (one
  row per dataset, with the kept links).
- **Scoring:** a true link counts as found when a kept (or flagged) link into
  the right target includes the true predictor. Noise links are links from the
  pure-noise columns N1, N2, … The collinear block counts once, by any member.

## Results

True links kept (kept or flagged), over the three datasets:

| Scenario | OLS | Elastic-Net | LightGBM | PLS | SVR |
|---|---|---|---|---|---|
| linear (with a depth-2 chain) | 12/12 (12) | 12/12 (12) | 7/12 (11) | 12/12 (12) | 10/12 (12) |
| collinear block | 6/6 (6) | 6/6 (6) | 5/6 (6) | 6/6 (6) | 6/6 (6) |
| small N, wide (N 150, 30 candidates) | 4/9 (8) | 7/9 (9) | 0/9 (6) | 3/9 (8) | 4/9 (8) |
| U-shape | 3/6 (3) | 3/6 (3) | 5/6 (6) | 3/6 (3) | **6/6 (6)** |
| pure interaction | 1/9 (3) | 2/9 (3) | 1/9 (8) | 2/9 (3) | 2/9 (**9**) |
| step | 4/6 (6) | 5/6 (6) | 3/6 (6) | 3/6 (6) | 3/6 (6) |
| dilution (4 weak among 40, N 300) | 0/12 (9) | 8/12 (11) | 0/12 (5) | 4/12 (10) | 1/12 (10) |
| heavy tails | 8/9 (9) | 9/9 (9) | 6/9 (9) | 8/9 (9) | 8/9 (9) |
| suppression (Y = X1 − X2) | 7/9 (9) | 8/9 (9) | 2/9 (5) | 7/9 (9) | 7/9 (9) |

- **Noise links kept:** none for PLS or SVR in any of the 30 datasets each
  (Elastic-Net 4, the others 0).
- **Null scenario:** nothing kept by any engine. Noise links flagged:
  SVR 0, PLS 4, OLS 2.
- **Network R² of Y in the U-shape:** SVR .32, LightGBM .24, linear engines
  about .12 (the B effect only).
- **Mean seconds per dataset on the runner:** PLS 2–14, SVR 18–98,
  OLS 3–21, Elastic-Net 6–63, LightGBM 28–173.

## Findings

### PLS: works, but it behaves like OLS

- It gives the same links as OLS in 7 of 10 scenarios, often identical to
  three decimals. Tuning picks enough components to reproduce OLS.
- It differs only where there are many candidates and weak effects
  (dilution: 4/12 kept against 0/12 for OLS). Elastic-Net is clearly better
  there (8/12).
- No new failure: suppression, where PLS needs a second component, was
  handled as well as by OLS. The collinear block was found every time.
- As expected for a linear engine, it misses the U-shape and the pure
  interaction.

### SVR: works, and fills the gap it was meant to

- **Curved effects at N = 500:** it found the U-shaped predictor in all three
  datasets, and its network explains the most (R² .32). LightGBM found it in
  all three but kept it in two of them. The linear engines missed it, as
  designed.
- **Pure interaction:** it saw A and B in all three datasets (kept or
  flagged), but kept them in only one. The effect is weak at this N.
- **Linear, collinear and heavy-tailed data:** close to OLS (linear: 10/12
  kept, every link at least flagged).
- **Clean on noise:** no noise link kept anywhere, and nothing kept or flagged
  in the null scenario.
- **Expected failure confirmed:** with 4 weak effects among 40 candidates at
  N = 300, it kept 1 of 12. It saw most of them (10/12 flagged or kept), but
  the kernel spreads over many columns and their unique losses stay under the
  80% line. Elastic-Net is the engine for that setting.
- **Runtime:** about 4–7 times OLS, and below LightGBM.

### Not about the new engines

In the pure-interaction scenario, the plain linear effect of C (.3) was often
missed by every engine, OLS included. A likely cause is that the heavy-tailed
A × B term makes fold-level R² noisy and raises the noise bar. It is not
investigated here.

## What this does not cover

- Only three datasets per cell, N up to 600 and depth 2.
- Confirmation was not run. An SVR network is confirmed with OLS + HC4 by
  default, which cannot confirm a U-shaped link; LightGBM's calibrated test
  (`confirm(..., engine="lightgbm")`) is the appropriate choice for curved
  links.
- Both engines stay experimental until they pass the exploratory validation
  study.
