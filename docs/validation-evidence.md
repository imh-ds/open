# Validation evidence

Four simulation studies support the method. Each has a frozen script, a
protocol, summary results in `examples/results/`, and raw outputs in a GitHub
release.

## 1. Exploratory-mode validation (the main evidence)

`examples/results/2026-10-19-exploratory-validation/interpretation.md`

**Design:**

- OLS, Elastic-Net and LightGBM.
- N = 300, 500 and 1,000.
- Linear, saturating, threshold and U-shaped effects.
- 200 datasets per cell: 7,200 in total.
- Each dataset has a known two-depth structure: three direct effects
  (r² .09, .04, .01), a substitute pair, an interaction, two second-depth links
  and pure noise.

**Findings:**

- **Solid lines are reliable.** Kept false links: about 0.015 per dataset for
  OLS and LightGBM, and 0.07 for Elastic-Net.
- **Dotted lines contain noise.** There are 1.1–1.4 flagged false interactions
  per dataset. Read dotted interactions as weak leads.
- **OLS and Elastic-Net** detect linear and monotone effects at N 500–1,000.
  The r² .09 effect is kept 70–100% of the time. Both miss U-shaped effects.
- **LightGBM** detects U-shaped effects (kept 81% at N = 1,000). At N 300–500 it
  mostly flags rather than keeps.
- **Small effects** (r² .01) and interactions are rarely kept at N ≤ 1,000.

## 2. Phase A: linear reference

`examples/results/2026-10-17-phase-a-linear-reference/`

This study compares linear tests (classical t, HC3, cross-fitted GCM with OLS
nuisances) as the reference for the later phases.

## 3. Phase A′: small-sample OLS calibration

`examples/results/2026-10-17-phase-a-prime-small-sample/interpretation.md`

It selects OLS with HC4 standard errors, which give honest p-values from
N ≥ 275. This is the OLS test used by `confirm`.

## 4. Phase B: engine-based p-values

`examples/results/2026-10-18-phase-b-local-detector/interpretation.md`

This is the cross-fitted GCM/PCM test with machine-learning engines. Supported
minimum N:

| Engine | Supported minimum N |
|---|---|
| LightGBM, with K = 10 fold-cluster variance (G2) | 300 |
| Random forest | 5,000 |
| scikit-learn gradient boosting | None |

LightGBM's combined test at N ≥ 300 is the LightGBM test used by `confirm`.

The bit-exact regeneration check failed for random-forest and gradient-boosting
values, consistent with floating-point differences across runner hardware. This
deviation was accepted and is documented in the interpretation; LightGBM and
OLS were unaffected.

## How the evidence maps to the software

| Software claim | Evidence |
|---|---|
| Offered engines: OLS, Elastic-Net, LightGBM | Studies 1 and 4; [engines.md](engines.md) |
| Default thresholds (80%, 40%, 95% bar) | Study 1 |
| `confirm` with OLS from N ≥ 275 | Study 3 |
| `confirm` with LightGBM from N ≥ 300 | Study 4 |
