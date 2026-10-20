# Validation evidence

Five simulation studies support the method. Each has a frozen script, a
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

**Threshold sensitivity** (re-analysis of the same records, depth 1;
`examples/results/2026-10-20-threshold-sensitivity/interpretation.md`):

- Keeping at 70% gives about 3–10 times more false solid lines than 80%.
  Keeping at 90% removes almost all of them, but cuts the true links kept by
  about 35–80%.
- Flagging at 50% instead of 40% roughly halves the false dotted interactions.
- The hide floor changes little at N ≤ 1,000.
- The defaults sit between these extremes, and no change is proposed.

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

## 5. Study v2: split-and-confirm, larger and correlated pools, comparators

`examples/results/2026-10-20-study-v2/interpretation.md`

**Design:**

- 33 cells × 100 datasets.
- Depth-3 truth, with pools of 13 or 25.
- Correlated proxies and null blocks; independent, skewed and Likert variants;
  linear and U-shaped effects.
- Each dataset is split 50/50: explored on one half and confirmed on the other.
- Compared with naive p < .05 recursion and lasso stability selection.
- False links are defined from the generating graph (d-separation given the
  pool).

**Findings:**

- **Solid lines stay reliable:**
  - false solid lines per network: 0.02–0.11 for OLS, 0.00–0.11 for LightGBM,
    and 0.10–0.62 for Elastic-Net;
  - correlated proxies are kept about as rarely as independent nulls.
- **Confirmation controls false confirmations where the engine fits the data:**
  - 0–2% of networks with kept links only, and 0–5% with flagged links too, in
    linear cells;
  - OLS + HC4 null p-values are uniform.
- **Linear engines on U-shaped data** can confirm links that only reflect
  unmodelled curvature: Elastic-Net networks, 8–36% of networks.
- **Dotted lines are mostly noise:** up to 12 false dotted interactions per
  network at pool 25.
- **Naive recursion** keeps 2–10 false links per network. Stability selection
  has more power than OPEN-OLS, but more false links and none of C → Y.
- **Networks vary noticeably with the seed:** 50–79% of kept true links are
  kept again under a second seed.

## How the evidence maps to the software

| Software claim | Evidence |
|---|---|
| Offered engines: OLS, Elastic-Net, LightGBM | Studies 1 and 4; [engines.md](engines.md) |
| Default thresholds (80%, 40%, 95% bar) | Study 1 and its threshold sensitivity re-analysis |
| `confirm` with OLS from N ≥ 275 | Studies 3 and 5 |
| `confirm` with LightGBM from N ≥ 300 | Studies 4 and 5 |
| Split-sample confirmation of whole networks | Study 5 |
| Comparison with simpler selection rules | Study 5 |
