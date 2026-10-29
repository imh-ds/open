# Validation evidence

Seven simulation studies support the method. Each has a frozen script, a
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

## 6. Study v2-R: repeated splits

`examples/results/2026-10-20-study-v2r/interpretation.md`

**Design:**

- Study v2's datasets 0–49 in 12 pool-25 cells (3 engines × N × shape).
- Each dataset explored with `repeats` 1 and 5, under two seeds each.

**Findings:**

- **R = 5 does not meet the pre-set stability target.** True links kept again
  under another seed rise only from .43–.74 to .60–.84 at depth 1.
- **It makes "kept" stricter.** True links near the 80% line move to flagged,
  so OLS at 500 exploration rows keeps .21 → .06 of true links.
- **It cleans the dotted tier.** False dotted links per network fall by half
  or more, and false solid links fall to 0 for OLS and LightGBM.
- **"Kept or flagged" is already stable at R = 1:** 86–97% of true links agree
  across seeds. The seed mainly moves links across the kept/flagged line.
- **Default kept at `repeats = 1`;** R > 1 is offered as an option.
- **Reproducibility:** the R = 1 runs reproduce study v2's runs of the same
  datasets, which used older code on other machines. OLS and LightGBM match
  exactly. Elastic-Net matches except for R² = 0 links, where its zeroed
  noise bar lets floating-point noise decide a pass; this changed one flagged
  interaction among about 115,000 links.

## 7. Study ablation: noise bar and recursion

`examples/results/2026-10-28-study-ablation/interpretation.md`

**Design:**

- Study v2's eight core cells (N 1,000 or 2,000, pool 13 or 25, linear or
  U-shaped), datasets 0–99, three engines.
- Discovery on the first half of each dataset, `max_depth` 3, four arms: the
  shipped rules (`full`); the noise bar replaced by zero (`nobar`, a link
  passes a fold when its held-out gain is above zero); and the depth-1 slices
  of each (`norec`, `screen`).
- 5,040 runs, every depth-1 slice checked against a `max_depth` 1 rerun
  (0 mismatches).

**Findings** (means per network over the eight cells):

| Engine | Arm | False solid links | True links kept |
|---|---|---:|---:|
| OLS | full | 0.04 | 2.00 |
| | no noise bar | 10.9 | 3.49 |
| | no recursion | 0.01 | 0.92 |
| Elastic-Net | full | 0.30 | 3.73 |
| | no noise bar | 3.5 | 4.14 |
| | no recursion | 0.08 | 1.62 |
| LightGBM | full | 0.02 | 1.25 |
| | no noise bar | 19.4 | 4.59 |
| | no recursion | 0.01 | 0.60 |

- **The noise bar is what keeps recursion safe.** Removing it at least doubles
  false solid links in all 24 engine-by-cell combinations, and the false links
  pile up at depth 2–3.
- **Recursion adds true links.** The full rules keep more true links than the
  no-recursion arm for every engine (+1.1 OLS, +2.1 Elastic-Net, +0.7
  LightGBM per network). Depth-2–3 true links are kept in 17 of the 24
  engine-by-cell combinations: every linear and LightGBM cell. OLS and
  Elastic-Net keep almost none in U-shaped cells at any depth.
- **Elastic-Net pays a small price.** With the bar on, recursion adds about
  0.2 false solid links per network (paired interval above zero in 7 of 8
  cells); OLS and LightGBM add none that is distinguishable from zero.
- **Screening alone is not enough.** With neither component, 1–4 false solid
  links per network are kept.

The "no noise bar" arm is one rule (gain above zero in 80% of folds). No other
variants were run.

## How the evidence maps to the software

| Software claim | Evidence |
|---|---|
| Offered engines: OLS, Elastic-Net, LightGBM | Studies 1 and 4; [engines.md](engines.md) |
| Default thresholds (80%, 40%, 95% bar) | Study 1 and its threshold sensitivity re-analysis |
| `confirm` with OLS from N ≥ 275 | Studies 3 and 5 |
| `confirm` with LightGBM from N ≥ 300 | Studies 4 and 5 |
| Split-sample confirmation of whole networks | Study 5 |
| Comparison with simpler selection rules | Study 5 |
| `repeats` option and the default of 1 | Study 6 |
| Noise bar and recursion each contribute | Study 7 |
