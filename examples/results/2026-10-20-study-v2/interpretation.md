# Study v2 (interpretation)

Actions run [39397875389](https://github.com/imh-ds/open/actions/runs/39397875389)
(attempt 2), workflow `.github/workflows/study-v2.yml`, script
`examples/study_v2.py`. Timing probe: run 39377209685. Raw shards: release
`study-v2-2026-10-20-39397875389`, asset
`study-v2-2026-10-20-39397875389-raw.zip` (SHA-256
`940f9313d3d78d9a669f23978af93cffbb15f3d0d21afdf9b1c25b59de81d336`).

It answers critique items 1–6 and 12. The plan is in
`outline/2026-10-20-anticipated-critiques-plan.md`, and the design in
`outline/2026-10-20-study-v2-design.md` (both private).

## Design

- **Cells:** 33 cells × 100 datasets, 3,300 in total, shared across engines.
  - **Core:** engine (OLS, Elastic-Net, LightGBM) × total N (1,000, 2,000) ×
    pool (13, 25) × shape (linear, U), with correlated normal predictors.
  - **Variants:** independent, skewed or Likert predictors, at N 2,000, pool 25,
    linear.
- **Truth:** a three-depth network, plus proxy nulls (P_A, P_E) and correlated
  null blocks (N1–N12):

  | Depth | Links (r²) |
  |---|---|
  | 1 | Y ← A (.09), B (.04), C (.01); S1/S2 shared (.04); C × D (.04) |
  | 2 | A ← E (.09); B ← F (.01); correlated condition only: A, B ← K (.30 each) |
  | 3 | E ← G (.09), H (.04) |

  The full generator is in the script's docstring.
- **Each dataset:**
  - split 50/50, so exploration uses 500 or 1,000 rows;
  - explored with defaults and `max_depth=3`;
  - confirmed on the other half with Holm, both for kept links only and for
    kept plus flagged links;
  - in core cells, also run through the comparators and, for datasets 0–49, a
    second seed.
- **Scoring:** a link X → T explored with pool P is *false* when X is
  d-separated from T given the rest of P in the generating graph, so X carries
  no information about T beyond the pool. Genuine but undesigned links (for
  example P_A → A) are "other", neither true nor false.
- **Runtime:** 4,991 runner-minutes, against a probe estimate of 6,437 and a 3×
  ceiling of 19,770.

### Integrity

- All 165 shards are present, with 3,300 unique datasets (100 per cell, reps
  0–99).
- One shard's results upload failed on GitHub's side (an artifact service
  timeout); the computation itself had finished. Only that shard was re-run.
  It is seeded, so the result is identical.
- **Two aggregation bugs were found and fixed before the tables here were
  made:**
  - **Double counting:** the re-run's aggregate job also downloaded the first
    attempt's results artifact, which counted 164 shards twice. The workflow
    now downloads shard artifacts only.
  - **Missing class:** the source class "null" was read as a missing value and
    dropped. The reader now treats only empty fields as missing.

  All tables here were re-aggregated from the 165 shard artifacts with the
  fixed script, then summarized with `examples/study_v2_report.py`.

## Findings

### 1. Solid (kept) lines stay reliable in larger, correlated, deeper networks (items 2 and 4)

False solid lines per network, summed over depths 1–3 and every kind
(`report-false-per-network.csv`):

| Engine | Range across the 11 conditions |
|---|---|
| OLS | 0.02–0.11 |
| LightGBM | 0.00–0.11 |
| Elastic-Net | 0.10–0.62 |

**Per tested false link** (`report-false-by-source-class.csv`):

- LightGBM kept 1 of 44,441 (it was an independent null).
- OLS kept 0.00–0.09% of them.
- Elastic-Net kept 0.24–0.56%.

**Elastic-Net's largest single source of false lines is A → B and B → A at
depth 2** (77 of its 162 false structural links kept). A and B share the parent
K, which is in the pool. Given K, neither adds anything, but Elastic-Net's
shrinkage under-adjusts for K. The rest are spread over independent nulls (143)
and proxies (41).

### 2. Dotted (flagged) lines are noisy, especially interactions in larger pools

False dotted links per network:

| Engine | Interactions (pool 13) | Interactions (pool 25) | Direct links |
|---|---|---|---|
| OLS | 0.7–3.3 | 1.5–12.2 | 0.6–7.1 |
| Elastic-Net | 0.3–2.7 | 1.3–10.4 | 0.1–1.6 |
| LightGBM | 0.6–2.0 | 1.7–6.8 | 0.1–1.0 |

This supports the decision to leave flagged interactions out of the graph by
default (commit 696de14). For OLS, about 5–6% of tested false direct links are
flagged, a little more than the 5% noise rate per fold.

### 3. Correlated proxies are not a problem for solid lines (item 3)

Proxies are noisy copies of a true predictor (r = .5). They were kept about as
rarely as independent nulls:

| Engine | Proxies kept | Independent nulls kept |
|---|---|---|
| OLS | 0.02% | 0.00% |
| LightGBM | 0% | 0% |
| Elastic-Net | 0.56% | 0.24–0.35% |

Proxies are flagged somewhat more often: OLS 6.4% vs 5.4%, and Elastic-Net
1.6% vs about 0.6%. So shuffled noise columns are a fair yardstick for solid
lines. No change to the noise construction is needed.

### 4. Confirmation holds its error rate where the engine fits the data (item 1)

**False confirmations per network**, at Holm .05
(`report-familywise-by-shape.csv`):

| Cells | Kept links only | Kept + flagged links |
|---|---|---|
| Linear, all engines | 0–2% | 0–5% |
| U-shape, LightGBM | 0% | ≤ 4% |
| U-shape, OLS | ≤ 5% | ≤ 9% |
| U-shape, Elastic-Net | **8–36%** | 12–39% |

**Null p-values** (`report-null-calibration.csv`):

- **OLS + HC4:** 4.65% below .05, 0.85% below .01, and uniform (KS p = .90),
  over 3,291 null links in linear cells. Elastic-Net networks, also confirmed
  with OLS, give 4.1%.
- **LightGBM:** 5.0% below .05 and 1.2% below .01 in linear cells. The
  distribution is non-uniform only because 25% of p-values are exactly 1, the
  combined test's Bonferroni cap, which makes it conservative.

**Why the Elastic-Net U-shape cells fail.** Nearly all of these false
confirmations are A ↔ B (and P_A → B) at depth 2.

- In U-shape cells, A and B depend on K², not K. Given K they are independent,
  so they are false by the graph.
- But a linear model can only adjust for K itself. B, which carries K², really
  does improve a *linear* prediction of A. So the OLS test is right about the
  question it asks: 98–100% of these links have p < .05.
- The result is a mismatch between a linear engine and curved data, not a
  miscalibrated test. LightGBM can model K² and seldom produces them: 4 of 12
  links tested.

**Lesson:** with a linear engine, a confirmed link means "predicts beyond the
pool *in a linear model*". Unmodelled curvature can create such links.

**One small LightGBM excess.** In U-shape cells, other null links fall below
.05 11.6% of the time (11 of 95, binomial p = .008). All of them were flagged,
not kept. With kept links only, no LightGBM network had a false confirmation.

### 5. Power: exploration is the bottleneck, and confirmation keeps nearly everything it is given

A true link that is kept is almost always confirmed: the kept-and-confirmed
rates (`report-kept-and-confirmed.csv`) are within .05 of the keep rates, and
usually equal to them.

Keep rates for the strongest link, A → Y (r² .09), in correlated linear cells:

| Engine | 500 exploration rows | 1,000 exploration rows |
|---|---|---|
| OLS | .33–.45 | .74–.90 |
| Elastic-Net | .77–.85 | .96–1.00 |
| LightGBM | .06–.14 | .27–.62 |

- **Correlation costs power.** With independent predictors, OLS keeps A → Y in
  100% of datasets, because the proxies and K absorb part of A's unique
  contribution.
- **The weak links** (C → Y and F → B, r² .01) are rarely kept.
- **U-shapes:**
  - OLS and Elastic-Net keep almost nothing.
  - LightGBM keeps A → Y in .42–.45 of datasets at 1,000 rows and flags it in
    nearly all the rest.
- **Skewed predictors** cut LightGBM's power markedly (A → Y .27 vs .46) and
  OLS's a little (.74 vs .87). Likert items did not hurt.
- **LightGBM flags rather than keeps:** kept-or-flagged rates are .72–1.00 for
  A → Y.

### 6. Comparison with simpler selection rules (item 12)

These are direct links only, scored the same way and run inside the same
recursion (`report-comparators.csv`):

| Method | False solid links per network | Designed links kept (mean, linear) |
|---|---|---|
| Naive recursion (HC4 p < .05) | 1.9–9.7 | .80–.89 |
| Lasso stability selection | 0.01–0.76 | .23–.76 |
| OPEN, OLS | 0.00–0.06 | .17–.55 |
| OPEN, Elastic-Net | 0.08–0.60 | .55–.85 |
| OPEN, LightGBM | 0.00 | .02–.22 |

- **Naive recursion is unusable.** It keeps 2–10 false links per network, more
  in larger pools.
- **Stability selection** has more power than OPEN-OLS in linear pools of 25.
  But it never keeps C → Y and seldom keeps the S1/S2 pair (.17). It also
  keeps 0.3–0.8 false links per network on U-shaped data.
- **OPEN-OLS trades power for very few false links.**
- **OPEN-Elastic-Net** gets about the power of stability selection or more,
  with a similar or somewhat higher false-link count.
- **OPEN-LightGBM** is the only method with no false solid links and some
  power on U-shapes.

### 7. The 90% noise bar (item 6)

Depth-1 links, per tested link (`noise-bar-90.csv`):

- **True links kept** rise a little: OLS .27 → .31, LightGBM .16 → .21.
- **False links kept** stay negligible: OLS .0001 → .0009.
- **False links flagged** roughly double: OLS 7% → 16%.

The 95% default is a reasonable choice. A 90% bar is an option for users who
want more power at depth 1.

### 8. Results depend noticeably on the seed (item 5)

**Kept-link replication.** Of the true depth-1 links kept with one seed, the
share also kept with a second seed (`report-seed-replication.csv`) was:

- Elastic-Net 79%;
- OLS 64%;
- LightGBM 50%.

**False kept links almost never replicate:** 0% for OLS and LightGBM, 17% for
Elastic-Net.

**Even maximal pass rates move.** A true link that passed 10 of 10 folds is
only flagged under the other seed 18% of the time (`report-seed-by-pass-rate.csv`).
The seed changes both the folds and the noise columns.

**Whole networks** (Jaccard overlap of kept links): 0.27–0.72 in linear cells.
U-shape cells look more stable only because both networks are often empty.

The design's rule was to consider repeated splits if agreement turned out low
near the thresholds. It did: repeating the 10-fold split R times and pooling
the fold passes would reduce this variation, and false links, which rarely
replicate, would benefit most. It would multiply exploration time by about R.
Whether to add it is an owner decision.

## What this means for the method

- **The core claims hold:**
  - solid lines carry very few false links in larger, correlated, skewed and
    Likert designs at depth 3;
  - correlated proxies do not break the noise yardstick;
  - split-sample confirmation controls false confirmations where the engine
    fits the data;
  - OPEN is far better than naive recursion.
- **Limits to state in the docs and the paper:**
  - Dotted lines, above all dotted interactions in large pools, are mostly
    noise.
  - Linear engines on curved data can find and confirm links that only reflect
    unmodelled curvature.
  - Elastic-Net under-adjusts for shared parents.
  - LightGBM needs about 1,000 exploration rows to keep links, and loses power
    with skewed predictors.
  - OPEN-OLS is conservative compared with stability selection.
  - Networks change noticeably with the seed.
- **Open decisions:**
  - repeated 10-fold splits (item 5);
  - how strongly to steer users between OLS (fewest false links), Elastic-Net
    (more power, more false links) and LightGBM (curved effects, larger N).
