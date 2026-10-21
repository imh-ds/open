# Study v2-R: repeated splits (interpretation)

Actions run [39480622940](https://github.com/imh-ds/open/actions/runs/39480622940),
workflow `.github/workflows/study-v2r.yml`, script `examples/study_v2r.py`.
Timing probe: run 39465126146. Raw shards: release
`study-v2r-2026-10-20-39480622940`. The plan is in
`outline/2026-10-20-repeated-splits-plan.md` (private).

## Question

In study v2, a second seed changed the network noticeably: only 50–79% of
kept true links were kept again. Does pooling R repeated 10-fold splits
(`ExploreConfig.repeats`) fix this?

## Design

- **Cells:** study v2's datasets 0–49 in 12 cells: OLS, Elastic-Net and
  LightGBM × total N 1,000 or 2,000 (500 or 1,000 rows explored) × linear or
  U-shaped, at pool 25 with correlated predictors.
- **Runs:** each dataset was explored with R = 1 and R = 5, each under two
  seeds. The seed-0 runs were also confirmed.
- **R = 3** is recomputed at depth 1 from the first three repeats.

**Success criteria, set before the run:**

1. True links kept again under the other seed ≥ 90% at R = 5.
2. False solid links per network not higher than at R = 1.
3. True-link keep rates not lower than at R = 1 by more than .05.

## Integrity

- **Complete:** 2,400 of 2,400 tasks, each exactly once.
- **Repeat 0 = R = 1:** repeat 0 of every R = 5 run reproduces the R = 1 run
  with the same seed. All 30,000 depth-1 tiers checked match
  (`integrity.json`).
- **Re-aggregation:** the tables were re-aggregated locally from the 188 shard
  artifacts, and matched the Actions tables numerically. The committed tables
  are the Actions ones, from the pinned environment.

### Bonus check: reproducibility across code versions and machines

The R = 1 runs here use the same datasets and seeds as study v2, which ran
before `repeats` existed, on other runners. Results are in
`report-reproducibility-vs-v2.csv`.

| Engine | Links compared | Pass rates differ | Tiers differ (drawn tiers) |
|---|---|---|---|
| OLS | 100,595 | 0 | 0 |
| LightGBM | 67,439 | 0 | 0 |
| Elastic-Net | 115,469 | 817 | 13 (1) |

- **R² values** agree to within 3.5 × 10⁻⁹ everywhere.
- **Elastic-Net's differences** come from one edge case. Its penalty shrinks
  weak columns, including all noise columns, to exactly zero. The noise bar is
  then about 0, and floating-point differences between machines (about 10⁻⁹)
  decide whether a column with R² ≈ 0 "passes" a fold.
- **Visible effect:** 12 of the 13 tier changes are between "dropped" and
  "hidden", for links with R² = 0, which the graph never draws. One flagged
  interaction differs, and flagged interactions are hidden by default. A small
  pass tolerance would remove this. It is not applied here, because it would
  change validated results.

## Findings

### 1. The pre-set criteria are not met, so R = 5 should not become the default

From `criteria.csv`:

- **Criterion 1** (true links kept again ≥ 90%) holds in 1 of 12 cells
  (Elastic-Net, N = 2,000, linear).
- **Criterion 2** (no more false solid links) holds in all 12.
- **Criterion 3** (keep rates not lower by more than .05) fails in 3 linear
  cells. For example, OLS at N = 1,000 keeps .21 of true links at R = 1 and
  .06 at R = 5.

### 2. Why: R = 5 makes "kept" stricter, rather than steadier

With 10 folds, a true link that passes about 70% of folds reaches the 80% bar
fairly often by luck: P(≥ 8 of 10) ≈ .38. With 50 folds the pass rate is
measured much more precisely, and such a link almost never reaches it:
P(≥ 40 of 50) ≈ .08. R = 5 turns the 80% rule into a sharp cut-off on each
link's true pass rate.

At these sample sizes, many true links pass 50–80% of folds, so they move from
kept to flagged. At depth 1 (`depth1-by-R.csv`), for OLS:

| | R = 1 | R = 3 | R = 5 |
|---|---|---|---|
| True links kept | .175 | .122 | .113 |
| True links flagged | .468 | .498 | .509 |
| False links kept | .0001 | 0 | 0 |
| False links flagged | .061 | .033 | .028 |

Most true links that leave "kept" move to "flagged" rather than disappearing.
The rest are dropped.

Part of the apparent gain in network stability comes from networks that are
emptier. In the OLS N = 1,000 linear cell, both seeds' networks are empty 66%
of the time at R = 5, against 12% at R = 1 (`stability-jaccard.csv`).

### 3. What R = 5 does improve: the dotted tier and exact tiers

**False links per network** (seed 0; `false-by-run.csv`):

| Engine | False dotted links, R = 1 → R = 5 | False solid links, R = 1 → R = 5 |
|---|---|---|
| OLS | 2.4–18.6 → 0.9–6.9 | 0.02–0.06 → 0 |
| Elastic-Net | 1.3–9.8 → 0.6–5.6 | 0.20–0.72 → 0.06–0.22 |
| LightGBM | 2.0–5.1 → 0.5–1.8 | 0–0.08 → 0 |

**Seed agreement for true depth-1 links** (`report-depth1-agreement.csv`):

| Engine | Kept again, R = 1 → R = 5 | Kept or flagged again | Same exact tier |
|---|---|---|---|
| OLS | .61 → .77 | .94 → .96 | .74 → .88 |
| Elastic-Net | .74 → .84 | .97 → .99 | .76 → .87 |
| LightGBM | .43 → .60 | .86 → .92 | .67 → .85 |

### 4. "Kept or flagged" is already stable at R = 1

A true link that reaches either tier under one seed reaches either tier under
the other 86–97% of the time, even at R = 1. The instability found in study
v2 is mostly in the exact split between kept and flagged for links near the
80% line, not in whether a link appears at all.

### 5. Confirmation is unaffected

False confirmations per network stay low at both R
(`confirmation-familywise-by-run.csv`).

- They fall with R = 5 in the Elastic-Net U-shape cells, from .10–.30 to
  .04–.16. Those are the nonlinear-proxy links found in study v2, and fewer of
  them are kept at R = 5.
- Elsewhere they are .00–.08 at both R.

## Conclusion

- **Keep `repeats = 1` as the default.** R = 5 does not reach the pre-set
  stability target, and it lowers the keep rate of true links, because it
  makes the 80% rule a sharper and stricter cut-off.
- **Offer `repeats` as an option** for users who want fewer false dotted links
  (roughly halved or better) and more reproducible exact tiers, at R times the
  exploration time and with fewer solid links.
- **Docs:** tell users that the kept/flagged split for links near the 80% line
  depends on the seed. "Kept or flagged" is the stable reading, and the pass
  rate itself is the informative number. Confirmation on independent data
  remains the way to decide.
