# Threshold sensitivity (interpretation)

Re-analysis of the exploratory validation study (Actions run 39184696821,
release `exploratory-validation-2026-10-19-39184696821`, raw zip SHA-256
`fc036001…d55b`). Nothing was re-run. The script is
`examples/threshold_sensitivity.py`.

## What was done

Each depth-1 link's fold pass rate and median R² are stored in the raw records.
Its tier was recomputed under 12 settings:

- keep ∈ {70%, 80%, 90%};
- flag ∈ {40%, 50%};
- hide floor (.001) on or off.

**Check.** The default setting (80%, 40%, floor on) reproduces every recorded
tier, and its true-link rates match `true-link-rates.csv` exactly.

**Scope.** Only depth-1 links (into Y) are re-scored.

- Changing `keep` changes which variables are promoted. So depth-2 links that a
  looser rule would test were never computed.
- False links below 40% of folds were not recorded. That does not matter here,
  because every flag value is at least 40%.
- The noise bar's quantile (95%) cannot be changed after the fact. A 90% arm is
  planned for study v2.

## Tables

- `summary-by-engine-n.csv`: by engine, N and setting, pooled over the four
  shapes and the five true depth-1 links (A, B, C, the S1/S2 group, C × D).
- `summary-by-engine.csv`: by engine and setting, with each true link's keep
  rate.
- `true-link-rates-by-setting.csv` and `false-links-by-setting.csv`: the full
  results by cell.

The true-keep rates are averages over the five true links, including the weak
C (r² .01), the interaction, and U-shapes, which OLS and Elastic-Net miss. They
are low in absolute terms for that reason. Read them relative to each other.

## Findings

**1. `keep` trades false solid lines against power, and 80% sits in the
middle.** For OLS at N = 1,000 (floor on, flag 40%):

| keep | True links kept | False kept per dataset (of which interactions) |
|---|---|---|
| 70% | .50 | .056 (.040) |
| **80%** | **.36** | **.008 (.006)** |
| 90% | .20 | .000 |

The other engines and sample sizes show the same pattern:

- 70% gives about 3–10 times more false solid lines than 80%: .04–.12 per
  dataset at depth 1, about half or more of them interactions.
- 90% removes almost all false solid lines but cuts the true links kept by
  about 35–80%.
- At 80%, false solid lines stay at or below .036 per dataset in every engine
  and N.

**2. `flag` mainly controls dotted noise.** Raising it from 40% to 50% roughly
halves the false dotted interactions, from about 0.6–1.1 to 0.3–0.6 per
dataset. It cuts false dotted direct links by about 15–65%. The share of true
links that are at least flagged falls a little: by .01–.10, and most for
LightGBM at N 300–500.

Flagged interactions are already left out of the graph by default (commit
696de14), so the choice matters mostly for dotted direct links.

**3. The hide floor changes little.** It removes up to about .08 false dotted
links of each kind (direct, interaction) per dataset, and has almost no effect
on true links. Its role is to suppress negligible effects in large samples,
which N ≤ 1,000 does not test.

## Conclusion

The defaults (80%, 40%, floor on) are a reasonable middle point. The tables
let users pick other cut-offs knowingly:

- **90%** for fewer false solid lines at the cost of power;
- **flag 50%** for fewer dotted lines.

No change to the defaults is proposed.
