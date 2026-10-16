# Tree-engine recursive null qualification

Study `tree-recursive-null-2026-10-15` used source `279c380c637ad7b0420df9d93f1e680480f2cf3e`.
An engine is eligible for a separately approved signal study only if both sample-size cells pass.

|Engine|N|Complete / 55|Unknown|False networks|Exact 95% upper bound|Gate|
|---|---:|---:|---:|---:|---:|---|
|gbm|80|55 / 55|0|0|0.064871|PASS|
|gbm|200|55 / 55|0|0|0.064871|PASS|
|lightgbm|80|55 / 55|0|0|0.064871|PASS|
|lightgbm|200|55 / 55|0|0|0.064871|PASS|
|random_forest|80|0 / 55|55|0|0.064871|STOP|
|random_forest|200|0 / 55|55|0|0.064871|STOP|

## Engine eligibility

- `random_forest`: stopped
- `gbm`: eligible for separate signal review
- `lightgbm`: eligible for separate signal review

Unknown cases remain in the denominator and prevent a cell from passing.
