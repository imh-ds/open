# Corrected no-fit audit of the multi-engine null study

This record corrects audit software only. Original producer run: [38846677048](https://github.com/imh-ds/open/actions/runs/38846677048), source `279c380c637ad7b0420df9d93f1e680480f2cf3e`, manifest `3a31a744ed6c38b7b497c5fb6e010ac715b7e597d3a1666f5c21448875c6dccb`.

## Repairs

1. The RF adapter now verifies the tree-null policy and RF grid, activates the existing singleton-grid verifier, and passes the original manifest unchanged to the historical replay engine. The historical `rf-singleton-v1` entrypoint still rejects other labels. Changing the manifest label itself would break the digest sealed into the original receipts; that shortcut was not used.
2. The null endpoint now counts promoted nodes, edges, and groups directly from the independently reconstructed network. The previous wrapper consulted an unused legacy scenario summary, which was absent for these case specifications. Regression tests demonstrate that a non-empty network can no longer silently receive an empty-network result.

The first regression reproduced the exact RF label error before the adapter repair. Three additional regressions reproduced the missed node, edge and group endpoints before their repair. These fixes were committed separately.

## Verification method

The local Python 3.11.9 installation has exactly the frozen package versions, including NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.2, scikit-learn 1.3.0 and LightGBM 4.6.0. All producer sources match the frozen hashes except the explicitly repaired audit adapter; the original adapter's hash is checked against the original Git commit. Both raw RF release bundles and all 110 inner archive hashes are checked against the published inventory.

Each RF case is independently replayed from its saved observations, predictions, splits and selected parameters. Original receipt run/source/attempt, input data, configuration, checkpoint arithmetic, tuning fit counts, fit-cost completeness, and recursive network are checked. The original manifest, seeds, thresholds, sample sizes and denominator are unchanged. Model fitting and the scientific case runner are explicitly disabled, also in each spawned worker. The 220 prior GBM/LightGBM rows already passed checkpoint replay; their endpoint is recomputed from their verified network fields. The original reports are retained unchanged.

Software verification: 54 focused tests passed in the pinned runtime, covering both fixes, historical RF audit guards, isolated sparse checkouts, original-manifest preservation, forbidden model fitting, missing-case denominators and reclassification of previously miscounted findings.

Replay command:

```powershell
& C:\Users\imhoh\AppData\Local\Programs\Python\Python311\python.exe -u examples/reaudit_tree_recursive_null.py --bundles .superpowers/tree-null-rf-reaudit/bundles --out .superpowers/tree-null-rf-reaudit/replay --workers 8
```

No GitHub Actions model run was repeated. The corrected report retains the producer source identity and separately records the replay source and repaired adapter hashes.

## Results

All 110 RF archives passed full independent verification: 55 at N=80 and 55 at N=200. None contained a false network. Reclassification of the 220 verified GBM/LightGBM networks also found zero false networks. All six cells are 55/55 complete, zero unknown, zero false networks; each exact 95% upper bound remains 6.49%, below the unchanged 10% gate.

All three engines are now eligible for a separately approved signal-recovery study. This supports avoiding invented links in the tested independent Gaussian populations; it does not establish recovery of real links, correlated-variable performance, interactions, causal effects, or general release readiness.

`study-report-corrected.json` is the corrected report. `rf-case-audits/` contains all 110 individual bound replay records. Each binds the original raw archive hash to the repaired auditor/replay hashes and the original producer run, source and manifest. Original GBM/LightGBM checkpoint audit rows are retained and their endpoints recomputed; no new model fits occurred. The original published report remains unchanged in the adjacent original-results directory and on the release.
