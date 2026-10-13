# Independent arithmetic review

The final hardened auditor completed successfully on all 10,800 selected datasets in 135 shards. Its authoritative outputs are in `independent-vectors-final/`; the earlier complete run is retained separately in `independent-vectors/`.

| Check | Verified |
|---|---:|
| Raw files, inventories, hashes and task contexts | 97,335 |
| Pair prediction risks and gains | 86,400 |
| Ideal-population predictions and risks | 43,200 |
| Curve anchoring, outcome normalization and training support | 86,400 |
| Pair records used for decisions and pattern summaries | 21,600 |
| Endpoint confidence bounds | 72 |
| Recomputed prediction-gain maximum difference | 0 |
| Combined qualified conditions | 4 of 24 |
| Model fits performed by this audit | 0 |

The auditor regenerates outcomes from the pinned seeds, reconstructs the three outer folds and inner training sizes, checks saved minimum-loss tuning winners, recalculates reference/additive/joint risks and pooled scores, and applies the strict frozen thresholds. It reconstructs anchored standardized curve contrasts and training-only support counts, common supported locations, fold-to-fold cosines, truth-pattern cosines, display decisions and exact directional bounds. It imports no project model or decision functions. Saved model predictions and tuning losses are inputs; this does not independently retrain or retune models.

Replay from the repository root, using the pinned runtime:

```powershell
python examples/results/2026-10-13-prospective-interaction-signal/recompute_vectors.py --output examples/results/2026-10-13-prospective-interaction-signal/independent-vectors-final
```

This requires the locally retained `raw/shard-000.zip` through `raw/shard-134.zip` and their receipt files. Public summary publication does not include these large raw archives yet; do not infer that a public clone already contains everything needed for full replay. Their local preservation commits and hashes are recorded in `raw-commit-receipts.json` and `raw-verification.json`.

## Review and regression checks

A fresh independent reviewer accepted the scope and guidance, but found two completeness gaps in the initial verifier: dictionary conversion could silently collapse duplicate dataset rows, and positional iteration could accept a shortened or mismatched condition list. The actual study inputs were complete and unique; these were verifier issues, not changed study results.

The corrected verifier validates original row cardinality, unique identities, all 24 expected condition keys and all 72 bounds. Conditions are matched by identity. Five added guard regressions were observed failing before these checks were implemented and then passing. The original eight curve/tuning tampering checks and three prior null-provenance checks also passed. The broader prospective suite passed **44 tests in 55.13 seconds**. After the fixes, the entire raw-vector audit was rerun successfully.

The review covered this independent auditor, the guidance and relevant frozen definitions. Earlier launch review covered the study implementation. The final reviewer did not rerun model fitting, review every historical branch change or certify remote raw preservation. No second reviewer was dispatched after the two fixes; regression checks and the complete hardened replay provide the fix evidence.

## Interpretation boundaries

These arithmetic checks confirm that the reported findings follow from the saved evidence and frozen rules. They do not establish universal detection reliability, a minimum N for real data, an individual finding's probability of being true, causal moderation, exact numerical effect-size accuracy, other engines or the complete recursive workflow. The historical failed confirmation result remains failed. Production confidence adoption remains a separate decision.
