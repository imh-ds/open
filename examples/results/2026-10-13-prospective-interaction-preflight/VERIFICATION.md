# Prospective interaction launch preflight

Implementation reviewed independently; no prospective scientific support has been established.

- 65 focused new/historical regression tests passed in47.99s. No full package suite claim.
- Final frozen manifest5a383f67f965c36ec9ffd07175e8153940d6b8844d5e021e25620c87862f5f4a; pinned runtime and60 required sources verified in a sparse fixture with no historical result archives. Public61 source/manifest Git blobs match development.
- Final two-worker80-task high-overlap p15/N300 nonlinear-additive shard179 completed in227.047s, inside720s computation cap. New-process resume5.797s added0fits. 55,360 fit/tune calls and1,600 verification refits, no failed tasks.
- Earlier pre-hardening run of the SAME80 planned identities took231.594s,55,360fit/tune+1,600verification refits; its raw archive is preserved under pre-hardening. Across both native runs110,720fit/tune+3,200verification refits;80 distinct datasets, not160 replications. Additional fit-backed development-test computation was not instrumented and is not included in those native totals.
- Independent explicit saved-vector arithmetic checked640pair risks,320population risks and160pooled rows for each run, max discrepancy0. No model refitting for arithmetic verification.
- Final partial audit80/14,400complete;14,320missing;0audit errors,0unknown attempt costs. No null condition qualifies from this preflight. Directory and compressed streaming audits agree on all11checked counts/endpoints/cost fields.
- Real lossless worker packaging722files/72,639,837raw bytes ->18,360,217compressed bytes in2.36s. One-shard streamed audit8.859s; linear StageA estimate26.58min within60min aggregate cap, not a guarantee. Conservative doubled compressed estimate6.61GB plus at most145.3MB extracted shard avoids the old approximately30.4GB dual raw extraction. All original compressed worker evidence stays preserved; temporary extracted shards alone are removed.
- First retained archive650files/18,561,188bytes SHA256256f0aec2209e7ba6b32942a7a07eb5a579e4a6c3470f32e537cd7771b0b5a84; final731files/18,616,251bytes SHA25635e5481bfd2515d6c005b0ed68efa4830741631092de1b460c32fbf749ff1e15. Inventory, CRC, hash and extraction verification commands are retained alongside each archive.

Review corrections: append-only failed/running/completed attempt costs survive eventual resume; unknown verification/fitting work never disappears after success. Broad prediction and broad curve-pattern flags are separate; generic broad support requires both. Observed-rate misses and imprecision are described separately without stronger inferential claims. All corrections have observed RED-GREEN regressions. Production and historical frozen sources remain unchanged; historical confirmation still failed.

Reproduce without fitting:

```
python verify_archive.py --output FRESH_DIRECTORY --extract
python recompute_results.py --input FRESH_DIRECTORY/.superpowers/prospective-native-preflight-final --output METRICS_DIRECTORY
```

Transport archive receipt/inventory can also be independently checked using examples/prospective_interaction_archive.py. Source freeze is reproducible with examples/prospective_interaction_study.py freeze in the exact source checkout and pinned runtime. Launch remains gated on verified public evidence/export. StageB remains blocked until complete independent StageA audit and committed matching selection.
