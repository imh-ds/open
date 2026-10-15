> **Public report projection:** Independent verification is complete. All 110 original
> raw outputs and the full 452-file packet are committed in the owner's local
> repository. Large archive uploads are timing out; this report branch does not
> yet contain every raw archive. SHA256 entries describe the full retained packet.
> See `publication-status.json` for the explicit preservation boundary.

# Completed no-fit chain verification

Read [interpretation.md](interpretation.md), [case-results.csv](case-results.csv),
[audit/audit.json](audit/audit.json) and [chain-recovery.png](chain-recovery.png).
The independently verified endpoint passed both 55-case sample-size conditions.
The original Actions workflow still has its original failed aggregate status.

## Reproduction

Use Windows with Python 3.11.9 and the pinned installed package/runtime from the
original chain workflow. The scientific source and settings remain bound by
`examples/study_manifests/chain-qualification.json`. The replay adapter prohibits
model fitting and calls the original frozen observation-loss auditor.

From the repository root, with that environment active:

```powershell
python -I examples/replay_chain_verification.py --inputs examples/results/2026-10-15-chain-verification-38690088093/raw/cases --out examples/results/2026-10-15-chain-verification-38690088093/audit --workers 8
python -I examples/results/2026-10-15-chain-verification-38690088093/summarize.py
```

Existing case caches are reused only when the raw archive, control, case identity,
original source/run/attempt, frozen manifest, replay source and result fingerprints
match. To repeat the full independent arithmetic replay, choose a fresh `--out`
directory instead. No fitting is done in either mode. Source line endings are
pinned to LF so cache fingerprints survive Windows/Unix Git checkout conversion;
the frozen auditor itself uses the original Windows runtime/path conventions.

## Files and provenance

- `raw/cases/`: all 110 original inner case ZIPs, controls and runtime records.
- `raw/download-inventory.json`: complete artifact inventory used for retrieval.
- `archive-locations.json` and `raw-sha256.json`: original archive locations/sizes
  and exact fingerprints for all 331 retained raw files.
- `audit/case-audits/`: 110 atomically written, resumable per-case verification
  records. None of the original analysis outputs is modified.
- `audit/audit.json`: fixed-denominator summary and per-case checked findings.
- `verification.json`: endpoint checks, cache-resumption proof and scope.
- `sha256.json`: fingerprints for the complete saved verification packet.

Original producer: run `38690088093`, attempt 1, source
`34729fd23f820711f31ea081c8e4b46769d3ed4f`.
The 32-minute timeout diagnosis and original aggregate ZIP/API/log evidence are
preserved separately in `../2026-10-15-chain-aggregate-timeout-38690088093/`.

Recovery used local computation only: zero model refits and zero additional
Actions dispatches. The prior 9,889/10,000 runner-minute reservation is unchanged.
The 110 correlated-predictor cases are pending; no first-release or universal
sample-size claim is made. Independent replay checks recorded data, splits,
configuration, loss arithmetic and retained recursive structure; it does not
independently refit the original predictions.
