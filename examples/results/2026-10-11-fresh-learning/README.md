# Durable fresh-learning evidence

Run [38133443512](https://github.com/imh-ds/open/actions/runs/38133443512),
source commit `31f85727a5e93528a4bec4970764bd0b4365f797`.
Read [REPORT.md](REPORT.md) for interpreted findings and scope.

- `actions-evidence.zip`: original combined artifact, 85,732,366 bytes,
  SHA256 `3ab8bb780b716c478eff2004f7e3967f98f5de86f7a6d90843d3ae4b2e98212f`.
  Contains all 960 tasks' receipts, 9,600 fold checkpoints, final results,
  saved-setting verification receipts, 1,920,000 test prediction rows,
  original audit/report and frozen manifest. CRC checked.
- `audit-from-actions.json` and `REPORT-from-actions.md`: untouched original
  aggregation outputs. `audit-local.json` exactly matches the original audit
  after local regeneration and checking of data, splits, identities, grids,
  tuning sizes and arithmetic. Local closeout did not rerun all tree refits.
- `manifest.json`: exact frozen task/source identities; generators and model
  source are retained in Git at the source commit. `execution.json` retains
  all 122 job identities/timestamps. `receipt.json` records archive integrity,
  audit scope, warnings, participant/prediction counts and execution totals.
- `cell-summary.csv` and `learning-overview.png`: descriptive summaries of all
  48 cells. Pointwise uncertainty is not simultaneous scientific validation.
  `render_closeout.py` reproduces these summaries from the evidence without fits.

To repeat the original aggregate audit offline, use an isolated checkout of the
source commit and the protocol's pinned Python environment. Extract the ZIP
into a temporary directory and run:

```powershell
python examples/fresh_learning.py aggregate --output <extracted>/.superpowers/fresh-learning
```

The archive's other root is `examples/study_manifests/fresh-learning.json`.
Do not flatten the ZIP or put extra directories inside the audit input.
Aggregate reproduces scores without expensive retuning or refits. On Actions,
every completed task already reproduced fold/final predictions at saved settings.
Neither audit independently replays tuning searches to verify their winners.

240 independent discovery datasets underlie these 960 paired method fits.
All methods share test participants within a dataset. This study runs no B100
network selection and supplies no minimum sample size or zero false-link claim.
Raw evidence remains in Git after the 30-day Actions artifact retention ends.
