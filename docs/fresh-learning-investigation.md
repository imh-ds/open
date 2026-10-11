# Fresh independent learner investigation

The previous investigation found that good signal could be difficult for tree
models to learn among nuisance predictors. Its examples cannot establish a
population failure rate. This development screen compares models across 240
fresh datasets, with 20 independent draws per scenario/design point.

See the [frozen protocol](../examples/fresh_learning_protocol.md) and
[source/task manifest](../examples/study_manifests/fresh-learning.json).
It tests current GBM, expanded-grid GBM, RF and a privileged known-function
linear benchmark. All methods share discovery/test data and outer folds;
hyperparameter selection uses discovery training partitions only.

This is the model-learning stage. It does **not** run B100 network selection,
estimate false-link rates or establish minimum sample sizes. Structural panels
and independent confirmation come after this screen; no package defaults change.

Run locally with the pinned environment from the protocol:

```powershell
python examples/fresh_learning.py freeze
python examples/fresh_learning.py shard --id 000 --output .superpowers/fresh-learning
python examples/fresh_learning.py aggregate --output .superpowers/fresh-learning
```

`freeze` checks the committed manifest; `--write` is only for protocol authors
before dispatch. Resume requires matching context and rechecks saved predictions.
Aggregation deliberately exits unsuccessfully when planned tasks are missing or
invalid, while writing the partial audit/report. A successful worker does not
mean that every study task completed.

The `fresh-learning.yml` workflow launches 120 shards with four workers each,
then combines the disjoint artifacts. Matrix parallelism is 256, subject to
GitHub's shared account capacity. The 60-minute shard timeout is this study's
budget, not GitHub's platform maximum. Worst-case allocated runner time is
121.25 hours; actual use includes setup and audit. No automatic retries.

Every completed fit saves nested-CV predictions/settings, independent-test
predictions and a receipt that saved-setting refits reproduced them. This audit
does not independently retune models. The combined artifact also contains the
manifest and partial evidence. Artifacts retain 30 days; after completion raw
evidence and checksums must be archived beside the interpreted repository report.
Do not publish sample-size advice from this screen.
