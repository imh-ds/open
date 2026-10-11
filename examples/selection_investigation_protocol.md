# Selection investigation and sharded pilot completion

Frozen development protocol, 2026-10-10. User authorized investigation of the
interaction misses/stable false input and maximum useful Actions parallelism.
No production selection policy, threshold, grid or historical denominator changes.

## Questions

1. Do pure-interaction misses reflect insufficient learner performance, nuisance
   predictors, model capacity, or filtering after useful prediction?
2. Does the small-chain x4 contribution persist under separately tuned model
   comparisons, or depend on fixed full-model settings used for LOCO ablation?
3. Can checkpoint-level sharding finish the interrupted chain while preserving
   its original scientific settings and all saved assessments?

This is diagnosis using already inspected development datasets/test draws.
Test comparisons are descriptive, not fresh confirmation or default selection.
Do not claim a causal explanation from model contrasts alone, an error rate from
one seed, or a minimum N. Any proposed method change requires fresh confirmation.

## Fixed input and source boundary

Use the archived 12-case pilot at public source dd8a176, whose evidence archive
SHA256 is a7d9c6a27d0318a52f8b51f293ab33212bbc8431eef6ad274dc59add96a5d13a.
The original pilot runner, production sources and original protocol stay unchanged.
New helper/protocol fingerprints accompany the operational continuation and
diagnostic tasks. Input arrays regenerate from the original seeds; all participants
and the 2,000 independent test rows remain fixed. Original 11/12 completion stays
in its historical report even if a later separate completion pass succeeds.

## Matched model diagnostics: 64 independent jobs

Four fixed discovery datasets: the three original interaction datasets
(N80/p5, N300/p5, N300/p15), plus chain N80/p5 with the false x4 link.
For each model comparison use two discovery scopes: all original discovery rows,
and the first original 80% stability subsample (run0). This is a train-size/row-
composition comparison, not a pure causal isolation of N. Match inputs/splits
across model variants within each scope. Tune only within training partitions.

Interaction datasets each receive seven variants:

- GBM shipped, diagnostic deeper GBM, and RF shipped, each with all candidate
  inputs and with known signal inputs x1,x2 only (six variants).
- Linear regression using the explicitly constructed known product x1*x2
  (one diagnostic positive control, not an automatic feature-generation proposal).

The chain dataset receives eleven variants:

- GBM shipped: all five inputs; x1; x1,x4; x4; all inputs except x4 (five).
- RF shipped: all inputs and x1,x4 (two).
- Linear regression: all inputs, x1, x1,x4, x4 (four).

Thus (3*7+11)*2=64 model jobs. Shipped trees retain their production grids and
100 estimators. Diagnostic deeper GBM has max_depth[2,3,5], learning_rate[.05,.1],
100 estimators, all other defaults unchanged. This experimental grid is confined
to the diagnostic runner; it does not modify the package or archived study.

Each job uses the original local five-fold/two-repeat prediction plan on its
declared discovery scope and three-fold tuning inside each training partition.
Refit preprocessing per inner fold. Save fold rows, parameters, predictions,
fold training-mean baselines, fitting/tuning sizes and training-only tuning scores.
Then tune/refit on the declared discovery scope and predict the fixed independent
test participants; save observed/baseline/oracle/predicted values and signed skill.
Both CV and test comparisons are required. Never choose defaults from test skill.
Four single-threaded outer-fold workers per standard four-vCPU Windows runner.

## Matched LOCO diagnostics: ten jobs

Replay the first ten original chain-N80/p5 root assessments (runs0..9), preserving
sample positions, original folds, seeds, preprocessing and GBM grid. Refit the
full five-input model and assert its CV risk matches the archived full risk.
Compare ablation of x4 with fixed full-model settings (current method) versus
independently tuning the four-input ablated model on the same training folds.
Also reproduce the original standalone x4 risk. Save all fold-level risks and
parameters, raw/baseline-relative/gain-relative contributions and comparison signs.
Assert the fixed-setting ablation and standalone risk reproduce archived contrasts.
Do not silently replace the current importance definition with retuned ablations.
The ten overlapping samples are not ten independent population replications.

## Operational completion of chain N300/p15

Reuse all116 original checkpoints:100root and16at the next target. Also transfer
the four native production smoke assessments at runs16–19, independently
validated with the original context/plans/arithmetic. Their fixed transfer archive
SHA256 is45cd4e528d9bd8fbb46b423a57833a1de629e79e603f96a7674a42ba3cb43406.
No scientific sources changed during these fits; the orchestration helper was
being revised, so its receipt records file hashes observed at receipt time rather
than asserting an immutable loaded helper. Production checkpoint contexts remain
the strict source/data/config identity. Derive the
frozen next target/candidate comparisons using production consensus aggregation.
Split outstanding assessments into nonoverlapping ranges of ten run IDs; each
shard uses four single-threaded workers and the original production assessment
function. Check all context/source/data/sample identities before reuse. With
120saved assessments the first unfinished layer requires eight shards (runs20–99).

Aggregate all100 planned assessments for every target in a layer, then finalize
the layer and derive the next frontier. Never advance from partial evidence or
pool different target comparisons. Preserve failed assessments as failures, not
missing tasks eligible for silent replacement. After all required layers finish,
run the actual production recursive engine reading the saved assessments and
the original train-only final refits; independently audit its full saved output.
The continuation has its own output/provenance and does not rewrite the initial
budget stop. It may still produce an empty/incorrect graph; completion is not recovery.

## Parallelism and evidence

GitHub's published limits: standard hosted job maximum6hours; matrix maximum256
jobs; account concurrency Free20/Pro40/Team60/Enterprise500. The authenticated
API does not expose the owner's plan. Set matrix max-parallel256 and let GitHub
enforce available account capacity, rather than capping at3. Account capacity is
shared with other workflows; do not cancel unrelated jobs or use paid larger runners.
Use standard public Windows2022 four-vCPU/16GB runners and pinned Python/dependencies.

Each diagnostic/assessment job has timeout360minutes; no90minute soft budget.
Assessments are independently checkpointed and final artifacts uploaded. Sharding
keeps normal jobs much shorter than the hard timeout. Layer dependencies remain
sequential because they are part of the scientific method. No automatic statistical
reruns, seed changes or replacement of failed results. Failed/missing jobs retain
their planned identities in aggregation; scientific results require complete evidence.

Stage aggregates reject nonidentical duplicate files, source/config/context mismatch,
overlapping/foreign run identities or missing required evidence. Model diagnostics
audit data/splits, prediction arithmetic and deterministic refits with saved settings;
completion uses the existing full pilot audit. Final report distinguishes diagnosis,
operational completion and unresolved scientific reliability. All warnings and errors
are retained. No broader sample-size calibration dispatched by this protocol.

Limits checked on2026-10-10:
[GitHub Actions limits](https://docs.github.com/en/actions/reference/limits),
[standard hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
