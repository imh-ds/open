# Selection diagnostics and sharded continuation

The [feasibility pilot](../examples/results/2026-10-10-sample-size-pilot/REPORT.md)
completed 11 of 12 analyses. Its unfinished chain stopped because our runner had
a 90-minute soft budget. The old workflow also imposed a three-job concurrency
cap and a 110-minute hard job timeout. These were project settings, not GitHub's
six-hour hosted-job limit.

The follow-up has two separate purposes:

- Diagnose the interaction misses and the stable noise input in the small chain,
  using matched model comparisons and ten original LOCO assessments.
- Finish the interrupted chain using its original data, settings and 116 saved
  assessments plus four verified smoke checkpoints. Report this continuation
  separately from the historical pilot.

The [frozen development protocol](../examples/selection_investigation_protocol.md)
specifies 64 model comparisons and ten LOCO comparisons. These reuse previously
inspected development datasets and test draws. They do not establish a minimum
sample size, validate scientific defaults, or provide fresh confirmation.

## Scheduling

The [workflow](../.github/workflows/selection-investigation.yml) gives each job a
360-minute timeout. Its matrices allow up to 256 simultaneous jobs; GitHub
enforces the account's actual shared concurrency limit. Standard hosted limits
are Free20, Pro40, Team60 and Enterprise500; a matrix can contain at most256jobs.
The authenticated API did not disclose this owner's plan. Standard public
Windows2022 runners provide four vCPUs and16GB RAM, so each fitting job runs four
single-threaded workers. No paid larger runners or other projects' cancellations.
[GitHub limits](https://docs.github.com/en/actions/reference/limits),
[runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

Each diagnostic model/scope combination is one independent job. Outstanding
recursive assessments are divided into fixed buckets of ten run IDs. The first
unfinished layer needs eight shards covering runs20–99; runs0–19 are reused.
Every target still requires all100planned assessments before its layer is
finalized. Only then can another layer's jobs be scheduled. Parallel execution
changes computation order, not sample identities or recursive selection.

Push registers the workflow without starting expensive fitting. Explicitly
dispatch `selection-investigation.yml` on `codex/actions-selection-reliability`.
Checkpoints and diagnostics are uploaded even after a failure. Aggregation
retains partial checkpoints, rejects conflicting/foreign identities and withholds
scientific completion when evidence is missing. No automatic statistical reruns.

## Evidence and interpretation

The final artifact, `selection-investigation-results`, contains a report,
all planned diagnostic identities, predictions, fold records, warnings/errors,
continuation checkpoints and its audited recursive graph. Shard and stage
artifacts are also retained for30days. Download evidence before expiry.

Model CV uses training-only inner tuning, then tests predictions in outer folds.
Full-scope refits predict the same fixed independent test participants. Signed
skill compares with a training-mean baseline; negative values mean worse
prediction than that baseline. The deeper GBM and known product feature are
diagnostic controls and do not alter production defaults.

LOCO contrasts compare removing x4 with fixed full-model settings against
separately tuning the reduced model. Both use the same rows/folds. Their ten
samples overlap and are not ten independent population replications. Differences
can suggest a contribution's sensitivity to model fitting; they cannot establish
why a population link exists or estimate a false-positive rate.

The audit regenerates data and splits, checks arithmetic, and deterministically
refits models with saved grid-valid parameters. It does not independently rerun
the tuning searches to verify their winning settings or inner tuning scores.
The continuation's finalization elapsed time excludes distributed assessment
computation; individual checkpoint metrics retain those assessment durations.
Completion of the interrupted chain does not guarantee recovery of its intended
structure. Original reports, source hashes, protocol and denominators stay intact.

## Verified follow-up results

[The completed investigation report](../examples/results/2026-10-10-selection-investigation/REPORT.md)
records 74/74 completed diagnostics and successful continuation of the interrupted
chain. All three intended chain links were retained in 100/100 assessments each,
with no extra links. The local offline audit passed with zero errors or warnings.

The diagnostic examples still expose scientific limitations: known signal inputs
helped the interaction models, and the small-chain noise input remained apparently
useful under retuned ablation but worsened test prediction in a matched GBM
comparison. No production repair or minimum sample-size guidance follows.
The run peaked at 20 concurrent jobs and finished in about 48 minutes.

The next [fresh independent learner screen](fresh-learning-investigation.md)
tests whether those model-learning patterns recur across independently generated
datasets. It is a development prediction study, preceding further network tests.
