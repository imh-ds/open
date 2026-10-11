# Fresh independent learner investigation — development stage 1

Frozen before dispatch, 2026-10-11. User authorized investigation and GitHub
Actions launch. This stage tests model learning; network selection and recursive
recovery require the next stage. No production defaults change.

## Exact design

240 independent discovery datasets: noise, additive, pure interaction and chain,
crossed with (N80,p5), (N300,p5), (N300,p15), with 20 replications per cell.
Use the unchanged sample-size pilot generator: clean continuous IID data;
root oracle R-squared .60 for signal; chain adjacent correlation .60.
Each dataset has 2,000 independently generated test observations. Discovery,
test and split seeds use disjoint namespaces starting at 8,000,000, 18,000,000
and 28,000,000. No previous draws are reused. All methods share each dataset
and its fixed 5-fold/two-repeat outer CV partitions. Training-only inner CV has
three folds; preprocessing is fitted inside partitions. Final tuning uses only
discovery observations. Neither test outcomes nor oracle predictions select
hyperparameters or features.

Four paired methods (960 planned fits): shipped GBM grid, expanded GBM grid
(depth 2/3/5, learning rate .05/.10), shipped RF grid, and a privileged linear
benchmark. The benchmark is given x1 for additive/chain, x1*x2 for interaction,
and a constant for noise. It diagnoses available signal; it is not a deployable
feature-selection method. Tree methods get all p predictors, never supplied
interaction terms. Reuse existing fitting helpers without editing their source.

## Outcomes and development decisions

Record nested-CV and independent-test MSE and signed baseline-relative skill,
oracle skill, actual train/tuning sizes, fitted settings, warnings, predictions,
runtime and saved-setting refit checks. Report positive-test-skill counts out of
all 20 planned datasets per cell, Wilson 95% intervals, skill distributions and
paired differences to shipped GBM. Missing/error cases remain in planned counts;
success counts are conservative lower counts and paired comparisons report
their completed-pair denominator. Two repetitions of CV are not independent
datasets. Test predictions are for new participants from the same simulated DGM.

Development nomination only: a candidate must improve interaction mean paired
test skill by at least .05 at both N300 points and must not worsen additive or
chain mean paired skill by more than .05 at any point. All affected cells must
be complete. This is a prespecified research heuristic, not a significance test,
scientific acceptance gate or permission to change defaults. Noise skill and
all reversals are reported. No minimum N, false-link rate, interaction-edge
recovery or general fold recommendation follows from this prediction screen.
Nomination and statistical uncertainty cannot validate a searched method;
network and fresh confirmation studies remain separate.

## Execution, audit and evidence

120 deterministic shards, two datasets/eight method tasks each. Four single-
threaded workers per standard Windows2022 job, maximum matrix parallelism 256;
GitHub applies shared account concurrency. No paid runners or automatic retries.
Each fit writes its receipt before fitting and atomic fold checkpoints after
each fold. Resume requires the identical frozen source/protocol/task context.
Every completed task checks all saved fold/final settings by deterministic
refitting; this checks predictions, not independent replay of tuning winners.
Aggregation regenerates data/splits, validates identities and arithmetic,
rejects foreign/conflicting evidence and preserves missing/error denominators.

Hard shard timeout 60 minutes (a deliberate bounded screen budget, not GitHub's
six-hour platform limit); preparation 15 minutes, aggregation 60 minutes.
Maximum allocation 7,275 runner minutes (121.25 hours), including setup and
upload within job timeouts. Actual cost is expected much lower; the manifest
does not authorize further runs or a larger confirmation panel. Upload on
failure retains completed checkpoints; unfinished tasks are explicitly missing.
Artifact retention 30 days. After completion, archive raw evidence, audit and
manifest alongside the report in the repository so findings survive expiry.
No partial run earns a nomination. Sources, workflow and protocol are hashed
in `study_manifests/fresh-learning.json` before dispatch.
