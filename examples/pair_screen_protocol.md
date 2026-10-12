# Pair interaction screen — frozen development study

Authorized by the user on 2026-10-11. This studies the opt-in fitted-model
interaction screen, not recursive network selection or causal moderation.
Freeze source/task/workflow hashes before dispatch. No package defaults change.

## Design

Six scenarios: noise, nonlinear additive, pure product, mixed main/product,
correlated nonlinear additive, correlated mixed. Each crosses (N80,p5),
(N300,p5), (N600,p5), (N300,p15), with 20 independent replications: 480 datasets.
RF and GBM use their unchanged shipped four-setting grids on identical datasets
and splits: 960 method tasks. No individual-importance filter or product feature
is supplied to either learner. Draw namespaces 38/48/58 million are new.

Predictors are standard normal; correlated scenarios set corr(x1,x2)=.7.
Additive mean: x1 + .6*(x2 squared - 1). Pure interaction: x1*x2.
Mixed mean: x1 + .6*x2 + .8*(x1*x2-rho). Independent Gaussian error has variance
two thirds of the analytic mean variance, yielding oracle R2=.60; noise is
independent unit Gaussian. True interacting pair is x1,x2 only in product/mixed
conditions. Other predictors are independent noise. These are continuous IID
simulation cases, not universal survey-data guidance.

Three outer folds, one repeat, with training-only three-fold inner tuning and
fold-local preprocessing. Each outer model screens its held-out covariates,
using discovery outcome SD for descriptive scaling as in the current supplement.
Final full-discovery tuning predicts 1,000 independently generated test rows.
Test outcomes never tune/select a model, pair, cutoff or screen configuration.
CV fits overlap; the dataset is the independent simulation unit.

Use unchanged PairScreenConfig defaults: 250 pairs, 64 screen rows, two bins,
five details, 256 detail rows, four bins, three observations per cell and
100,000 prediction rows per outer model. In small samples many detailed grids
may be unsupported; preserve that outcome instead of interpreting it as zero.

## Outcomes and interpretation

Record all raw per-fold screens, support counts/statuses, settings, predictions,
warnings, actual training/inner sizes, timings and prediction budgets. Primary
descriptive outcomes: true-pair rank/top-one/top-five, true supported detail,
incidental supported details, null probability of any displayed candidate,
fraction of pairs with complete coarse support, and independent-test skill.
Ranks require complete coarse support. A displayed candidate requires complete
coarse support, shortlist membership and at least one supported detail, matching
the current report. Ranking is not statistical detection or a stability rate.

Also report score-cutoff sensitivity at .05,.10,.20,.30,.40,.50 outcome-SD units
using complete mean coarse RMS, independently of detail availability. Record
true-pair exceedances and any unrelated-pair exceedances, with assessed/missing
counts; do not label unassessed comparisons negative. No cutoff nomination,
significance claim or automated retention rule is earned by this development
panel. A disjoint, separately frozen confirmation is required before adopting
any rule. Twenty datasets per cell give preliminary, imprecise estimates.
Report Wilson intervals for independent binary dataset counts when complete;
missing/error tasks stay in planned denominators and count bounds.
Unassessed true-pair ranks/cutoff comparisons are unknown; their possible successes
enter the upper count bound. Incidental cutoff bounds include unassessed pairs.
Operational display coverage is observed even if support prevents display; it
does not establish population absence.

## Execution and evidence

240 deterministic shards of four tasks, two single-threaded worker processes per
standard Windows2022 runner; maximum parallel 256, subject to shared account
capacity. Each shard has a deliberately bounded 30-minute hard timeout; prepare
15 and aggregate 60 minutes, total maximum allocation 7,275 runner minutes.
No paid runners or automatic retries. Push registers workflow; explicit dispatch
starts fits. Dedicated setup avoids the old frozen-study source gates.
Shard setup has a four-minute step limit and fits a 22-minute step limit within
the 30-minute job, reserving time for failure uploads before hard cancellation.
These are study limits, not platform maximums. Runtime Python/package versions
are checked against the frozen environment and retained/validated in receipts.

Write receipts before fits, atomic fold checkpoints, errors and verification
records. Resume only identical frozen context. Workers regenerate data/splits
and deterministically refit saved settings, reproducing predictions and pair
screens; this does not independently replay tuning winners. Aggregation checks
identities, source context, arithmetic, planned folds/pairs and verification
receipts. Download shards separately and reject conflicting duplicate evidence.
Verification receipts bind hashes of results, runtime receipts and prediction
files; changed evidence cannot reuse an old saved-setting refit certification.
Upload partial output on failure; missing tasks stay missing.

Retain artifacts 30 days. Preserve raw evidence archive, SHA256, frozen manifest,
audit, execution provenance, summary CSV/report/graph in the repository after
completion, before artifact expiration. The launch does not constitute results.
