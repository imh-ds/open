# Synthetic OPEN demonstration

The current layer-consensus survey runner accepts `--method layer_consensus`
and `--sensitivity`. It finalizes a fixed-candidate comparison at each depth
before recursion. Its seven-file bundles and one reference graph are distinct
from the historical whole-pipeline demonstrations described below.
Threshold sensitivity reconstructs later layers rather than filtering a graph.
Local CV performance remains conditional on discovered candidates.

```powershell
python examples/run_survey_demo.py --output NEW_SURVEY_DIRECTORY --resamples 20 --max-depth 5 --method layer_consensus --sensitivity
python examples/audit_layer_demo.py --output NEW_SURVEY_DIRECTORY --historical examples/results/2026-10-07-survey-exploratory
python examples/run_layer_fixtures.py --output NEW_FIXTURE_DIRECTORY --resamples 20
```

Saved layer results: [six-variable survey](results/2026-10-08-survey-layer-consensus/REPORT.md)
and [chain, missing-outcome and joint-predictor checks](results/2026-10-08-layer-fixtures/REPORT.md).
These are conditional reference networks; historical reports below retain their
original whole-pipeline interpretation.


Run `python examples/run_synthetic_demo.py --output NEW_DIRECTORY --resamples 20`
from this checkout after installing the package with its visualization extra.
The output directory must not already exist. The script explicitly imports this
checkout's source so a separate, older checkout cannot supply the engine.

The saved [corrected demonstration report](results/2026-10-07-fixed/REPORT.md) includes actual
whole-pipeline counts, local predictive risks, and the exported consensus plots.
All samples contain 80 independent synthetic rows:

- `chain`: seed 271828; Stress is standard normal, Depression is Stress plus
  normal noise with SD 0.7, and Anxiety is Depression plus noise with SD 0.3.
- `chain_missing`: the identical sample, with Depression missing in row 0. This
  reproduces the previously blocked case where an incomplete predictor becomes
  a recursive target.
- `shared_predictors`: seed 120; Scale A is standard normal, Scale B is a near
  copy with noise SD 0.01, and Outcome is Scale A plus noise SD 0.15.

All analyses use the fixed elastic-net learner, two outer folds, two tuning
folds, depth cap 3, 80% subsampling without replacement, 20 development runs,
unique relevance cutoff 0.15, and consensus cutoff 0.70. These match the review's
development settings; thresholds were not selected from the resulting graphs.

Each output case contains its input CSV and configuration, a full-sample
single-run diagnostic, six exported OPEN files under `bundle/`, captured warning
counts and messages, and PNG/SVG graphs rendered from the saved bundle.
`summary.json` retains input hashes, generation seeds, engine revision, counts,
and empirical complete-path frequencies. `local_targets.csv` records effective
observed/missing target counts for each run.

Solid arrows show unique predictive contributions; dashed arrows show screened
relations. Shared groups retain all individual member nodes. Normalized broad,
unique, and group relevance ratios describe fractions of each local model's
predictive gain; they are not additive percentages of explained variance.
Different targets can have different observed-row subsets and gain denominators.

This is a small software demonstration. It does not establish calibrated
selection rates, causal direction, or validity on real observations. The full
path frequency is measured from complete networks, not multiplied from edge
probabilities. Numerical warnings remain available for inspection.

The [earlier run](results/2026-10-07/REPORT.md) is retained as pre-convergence-fix
evidence, with its original engine revision and warning counts. Its shared-case
diagnostic found 12 of 200 selected outer fits truncated at 10,000 iterations.
The identical-seed diagnostic at 100,000 iterations found no convergence warnings;
the hardest selected fit needed 44,861 iterations. The corrected run uses that
higher cap without changing optimizer tolerance, analytical thresholds, or data.

## Recovered six-variable survey example

The [survey report](results/2026-10-07-survey/REPORT.md) reruns the earlier
180-row simulation with Well-being as the root and Social Support, Sleep Quality,
Perceived Stress, Physical Activity, and Work Demands as candidates. It uses the
original seed 613, equations, two-fold outer/tuning CV, two-level depth cap, and
unique relevance threshold 0.05. It expands the original five resamples to 20;
the first five seeds and sampled row positions are also compared with the saved
earlier outputs. These settings differ from the three small fixtures above.

Run `python examples/run_survey_demo.py --output NEW_DIRECTORY --resamples 20`.
The optional `--original-directory DIRECTORY` checks the recovered generator
against the earlier rounded input CSV and compares local-regression risks and
sampled row positions from all five original runs. The new input retains full precision,
matching the earlier generator's in-memory analysis rather than refitting its
rounded CSV. Existing output directories are never overwritten.

The runner composes the existing batch, stability, consensus, and export APIs;
memory profiling is not enabled. It saves all recursive regressions and stop
reasons, the standard six-file bundle, exploratory and explicit consensus networks,
an edge evidence table, and a heatmap of
unique edges across completed networks, including edges below the display
threshold. Simulated scores are clipped continuous composites on a 1–5 scale;
they are not observed respondents, raw ordinal items, or validated instruments.

The [depth-5 report](results/2026-10-07-survey-depth5/REPORT.md) uses the same
20 sampled row sets and thresholds with enough depth for all six variables.
Run with `--max-depth 5` to reproduce it. Its branches stop naturally without
hitting the cap. The report includes both complete-network edge frequencies
and descriptive frequencies conditional on the source being eligible for that
target. The latter help explain the cascade but do not replace the whole-network
consensus denominator. The pictured depth-3 network is an actual illustrative
run, not the consensus; its complete Work Demands → Stress → Sleep → Well-being
path occurred in 1/20 runs.

## Exploratory display comparison

The [updated display report](results/2026-10-07-survey-exploratory/REPORT.md)
uses the same 180 rows, 20 sampled row sets, depth-5 limit and analytical thresholds.
Run `python examples/run_survey_demo.py --output NEW_DIRECTORY --resamples 20 --max-depth 5`.
It shows all observed relations by default and retains a separate consensus plot.
Whole-network frequency controls opacity; the consensus subset is highlighted.
Unique width uses median signed per-run loss reduction relative to target baseline
risk, including scored nonselections. The fixed scale prevents weak sole predictors
from receiving maximum width merely because they account for all model gain.

Selected/complete and selected/eligible counts appear separately in the plot's
evidence panel and `edge_evidence.csv`. Eligibility follows actual layer-frozen
candidate sets and includes low-gain stops. Conditional counts explain opportunity;
they are not a new consensus rule or a calibrated confidence measure. Pooled
arrows with modal-depth positions may include cycles; actual single-run graphs
and measured complete-path frequencies remain the evidence for recursive pathways.
Historical reports are retained without rewriting their graphs or results.

## Comparable effects across engines

Run `python examples/run_engine_comparison.py --output NEW_DIRECTORY` after
installing `.[visualization,lightgbm]`. This compares elastic net, random forest,
GBM and LightGBM on the same saved 180-row survey and a nonlinear U-shaped
fixture. Use `--engines elastic_net random_forest gbm` if LightGBM is not installed;
requesting an unavailable engine fails explicitly. Existing output is preserved.

The [matched comparison report](results/2026-10-08-engine-comparison/REPORT.md)
contains standardized ALE curves/contrasts in outcome SD units, unique and joint
ΔCV R², overall CV R², and complete fold/reference/tuning provenance. This is a
new fixed-target comparison with shared three-fold outer CV and two-fold tuning,
not the earlier 20-resample network. The nonlinear fixture demonstrates why a
near-zero endpoint contrast still needs its curve. Tree engines have no surrogate
beta coefficients. Each engine can independently run the existing recursive APIs
by selecting its name in `learner.library`; effect comparisons hold candidate sets
fixed and do not pool different discovered networks. See the
[method and API usage](../docs/engines-and-effect-sizes.md).

The [curve-magnitude supplement](results/2026-10-08-curve-magnitude/REPORT.md)
adds the unsigned typical spread of each ALE curve in outcome SD units.
The U-shaped fixture has near-zero endpoint contrasts but substantial curve
magnitudes for all three tree engines. Existing prediction scores, curves and
contrasts match the preceding engine comparison. A flat curve yields zero;
unavailable support has a separate magnitude status and remains blank.

The [percentile-profile report](results/2026-10-09-percentile-effects/REPORT.md)
adds curve points at P0, P10 … P100 and signed adjacent-interval changes, with
separate CSV tables and fold evidence. The runner defaults to `--bins 10` and
reduces estimation resolution before fitting if held-out bins lack observations.
This demonstration selected eight nominal bins. Decile points interpolate those
supported bins; they do not create additional independently estimated detail.
Existing Q25→Q75, magnitude and predictive scores remain reported. Prediction
scores/parameters and unique/joint contributions match the earlier run; finer
ALE resolution can change curves and their effect-size summaries.

The [within-depth demonstration](results/2026-10-09-within-depth-demonstration/REPORT.md)
freezes the existing survey layers and separately assesses peers using elastic
net and GBM on the same twenty samples. It shows additional predictive relations
between Work Demands and Stress, with all nonretained peer comparisons reported.
This exploratory supplement does not add automatic peer scheduling, infer causal
direction, or estimate mediated effects. Its saved reproduction script uses the
existing local-assessment APIs; the core recursive implementation is unchanged.

The [complete-procedure evaluation](results/2026-10-09-procedure-evaluation/REPORT.md)
rebuilds discovery within outer training folds and evaluates retained root
predictors against the training mean and all original candidates. Its unified
survey report separates held-out prediction from the full-data reference network
and root effects. The predefined noise/direct/redundant/U/interaction pilot
preserves unfavorable results, including an interaction selection failure.
Run `python examples/run_procedure_evaluation.py --output NEW_DIRECTORY` and
`python examples/audit_procedure_evaluation.py NEW_DIRECTORY` to reproduce and
recount the evidence. This is development validation, not final P6 calibration.

The [interaction-selection investigation](results/2026-10-10-interaction-investigation/REPORT.md)
replays the failed pilot's root decisions and compares assessment count, local
CV training size, and fresh small/larger interaction examples at unchanged
thresholds. It preserves a noise check and all negative results. Reproduction
and offline audit commands are in the report; no production rule is changed.

The [research-settings reliability report](results/2026-10-10-selection-reliability/REPORT.md)
extends that investigation to 48 independent datasets, 4,800 local assessments
and 48,000 independent test predictions. It preserves one small-sample interaction
miss alongside successful ordinary, redundant and larger interaction examples.
See the [parallel Actions workflow](selection_actions.md) for validated checkpoint
transfer and execution provenance. No selection threshold or production rule was
changed; this remains targeted development evidence rather than final calibration.
