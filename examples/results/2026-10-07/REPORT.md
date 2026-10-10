# OPEN synthetic sample demonstration

Three deterministic 80-row fixtures; 20 development resamples per fixture by default.
Each resample uses 80% of rows without replacement, two outer folds, two inner folds,
maximum depth 3, unique relevance threshold 0.15, and consensus threshold 0.70.
Settings are fixed before running; no thresholds were tuned to these outputs.

These fixtures demonstrate software behavior. They do not establish calibrated
scientific performance, real-data validity, or causal direction. Edge probabilities
are empirical frequencies across the completed whole-pipeline runs.

![Overview of the measured consensus graphs](overview.png)

## chain

Completed: 20; failed: 0; structural denominator: 20.

Engine revision: `8daca3f2cca13a9fed6ea0f1c1d254f3fec30805`.

![Saved-bundle consensus network](chain/network.png)

### One complete-data-size run

| Target | Baseline MSE | Full MSE | Relative gain | Observed / missing target rows |
|---|---:|---:|---:|---:|
| Anxiety | 1.2483 | 0.0594 | 95.2% | 80 / 0 |
| Depression | 1.2488 | 0.3812 | 69.5% | 80 / 0 |

### Whole-pipeline results

| Relation | Count / completed runs | Probability |
|---|---:|---:|
| Depression -> Anxiety (PROXIMAL_UNIQUE) | 20 / 20 | 100% |
| Stress -> Anxiety (SCREENED_BY_PARTNER) | 20 / 20 | 100% |
| Stress -> Depression (PROXIMAL_UNIQUE) | 20 / 20 | 100% |
| Stress -> Depression -> Anxiety (complete unique-edge path) | 20 / 20 | 100% |

Interpretation: the generated fixture has Stress feeding Depression, then Anxiety.
Inspect the single-run classifications: Stress can predict Anxiety marginally but is
screened when Depression is available; it remains eligible and is evaluated as a
predictor of Depression at the next layer. The table reports how often the entire
unique-edge path appeared across independently rebuilt networks, rather than a
product of separate edge probabilities.

Captured numerical warnings: `{}`. Full messages/counts are
retained in [chain/warnings.json](chain/warnings.json); none were silently discarded.

## chain_missing

Completed: 20; failed: 0; structural denominator: 20.

Engine revision: `8daca3f2cca13a9fed6ea0f1c1d254f3fec30805`.

![Saved-bundle consensus network](chain_missing/network.png)

### One complete-data-size run

| Target | Baseline MSE | Full MSE | Relative gain | Observed / missing target rows |
|---|---:|---:|---:|---:|
| Anxiety | 1.2483 | 0.0628 | 95.0% | 80 / 0 |
| Depression | 1.2637 | 0.3840 | 69.6% | 79 / 1 |

### Whole-pipeline results

| Relation | Count / completed runs | Probability |
|---|---:|---:|
| Depression -> Anxiety (PROXIMAL_UNIQUE) | 20 / 20 | 100% |
| Stress -> Anxiety (SCREENED_BY_PARTNER) | 20 / 20 | 100% |
| Stress -> Depression (PROXIMAL_UNIQUE) | 20 / 20 | 100% |
| Stress -> Depression -> Anxiety (complete unique-edge path) | 20 / 20 | 100% |

Interpretation: the generated fixture has Stress feeding Depression, then Anxiety.
Inspect the single-run classifications: Stress can predict Anxiety marginally but is
screened when Depression is available; it remains eligible and is evaluated as a
predictor of Depression at the next layer. The table reports how often the entire
unique-edge path appeared across independently rebuilt networks, rather than a
product of separate edge probabilities.

Depression is missing in sample row 0. When Depression becomes a target,
its risk uses only observed Depression values within the existing folds.
In resamples that omit row 0, no target rows need exclusion. No target value
is imputed. This is the formerly blocked recursive case.

Captured numerical warnings: `{}`. Full messages/counts are
retained in [chain_missing/warnings.json](chain_missing/warnings.json); none were silently discarded.

## shared_predictors

Completed: 20; failed: 0; structural denominator: 20.

Engine revision: `8daca3f2cca13a9fed6ea0f1c1d254f3fec30805`.

![Saved-bundle consensus network](shared_predictors/network.png)

### One complete-data-size run

| Target | Baseline MSE | Full MSE | Relative gain | Observed / missing target rows |
|---|---:|---:|---:|---:|
| Outcome | 0.9385 | 0.0200 | 97.9% | 80 / 0 |

### Whole-pipeline results

| Relation | Count / completed runs | Probability |
|---|---:|---:|
| {Scale_A, Scale_B} -> Outcome (shared) | 20 / 20 | 100% |

Interpretation: the two scales are near copies. Individually removing either
loses little prediction because the other remains; joint removal loses the
shared predictive signal. Both members can therefore be promoted together.
The group is a predictive relation and display object, not a synthetic target.

Captured numerical warnings: `{'ConvergenceWarning': 268}`. Full messages/counts are
retained in [shared_predictors/warnings.json](shared_predictors/warnings.json); none were silently discarded.

## Reproduce

Run from the developed checkout with the visualization extra installed:

```powershell
python examples/run_synthetic_demo.py --output NEW_DIRECTORY --resamples 20
```

The runner refuses to overwrite an existing directory. Each case saves sample data,
configuration, a single-run diagnostic, the six-file OPEN bundle, and PNG/SVG plots.
summary.json records hashes, seeds, code revision, and exact structural denominators.
Convergence warnings in collinear tuning candidates can occur; a completed batch is
an execution result, not a claim that every candidate fit converged.
