# OPEN quick tutorial

This tutorial creates a small synthetic study, explores one outcome, then runs
the same workflow for two outcomes. It uses OLS and does not require a plotting
or LightGBM installation. The examples save tables and settings; plotting is
shown as an optional final step.

## Install

From a clone of the repository, install OPEN and its core dependencies:

```bash
python -m pip install .
```

To save or display network figures, install the plotting extra:

```bash
python -m pip install ".[visualization]"
```

## Make example data

The data has one row per observation, five numeric predictors, and two numeric
outcomes. Replace this block with `pd.read_csv("my_study.csv")` to use your own
data.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1_200
x = rng.normal(size=(n, 5))
data = pd.DataFrame(x, columns=[f"X{i}" for i in range(1, 6)])
data["Y1"] = 1.0 * x[:, 0] + 0.7 * x[:, 1] + rng.normal(size=n)
data["Y2"] = 0.8 * x[:, 0] - 0.6 * x[:, 2] + rng.normal(size=n)
predictors = [f"X{i}" for i in range(1, 6)]
```

The selected outcome columns must be numeric and complete. Predictor missing
values can be imputed by the models. Categorical variables should be
dummy-coded first.

## Analyze one outcome

`analyze` splits the rows before exploration. It builds the network on the
exploration portion, then tests the frozen links on the independent confirmation
portion. This example limits depth and disables interaction screening to keep
the tutorial run modest:

```python
from open.api import ExploreConfig, analyze, save_split

config = ExploreConfig(folds=5, max_depth=2, interactions=False, seed=42)
single = analyze(
    data,
    outcome="Y1",
    predictors=predictors,
    config=config,
    split_seed=42,
)

print(single.summary())
save_split(single, "open-single-output", plot=False)
```

The output directory contains the explored links and nodes, fold-level
assessments, settings, a text summary, and the confirmation table. A link marked
confirmed passed the held-out test after multiple-testing correction. A link
that was not confirmed remains in the exploratory results, but should not be
described as independently confirmed.

## Analyze several outcomes

Use `outcomes` to let the same `analyze` function coordinate separate networks
for multiple outcomes. The workflow uses one row split and shares upstream
assessments only when their complete modeling context matches. Each outcome
still has its own predictors and recursion.

```python
from open.api import analyze, save_batch

batch = analyze(
    data,
    outcomes=["Y1", "Y2"],
    predictors=predictors,
    config=config,
    split_seed=42,
)

print(batch.by_outcome["Y1"].summary())
print(batch.confirmation.head())
save_batch(batch, "open-multiple-output")
```

If you installed the visualization extra, plot the separate networks as panels
or together in one graph:

```python
from open.api import plot_outcomes

plot_outcomes(batch, path="outcome-panels.png")
plot_outcomes(batch, layout="combined", path="outcome-combined.png")
```

The combined view overlays independently estimated predictive networks. It is
not a joint outcome model and does not estimate correlations among outcome
errors. OPEN's arrows describe prediction, not causal direction. See the
[method guide](method.md) for how the exploration, confirmation, and graph
labels work, and the [API guide](api.md) for advanced options and export details.
