# Python API

Import the common workflows from `open.api`:

```python
from open.api import ExploreConfig, explore_network, save_exploration

result = explore_network(data, outcome="Y", config=ExploreConfig(max_depth=3))
print(result.summary())
save_exploration(result, "open-output")
```

## Choose the function for your task

| Function in `open.api` | Purpose and return value | Existing import, still supported |
|---|---|---|
| `explore_network` | Explore a predictive network; returns `ExploreResult` | `open.explore.explore` |
| `confirm_network` | Test frozen links on independent data; returns a confirmation DataFrame | `open.confirm.confirm` |
| `explore_and_confirm` | Split one dataset before exploration and confirmation; returns `SplitResult` | `open.confirm.explore_and_confirm` |
| `evaluate_confirmation_fit` | Cross-validate the frozen network's predictive fit on new data; returns a fit DataFrame | `open.confirm.confirmation_fit` |
| `split_sample` | Split data before analysis; returns exploration and confirmation DataFrames | `open.confirm.split_sample` |
| `plot_network` | Draw the graph; returns a Figure, or a Path when saving to a path | `open.explore_report.plot_explore` |
| `save_exploration` | Save exploration tables, settings, summary and optional graph; returns a Path | `open.explore_report.write_explore` |
| `save_split_analysis` | Save a `SplitResult`, including confirmation and split settings; returns a Path | `open.explore_report.write_split` |

`ExploreConfig`, `ExploreResult`, `SplitResult` and `GraphStyle` are available
from the same entry point. These are direct aliases: all existing arguments,
defaults, exceptions and return types are preserved. There is no deprecation
or migration requirement. Detailed help is shared with the original functions,
so `help(explore_network)` displays `explore` and its existing signature.

## Explore now, confirm on a new sample

```python
from open.api import (
    confirm_network, evaluate_confirmation_fit, explore_network,
    save_exploration,
)

result = explore_network(data, outcome="Y", engine="ols")
confirmation = confirm_network(result, new_sample)
fit = evaluate_confirmation_fit(result, new_sample)
save_exploration(result, "confirmed-output",
                 confirmation=confirmation, confirmation_fit=fit)
```

Use data independent of exploration for both confirmation and confirmation
fit. Shared groups and interactions remain explicitly untested in confirmation.
Predictive fit is a separate cross-validation summary, not another significance
test. See [the method](method.md#confirmation-openconfirmconfirm).

## Split one dataset before analysis

```python
from open.api import explore_and_confirm, save_split_analysis

split = explore_and_confirm(data, outcome="Y", split_seed=1)
print(split.summary())
save_split_analysis(split, "split-output")
```

`save_split_analysis` saves an already computed `SplitResult`; the splitting
happens in `explore_and_confirm`. Splitting is intended for samples of about
1,000 rows or more; see the [README](../README.md#confirming-links) for context.

## Plotting and saving

```python
from open.api import GraphStyle, plot_network, save_exploration

plot_network(result, "network.png", style=GraphStyle(spacing=.8))
save_exploration(result, "tables-only", plot=False)
```

`plot_network` draws only the graph. The save functions write tables, settings
and summary, plus a graph by default. Reusing a directory follows the
[documented overwrite behavior](method.md#saving-and-overwriting-exports).
Importing `open.api` and saving with `plot=False` require neither matplotlib
nor LightGBM. Plotting requires the visualization extra; selecting LightGBM
requires its extra. Construct `GraphStyle` when plotting is needed: its color
validation also requires matplotlib.
