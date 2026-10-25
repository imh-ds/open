# Python API

Import the common workflows from `open.api`. Short names are available for the
most common operations:

```python
from open.api import ExploreConfig, analyze, explore, save_split

result = analyze(data, outcome="Y", config=ExploreConfig(max_depth=3))
print(result.summary())
save_split(result, "open-output", plot=False)
```

## Choose the function for your task

| Function in `open.api` | Purpose and return value | Descriptive name, still supported |
|---|---|---|
| `explore` | Explore a predictive network; returns `ExploreResult` | `explore_network` |
| `confirm` | Test frozen links on independent data; returns a confirmation DataFrame | `confirm_network` |
| `evaluate_fit` | Cross-validate the frozen network's predictive fit on new data; returns a fit DataFrame | `evaluate_confirmation_fit` |
| `analyze` / `analyse` | Split data, explore and confirm; returns `SplitResult` for one outcome or `MultiOutcomeResult` for several | `explore_and_confirm` / `explore_and_confirm_outcomes` |
| `split_sample` | Split data before analysis; returns exploration and confirmation DataFrames | `open.confirm.split_sample` |
| `plot_network` | Draw the graph; returns a Figure, or a Path when saving to a path | `open.explore_report.plot_explore` |
| `save_exploration` | Save exploration tables, settings, summary and optional graph; returns a Path | `open.explore_report.write_explore` |
| `save_split` | Save a `SplitResult`, including confirmation and split settings; returns a Path | `save_split_analysis` |
| `plot_outcomes` | Draw radial facets or a combined outcome overlay; returns a Figure or Path | `open.explore_report.plot_outcomes` |
| `save_batch` | Save all outcome analyses and aggregate tables; returns a Path | `save_multi_outcome_analysis` |

`analyze` accepts exactly one of `outcome` or `outcomes`. A string outcome runs
the single-outcome workflow. A one-item list also runs that workflow; a list
with multiple outcome names runs the coordinated multi-outcome workflow:

```python
single = analyze(data, outcome="Y")
several = analyze(data, outcomes=["Y1", "Y2"])
british_spelling = analyse(data, outcomes=["Y1", "Y2"])
```

The longer workflow functions remain available for options specific to one
workflow, including `include_flagged` and `correction_scope`. The shorter names
are aliases or convenience wrappers; existing imports and behavior remain
supported. `ExploreConfig`, `ExploreResult`, `SplitResult`, `MultiOutcomeResult`
and `GraphStyle` are also available from `open.api`.

## Explore now, confirm on a new sample

```python
from open.api import (
    confirm, evaluate_fit, explore, save_exploration,
)

result = explore(data, outcome="Y", engine="ols")
confirmation = confirm(result, new_sample)
fit = evaluate_fit(result, new_sample)
save_exploration(result, "confirmed-output",
                 confirmation=confirmation, confirmation_fit=fit)
```

Use data independent of exploration for both confirmation and confirmation
fit. Shared groups and interactions remain explicitly untested in confirmation.
Predictive fit is a separate cross-validation summary, not another significance
test. See [the method](method.md#confirmation-openconfirmconfirm).

## Split one dataset before analysis

```python
from open.api import analyze, save_split

split = analyze(data, outcome="Y", split_seed=1)
print(split.summary())
save_split(split, "split-output")
```

`save_split` saves an already computed `SplitResult`; the splitting
happens in `analyze`. Splitting is intended for samples of about
1,000 rows or more; see the [README](../README.md#confirming-links) for context.

## Several outcomes in one call

```python
import pandas as pd
from open.api import (
    ExploreConfig, analyze, plot_outcomes, save_batch,
)

data = pd.read_csv("my_study.csv")
predictors = [f"X{i}" for i in range(1, 11)]
batch = analyze(
    data,
    outcomes=["Y1", "Y2"],
    predictors=predictors,
    config=ExploreConfig(max_depth=3, seed=42),
    split_seed=42,
    correction="holm",
    correction_scope="all_outcomes",
)
print(batch.summary())
y1 = batch.by_outcome["Y1"]       # ordinary SplitResult for this outcome
print(batch.confirmation)         # aggregate table, labeled root_outcome
print(batch.fit)                  # separate fit estimates, labeled root_outcome
plot_outcomes(batch, path="outcome-panels.png")
plot_outcomes(batch, layout="combined", path="outcome-overlay.png")
save_batch(batch, "multi-output", plot="facets")
```

All selected outcomes must be numeric and have no missing values. The data is
split once, and all outcomes use the same exploration configuration and folds.
LightGBM confirmation and confirmation-fit folds align whenever the eligible
complete rows match; predictor
missingness can produce different eligible row sets. Model random streams remain
target-specific. Duplicate DataFrame index labels do not define row identity.

Each outcome has its own candidate pool, promoted variables, depths, parents,
and stop reasons. With `predictors=None`, every selected outcome, control, and
covariate is excluded from the ordinary predictor pool. Explicit predictors
cannot include any selected outcome. A target assessment is reused within the
call only when its entire modeling context matches, including target, ordered
pool, rows and values, folds, depth, controls, covariates, engine and config.
The cache does not persist across calls. Different pools can still give the same
upstream variable different effects.
Each contained exploration exposes `assessment_ids`, keyed by `(depth, target)`.
An ID identifies one assessment within this batch; IDs are not comparable across
separate calls. Scalar explorations leave this mapping empty.

`correction_scope="all_outcomes"` adjusts the raw testable p-values across all
outcome networks by default. `"per_outcome"` adjusts each network separately;
those p-values do not represent adjustment across the whole batch. Untestable
rows retain their notes and missing p-values. Holm is the default. Selecting BH
does not establish FDR control for arbitrary dependence among outcomes.
If a batch target has no complete rows, a constant outcome, or too few rows
for its confirmation-fit folds, its fit row retains the sample size and a
`note` explaining why fit was not evaluated, with missing fit values. Unsupported
LightGBM test fold plans likewise retain untested links with a reason.

Facets retain a separate root and depth layout for each outcome. The combined
view uses the same radial tree and spring layout, node styling, fit rings, and
annotated edges as the single-network plot, with multiple outcome roots. It
draws shared variables once and retains outcome colors for overlapping or
reversed edges. Links from the same reused assessment with identical effects and
confirmation status draw once in neutral grey, with the ordinary label; the
legend and footnote explain the dark grey edges, and the tables list which outcomes
share each link. Independent
assessments stay separate even when their printed effect labels match. Multiple
distinct links between a node pair fan into curves with separate triangle
arrowheads; a lone link remains straight. Differing fit estimates are labeled by
outcome rather than averaged. Group and interaction junctions retain their relation types.
It overlays separate predictive analyses; it does not fit a joint model,
estimate cross-outcome residual covariance, or establish causal direction.

`save_batch` defaults to `plot=False`. It writes aggregate
`outcomes-confirmation.csv`, `outcomes-fit.csv`, `outcomes-settings.json`, and
`outcomes-summary.txt`, plus an optional `outcomes-network.png`. Each outcome
has a safe generated folder under `outcomes/`; the settings file maps the exact
original labels to those folders. Folder names include a digest of the exact
label and remain stable when outcomes are reordered. Per-outcome batch
`links.csv` files include `assessment_id`,
and `outcomes-settings.json` maps these IDs to their full outcome memberships
under `assessment_outcomes`. Each outcome's complete links and confirmation
rows remain separate in the exports. Reusing a directory
replaces managed files
and removes a stale aggregate plot when plotting is disabled. Unrelated files
and outcome directories from earlier batches remain. Publication recovers per
outcome directory and per aggregate file set, so a later failure can leave
earlier outcome directories updated.

## Plotting and saving

```python
from open.api import GraphStyle, plot_network, save_exploration

plot_network(result, "network.png", style=GraphStyle(spacing=.8))
save_exploration(result, "tables-only", plot=False)
```

`plot_network` draws only the graph. The save functions write tables, settings
and summary, plus a graph by default. Reusing a directory follows the
[documented overwrite behavior](method.md#saving-and-overwriting-exports).
Nodes are also placed adaptively, to make room for labels rather than move
them away. After the initial radial/spring layout, a bounded local search
repositions crowded nodes and interaction junctions to clear nearby nodes, fit
rings and unrelated edges. Then each label is sized from its text and given a
spot on its own edge (or beside its own node). Where a label would still
collide, only the nodes involved move: an edge too short for its label is
lengthened, edges whose labels collide are pushed apart and outwards, a node
in a label's way steps aside, and edges that leave a node in nearly the same
direction are spread so they do not run on top of each other. Quiet parts of
the graph keep their place, and the result is deterministic. The combined view
uses the visible graph, without its invisible layout root. Set
`GraphStyle(dodge_nodes=False)` to use the initial layout alone.

A final pass checks each label against the rendered text, nodes and fit
rings. An edge label stays on its own edge, wherever along it is free; only
when every spot on the edge is blocked does it move beside the edge, with a
thin leader line. Node-fit labels stay next to their own node. This pass does
not move nodes or resize the figure. Very dense graphs can still need more
space or fewer displayed annotations. Set `GraphStyle(dodge_labels=False)` to
skip it.

A checkmark means the link passed confirmation on independent held-out data
after the selected multiple-testing correction. Solid/dashed lines instead
describe exploratory fold stability. No checkmark can mean either unconfirmed
or untested; the confirmation table distinguishes them. To hide the symbols
without changing any tests or results:

```python
plot_outcomes(batch, layout="combined", style=GraphStyle(show_confirmation=False))
```

Importing `open.api` and saving with `plot=False` require neither matplotlib
nor LightGBM. Plotting requires the visualization extra; selecting LightGBM
requires its extra. Construct `GraphStyle` when plotting is needed: its color
validation also requires matplotlib.
