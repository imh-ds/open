# OPEN

**Outcome-centered Predictive Expansion Network.** OPEN builds a network of
predictors around one outcome, one depth at a time:

1. Find which variables predict the outcome reliably better than noise (depth 1).
2. Treat each of those as a new target and find what predicts it, from the
   variables not yet placed (depth 2), and so on.

Each link shows how consistently it beats noise across 10 cross-validation
folds, and how much held-out R² it carries. The result is a discovery-stage,
hypothesis-generating map. Links can then be confirmed on independent data.

## Install

Clone the repository and install the core package:

```bash
git clone https://github.com/imh-ds/open.git
cd open
python -m pip install .
```

Requires Python 3.11 or later. Install optional dependencies only when needed:

```bash
python -m pip install ".[visualization]"  # plotting
python -m pip install ".[lightgbm]"       # LightGBM engine
```

## Quick start

Load a numeric dataset with one row per observation. `analyze` splits the rows,
discovers the network on one part, and confirms selected links on the other:

```python
import pandas as pd
from open.api import DiscoveryConfig, analyze, save_split

data = pd.read_csv("my_study.csv")
result = analyze(
    data,
    outcome="Y",
    predictors=["X1", "X2", "X3"],
    config=DiscoveryConfig(max_depth=3),
    split_seed=42,
)
print(result.summary())
save_split(result, "open-output", plot=False)
```

The outcome must be numeric and complete. For details and runnable examples,
see the [OPEN field guide](docs/index.html) or its [Markdown entry point](docs/quickstart.md).
The [API guide](docs/api.md) covers other workflows and the longer function names,
which remain supported.

**Reading the output.** The graph is a radial network: the outcome sits in
the centre, and each target's predictors branch outwards from it. Node colour
shows the depth, and grey dashed nodes are predictors that were only flagged.

- **Solid line (kept):** the predictor beat the noise bar in at least 80% of
  folds. It becomes a target at the next depth.
- **Dotted line (flagged):** it beat the noise bar in 40–79% of folds. Treat it
  as a weak lead. It returns to the candidate pool.
- **R² lost:** held-out R² lost when the predictor is left out, which is its
  unique predictive contribution. **R² above noise:** R² lost minus the noise
  bar.
- **Edge labels:** a solid edge is labelled with its held-out ΔR² (R² lost)
  and the folds it passed, for example ".120 (9/10)", with ✓ when it was
  confirmed on new data. R² is printed APA style (three decimals, no leading
  zero). Every value is in the tables.
- **Blue band:** correlated predictors that matter together (a shared group).
  **Orange diamond:** a pairwise interaction. Solid lines from the two variables
  meet in the diamond (labelled with the pair, e.g. "F·N7"); the line on to the
  target is solid if the interaction is kept and dashed if flagged. Flagged
  interactions are listed in the tables but not drawn by default, because most
  were false in validation.
- **Arrows show prediction, not causation.** An arrow X → T means X predicts T
  on held-out data, beyond T's other candidates. It shows the direction of
  prediction in this analysis, not of causation: reversed arrows can be
  predictive too, for example a noisy measure predicting its source. It is not a
  partial correlation either. Depth is predictive distance from the outcome,
  not a causal order.
- **Tan nodes** are covariates (see below), and the footnote lists any
  controls.
- **Network fit (green ring around each target):** the filled share of the
  ring is the held-out R² of the target from its solid lines, also printed
  below it. With confirmation, the R² cross-validated on the new data is shown
  too ("new .29"). This is OPEN's counterpart of a fit index (see
  [docs/method.md](docs/method.md)).
- **The fold tally is an exploratory rule, not a p-value.** Formal tests come
  from confirmation on independent data (below).

**Adjusting the graph.** Every display setting is a `GraphStyle` field. Pass
the fields as keywords to `plot_discovery`, `save_discovery` or
`save_split`, or
pass a `style=GraphStyle(...)`. Sizes are multipliers of the defaults:

```python
from open.api import GraphStyle, plot_discovery

plot_discovery(result, "network.png",
             spacing=.7,           # < 1 more compact, > 1 more spread out (nodes never overlap)
             node_size=1.2,        # circles, fit rings and interaction diamonds
             edge_width=.8,        # thickness of every edge
             label_size=1.1,       # all text; this is now the default
             abbreviate_node_labels=False, # keep the full words in node names
             wrap_node_labels=False, # keep each node name on one line
             show_flagged=False,   # hide dashed (flagged) links and flagged-only nodes
             edge_labels="kept",   # "kept", "all" or None
             label_outline="white",       # colour of the solid outline around label text
             label_outline_width=5)       # its thickness in points; 0 for none
```

By default, graph node labels use Title Case, shorten words longer than 10 characters at a
consonant boundary, then wrap at word boundaries to a 10-character line width.
All words remain visible, and the full variable names remain in result tables.
Use `format_node_label(label, max_word_length=10, max_line_length=10,
abbr_length=5)` to apply this formatting to a string. Set
`abbreviate_node_labels=False` to keep full words and `wrap_node_labels=False`
to keep them on one line. `label_size` defaults to 1.1, making all graph text
slightly larger. `show_flagged_interactions=True` also draws flagged
interactions, and `dpi` sets the image resolution.

See [docs/method.md](docs/method.md) for the full procedure and its settings.

## Engines

| Engine | Use it for |
|---|---|
| `ols` | Typical samples (N 300–1,000) with roughly straight-line or monotone effects |
| `elastic_net` | The same, with many or correlated predictors |
| `lightgbm` | N of about 1,000 or more, or when effects may be U-shaped or otherwise curved |

Every engine offered was validated in simulation
([docs/validation-evidence.md](docs/validation-evidence.md)). Random forest and
gradient boosting were evaluated and dropped ([docs/engines.md](docs/engines.md)).

## Controls and covariates

Variables that should be in every model, at every depth, whatever they
predict, go in one of two arguments of `discover` or `analyze`:

```python
result = discover(data, outcome="Y", engine="ols",
                 controls=["cohort"],      # adjust only: never tested, drawn or used as a target
                 covariates=["gender"])    # also tested and drawn (tan), but never a target
```

- **`controls`:** nuisance variables you only want to adjust for. Every link
  is then "beyond the controls"; the graph's footnote lists them.
- **`covariates`:** variables you want adjusted for *and* reported, such as
  known group differences. They are tested against the noise bar for every
  target, drawn when they pass, and never become targets themselves.

Without either, a variable such as age is an ordinary candidate. If it is
kept, it becomes a target ("what predicts age?") and leaves the pools of
deeper depths. Confirmation adjusts for the same controls and covariates.
Dummy-code categorical variables first.

## Confirming links

Discovery and confirmation must use different data. To confirm, either split
one dataset before any analysis or use a new sample:

**One dataset (N of about 1,000 or more):** split once, discover on one part and
confirm on the other.

```python
from open.api import analyze, save_split

split = analyze(data, outcome="Y", engine="ols", split_seed=1)
print(split.summary())
save_split(split, "open-output")      # adds confirmation.csv; the graph marks confirmed links
```

**A new sample:** discover on all of your current data, then confirm on the new
sample.

```python
from open.api import confirm, discover

result = discover(data, outcome="Y", engine="ols")
table = confirm(result, new_sample)     # Holm-adjusted p-values for the frozen links
```

`confirm` tests only the links found during discovery, adjusting for the same
candidate pool. It uses OLS + HC4 (calibrated from N ≥ 275) or LightGBM's
combined GCM/PCM test (calibrated from N ≥ 300).

## Multiple outcomes

Use one call to coordinate separate networks on the same row split:

```python
from open.api import analyze, plot_outcomes, save_batch

batch = analyze(data, outcomes=["Y1", "Y2"],
                predictors=[f"X{i}" for i in range(1, 11)], split_seed=42)
y1 = batch.by_outcome["Y1"]
plot_outcomes(batch, path="outcome-panels.png")
plot_outcomes(batch, layout="combined", path="outcome-overlay.png")
save_batch(batch, "multi-output", plot=False)
```

Outcomes must be numeric and complete. Each network retains its own predictor
pools and recursion; matching eligible rows share folds. Confirmation adjusts
across all outcome networks by default. The combined graph overlays separate
predictive analyses and does not estimate correlated outcome errors. See the
[multi-outcome API guide](docs/api.md#several-outcomes-in-one-call) for correction
scopes, exact reuse, result tables, and export behavior.
Plotting requires the `visualization` extra described above.

## Reproducing the evidence

[docs/reproducing.md](docs/reproducing.md) lists each study, the command or
GitHub Actions workflow that runs it, pinned package versions, and the release
holding its raw outputs.

## Citation

See [CITATION.cff](CITATION.cff). Development history is archived in
[imh-ds/open-archive](https://github.com/imh-ds/open-archive)
([docs/provenance.md](docs/provenance.md)).

See [CHANGELOG.md](CHANGELOG.md) for developer-preview release notes.

## License

GPL-3.0. See [LICENSE](LICENSE).
