# OPEN

**Outcome-centered Predictive Expansion Network.** OPEN builds a network of
predictors around one outcome, one depth at a time:

1. Find which variables predict the outcome reliably better than noise (depth 1).
2. Treat each of those as a new target and find what predicts it, from the
   variables not yet placed (depth 2), and so on.

Each link shows how consistently it beats noise across 10 cross-validation
folds, and how much held-out R² it carries. The result is an exploratory,
hypothesis-generating map. Links can then be confirmed on independent data.

## Install

```bash
pip install "open-predictive-expansion[lightgbm,visualization] @ git+https://github.com/imh-ds/open"
```

Requires Python 3.11 or later. LightGBM and matplotlib are optional extras.

## Quick start

```python
import pandas as pd
from open.explore import ExploreConfig, explore
from open.explore_report import write_explore

data = pd.read_csv("my_study.csv")            # numeric columns, one row per person
result = explore(data, outcome="Y", engine="ols", config=ExploreConfig(max_depth=3))
print(result.summary())
write_explore(result, "open-output")           # tables plus network.png
```

**Reading the output:**

- **Solid line (kept):** the predictor beat the noise bar in at least 80% of
  folds. It becomes a target at the next depth.
- **Dotted line (flagged):** it beat the noise bar in 40–79% of folds. Treat it
  as a weak lead. It returns to the candidate pool.
- **R² lost:** held-out R² lost when the predictor is left out, which is its
  unique predictive contribution. **R² above noise:** R² lost minus the noise
  bar.
- **Blue box:** correlated predictors that matter together (a shared group).
  **× marker:** a pairwise interaction.

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

## Confirming links

Exploration and confirmation must use different data. To confirm, either split
one dataset before any analysis or use a new sample:

```python
from open.confirm import confirm

explore_half = data.sample(frac=0.5, random_state=1)
confirm_half = data.drop(explore_half.index)
result = explore(explore_half, outcome="Y", engine="ols")
table = confirm(result, confirm_half)           # Holm-adjusted p-values for the frozen links
```

`confirm` tests only the links found by the exploration, adjusting for the same
candidate pool. It uses OLS + HC4 (calibrated from N ≥ 275) or LightGBM's
combined GCM/PCM test (calibrated from N ≥ 300).

## Reproducing the evidence

[docs/reproducing.md](docs/reproducing.md) lists each study, the command or
GitHub Actions workflow that runs it, pinned package versions, and the release
holding its raw outputs.

## Citation

See [CITATION.cff](CITATION.cff). Development history is archived in
[imh-ds/open-archive](https://github.com/imh-ds/open-archive)
([docs/provenance.md](docs/provenance.md)).

## License

GPL-3.0. See [LICENSE](LICENSE).
