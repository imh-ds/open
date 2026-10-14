# open
Outcome-centered Predictive Expansion Network

See the [research-preview release scope and open gates](docs/research-preview-readiness.md).

## Development usage

OPEN is a development package. The current configuration contract covers one
cross-sectional analysis with independent rows, a continuous outcome using MSE,
numeric continuous or ordinal scale-level predictors, and one fixed seeded
engine: elastic net (default), random forest, GBM, or optional LightGBM.
The current prototype includes analysis and visualization
workflows; these are software implementations, not evidence of scientific
validity.

For reproducible chain, missing-value recursion, and shared-predictor examples,
see the [synthetic demonstration](examples/README.md).

See [engines and comparable predictive effect sizes](docs/engines-and-effect-sizes.md)
for standardized ALE contrasts/curves, matched cross-validated R², and unique or
joint ΔCV R². These shared measures describe predictions; elastic-net coefficients
remain an additional engine-specific output.

See [complete-procedure evaluation and the researcher report](docs/procedure-evaluation.md)
for training-isolated discovery, held-out root prediction, and a unified report
that keeps descriptive network/effect evidence distinct from generalization.

After installing the package in a Python 3.11+ environment, load and inspect the
example configuration:

```python
from open.config import load_config

config = load_config("configs/default.yaml")
print(config.analysis.root, config.performance_cv.folds)
```

The shipped configuration uses `stability.scope: layer_consensus`. OPEN repeats
one fixed candidate comparison for the root, finalizes its retained unique and
joint predictors, then repeats the same process for the next layer. Every evaluated
target gets the planned number of assessments. A failed or infeasible assessment
leaves that layer unfinished; the finalized prefix and failure evidence remain
available. Valid low-gain fits count as completed nonselections.

To render its saved seven-file bundle, install `pip install -e ".[visualization]"`:

```python
from open.visualization import load_visualization_data, plot_open_network

bundle = load_visualization_data("path/to/bundle")
figure = plot_open_network(bundle)
figure.savefig("open-network.png", dpi=150)
```

The main graph shows finalized depths and retained unique arrows or joint group
connections, labeled selected/planned. Frequencies describe stability **conditional
on the earlier finalized layers**, rather than confidence or whole-path frequency.
Widths use signed median loss reduction relative to each target's baseline, clipped
only for drawing. Raw values and measurement counts remain in the evidence panel.
To change retention, reconstruct the analysis with another threshold; filtering an
image cannot evaluate the different downstream candidate sets.

`fit_layer_sensitivity` reconstructs thresholds .60, .70 and .80 independently.
Local cross-validation performance is conditional on discovered candidate sets;
it does not evaluate the complete discovery procedure on untouched data.

Omitted scope preserves the historical `whole_pipeline` API. Its six-file bundles
support exploratory and consensus views, selected/eligible diagnostics, and an
optional `min_frequency` display filter. Those whole-pipeline frequencies answer
a different question and must not be relabeled as layer-conditional frequencies.
See the historical [display comparison](examples/results/2026-10-07-survey-exploratory/REPORT.md).

Set `analysis.root` to the outcome column and set `analysis.random_seed` explicitly.
Both performance and tuning cross-validation require at least two folds. The
root outcome must be fully observed. When a predictor becomes a recursive target,
rows missing that target are excluded from each original training and test fold;
target values are never imputed. All models compared for that target use the same
observed rows, while predictor imputation remains training-fold local. If any
observed training partition cannot support tuning or an observed test partition
is empty, the current layer remains incomplete in layer mode; planned sibling assessments still finish. Historical whole-pipeline mode stops that branch and continues others. Exported local
target rows include observed and missing outcome counts; the run ledger records
per-target stop reasons. Risks for different targets may use different row subsets.

Thresholds are configurable engineering defaults, not calibrated scientific
constants. An OPEN relation represents predictive contribution under a stated
learner, candidate set, loss, splits, and thresholds; it does not establish
causal direction. Fitted preprocessing, dependence estimation, and tuning must
use training folds only. This prototype has not been validated for methodological
performance, simulation calibration, or applied use.
