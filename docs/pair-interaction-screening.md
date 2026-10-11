# Bounded interaction explanation

This optional supplement examines a fitted prediction model without creating
product predictors or fitting a model for every pair. It screens pairs cheaply,
then allocates detailed prediction queries to a short list. Individual feature
importance does not filter eligibility: pure interactions can involve variables
with weak individual associations.

The initial implementation measures a **local mixed prediction contrast**:

`f(high A, high B) - f(high A, low B) - f(low A, high B) + f(low A, low B)`.

Other inputs remain at reference observations. Any additive prediction function,
including separate nonlinear curves, cancels in this contrast. A nonzero value
describes nonadditivity in the fitted function, not an established population
interaction. RMS over the contrasts avoids opposite signs cancelling. This is
not a beta, Cohen's d, variance explained, p-value, or a complete second-order
ALE surface. It can include higher-order effects involving other variables.

Default per-model caps:250 pairs;64 reference rows for a two-bin screen;five
detailed pairs using up to256 rows/four bins;100000 total prediction rows.
Minimum three observed reference rows are required in every pair bin cell for
a supported score. Missing/constant predictors, sparse joint support and budget
exhaustion remain unavailable rather than zero. Bin corners can still leave the
exact observed manifold; correlation is not solved by this screen.

Screening uses deterministic sampling, independent of outcome values, and
canonical pair ordering. A pair cap prioritizes names rather than importance;
unexamined pairs are explicitly listed. Shortlisting is computational triage,
not statistical acceptance. Detailed bin resolution differs from screening, so
their scores must not be treated as directly interchangeable.

In engine comparisons the existing outer-fold fitted model is reused, with
held-out reference covariates. No extra models or tuning searches are required.
Fold scores/support are reported with the planned denominator. Overlapping CV
folds do not provide independent replications, stability probabilities or p-values.

Candidate interactions belong in a separate exploratory graph using a junction
fed by both variables and connected to the target. Ordinary graph edges still
mean predictive contribution, with interaction involvement undetermined. This
supplement neither certifies separate main effects nor changes the recursive
network. No marker means only no displayed candidate, not proof of no interaction.

Scientific validation remains pending: false discoveries, screening recall,
held-out value of interactions and support across independent fitted datasets.
Explaining a fitted interaction cannot recover one the learner failed to learn.
The local mixed differences are motivated by the differences used in
[Apley and Zhu's ALE methodology](https://arxiv.org/abs/1612.08468); this bounded
screen deliberately does not claim to implement their full interaction estimator.

## Using the optional workflow

```python
from open.interactions import PairScreenConfig
from open.effects import compare_target_engines, save_engine_comparison
from open.interaction_visualization import save_interaction_report

budget = PairScreenConfig(max_pairs=250, max_details=5,
                        screen_rows=64, detail_rows=256,
                        prediction_rows=100_000)
comparison = compare_target_engines(
    data, target, candidates, config, split_plan,
    engines=("random_forest",), interaction_screen=budget)
save_engine_comparison(comparison, output_directory)
save_interaction_report(comparison, output_directory, plots=True)
```

`plots=True` requires the visualization extra. Omitting `interaction_screen`
preserves the previous effect API and fit counts. The optional screen has zero
additional fits/tuning searches; the effect comparison retains its usual full,
ablation and explanatory fits. Budgets apply separately to each fitted outer-fold
model; multiply the cap by planned folds and engines for an overall upper bound.
Pair enumeration and its complete status ledger remain quadratic in p: prediction
caps do not guarantee constant memory or wall time for arbitrarily large inputs.

Outputs include `interaction_pairs.csv`, `interaction_cells.csv` and
`INTERACTIONS.md`, alongside the versioned comparison JSON. The report draws a
separate candidate-junction view only for pairs with complete coarse support and
at least one supported detailed fold. It ranks them by descriptive mean coarse
RMS and displays at most five, independent of statistical acceptance. Changing
`max_displayed` changes only the report, not which pairs were examined. Fold
shortlist counts are not the network's 70% stability criterion.

Optional heatmaps show the first supported detailed fold in saved order, its
signed cell means, within-cell RMS and sampled support counts. They are not an
averaged interaction surface or a plot of absolute predicted outcomes. Other
folds and boundaries remain in JSON. Zero coarse scores can reflect failure to
learn, cancellation at the chosen resolution or absence of a fitted interaction;
they never establish population absence.

For an already fitted model, call `screen_pair_interactions(model, reference,
candidates, outcome_sd=..., config=budget)`. `candidates` must list all model
features in fitted order; reference columns are aligned to that order. Keep a
separate reference frame without the target. Summaries are unit-scale invariant
for corresponding scaled predictions, but refitting/tuning invariance is not
claimed. Unrepresentable numeric summaries raise an error, not infinite scores.

Use a fresh output directory to preserve old evidence. Saving an opt-out effect
comparison in a previously enabled directory clears the known generated pair
CSVs and Markdown view, preventing an old supplement from describing new data.

## Checked demonstration

[Synthetic analytic functions and RF example](../examples/results/2026-10-11-pair-interactions/REPORT.md)
show pure, mixed and absent fitted interactions. The RF example ranks the known
Support–Stress interaction first, but also shortlists incidental patterns among
other pairs. This demonstrates why the view remains exploratory rather than a
validated population interaction detector.
