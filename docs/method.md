# The OPEN method

## Exploration (`open.explore.explore`)

**1. Folds.** The rows are split into `folds` (default 10) once, with a fixed
seed. Every regression in the network uses the same folds.

**2. Each regression.** For a target and its candidate pool, in every fold:

- One **noise column** is added per candidate: a shuffled copy, with the same
  distribution and no relationship to anything.
- The engine is tuned and fitted on the training folds.
- On the held-out fold, everything is measured in R² units relative to the
  mean-only model:
  - **unique:** R² lost when one column is left out (LOCO);
  - **broad:** R² gained by a column on its own;
  - **group:** R² lost when correlated candidates are left out together.
- Every comparison reuses the full model's tuned settings, so real and noise
  columns are treated identically.

**3. Noise bar and tiers.**

- **Noise bar:** the `noise_quantile` (default 95th percentile) of the noise
  columns' values, pooled over the folds.
- **Passing a fold:** a real column passes when its value is above the bar.
  A predictor with no effect passes about 5% of folds, whatever the number of
  candidates.

| Share of folds passed | Tier | Graph |
|---|---|---|
| ≥ `keep` (80%) | Kept | Solid line |
| ≥ `flag` (40%) | Flagged | Dotted line |
| Lower | Dropped | None |

**4. Hide floor.** A kept or flagged relation whose median R² is below
`hide_below` (default .001, about r = .03) is marked hidden. Hidden relations
are not drawn and promote nothing. Set `hide_below=None` to turn this off.

**5. Correlated predictors** (|Spearman| ≥ `dependence_pair`, default .70).
Each pair is classed per fold, then tallied into the same tiers:

- **both proximal:** each passes on its own;
- **partner screening:** one passes, and the other only helps alone, so it is
  screened by its partner;
- **shared redundant:** neither passes alone, but dropping both does.

A shared group counts only when no member is kept alone. A kept group promotes
all its members.

**6. Interactions.** Each kept or flagged source is paired with every other
candidate.

- **Gain:** the held-out R² gained by the pair's joint term (a centered
  product plus a spline tensor, with its own cross-validated ridge penalty)
  over an additive spline model.
- **Noise comparison:** each real pair is compared against a pair with one
  member replaced by its noise twin, using the same bar and tiers.
- **Drawing only:** interactions promote nothing. Kept interactions are drawn.
  Flagged interactions stay in the tables but are left out of the graph by
  default, because in the validation study most of them were false (1.1–1.4
  per dataset). Draw them with `show_flagged_interactions=True`.

**7. Recursion.**

- Kept variables, and members of kept groups, become the next depth's targets.
- They are regressed on every variable not yet placed, including flagged and
  dropped ones. Variables at the same depth never predict each other.
- The recursion stops when nothing is kept, the pool is empty, or `max_depth`
  is reached.

**Missing values.** Rows with a missing outcome are not allowed. Missing
predictor values are median-imputed within each training fold. Rows with a
missing deeper target are left out of that target's folds.

## Confirmation (`open.confirm.confirm`)

Testing links on the data that selected them inflates their significance, so
confirmation always uses independent data:

- **One dataset:** split it at random before any analysis (50/50 or 60/40), and
  record the seed. Explore on one part and confirm on the other. This suits
  N ≥ about 1,000; below that, both parts lose power.
- **Smaller samples:** explore on all the data and confirm in a new sample.

**In code:**

- `open.confirm.split_sample(data, explore_fraction, seed)` returns the two
  parts.
- `open.confirm.explore_and_confirm(data, outcome, ..., split_seed=...)` runs
  the whole procedure and returns the exploration, the confirmation table and
  the split settings.
- `open.explore_report.write_split` saves them.
- With a new sample, call `open.confirm.confirm(result, new_sample)`.

**Procedure:**

1. **Freeze.** The exploratory network's kept links (or kept and flagged, with
   `include_flagged=True`) are frozen, together with the candidate pool each
   target was explored with.
2. **Fit, don't explore.** On the confirmation data, each frozen target is
   fitted on its saved pool, and only the frozen links are tested. The recursion
   is not re-run.
3. **Tests:**
   - **OLS + HC4** (t reference): calibrated from N ≥ 275.
   - **LightGBM:** the combined GCM/PCM test (Bonferroni of the two), with
     K = 10 cross-fitting folds and fold-cluster variance on a t(9) reference.
     Calibrated from N ≥ 300.
   - Networks explored with Elastic-Net are confirmed with OLS + HC4.
4. **Multiple testing:** p-values are adjusted across the frozen links with
   Holm (default) or Benjamini–Hochberg.
5. **Not yet tested:** shared groups and interactions are listed as not tested,
   because no calibrated test exists for them yet. They remain exploratory
   results, even after confirmation.

**Why LightGBM uses different settings in exploration and confirmation.**

- **Exploration** refits the engine many times per target: every fold, every
  left-out column and every noise column. It uses a fast LightGBM (100 trees,
  a 4-point tuning grid), whose behaviour was checked in the validation study.
- **Confirmation** must use exactly the configuration whose p-values were
  calibrated in Phase B (F3: up to 1,000 trees with early stopping). Changing it
  would lose that calibration.
- Using F3 in exploration would multiply its runtime. Each setting is the one
  validated for its job.

## What the results mean

**Arrows are predictive, not causal.**

- An arrow X → T means that X predicts T beyond the other candidates in T's pool,
  on held-out data. It does not mean that X causes T.
- Depth is predictive distance from the outcome, not a causal or temporal order.
  A depth-2 variable predicts a depth-1 variable; it is not shown to act through
  it (mediation).
- Confirmation tests the same predictive claim on new data. It does not make the
  claim causal.

**The fold tally is a rule, not a probability.**

- The 10 folds share most of their training data (each pair of training sets
  overlaps by about 89%). "Kept in 9 of 10 folds" is therefore not 9 independent
  replications, and the tally is not a p-value or an error rate.
- How often the rule keeps false links is known from simulation
  ([validation-evidence.md](validation-evidence.md)): about 0.01–0.07 kept false
  links per dataset in the validation study.
- Formal error control comes only from confirmation on independent data.

## Scope and assumptions

- **Outcome:** continuous, with no missing values.
- **Predictors:** numeric. Dummy-code categorical predictors before the
  analysis.
- **Missing predictor values:** median-imputed within each training fold. This
  is reasonable when few values are missing and missingness is unrelated to the
  values. With more missing data, impute beforehand (for example, multiple
  imputation).
- **Confirmation** uses complete rows only (listwise deletion) for each target
  and its pool.
- **Validated range:**
  - The validation studies cover 10–25 candidates, depth up to 3, and
    exploration samples of 300–1,000.
  - They include independent, correlated (including noisy copies of true
    predictors), skewed and 5-point Likert predictors, with linear and curved
    effects.
  - Confirmation was checked end to end on split samples of 500 and 1,000
    rows.
  - Larger pools and networks are extrapolation.
- **Known limits:**
  - With OLS or Elastic-Net on curved data, a link can be found and confirmed
    because it stands in for an effect the linear model cannot represent.
    Prefer LightGBM when effects may be curved.
  - Elastic-Net can keep links between two variables that share a parent in
    the pool.
  - The network changes noticeably with `seed`: re-run with another seed to
    see which kept links are stable.

## Defaults

| Setting | Default |
|---|---|
| `folds` | 10 |
| `keep` | 0.80 |
| `flag` | 0.40 |
| `noise_quantile` | 0.95 |
| `hide_below` | 0.001 |
| `dependence_pair` | 0.70 |
| `interactions` | True |
| `max_interaction_pairs` | 100 |
| `max_depth` | None |
| `seed` | 0 |
