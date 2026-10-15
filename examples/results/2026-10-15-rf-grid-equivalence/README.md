# RF singleton tuning equivalence probe

All **48 matched comparisons passed**, across all 12 conditions. Selected
settings, CV scores and held-out predictions were exactly equal. Actual singleton
tuning fits fell from 12 to 6 per search. The median reduction in tuning time
across conditions was **50.3%** (condition medians ranged from 44.2% to 60.9%).
The final timing run used one numerical thread and no concurrent task-owned
test process. A preliminary run verified parity but was superseded for timing
because a broader test suite was active. These measurements describe this local
machine, not a guarantee of a particular runtime saving on GitHub runners.

This is a software and runtime comparison. It is separate from the frozen
recursive reliability studies and does not change production defaults.

The current RF grid considers `max_features=0.7` and `max_features=1.0` for
each of `min_samples_leaf=2` and `min_samples_leaf=5`. With one numeric predictor,
both feature fractions resolve to one feature. This probe keeps the first
fraction, 0.7, and both leaf settings. It does not change the multi-predictor
grid, preprocessing, 100-tree count, random seed, folds, or loss function.

The probe calls the existing `TreePredictor.tune`, `fit`, and `predict` methods.
A scoped, serial-only test patch supplies the reduced grid; it restores the
original grid after every run. An instrumented RF fit method delegates every
call to the real estimator and counts actual fits. The instrumented search
likewise delegates to real GridSearchCV and records every inner-fold score.
No prediction or fitting result is mocked. Timing instrumentation is the same
for both grids. The retained first parameter choice preserves original tie order.

There are 12 software test conditions: N=80 and N=200, crossed with a linear
response, U-shaped response, 20% predictor missingness, constant predictor,
entirely missing predictor, and constant outcome. Each uses the first 80% of rows
for training (64 or 160 rows) and the remaining 20% for prediction comparison.
Generation uses NumPy seed 615+N, a standard-normal predictor, and Gaussian noise
with SD 0.5. The linear outcome is `2*x+noise`; the U-shaped outcome is
`x*x+noise`. Other conditions modify the linear fixture as specified in
`fixture()` in the probe. These are deliberately chosen software edge cases,
not a simulation protocol for estimating scientific reliability or sample size.

Within each condition, both searches use identical three-fold shuffled CV,
the same model seed, and the same input arrays. Four timed repetitions alternate
which search runs first. The equivalence gate requires exact equality, with no
numerical tolerance, for selected parameters, mean CV scores, each inner-fold
score, and every held-out prediction. It also verifies that the removed
candidates actually have duplicate scores. Each original search performs 12
tuning fits; each reduced search performs six. Both perform one final fit.

`comparison.json` retains every timing, score, prediction, selected setting,
actual/reported fit count, runtime version and source fingerprint. The 12 NPZ
files preserve the exact training inputs/outcomes and held-out predictor values;
their hashes are in the report. Four timing repeats on the same data are not
four independent scientific datasets. Failures abort the comparison; they are
never silently filtered from a successful report.

Reproduce from the repository root with the pinned preflight environment
(Python 3.11.9, sklearn 1.3.0, NumPy 1.26.4), setting OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 before starting Python:

```powershell
$env:PYTHONPATH='src'
.superpowers/multiengine-venv/Scripts/python.exe examples/rf_grid_equivalence.py --output examples/results/2026-10-15-rf-grid-equivalence --repeats 4
```

The singleton tuning measurement does not establish an equal percentage saving
for a full recursive analysis. Full analyses also include multi-predictor tuning,
outer fits, group checks and bookkeeping. Earlier fit-count accounting estimates
that singleton deduplication could remove 54,000 of 144,000 tuning fits in the
specific completed three-target chain (about one-third of its ordinary total
fits). A full-run runtime saving has not been measured.

The next implementation step, if this probe passes, is to make the effective
grid depend on predictor count and ensure every risk/effect report describes
the grid actually searched. Keep historical manifests, checkpoints and results
unchanged. Updated implementation would need a fresh source manifest for future
studies. These edge-case comparisons do not supply false-positive rates or
recovery rates across independent datasets.

Four focused tests passed, including exact tie preservation, rejection of
multi-predictor inputs, a negative test that rejects changed predictions, and
real risk evaluation with identical fold losses but 24 versus 12 tuning fits
(two outer folds). The broader pytest run was stopped at approximately five
minutes, with a diagnostic identifying an existing recursive RF integration
fit; it did not produce a full-suite pass. See verification.json. Production
source and default configuration are unchanged.
