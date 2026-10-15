"""Guard production equivalence against the explicit historical grid reference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_probe():
    path = Path(__file__).parents[1] / 'examples' / 'rf_grid_equivalence.py'
    spec = importlib.util.spec_from_file_location('rf_grid_equivalence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_tuning_tie_preserves_parameters_predictions_and_actual_fit_counts():
    probe = load_probe()
    from open.learners.trees import GRIDS
    original = {k: dict(v) for k, v in GRIDS.items()}
    X = np.arange(36, dtype=float).reshape(-1, 1)
    # All scores tie: deduplication must preserve the first original setting.
    result = probe.compare(X[:27], np.ones(27), X[27:], seed=17, repeats=1)
    assert result['exact_parity']
    assert result['pairs'][0]['original']['selected'] == {
        'max_features': .7, 'min_samples_leaf': 2}
    assert result['pairs'][0]['original']['actual_tuning_fits'] == 12
    assert result['pairs'][0]['deduplicated']['actual_tuning_fits'] == 6
    assert result['pairs'][0]['original']['actual_final_fits'] == 1
    assert result['pairs'][0]['deduplicated']['actual_final_fits'] == 1
    assert result['pairs'][0]['deduplicated']['reported_tuning_fits'] == 6
    assert GRIDS == original


def test_probe_rejects_more_than_one_predictor_without_fitting():
    probe = load_probe()
    with pytest.raises(ValueError, match='one predictor'):
        probe.compare(np.ones((12, 2)), np.arange(12.), np.ones((3, 2)), seed=1, repeats=1)


def test_mismatched_predictions_fail_the_equivalence_gate():
    probe = load_probe()
    base = dict(selected={'max_features': .7, 'min_samples_leaf': 2},
                scores=[-1., -2., -1., -2.], predictions=[0., 1.])
    reduced = dict(selected=base['selected'], scores=[-1., -2.], predictions=[0., 1.01])
    with pytest.raises(AssertionError, match='predictions'):
        probe.require_parity(base, reduced)


def test_risk_reports_actual_reduced_tuning_work_with_identical_losses():
    """Catch stale grid-size accounting at the real risk/clone boundary."""
    from unittest.mock import patch
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from open.learners import trees
    from open.risk import evaluate_risk
    from open.splits import SplitFold, SplitPlan
    frame = pd.DataFrame({'x': np.arange(24.), 'y': np.arange(24.)**2})
    plan = SplitPlan(n_samples=24, folds=2, repeats=1, random_seed=17,
                     splits=(SplitFold(0, 0, tuple(range(12)), tuple(range(12, 24))),
                             SplitFold(0, 1, tuple(range(12, 24)), tuple(range(12)))))
    original_fit = RandomForestRegressor.fit
    results, counts = [], []
    for predictor_class in (load_probe().OriginalGridPredictor, trees.TreePredictor):
        count = 0

        def counted_fit(estimator, *args, **kwargs):
            nonlocal count
            count += 1
            return original_fit(estimator, *args, **kwargs)

        with patch.object(RandomForestRegressor, 'fit', counted_fit):
            results.append(evaluate_risk(frame, target='y', candidate_set=('x',),
                predictors=('x',), split_plan=plan,
                predictor=predictor_class('random_forest', random_seed=17),
                tuning_folds=3))
        counts.append(count)
    assert counts == [26, 14]  # 24/12 inner fits plus two outer fits
    assert [r.tuning_model_fit_count for r in results] == [24, 12]
    assert [r.outer_model_fit_count for r in results] == [2, 2]
    assert results[0].aggregate_risk == results[1].aggregate_risk
    assert results[0].folds == results[1].folds
