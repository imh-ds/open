"""Guard a matched optimization probe; production grids remain unchanged."""
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
