"""Production RF tuning must skip equivalent work and report actual fits."""
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from open.learners.trees import TreePredictor


@pytest.mark.parametrize('width,expected', [(1, 6), (2, 12)])
def test_actual_fit_count_tracks_current_width_and_clone(width, expected):
    X = np.arange(48., dtype=float).reshape(24, 2)[:, :width]
    y = np.ones(24)  # tie must keep .7 and leaf=2
    original_fit = RandomForestRegressor.fit
    calls = []

    def counted_fit(estimator, *args, **kwargs):
        calls.append(1)
        return original_fit(estimator, *args, **kwargs)

    model = TreePredictor('random_forest', random_seed=17)
    with patch.object(RandomForestRegressor, 'fit', counted_fit):
        selected = model.tune(X, y, KFold(3))
    assert len(calls) == expected
    assert model.tuning_grid_size * 3 == expected
    assert selected == {'max_features': .7, 'min_samples_leaf': 2}
    # Reusing an instance for a different target width must refresh accounting.
    other = np.column_stack((X[:, 0], -X[:, 0])) if width == 1 else X[:, :1]
    model.tune(other, y, KFold(3))
    assert model.tuning_grid_size == (4 if width == 1 else 2)
    clone = model.clone()
    clone.tune(X, y, KFold(3))
    assert clone.tuning_grid_size * 3 == expected


def test_parity_when_only_one_inner_training_partition_is_all_missing():
    from test_rf_grid_equivalence import load_probe
    X = np.linspace(-2, 2, 18).reshape(-1, 1)
    y = X[:, 0]**2
    splits = list(KFold(3, shuffle=True, random_state=17).split(X))
    X[splits[0][0]] = np.nan
    assert np.isnan(X[splits[0][0]]).all()
    assert np.isfinite(X).any()
    result = load_probe().compare(X, y, np.array([[np.nan], [-1.], [1.]]),
                                  seed=17, repeats=1)
    assert result['exact_parity']
    pair = result['pairs'][0]
    assert pair['original']['actual_tuning_fits'] == 12
    assert pair['deduplicated']['actual_tuning_fits'] == 6
