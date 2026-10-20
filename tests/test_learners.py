"""The three validated engines: OLS, Elastic-Net and LightGBM."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from open.learners import ENGINES, make_predictor


def _data(n=200, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, 3)), columns=['a', 'b', 'c'])
    y = 1 + 2 * X.a - .5 * X.b + .1 * r.normal(size=n)
    return X, y.to_numpy()


def test_factory_offers_only_validated_engines():
    assert ENGINES == ('ols', 'elastic_net', 'lightgbm')
    for name in ('random_forest', 'gbm'):
        with pytest.raises(ValueError, match='unknown engine'):
            make_predictor(name)


def test_ols_matches_least_squares():
    X, y = _data()
    m = make_predictor('ols').fit(X, y, params={})
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    np.testing.assert_allclose(m.predict(X), A @ beta, atol=1e-10)
    with pytest.raises(RuntimeError):
        m.clone().predict(X)


def test_elastic_net_tunes_and_predicts():
    X, y = _data()
    m = make_predictor('elastic_net')
    params = m.tune(X, y, KFold(3, shuffle=True, random_state=0), 'mse')
    assert set(params) == {'alpha', 'l1_ratio'}
    pred = m.fit(X, y, params).predict(X)
    assert np.corrcoef(pred, y)[0, 1] > .99


def test_lightgbm_is_deterministic_and_tunes():
    pytest.importorskip('lightgbm')
    X, y = _data(300)
    a = make_predictor('lightgbm', random_seed=3)
    params = a.tune(X, y, KFold(3, shuffle=True, random_state=0), 'mse')
    assert set(params) == {'num_leaves', 'learning_rate'}
    pa = a.fit(X, y, params).predict(X)
    pb = make_predictor('lightgbm', random_seed=3).fit(X, y, params).predict(X)
    np.testing.assert_array_equal(pa, pb)
    assert np.corrcoef(pa, y)[0, 1] > .9
