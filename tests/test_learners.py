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


# ---- backend engines (PLS, SVR): piloted, dropped, still runnable by name

def test_backend_engines_are_not_offered():
    from open.learners import BACKEND_ENGINES
    assert BACKEND_ENGINES == ('pls', 'svr')
    assert not set(BACKEND_ENGINES) & set(ENGINES)


def test_pls_tunes_caps_components_and_equals_ols_at_full_rank():
    X, y = _data()
    m = make_predictor('pls')
    params = m.tune(X, y, KFold(3, shuffle=True, random_state=0), 'mse')
    assert set(params) == {'n_components'} and 1 <= params['n_components'] <= 3
    full = m.fit(X, y, {'n_components': 8}).predict(X)            # capped at 3 columns: OLS
    np.testing.assert_allclose(full, make_predictor('ols').fit(X, y, {}).predict(X), atol=1e-8)
    assert m.fit(X[['a']], y, params).predict(X[['a']]).shape == (len(X),)   # fewer columns than components
    with pytest.raises(RuntimeError):
        m.clone().predict(X)


def test_svr_tunes_and_fits_a_curve_linear_models_miss():
    r = np.random.default_rng(1)
    X = pd.DataFrame(r.normal(size=(400, 2)), columns=['a', 'b'])
    y = 3 * (X.a ** 2) + .2 * r.normal(size=400)
    m = make_predictor('svr')
    params = m.tune(X, y, KFold(3, shuffle=True, random_state=0), 'mse')
    assert set(params) == {'C', 'gamma'}
    test = pd.DataFrame(r.normal(size=(400, 2)), columns=['a', 'b'])
    truth = 3 * test.a ** 2
    svr = np.corrcoef(m.fit(X, y, params).predict(test), truth)[0, 1]
    ols = np.corrcoef(make_predictor('ols').fit(X, y, {}).predict(test), truth)[0, 1]
    assert svr > .95 and abs(ols) < .3
    # same predictions at any outcome scale: the outcome is standardized inside
    np.testing.assert_allclose(m.fit(X, 100 * y, params).predict(test) / 100,
                               m.fit(X, y, params).predict(test), rtol=1e-6, atol=1e-6)


def test_explore_accepts_experimental_engines_with_a_warning():
    from open.explore import ExploreConfig, explore
    X, y = _data(300)
    data = X.assign(Y=y)
    for engine in ('pls', 'svr'):
        with pytest.warns(UserWarning, match='experimental'):
            res = explore(data, 'Y', engine=engine, config=ExploreConfig(max_depth=1, interactions=False))
        assert res.settings['engine'] == engine
        assert 'a' in set(res.links[res.links.tier == 'kept'].source)
