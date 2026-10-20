"""confirm(): frozen links tested on independent data with the calibrated tests."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from open.confirm import _adjust, _folds, confirm, frozen_network, gcm_pcm_pvalues, hc4_pvalues
from open.explore import ExploreConfig, explore

EXAMPLES = Path(__file__).parents[1] / "examples"


def _phase_b():
    sys.path.insert(0, str(EXAMPLES))
    import phase_b_actions
    return phase_b_actions.frozen()


def _chain(n, seed):
    r = np.random.default_rng(seed)
    d = {f'X{i}': r.normal(size=n) for i in (2, 4, 5, 6, 7)}
    d['X1'] = .5 * d['X4'] + r.normal(size=n)
    d['X3'] = .5 * d['X2'] + r.normal(size=n)
    d['Y'] = .35 * d['X1'] + .35 * d['X3'] + r.normal(size=n)
    return pd.DataFrame(d)


def test_hc4_matches_frozen_phase_b():
    pb = _phase_b()
    r = np.random.default_rng(1)
    X = r.normal(size=(300, 4))
    y = X @ [.2, 0, -.1, 0] + r.normal(size=300) * (1 + X[:, 0] ** 2)
    np.testing.assert_allclose(hc4_pvalues(X, y), pb.hc4_p(X, y), rtol=0, atol=1e-12)


def test_lightgbm_combined_test_matches_frozen_phase_b():
    pytest.importorskip("lightgbm")
    pb = _phase_b()
    r = np.random.default_rng(2)
    X = r.normal(size=(160, 3))
    T = np.tanh(X[:, 0]) + r.normal(size=160)
    folds = _folds(160, 99)
    mine = gcm_pcm_pvalues(X, T, folds, 99)
    eT, eX = pb.gcm_residuals("F3", X, T, folds["k10"], 10, 99)
    gcm = pb.stat_cluster(eT * eX, folds["k10"], 10)
    pcm = pb.pcm_p(pb.pcm_products("F3", X, T, folds["half"], folds["k10"], 10, 99), 10, "cluster")
    np.testing.assert_allclose(mine, np.minimum(1, 2 * np.minimum(gcm, pcm)), rtol=0, atol=1e-12)


def test_split_sample_confirmation_with_ols():
    data = _chain(1200, 5)
    explore_half, confirm_half = data.iloc[:600], data.iloc[600:]
    res = explore(explore_half, 'Y', engine='ols', config=ExploreConfig(max_depth=2, interactions=False))
    frozen = frozen_network(res)
    assert set(zip(frozen.source, frozen.target)) == {('X1', 'Y'), ('X3', 'Y'), ('X4', 'X1'), ('X2', 'X3')}
    y_pool = frozen[frozen.target == 'Y'].pool.iloc[0]
    assert set(y_pool) == {'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7'}
    out = confirm(res, confirm_half)
    assert len(out) == 4 and out.confirmed.all()
    assert (out.p_adjusted >= out.p).all() and set(out.engine) == {'ols'}
    assert out.attrs['links_tested'] == 4


def test_untestable_kinds_are_listed_not_tested_and_engine_defaults():
    r = np.random.default_rng(11)
    n = 800
    a = r.normal(size=n)
    d = pd.DataFrame({'S1': a + .1 * r.normal(size=n), 'S2': a + .1 * r.normal(size=n),
                      'P': r.normal(size=n), 'N1': r.normal(size=n)})
    d['Y'] = .35 * a + .4 * d.P + r.normal(size=n)
    res = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    out = confirm(res, d)
    shared = out[out.kind == 'shared']
    assert len(shared) and shared.p.isna().all() and shared.note.str.contains('not tested').all()
    assert not shared.confirmed.any()


def test_small_confirmation_sample_warns():
    data = _chain(900, 8)
    res = explore(data.iloc[:600], 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    with pytest.warns(UserWarning, match='below the calibrated minimum'):
        confirm(res, data.iloc[600:800])
    with pytest.raises(ValueError):
        confirm(res, data, engine='elastic_net')


def test_holm_and_bh_adjustments():
    p = np.array([.01, .04, .03, .005])
    np.testing.assert_allclose(_adjust(p, 'holm'), [.03, .06, .06, .02])
    np.testing.assert_allclose(_adjust(p, 'bh'), [.02, .04, .04, .02])


def test_elastic_net_network_is_confirmed_with_ols():
    data = _chain(900, 9)
    res = explore(data.iloc[:600], 'Y', engine='elastic_net',
                  config=ExploreConfig(max_depth=1, interactions=False, folds=5))
    out = confirm(res, data.iloc[600:])
    assert len(out) and set(out.engine) == {'ols'}
