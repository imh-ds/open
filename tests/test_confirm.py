"""confirm(): frozen links tested on independent data with the calibrated tests."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from open.confirm import (_adjust, _fit_predict, _folds, _gcm_residuals, _seed, confirm,
                          frozen_network, gcm_pcm_pvalues, hc4_pvalues)
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


@pytest.mark.parametrize("value", [0.0, 1.0, 0.1, -3.25])
def test_hc4_rejects_constant_outcomes(value):
    X = np.random.default_rng(2).normal(size=(300, 2))
    with pytest.raises(ValueError, match="constant|variation"):
        hc4_pvalues(X, np.full(300, value))


def test_hc4_accepts_small_nonzero_outcome_variation():
    r = np.random.default_rng(7)
    X = r.normal(size=(300, 2))
    y = 1e-12 * (X[:, 0] + r.normal(size=300))
    p = hc4_pvalues(X, y)
    assert p.shape == (2,) and np.isfinite(p).all()
    assert p[0] < .001


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


def test_scalar_lightgbm_confirmation_keeps_target_seeded_fold_plan(monkeypatch):
    from open.explore import ExploreResult

    n = 40
    data = pd.DataFrame({"Y": np.linspace(0, 1, n), "X": np.sin(np.arange(n))})
    links = pd.DataFrame([dict(depth=1, target="Y", source="X", kind="unique", tier="kept",
                               pass_rate=1.0, r2_median=0.2)])
    folds_table = pd.DataFrame([dict(depth=1, target="Y", source="X", contrast="unique", noise=False)])
    empty = pd.DataFrame()
    result = ExploreResult(empty, links, empty, folds_table, empty, {},
                           {"engine": "ols", "controls": ()}, empty)
    seen = {}

    def gcm_stub(X, T, folds, dseed, columns=None):
        seen["folds"] = {key: value.copy() for key, value in folds.items()}
        seen["seed"] = dseed
        return np.array([0.1])

    monkeypatch.setattr("open.confirm.gcm_pcm_pvalues", gcm_stub)
    with pytest.warns(UserWarning, match="below the calibrated minimum"):
        confirm(result, data, engine="lightgbm", seed=31)

    expected_seed = _seed(31, "confirm", 1, "Y")
    expected = _folds(n, expected_seed)
    assert seen["seed"] == expected_seed
    for key in ("k10", "half"):
        np.testing.assert_array_equal(seen["folds"][key], expected[key])


def test_scalar_confirmation_fit_keeps_target_seeded_fold_ids(monkeypatch):
    from open.confirm import confirmation_fit
    from open.explore import ExploreResult

    n = 39
    data = pd.DataFrame({"Y": np.linspace(0, 1, n), "X": np.cos(np.arange(n))})
    folds_table = pd.DataFrame([dict(depth=1, target="Y", source="X", contrast="unique", noise=False)])
    fit = pd.DataFrame([dict(depth=1, target="Y", n=n, network=("X",), network_size=1,
                             pool_size=1, r2_network=0.2)])
    empty = pd.DataFrame()
    result = ExploreResult(empty, empty, empty, folds_table, empty, {},
                           {"engine": "ols", "controls": ()}, fit)
    seen = {}

    def network_fit_stub(data, target, pool, network, fold_ids, *args, **kwargs):
        seen["fold_ids"] = np.asarray(fold_ids).copy()
        return {"network": tuple(network), "network_size": 1, "pool_size": 1,
                "r2_network": 0.2, "r2_pool": 0.3, "share_of_pool": 0.5}

    monkeypatch.setattr("open.explore.network_fit", network_fit_stub)
    confirmation_fit(result, data, folds=3, seed=31)
    expected = np.arange(n) % 3
    np.random.default_rng(_seed(31, "confirmation-fit", 1, "Y")).shuffle(expected)
    np.testing.assert_array_equal(seen["fold_ids"], expected)


def test_elastic_net_network_is_confirmed_with_ols():
    data = _chain(900, 9)
    res = explore(data.iloc[:600], 'Y', engine='elastic_net',
                  config=ExploreConfig(max_depth=1, interactions=False, folds=5))
    out = confirm(res, data.iloc[600:])
    assert len(out) and set(out.engine) == {'ols'}


def test_split_sample_is_reproducible_and_disjoint():
    from open.confirm import split_sample
    data = _chain(1000, 3)
    a1, b1 = split_sample(data, .6, seed=4)
    a2, b2 = split_sample(data, .6, seed=4)
    assert a1.index.equals(a2.index) and b1.index.equals(b2.index)
    assert len(a1) == 600 and len(b1) == 400
    assert not set(a1.index) & set(b1.index) and len(set(a1.index) | set(b1.index)) == 1000
    a3, _ = split_sample(data, .6, seed=5)
    assert not a3.index.equals(a1.index)
    with pytest.raises(ValueError):
        split_sample(data, 1.0)


def test_explore_and_confirm_on_one_dataset():
    from open.confirm import explore_and_confirm
    data = _chain(1400, 6)
    res = explore_and_confirm(data, 'Y', engine='ols', split_seed=2,
                              config=ExploreConfig(max_depth=2, interactions=False))
    assert res.settings['n_explore'] == 700 and res.settings['n_confirm'] == 700
    assert res.settings['confirm_engine'] == 'ols'
    assert set(zip(res.confirmation.source, res.confirmation.target)) >= {('X1', 'Y'), ('X3', 'Y')}
    assert res.confirmation.confirmed.all()
    text = res.summary()
    assert 'Confirmation:' in text and 'confirmed (p =' in text


def test_explore_and_confirm_warns_for_small_n_and_null_links_rarely_confirm():
    from open.confirm import explore_and_confirm
    r = np.random.default_rng(12)
    data = pd.DataFrame(r.normal(size=(700, 8)), columns=[f'N{i}' for i in range(8)])
    data['Y'] = r.normal(size=700)
    with pytest.warns(UserWarning, match='small parts'):
        res = explore_and_confirm(data, 'Y', config=ExploreConfig(max_depth=1, interactions=False, flag=.1),
                                  include_flagged=True)
    assert not res.confirmation.confirmed.any()


def test_write_split_saves_confirmation(tmp_path):
    from open.confirm import explore_and_confirm
    from open.explore_report import write_split
    res = explore_and_confirm(_chain(1200, 7), 'Y', split_seed=1,
                              config=ExploreConfig(max_depth=1, interactions=False))
    out = write_split(res, tmp_path / 'out')
    for name in ('confirmation.csv', 'split-settings.json', 'summary.txt', 'network.png', 'links.csv'):
        assert (out / name).stat().st_size > 0
    table = pd.read_csv(out / 'confirmation.csv')
    assert {'p', 'p_adjusted', 'confirmed', 'explore_tier'} <= set(table.columns)
    assert 'Confirmation:' in (out / 'summary.txt').read_text(encoding='utf-8')


def test_lightgbm_column_subset_matches_full_computation():
    pytest.importorskip("lightgbm")
    r = np.random.default_rng(3)
    X = r.normal(size=(160, 4))
    T = np.tanh(X[:, 0]) + .3 * X[:, 2] + r.normal(size=160)
    folds = _folds(160, 7)
    full = gcm_pcm_pvalues(X, T, folds, 7)
    np.testing.assert_array_equal(gcm_pcm_pvalues(X, T, folds, 7, columns=[2, 0]), full[[2, 0]])
    with pytest.raises(ValueError):
        gcm_pcm_pvalues(X, T, folds, 7, columns=[1, 1])


def test_confirmation_fit_is_cross_validated_on_the_new_data():
    from open.confirm import confirmation_fit, explore_and_confirm
    data = _chain(1600, 8)
    split = explore_and_confirm(data, 'Y', engine='ols',
                                config=ExploreConfig(max_depth=2, interactions=False), split_seed=2)
    fit = split.fit.set_index('target')
    assert set(fit.index) == set(split.exploration.fit.target)
    assert (fit.n == 800).all()
    assert .1 < fit.loc['Y', 'r2_network'] < .4                       # true R² about .24
    assert 'Network fit on the confirmation data' in split.summary()
    again = confirmation_fit(split.exploration, data.iloc[800:], seed=2)
    assert again.r2_network.notna().all()
    with pytest.raises(ValueError):
        confirmation_fit(split.exploration, data.drop(columns='X2'))


@pytest.fixture(scope="module")
def constant_target_network():
    return explore(_chain(900, 5), "Y",
                   config=ExploreConfig(folds=5, max_depth=2, interactions=False))


@pytest.mark.parametrize("engine", ["ols", "lightgbm"])
@pytest.mark.parametrize("correction", ["holm", "bh"])
def test_constant_confirmation_targets_are_retained_as_untested(constant_target_network, engine, correction):
    frozen = frozen_network(constant_target_network)
    new = _chain(600, 2)
    for target in frozen.target.unique():
        new[target] = .1
    out = confirm(constant_target_network, new, engine=engine, correction=correction)
    assert len(out) == len(frozen) == 4
    assert out.p.isna().all() and out.p_adjusted.isna().all()
    assert not out.confirmed.any()
    assert out.note.str.contains("not tested.*no variation").all()
    assert out.attrs["links_tested"] == 0


def test_confirmation_checks_outcome_variation_after_complete_case_filtering(constant_target_network):
    new = _chain(601, 2)
    new["Y"] = .1
    new.loc[600, "Y"] = 2.0
    new.loc[600, "X2"] = np.nan
    assert new.Y.nunique() == 2
    out = confirm(constant_target_network, new)
    skipped = out[out.target == "Y"]
    assert len(skipped) == 2 and (skipped.n == 600).all()
    assert skipped.p.isna().all() and skipped.p_adjusted.isna().all()
    assert not skipped.confirmed.any()
    assert skipped.note.str.contains("no variation").all()


@pytest.mark.parametrize("correction", ["holm", "bh"])
def test_constant_target_does_not_prevent_other_targets_being_confirmed(constant_target_network, correction):
    new = _chain(900, 11).assign(Y=.1)
    out = confirm(constant_target_network, new, correction=correction)
    skipped, tested = out[out.target == "Y"], out[out.target != "Y"]
    assert len(skipped) == len(tested) == 2
    assert skipped.p.isna().all() and not skipped.confirmed.any()
    assert set(tested.target) == {"X1", "X3"}
    assert np.isfinite(tested[["p", "p_adjusted"]]).all().all()
    assert tested.confirmed.all() and (tested.p_adjusted >= tested.p).all()
    assert out.attrs["links_tested"] == 2


def test_constant_target_keeps_small_sample_warning_and_note(constant_target_network):
    new = _chain(200, 2)
    for target in frozen_network(constant_target_network).target.unique():
        new[target] = 1.0
    with pytest.warns(UserWarning, match="below the calibrated minimum"):
        out = confirm(constant_target_network, new)
    assert out.note.str.contains("below the calibrated minimum").all()
    assert out.note.str.contains("not tested.*no variation").all()
    assert not out.confirmed.any()


def test_constant_target_preserves_unsupported_link_kind_notes():
    r = np.random.default_rng(11)
    a = r.normal(size=800)
    data = pd.DataFrame({"S1": a + .1 * r.normal(size=800), "S2": a + .1 * r.normal(size=800),
                         "P": r.normal(size=800), "N1": r.normal(size=800)})
    data["Y"] = .35 * a + .4 * data.P + r.normal(size=800)
    result = explore(data, "Y", config=ExploreConfig(max_depth=1, interactions=False))
    out = confirm(result, data.assign(Y=.1))
    shared = out[out.kind == "shared"]
    assert len(shared) == 1
    assert shared.note.str.contains("not tested: no calibrated shared test").all()
    assert out.p.isna().all() and out.p_adjusted.isna().all() and not out.confirmed.any()


@pytest.mark.parametrize("correction", ["holm", "bh"])
@pytest.mark.parametrize("case,reason", [
    ("constant", "rank"), ("duplicate", "rank"), ("dummy_set", "rank"),
    ("empty", "no.*rows"), ("all_missing", "no.*rows"),
    ("saturated", "degrees of freedom"), ("underdetermined", "degrees of freedom"),
    ("infinite_predictor", "finite"), ("infinite_outcome", "finite"),
    ("unit_leverage", "leverage"),
])
def test_invalid_ols_targets_are_reported_without_changing_frozen_pools(
        constant_target_network, correction, case, reason):
    new = _chain(900, 11)
    if case == "constant":
        new["X1"] = 1.0
    elif case == "duplicate":
        new["X3"] = new.X1
    elif case == "dummy_set":
        new[["X2", "X4", "X5"]] = np.eye(3)[np.arange(900) % 3]
    elif case == "empty":
        new = new.iloc[:0]
    elif case == "all_missing":
        new["X2"] = np.nan
    elif case == "saturated":
        new = new.iloc[:8]  # Y's seven frozen predictors plus the intercept
    elif case == "underdetermined":
        new = new.iloc[:3]
    elif case == "infinite_predictor":
        new.loc[0, "X1"] = np.inf
    elif case == "infinite_outcome":
        new.loc[0, "Y"] = np.inf
    elif case == "unit_leverage":
        new["X5"] = (np.arange(900) == 0).astype(float)
    if case in {"empty", "all_missing", "saturated", "underdetermined"}:
        with pytest.warns(UserWarning, match="below the calibrated minimum"):
            out = confirm(constant_target_network, new, correction=correction)
    else:
        out = confirm(constant_target_network, new, correction=correction)
    root = out[out.target == "Y"]
    assert len(out) == 4 and len(root) == 2
    assert (root.pool_size == 7).all()
    assert root.p.isna().all() and root.p_adjusted.isna().all() and not root.confirmed.any()
    assert root.note.str.contains("not tested").all() and root.note.str.contains(reason).all()
    assert out.attrs["links_tested"] == int(out.p.notna().sum())
    if case in {"constant", "duplicate", "infinite_predictor", "infinite_outcome", "saturated"}:
        valid = out[out.target == "X3"]
        assert len(valid) == 1 and np.isfinite(valid[["p", "p_adjusted"]]).all().all()
        assert (valid.p_adjusted >= valid.p).all()
    if case in {"empty", "all_missing"}:
        assert (out.n == 0).all() and out.attrs["links_tested"] == 0
        assert out.note.str.contains("below the calibrated minimum").all()


def test_dropped_constant_predictor_cannot_crash_ols_confirmation():
    r = np.random.default_rng(12)
    x = r.normal(size=600)
    data = pd.DataFrame({"Y": 3 * x + r.normal(size=600), "X": x, "constant": 1.0})
    result = explore(data.iloc[:300], "Y", config=ExploreConfig(folds=3, max_depth=1, interactions=False))
    assert result.links.set_index("source").loc["constant", "tier"] == "dropped"
    out = confirm(result, data.iloc[300:])
    assert len(out) == 1 and out.source.iloc[0] == "X" and out.pool_size.iloc[0] == 2
    assert out.p.isna().all() and out.p_adjusted.isna().all() and not out.confirmed.any()
    assert out.note.str.contains("not tested.*rank").all()


def test_zero_feature_nuisance_predictions_use_training_mean_without_lightgbm(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    predicted = _fit_predict(np.empty((3, 0)), np.array([2., 4., 6.]),
                             [np.empty((2, 0)), np.empty((0, 0)), np.empty((1, 0))], seed=17)
    assert len(predicted) == 3
    np.testing.assert_array_equal(predicted[0], [4., 4.])
    assert predicted[1].shape == (0,)
    np.testing.assert_array_equal(predicted[2], [4.])
    assert all(p.dtype.kind == "f" for p in predicted)


@pytest.mark.parametrize("training_y", [np.array([]), np.array([1., np.nan]), np.array([1., np.inf])])
def test_zero_feature_nuisance_rejects_undefined_training_mean(training_y):
    with pytest.raises(ValueError, match="nonempty|finite"):
        _fit_predict(np.empty((len(training_y), 0)), training_y, [np.empty((2, 0))], seed=0)


@pytest.mark.parametrize("training_y,expected", [
    ([1e308, 1e308], 1e308), ([-1e308, -1e308], -1e308),
    ([1e308, -1e308], 0.0), ([1e-308, 1e-308], 1e-308), ([0., 0.], 0.0),
])
def test_zero_feature_nuisance_mean_remains_finite_across_outcome_scales(training_y, expected):
    with np.errstate(over="raise", invalid="raise"):
        predicted = _fit_predict(np.empty((2, 0)), np.array(training_y), [np.empty((2, 0))], seed=0)
    np.testing.assert_array_equal(predicted[0], [expected, expected])


def test_one_predictor_gcm_residuals_use_only_training_outcomes():
    X = np.arange(1., 7.)[:, None]
    T = np.arange(10., 70., 10.)
    fold = np.array([0, 1, 0, 1, 0, 1])
    eT, eX = _gcm_residuals(X, T, fold, 2, 5)
    np.testing.assert_array_equal(eT, [[-30., -10., -10., 10., 10., 30.]])
    np.testing.assert_array_equal(eX, [[-3., -1., -1., 1., 1., 3.]])
    changed = T.copy()
    changed[fold == 0] += 1000
    new_eT, _ = _gcm_residuals(X, changed, fold, 2, 5)
    np.testing.assert_array_equal(new_eT[:, fold == 0], eT[:, fold == 0] + 1000)


@pytest.mark.parametrize("recursive", [False, True])
def test_lightgbm_confirmation_supports_one_predictor_pools(recursive):
    pytest.importorskip("lightgbm")
    r = np.random.default_rng(12)
    z = r.normal(size=1200)
    x = 2 * z + r.normal(size=1200)
    data = pd.DataFrame({"Y": 3 * x + r.normal(size=1200), "X": x})
    if recursive:
        data["Z"] = z
    result = explore(data.iloc[:600], "Y", config=ExploreConfig(folds=5, max_depth=2, interactions=False))
    frozen = frozen_network(result)
    target = "X" if recursive else "Y"
    row = frozen[frozen.target == target]
    assert len(row) == 1 and len(row.pool.iloc[0]) == 1
    if recursive:
        assert row.depth.iloc[0] == 2
    out = confirm(result, data.iloc[600:], engine="lightgbm", seed=2)
    assert len(out) == len(frozen)
    assert np.isfinite(out[["p", "p_adjusted"]]).all().all()
    assert out.p.between(0, 1).all() and out.p_adjusted.between(0, 1).all()
    assert out.loc[out.target == target, "confirmed"].all()
    assert out.attrs["links_tested"] == len(out)
