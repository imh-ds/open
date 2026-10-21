"""Exploratory mode: fold-level noise bar, tiers and depth-by-depth recursion."""
import numpy as np
import pandas as pd
import pytest

from open.explore import ExploreConfig, assess_target, explore, tier
from open.learners.ols import OLSPredictor


def _chain(n=800, seed=2026):
    # X4 -> X1 -> Y <- X3 <- X2 ; X5/X11 substitute pair -> Y ; X6..X10 noise
    r = np.random.default_rng(seed)
    d = {f'X{i}': r.normal(size=n) for i in (2, 4, 5, 6, 7, 8, 9, 10)}
    d['X1'] = .5 * d['X4'] + r.normal(size=n)
    d['X3'] = .5 * d['X2'] + r.normal(size=n)
    d['X11'] = d['X5'] + .3 * r.normal(size=n)
    d['Y'] = .3 * d['X1'] + .3 * d['X3'] + .25 * (d['X5'] + d['X11']) / 2 + r.normal(size=n)
    return pd.DataFrame(d)


@pytest.fixture(scope='module')
def chain_result():
    return explore(_chain(), 'Y', engine='ols', config=ExploreConfig(max_depth=2, interactions=False))


def test_tier_boundaries_are_inclusive():
    assert tier(.8, .8, .4) == 'kept'
    assert tier(.79, .8, .4) == 'flagged'
    assert tier(.4, .8, .4) == 'flagged'
    assert tier(.39, .8, .4) == 'dropped'
    assert tier(8 / 10, .8, .4) == 'kept'


def test_recovers_planted_chain_and_shared_pair(chain_result):
    res = chain_result
    depth = dict(zip(res.nodes.variable, res.nodes.depth))
    assert depth == {'Y': 0, 'X1': 1, 'X3': 1, 'X5': 1, 'X11': 1, 'X4': 2, 'X2': 2}
    kept = res.links[res.links.tier == 'kept']
    assert set(zip(kept.source, kept.target, kept.kind)) == {
        ('X1', 'Y', 'unique'), ('X3', 'Y', 'unique'), ('X5+X11', 'Y', 'shared'),
        ('X4', 'X1', 'unique'), ('X2', 'X3', 'unique')}
    parents = dict(zip(res.nodes.variable, res.nodes.parents))
    assert parents['X5'] == parents['X11'] == ('Y',)
    assert res.stops['X4'] == res.stops['X2'] == 'MAX_DEPTH'
    assert '[solid ] X4 -> X1' in res.summary()


def test_pool_excludes_kept_but_returns_flagged_and_dropped(chain_result):
    res = chain_result
    d1 = res.links[res.links.depth == 1]
    not_kept = set(d1.source[(d1.kind == 'unique') & (d1.tier != 'kept')]) - {'X5', 'X11'}
    d2 = res.folds[(res.folds.depth == 2) & ~res.folds.noise & (res.folds.contrast == 'unique')]
    for target in ('X1', 'X3', 'X5', 'X11'):
        pool = set(d2.source[d2.target == target])
        assert pool == not_kept
        assert not pool & {'Y', 'X1', 'X3', 'X5', 'X11'}


def test_noise_bar_is_pooled_quantile_and_pass_is_strictly_above(chain_result):
    f = chain_result.folds
    f = f[(f.depth == 1) & (f.contrast == 'unique')]
    bar = f[f.noise].value.quantile(.95)
    assert np.allclose(f.bar, bar)
    assert (f[f.noise].shape[0]) == 10 * 11
    assert (f.passed == (f.value > bar)).all()
    links = chain_result.links
    x1 = links[(links.depth == 1) & (links.source == 'X1')].iloc[0]
    real = f[(f.source == 'X1') & ~f.noise]
    assert x1.pass_rate == pytest.approx(real.passed.mean())
    assert x1.r2_median == pytest.approx(real.value.median())


def test_pure_noise_outcome_keeps_nothing():
    r = np.random.default_rng(7)
    df = pd.DataFrame(r.normal(size=(600, 9)), columns=[f'X{i}' for i in range(1, 10)])
    df['Y'] = r.normal(size=600)
    res = explore(df, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    assert (res.links.tier != 'kept').all()
    assert res.stops == {'Y': 'NOTHING_KEPT'}
    assert list(res.nodes.variable) == ['Y']


def test_missing_target_rows_are_left_out_of_that_targets_folds():
    df = _chain(n=400)
    df.loc[:19, 'X1'] = np.nan
    fold_ids = np.arange(400) % 10
    folds, links, _ = assess_target(df, 'X1', ('X2', 'X4', 'X6'), fold_ids, OLSPredictor(),
                                    ExploreConfig(interactions=False), depth=2)
    assert set(folds.fold) == set(range(10))
    assert 'X4' in set(links.source[links.tier == 'kept'])


def test_seed_reproducible_and_engine_instance_accepted():
    df = _chain(n=400, seed=3)
    a = explore(df, 'Y', engine='ols', config=ExploreConfig(max_depth=1, seed=5))
    b = explore(df, 'Y', engine=OLSPredictor(), config=ExploreConfig(max_depth=1, seed=5))
    pd.testing.assert_frame_equal(a.links, b.links)
    pd.testing.assert_frame_equal(a.folds, b.folds)
    c = explore(df, 'Y', engine='ols', config=ExploreConfig(max_depth=1, seed=6, interactions=False))
    unique = [r.folds[r.folds.contrast == 'unique'].value.to_numpy() for r in (a, c)]
    assert not np.allclose(*unique)


def test_elastic_net_engine_runs():
    res = explore(_chain(n=400), 'Y', engine='elastic_net',
                  config=ExploreConfig(max_depth=1, folds=5, interactions=False))
    assert {'X1', 'X3'} <= set(res.links.source[res.links.tier != 'dropped'])


def test_invalid_inputs_raise():
    df = _chain(n=400)
    with pytest.raises(ValueError):
        ExploreConfig(keep=.4, flag=.8)
    with pytest.raises(ValueError):
        ExploreConfig(noise_quantile=1)
    with pytest.raises(ValueError):
        explore(df, 'Y', predictors=['Y', 'X1'])
    bad = df.copy()
    bad.loc[0, 'Y'] = np.nan
    with pytest.raises(ValueError, match='missing'):
        explore(bad, 'Y')
    with pytest.raises(ValueError, match='too small'):
        explore(df.iloc[:40], 'Y')
    with pytest.raises(ValueError, match='reserved|without'):
        explore(df.rename(columns={'X1': 'X1+X2'}), 'Y')


def test_pair_classes_shared_and_partner_screening():
    r = np.random.default_rng(11)
    n = 800
    a = r.normal(size=n)
    d = pd.DataFrame({'S1': a + .1 * r.normal(size=n), 'S2': a + .1 * r.normal(size=n),  # substitutes
                      'P': r.normal(size=n)})
    d['Q'] = d.P + .6 * r.normal(size=n)                                   # proxy of P only
    d['N1'], d['N2'] = r.normal(size=n), r.normal(size=n)
    d['Y'] = .35 * a + .4 * d.P + r.normal(size=n)
    res = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    pairs = res.pairs.set_index(['a', 'b'])
    assert pairs.loc[('S1', 'S2'), 'relation'] == 'shared_redundant'
    assert pairs.loc[('P', 'Q'), 'relation'] == 'partner_screening'
    assert pairs.loc[('P', 'Q'), 'screened_member'] == 'Q'
    assert pairs.loc[('P', 'Q'), 'screened_by'] == 'P'
    kept = res.links[res.links.tier == 'kept']
    assert set(kept.source) >= {'P', 'S1+S2'}
    assert 'Q' not in set(res.nodes.variable)
    assert 'Correlated pairs:' in res.summary()


def test_planted_interaction_is_detected_without_main_effect_of_partner():
    r = np.random.default_rng(4)
    n = 800
    d = pd.DataFrame(r.normal(size=(n, 6)), columns=['A', 'B', 'C', 'D', 'E', 'F'])
    d['Y'] = .4 * d.A + .3 * d.C + .35 * d.A * d.B + r.normal(size=n)
    res = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1))
    inter = res.links[res.links.kind == 'interaction'].set_index('source')
    assert inter.loc['A*B', 'tier'] == 'kept'
    assert inter.loc['A*B', 'members'] == ('A', 'B')
    # .35 * A * B explains about .1 of Y's variance; the held-out gain must be of that order
    assert inter.loc['A*B', 'r2_median'] > .05
    f = res.folds[res.folds.contrast == 'interaction']
    assert f.noise.sum() == f[~f.noise].shape[0]            # one noise comparator per real pair
    assert 'B' not in set(res.nodes.variable)              # interactions promote nothing


def test_write_explore_outputs(tmp_path):
    from open.explore_report import write_explore
    res = explore(_chain(n=400), 'Y', engine='ols', config=ExploreConfig(max_depth=2, interactions=False))
    out = write_explore(res, tmp_path / 'out')
    for name in ('nodes.csv', 'links.csv', 'pairs.csv', 'folds.csv', 'noise-bars.csv', 'settings.json',
                 'summary.txt', 'network.png'):
        assert (out / name).stat().st_size > 0
    links = pd.read_csv(out / 'links.csv')
    assert {'pass_rate', 'fold_tier', 'tier', 'r2_median', 'r2_above_noise', 'x_noise', 'noise_bar'}         <= set(links.columns)


def test_r2_above_noise_and_ratio_only_with_a_real_noise_bar(chain_result):
    links = chain_result.links
    assert np.allclose(links.r2_above_noise, links.r2_median - links.noise_bar)
    small = links.noise_bar < .001
    assert links.x_noise[small].isna().all()
    assert np.allclose(links.x_noise[~small], (links.r2_median / links.noise_bar)[~small])


def test_hide_floor_hides_tiny_effects_and_stops_their_promotion():
    r = np.random.default_rng(21)
    n = 3000
    d = pd.DataFrame(r.normal(size=(n, 5)), columns=['A', 'T', 'C', 'D', 'E'])
    d['Y'] = .5 * d.A + .12 * d['T'] + r.normal(size=n)    # T explains about .011 of Y
    on = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False, hide_below=.02))
    off = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False, hide_below=None))
    t_on = on.links.set_index('source').loc['T']
    t_off = off.links.set_index('source').loc['T']
    assert t_off.tier == t_off.fold_tier != 'dropped'
    assert t_on.tier == 'hidden' and t_on.fold_tier == t_off.fold_tier
    assert 'A' in set(on.nodes.variable)
    assert ('T' in set(off.nodes.variable)) == (t_off.tier == 'kept')
    assert 'T' not in set(on.nodes.variable)
    assert 'Hidden: 1' in on.summary()
    with pytest.raises(ValueError):
        ExploreConfig(hide_below=1.5)


@pytest.mark.parametrize('engine', ['random_forest', 'gbm'])
def test_unvalidated_engines_are_rejected(engine):
    with pytest.raises(ValueError, match='not validated'):
        explore(_chain(n=400), 'Y', engine=engine, config=ExploreConfig(max_depth=1, interactions=False))


def test_flagged_interactions_are_hidden_from_the_graph_by_default():
    from open.explore_report import _drawn
    links = pd.DataFrame({"kind": ["unique", "unique", "interaction", "interaction", "unique"],
                          "tier": ["kept", "flagged", "kept", "flagged", "dropped"]})
    assert list(_drawn(links, True, False).index) == [0, 1, 2]
    assert list(_drawn(links, True, True).index) == [0, 1, 2, 3]
    assert list(_drawn(links, False, True).index) == [0, 2]


def test_network_fit_reports_held_out_r2_per_explored_target(chain_result):
    fit = chain_result.fit.set_index('target')
    explored = set(chain_result.folds.target)
    assert set(fit.index) == explored
    y = fit.loc['Y']
    assert y.network_size == len(chain_result.nodes) - 1 - (chain_result.nodes.depth == 2).sum()
    assert .1 < y.r2_network <= y.r2_pool + .02 and y.folds == 10
    assert 'Network fit' in chain_result.summary()


def test_network_fit_is_zero_without_kept_predictors():
    r = np.random.default_rng(9)
    d = pd.DataFrame(r.normal(size=(400, 4)), columns=['Y', 'A', 'B', 'C'])
    res = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    row = res.fit.iloc[0]
    assert row.network_size == 0 and row.r2_network == 0 and row.r2_pool < .02


def test_graph_layout_is_centred_spaced_and_deterministic(chain_result):
    import math
    from open.explore_report import MIN_SEP, _drawn, _layout
    links = _drawn(chain_result.links, True, True)
    pos, ring, candidate, junctions = _layout(chain_result, links)
    assert pos['Y'] == (0.0, 0.0)
    assert set(junctions) == {(int(r.depth), r.target, r.source)
                              for r in links[links.kind == 'interaction'].itertuples()}
    assert {m for ms in links.members for m in ms} | set(links.target) <= set(pos)
    names = list(pos)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert math.dist(pos[a], pos[b]) >= .98 * MIN_SEP, (a, b)
    for j in junctions.values():                       # diamonds never sit on a node
        assert all(math.dist(j, q) >= .98 * MIN_SEP for q in pos.values())
    assert _layout(chain_result, links)[0] == pos


def test_plot_draws_interactions_groups_and_confirmation(tmp_path):
    from open.explore_report import plot_explore
    r = np.random.default_rng(4)
    n = 800
    d = pd.DataFrame(r.normal(size=(n, 6)), columns=['A', 'B', 'C', 'D', 'E', 'L'])
    d['S1'], d['S2'] = d.L + .1 * r.normal(size=n), d.L + .1 * r.normal(size=n)
    d['Y'] = .4 * d.A + .3 * d.L + .4 * d.A * d.B + r.normal(size=n)
    res = explore(d.drop(columns='L'), 'Y', engine='ols', config=ExploreConfig(max_depth=1, max_interaction_pairs=6))
    conf = pd.DataFrame(dict(depth=[1], target=['Y'], source=['A'], p=[.001], confirmed=[True]))
    out = plot_explore(res, tmp_path / 'g.png', show_flagged_interactions=True, confirmation=conf)
    assert out.stat().st_size > 10_000


def test_graph_style_settings_scale_the_layout_and_validate(chain_result, tmp_path):
    import math
    from open.explore_report import GraphStyle, _drawn, _layout, plot_explore, write_explore
    links = _drawn(chain_result.links, True, True)

    def spread(style):
        pos = _layout(chain_result, links, style)[0]
        names = list(pos)
        closest = min(math.dist(pos[a], pos[b]) for i, a in enumerate(names) for b in names[i + 1:])
        return max(math.dist(p, (0, 0)) for p in pos.values()), closest

    compact, closest_compact = spread(GraphStyle(spacing=.4))
    wide, _ = spread(GraphStyle(spacing=1.6))
    assert compact < wide
    assert closest_compact >= .98 * GraphStyle(spacing=.4).min_sep() >= 2 * GraphStyle().node_r()
    big = GraphStyle(node_size=2, spacing=.4)
    assert spread(big)[1] >= .98 * big.min_sep()                  # bigger nodes never overlap
    kept = _drawn(chain_result.links, False, False)
    assert not _layout(chain_result, kept, GraphStyle(show_flagged=False))[2]   # no flagged-only nodes
    for bad in (dict(node_size=0), dict(spacing=-1), dict(label_size=True), dict(edge_labels="some"),
                dict(show_flagged="yes"), dict(dpi=10), dict(label_outline="not-a-colour"),
                dict(label_outline_width=-1)):
        with pytest.raises(ValueError):
            GraphStyle(**bad)
    with pytest.raises(TypeError):
        plot_explore(chain_result, tmp_path / 'x.png', colour='red')
    out = plot_explore(chain_result, tmp_path / 'a.png', style=GraphStyle(spacing=.7), node_size=1.3,
                       edge_width=.6, label_size=1.2, show_flagged=False, edge_labels=None, dpi=90,
                       label_outline="#fff3bf", label_outline_width=0)
    assert out.exists()
    assert (write_explore(chain_result, tmp_path / 'w', spacing=.8) / 'network.png').exists()


def test_edge_labels_use_apa_r2_and_folds_passed():
    from open.explore_report import _apa, _passes
    assert [_apa(v) for v in (.12, .1204, .0042, .0003, 0, -.0042, -.0002)] == \
        ['.120', '.120', '.004', '<.001', '.000', '−.004', '.000']
    assert _passes(.9, 10) == '9/10' and _passes(.82, 50) == '41/50'
