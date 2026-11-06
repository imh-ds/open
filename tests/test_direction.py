"""Signed effects: learner coefficients, per-fold direction summaries, confirmation slopes and the graph option."""
from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from opennet import explore_report as report
from opennet.api import DiscoveryConfig, analyze, effects_table
from opennet.confirm import hc4_pvalues, hc4_slopes
from opennet.explore import DIRECTION_COLUMNS, ExploreConfig, _summarize_directions, explore
from opennet.learners.base import Predictor
from opennet.learners.elastic_net import ElasticNetPredictor
from opennet.learners.ols import OLSPredictor


def _data(n=2400, seed=3):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"P": r.normal(size=n), "N1": r.normal(size=n)})
    d["A"] = .7 * d.P + .7 * r.normal(size=n)
    d["B"] = -.6 * d.P + .7 * r.normal(size=n)
    d["Y"] = .5 * d.A - .4 * d.B + r.normal(size=n)
    return d


@pytest.mark.parametrize("learner, params", [(OLSPredictor, {}), (ElasticNetPredictor, {"alpha": .01, "l1_ratio": .5})])
def test_linear_learners_return_signed_coefficients_in_both_units(learner, params):
    r = np.random.default_rng(0)
    X = pd.DataFrame({"a": r.normal(size=500) * 3, "b": r.normal(size=500)})
    y = 2 * X.a - X.b + r.normal(size=500)
    model = learner().fit(X, y, params)
    standardized, raw = model.coefficients()
    assert model.supports_coefficients and standardized[0] > 0 > standardized[1]
    assert raw[0] == pytest.approx(2, abs=.15) and raw[1] == pytest.approx(-1, abs=.15)
    assert standardized[0] == pytest.approx(raw[0] * X.a.std(), rel=.05)         # per SD versus per unit


def test_other_learners_give_no_coefficients_rather_than_a_guess():
    class Dummy(Predictor):
        engine_name = "dummy"
        tuning_grid_size = 0

        def tune(self, X, y, cv, loss):
            return {}

        def fit(self, X, y, params=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def clone(self):
            return Dummy()

    assert Dummy.supports_coefficients is False and Dummy().coefficients() is None
    from opennet.learners.lightgbm import LightGBMPredictor
    assert LightGBMPredictor.supports_coefficients is False


@pytest.mark.parametrize("engine", ["ols", "elastic_net"])
def test_discovery_records_the_sign_of_each_candidate_in_each_targets_full_model(engine):
    res = explore(_data(), "Y", engine=engine, config=ExploreConfig(folds=5, interactions=False))
    table = res.directions
    assert list(table.columns) == DIRECTION_COLUMNS and (table.model == engine).all()

    def row(target, source):
        return table[(table.target == target) & (table.source == source)].iloc[0]

    assert row("Y", "A").direction == "positive" and row("Y", "A").agreement == 1
    assert row("Y", "B").direction == "negative" and row("Y", "B").coef < 0
    assert row("A", "P").direction == "positive" and row("B", "P").direction == "negative"
    a = row("Y", "A")
    assert a.n_fits == 5 and a.coef_std_min <= a.coef_std <= a.coef_std_max
    assert row("Y", "N1").direction in ("mixed", "zero", "positive", "negative")   # noise has no promised sign


def test_summary_rules_use_no_threshold():
    def rows(signs):
        return [dict(depth=1, target="Y", source="X", repeat=0, fold=i, coef_std=v, coef=v)
                for i, v in enumerate(signs)]

    def summarize(signs):
        return _summarize_directions(rows(signs), "ols").iloc[0]

    assert summarize([.1, .2, .3]).direction == "positive"
    assert summarize([-.1, -.2]).direction == "negative"
    mixed = summarize([.1, .2, .3, .4, -.01])
    assert mixed.direction == "mixed" and mixed.agreement == .8            # one disagreeing fit is not hidden
    assert summarize([0, 0, 0]).direction == "zero"
    assert summarize([.2, 0, .3]).direction == "mixed"                      # a zero among signed fits is not consensus
    assert _summarize_directions([], "ols").empty


def test_confirmation_reports_the_frozen_ols_slope_with_hc4_standard_error():
    r = np.random.default_rng(1)
    n = 500
    X = np.column_stack([r.normal(size=n) * 5 + 3, r.normal(size=n) * .2])
    y = 2 * X[:, 0] - 40 * X[:, 1] + r.normal(size=n) * 10 + 7
    slope, se, p, df = hc4_slopes(X, y)
    assert np.allclose(slope, np.linalg.lstsq(np.column_stack([np.ones(n), X]), y, rcond=None)[0][1:])
    assert (se > 0).all() and df == n - 3 and np.allclose(p, hc4_pvalues(X, y))
    res = analyze(_data(), outcome="Y", config=DiscoveryConfig(folds=5, interactions=False))
    c = res.confirmation[res.confirmation.kind == "unique"]
    tested = c[c.p.notna()]
    assert {"coef", "se", "ci_low", "ci_high"} <= set(c.columns)
    assert (tested.ci_low < tested.coef).all() and (tested.coef < tested.ci_high).all()
    a = tested[(tested.source == "A") & (tested.target == "Y")].iloc[0]
    assert a.coef == pytest.approx(.5, abs=.12)


def test_effects_table_carries_signed_estimates_and_keeps_the_two_sources_apart():
    res = analyze(_data(), outcome="Y", config=DiscoveryConfig(folds=5, interactions=False,
                                                                blocks={"ab": ["A", "B"]}))
    t = effects_table(res)
    unique = t[(t.kind == "unique") & (t.target == "A") & (t.predictor == "P")].iloc[0]
    assert unique.direction == "positive" and unique.estimate == pytest.approx(.7, abs=.1)
    assert unique.slope == pytest.approx(.7, abs=.1) and unique.slope_ci_low < unique.slope < unique.slope_ci_high
    block = t[t.kind == "block"].iloc[0]
    assert block.direction.startswith("none") and np.isnan(block.estimate) and np.isnan(block.slope)
    assert {"estimate", "estimate_std", "direction", "direction_agreement", "slope", "slope_se"} <= set(t.columns)


def test_directions_survive_the_shared_cache_of_a_multi_outcome_run():
    data = _data()
    data["Z"] = .6 * data.P + np.random.default_rng(1).normal(size=len(data))
    res = analyze(data, outcomes=["Y", "Z"], config=DiscoveryConfig(folds=5, interactions=False))
    for split in res.by_outcome.values():
        directions = split.discovery.directions
        assert len(directions) and directions.target.isin(split.discovery.nodes.variable).all()


def _line_colours(fig):
    return [ln.get_color() for ln in fig.axes[0].lines if isinstance(ln, Line2D)]


def test_direction_colours_are_opt_in_and_redundantly_marked():
    res = explore(_data(), "Y", config=ExploreConfig(folds=5, interactions=False))
    off = report.plot_explore(res)
    assert report.POSITIVE not in _line_colours(off) and report.NEGATIVE not in _line_colours(off)
    plt.close(off)
    on = report.plot_explore(res, show_direction=True)
    colours = _line_colours(on)
    assert report.POSITIVE in colours and report.NEGATIVE in colours
    texts = [t.get_text() for t in on.findobj(matplotlib.text.Text)]
    assert any(t.startswith("+ ") for t in texts) and any(t.startswith("− ") for t in texts)
    assert any("positive sign" in t for t in texts) and any("not tested or confirmed" in t for t in texts)
    plt.close(on)


def test_direction_agreement_decides_when_a_majority_sign_is_drawn_and_is_validated():
    res = explore(_data(), "Y", config=ExploreConfig(folds=5, interactions=False))
    forged = res.directions.copy()
    mask = (forged.target == "Y") & (forged.source == "A")
    forged.loc[mask, "direction"] = "mixed"
    forged.loc[mask, "agreement"] = .8
    forged.loc[mask, "coef_std"] = .4
    mixed = replace(res, directions=forged)
    strict = report.plot_explore(mixed, show_direction=True)
    relaxed = report.plot_explore(mixed, show_direction=True, direction_agreement=.8)
    assert report.MIXED_SIGN in _line_colours(strict) and report.MIXED_SIGN not in _line_colours(relaxed)
    plt.close(strict)
    plt.close(relaxed)
    for bad in (.5, 1.1, True, "1"):
        with pytest.raises(ValueError, match="direction_agreement"):
            report.GraphStyle(direction_agreement=bad)
    with pytest.raises(ValueError, match="show_direction"):
        report.GraphStyle(show_direction="yes")


def test_bundled_arrows_into_a_block_carry_a_sign_only_when_the_members_agree():
    res = analyze(_data(), outcome="Y", config=DiscoveryConfig(folds=5, interactions=False,
                                                                blocks={"ab": ["A", "B"]}))
    fig = report.plot_explore(res.exploration, show_direction=True)
    texts = [t.get_text() for t in fig.axes[0].texts]
    bundled = [t for t in texts if "of 2)" in t]
    assert bundled and all(t.startswith(("+ ", "− ", "± ")) for t in bundled)
    plt.close(fig)
