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
from opennet.explore import (DIRECTION_COLUMNS, ExploreConfig, _local_effects, _profile_rows,
                             _summarize_directions, explore)
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
    assert _summarize_directions([], "ols").empty


def test_a_fold_inside_the_noise_range_abstains_instead_of_vetoing_the_sign():
    def summarize(values, real):
        rows = [dict(depth=1, target="Y", source="X", repeat=0, fold=i, coef_std=v, coef=v, real=ok)
                for i, (v, ok) in enumerate(zip(values, real))]
        return _summarize_directions(rows, "ols").iloc[0]

    wobble = summarize([.324, .3, .33, .31, -.05], [True, True, True, True, False])
    assert wobble.direction == "positive" and wobble.n_noise == 1 and wobble.agreement == 1
    assert wobble.n_positive == 4 and wobble.n_negative == 0
    split = summarize([.3, .3, -.3, .3], [True, True, True, True])             # a real opposing fold still counts
    assert split.direction == "mixed" and split.agreement == .75 and split.n_noise == 0
    assert summarize([.02, -.03], [False, False]).direction == "zero"
    assert np.isnan(summarize([.02], [False]).agreement)
    shrunk = summarize([.2, 0, .3], [True, False, True])                         # Elastic-Net zero abstains too
    assert shrunk.direction == "positive" and shrunk.n_noise == 1
    minority = summarize([.3, .3, .01, .02, -.02], [True, True, False, False, False])
    assert minority.direction == "zero" and minority.n_positive == 2     # 2 of 5 real folds: no stable sign


def test_a_small_dip_inside_a_long_climb_is_not_a_fall():
    frame = _frame()

    def formula(f):                                   # steady climb with a shallow wiggle in the middle
        return (2 * f.up + .05 * np.sin(6 * f.up) + .01 * f.noise_up - .01 * f.noise_down).to_numpy()

    rows = _profile_rows(_Function(formula), frame, ["up"], ["noise_up", "noise_down"], .95, 1, "Y", 0, 0)
    assert rows[0]["sign"] == 1


def test_linear_discovery_marks_chance_sized_coefficients_with_the_noise_columns():
    res = explore(_data(), "Y", config=ExploreConfig(folds=5, interactions=False))
    t = res.directions
    assert {"n_noise"} <= set(t.columns) and (t.n_positive + t.n_negative + t.n_noise == t.n_fits).all()
    strong = t[(t.target == "Y") & (t.source == "A")].iloc[0]
    assert strong.direction == "positive" and strong.n_noise == 0


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


# ------------------------------------------------------------------ LightGBM: average local effect profiles

class _Function:
    """A fitted-model stand-in: predicts from a formula, so the true profile is known."""

    def __init__(self, formula):
        self.formula = formula

    def predict(self, frame):
        return self.formula(frame)


def _frame(n=600, seed=0):
    r = np.random.default_rng(seed)
    cols = {c: r.normal(size=n) for c in ("up", "down", "ushape", "other", "noise_up", "noise_down")}
    return pd.DataFrame(cols)


def test_local_effects_recover_the_slope_of_a_known_function():
    frame = _frame()
    effects = _local_effects(lambda f: 2 * f.up.to_numpy(), frame, "up")
    quantiles = np.quantile(frame.up, np.linspace(0, 1, 11))
    assert (effects > 0).all() and effects.sum() == pytest.approx(2 * (quantiles[-1] - quantiles[0]), rel=.02)
    assert _local_effects(lambda f: f.up.to_numpy(), frame.assign(up=1.0), "up") is None   # nothing to profile


def test_profiles_are_judged_against_the_models_own_noise_columns():
    frame = _frame()
    jitter = np.random.default_rng(1).normal(size=len(frame)) * .02

    def formula(f):
        return (1.0 * f.up - .8 * f.down + .7 * f.ushape ** 2 + .01 * f.noise_up - .01 * f.noise_down).to_numpy()

    rows = _profile_rows(_Function(formula), frame, ["up", "down", "ushape", "other"],
                         ["noise_up", "noise_down"], .95, 1, "Y", 0, 0)
    by = {r["source"]: r for r in rows}
    assert by["up"]["sign"] == 1 and by["down"]["sign"] == -1
    assert by["ushape"]["sign"] == 9                         # rises and falls: neither sign
    assert by["other"]["sign"] == 0                          # no movement beyond the noise columns
    assert by["up"]["ale_net"] > 0 > by["down"]["ale_net"] and by["ushape"]["ale_range"] > 1
    assert all(r["basis"] == "ale" for r in rows)


def test_a_profile_that_rises_in_one_range_and_falls_in_another_is_mixed_not_averaged_away():
    frame = _frame()
    rows = _profile_rows(_Function(lambda f: (np.sin(2 * f.up)).to_numpy()), frame, ["up"],
                         ["noise_up", "noise_down"], .95, 1, "Y", 0, 0)
    assert rows[0]["sign"] == 9


@pytest.fixture(scope="module")
def lightgbm_result():
    pytest.importorskip("lightgbm")
    r = np.random.default_rng(4)
    n = 1500
    d = pd.DataFrame({k: r.normal(size=n) for k in ["Up", "Down", "U", "Inter1", "Inter2", "Noise1"]})
    d["Y"] = 1.0 * d.Up - .8 * d.Down + .7 * d.U ** 2 + .6 * d.Inter1 * d.Inter2 + r.normal(size=n)
    return explore(d, "Y", engine="lightgbm", config=ExploreConfig(folds=5, interactions=False, max_depth=1))


def test_lightgbm_directions_come_from_the_profile_and_keep_the_shape(lightgbm_result):
    table = lightgbm_result.directions
    assert (table.basis == "ale").all() and (table.model == "lightgbm").all() and table.coef.isna().all()
    direction = dict(zip(table.source, table.direction))
    assert direction["Up"] == "positive" and direction["Down"] == "negative"
    assert direction["U"] == "mixed"                             # U-shaped: not forced into a sign
    assert direction["Inter1"] in ("mixed", "zero") and direction["Inter2"] in ("mixed", "zero")
    assert direction["Noise1"] in ("zero", "mixed")
    up = table[table.source == "Up"].iloc[0]
    assert up.agreement == 1 and up.ale_net > 0 and up.ale_range >= up.ale_net


def test_lightgbm_effects_table_and_graph_state_what_the_sign_is(lightgbm_result):
    table = effects_table(lightgbm_result)
    up = table[table.predictor == "Up"].iloc[0]
    assert up.direction == "positive" and up.direction_basis == "ale" and up.profile_net > 0
    assert np.isnan(up.estimate)                                # no coefficient is invented
    fig = report.plot_explore(lightgbm_result, show_direction=True)
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert any("average local effect (ALE) profile" in t for t in texts)
    assert report.POSITIVE in _line_colours(fig) and report.MIXED_SIGN in _line_colours(fig)
    plt.close(fig)
