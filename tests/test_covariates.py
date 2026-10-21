"""Controls (adjust only) and covariates (tested and drawn, never targets)."""
import numpy as np
import pandas as pd
import pytest

from open.confirm import confirm, confirmation_fit, frozen_network
from open.explore import ExploreConfig, explore

CFG = ExploreConfig(max_depth=2, interactions=False)


def _data(n=900, seed=3):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"cohort": r.normal(size=n), "gender": r.integers(0, 2, n).astype(float),
                      "E": r.normal(size=n), "N1": r.normal(size=n)})
    d["X1"] = .7 * d.cohort + r.normal(size=n)          # a proxy of cohort, no effect of its own
    d["A"] = .5 * d.E + .6 * d.gender + r.normal(size=n)
    d["Y"] = .5 * d.cohort + .45 * d.A + r.normal(size=n)
    return d


@pytest.fixture(scope="module")
def adjusted():
    return explore(_data(), "Y", engine="ols", config=CFG, controls=["cohort"], covariates=["gender"])


def test_without_adjustment_the_nuisance_variable_becomes_a_target():
    res = explore(_data(), "Y", engine="ols", config=CFG)
    assert "cohort" in set(res.nodes.variable)
    assert ("X1", "cohort") in set(zip(res.links.source, res.links.target))


def test_controls_are_in_every_model_but_never_tested_or_drawn(adjusted):
    assert "cohort" not in set(adjusted.links.source) | set(adjusted.nodes.variable)
    assert "cohort" not in set(adjusted.folds.source)
    x1 = adjusted.links[(adjusted.links.source == "X1") & (adjusted.links.target == "Y")]
    assert x1.tier.iloc[0] == "dropped"                 # its proxy signal is adjusted away
    assert adjusted.settings["controls"] == ["cohort"]
    assert "Adjusted for (controls" in adjusted.summary()


def test_covariates_are_tested_in_every_pool_and_never_become_targets(adjusted):
    pools = (adjusted.folds[(adjusted.folds.contrast == "unique") & ~adjusted.folds.noise]
             .groupby("target").source.unique().map(set))
    assert set(pools.index) == {"Y", "A"} and all("gender" in p for p in pools)
    assert "gender" not in set(adjusted.nodes.variable)
    kept = adjusted.links[adjusted.links.tier == "kept"]
    assert ("gender", "A") in set(zip(kept.source, kept.target))
    fit = adjusted.fit.set_index("target")
    assert "gender" in fit.loc["A", "network"] and "cohort" not in fit.loc["Y", "network"]


def test_confirmation_uses_the_same_adjustment(adjusted):
    frozen = frozen_network(adjusted)
    assert all("cohort" in p and "gender" in p for p in frozen.pool)
    new = _data(seed=8)
    out = confirmation_fit(adjusted, new)
    assert out.r2_network.notna().all()
    with pytest.raises(ValueError):
        confirmation_fit(adjusted, new.drop(columns="cohort"))


@pytest.mark.parametrize("kw", [dict(controls=["Y"]), dict(controls=["A"], covariates=["A"]),
                                dict(predictors=["A", "E"], covariates=["A"]), dict(controls=["cohort", "cohort"])])
def test_invalid_controls_and_covariates_raise(kw):
    with pytest.raises(ValueError):
        explore(_data(300), "Y", engine="ols", config=CFG, **kw)


def test_no_controls_or_covariates_leaves_settings_unchanged():
    res = explore(_data(300), "Y", engine="ols", config=ExploreConfig(max_depth=1, interactions=False))
    assert "controls" not in res.settings and "covariates" not in res.settings


def _covariate_chain(n=900, seed=4):
    r = np.random.default_rng(seed)
    c, d = r.normal(size=(2, n))
    x = 2 * c + .5 * d + r.normal(size=n)
    return pd.DataFrame({"Y": 2 * x + .8 * d + r.normal(size=n), "X": x, "C": c, "D": d})


def test_final_target_is_evaluated_against_covariates_and_controls():
    res = explore(_covariate_chain(), "Y", predictors=["X"], covariates=["C"], controls=["D"],
                  config=ExploreConfig(folds=5, interactions=False))
    assert set(res.fit.target) == {"Y", "X"}
    kept = res.links[res.links.tier == "kept"]
    assert ("C", "X") in set(zip(kept.source, kept.target))
    assert set(res.nodes.variable) == {"Y", "X"}
    assert res.nodes.set_index("variable").loc["X", "parents"] == ("Y",)
    assert set(res.folds.target) == {"Y", "X"} and res.stops == {"X": "NOTHING_KEPT"}
    fit = res.fit.set_index("target").loc["X"]
    assert fit.pool_size == fit.network_size == 1 and fit.network == ("C",)
    frozen = frozen_network(res)
    assert frozen.loc[frozen.target == "X", "pool"].iloc[0] == ("C", "D")
    new = _covariate_chain(seed=12)
    confirmed = confirm(res, new)
    terminal = confirmed[(confirmed.target == "X") & (confirmed.source == "C")]
    assert len(terminal) == 1 and terminal.confirmed.iloc[0]
    new_fit = confirmation_fit(res, new)
    assert set(new_fit.target) == {"Y", "X"} and np.isfinite(new_fit.r2_network).all()


@pytest.mark.parametrize("predictors", [None, []])
@pytest.mark.parametrize("adjust", [False, True])
def test_initial_covariate_only_pool_is_evaluated_once(predictors, adjust):
    data = _covariate_chain()[["Y", "C", "D"] if adjust else ["Y", "C"]]
    res = explore(data, "Y", predictors=predictors, covariates=["C"], controls=["D"] if adjust else None,
                  config=ExploreConfig(folds=5, interactions=False))
    kept = res.links[res.links.tier == "kept"]
    assert list(kept.source) == ["C"] and list(kept.target) == ["Y"]
    assert list(res.nodes.variable) == ["Y"] and list(res.fit.target) == ["Y"]
    assert set(res.folds.depth) == {1} and res.stops == {"Y": "NOTHING_KEPT"}
    assert res.fit.network.iloc[0] == ("C",)
    expected_pool = ("C", "D") if adjust else ("C",)
    assert frozen_network(res).pool.iloc[0] == expected_pool


@pytest.mark.parametrize("predictors", [None, []])
def test_controls_alone_do_not_form_a_tested_pool(predictors):
    with pytest.raises(ValueError, match="predictor|covariate|candidate"):
        explore(_covariate_chain()[["Y", "D"]], "Y", predictors=predictors, controls=["D"],
                config=ExploreConfig(folds=5, interactions=False))


def test_final_covariate_pool_respects_max_depth():
    res = explore(_covariate_chain(), "Y", predictors=["X"], covariates=["C"], controls=["D"],
                  config=ExploreConfig(folds=5, max_depth=1, interactions=False))
    assert set(res.nodes.variable) == {"Y", "X"} and list(res.fit.target) == ["Y"]
    assert set(res.folds.target) == {"Y"} and res.stops == {"X": "MAX_DEPTH"}


def test_empty_pool_without_covariates_keeps_existing_stop_behavior():
    res = explore(_covariate_chain(), "Y", predictors=["X"], controls=["D"],
                  config=ExploreConfig(folds=5, interactions=False))
    assert set(res.nodes.variable) == {"Y", "X"} and list(res.fit.target) == ["Y"]
    assert res.stops == {"X": "NO_CANDIDATES"}
