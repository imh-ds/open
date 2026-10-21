"""Controls (adjust only) and covariates (tested and drawn, never targets)."""
import numpy as np
import pandas as pd
import pytest

from open.confirm import confirmation_fit, frozen_network
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
