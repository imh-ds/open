"""Repeated splits: ``repeats=1`` is bit-identical to the original exploration."""
import numpy as np
import pandas as pd
import pytest

from open.explore import ExploreConfig, assess_target, explore
from reference import explore_r1 as original


def _data(n=400, seed=7):
    # X1 -> Y <- X3 ; X4 -> X1 ; substitutes S1/S2 -> Y ; Q a proxy of X3 ; A x B interaction ; N noise
    r = np.random.default_rng(seed)
    d = {v: r.normal(size=n) for v in ("X3", "X4", "L", "A", "B", "N")}
    d["X1"] = .6 * d["X4"] + r.normal(size=n)
    d["Q"] = d["X3"] + .6 * r.normal(size=n)
    d["S1"], d["S2"] = d["L"] + .1 * r.normal(size=n), d["L"] + .1 * r.normal(size=n)
    d["Y"] = (.35 * d["X1"] + .35 * d["X3"] + .3 * d["L"] + .35 * d["A"] * d["B"] + .2 * d["A"]
              + r.normal(size=n))
    return pd.DataFrame(d).drop(columns="L")


def _compare(new, old):
    for name in ("nodes", "links", "pairs", "bars"):
        pd.testing.assert_frame_equal(getattr(new, name), getattr(old, name), check_exact=True)
    pd.testing.assert_frame_equal(new.folds.drop(columns="repeat"), old.folds, check_exact=True)
    assert (new.folds.repeat == 0).all()
    assert new.stops == old.stops
    settings = dict(new.settings)
    assert settings.pop("repeats") == 1
    assert settings == old.settings


# A few interaction pairs keep the tests fast; the identity holds for any setting.
CASES = [("ols", dict(max_depth=2, max_interaction_pairs=4)), ("ols", dict(max_depth=1, seed=3, max_interaction_pairs=4)),
         ("elastic_net", dict(max_depth=1, interactions=False)), ("lightgbm", dict(max_depth=1, max_interaction_pairs=2))]


@pytest.mark.parametrize("engine, settings", CASES)
def test_repeats_1_is_bit_identical_to_the_original(engine, settings):
    if engine == "lightgbm":
        pytest.importorskip("lightgbm")
    d = _data()
    new = explore(d, "Y", engine=engine, config=ExploreConfig(**settings))
    old = original.explore(d, "Y", engine=engine, config=original.ExploreConfig(**settings))
    assert len(new.links) and len(new.pairs)
    if settings.get("interactions", True):
        assert (new.links.kind == "interaction").any()
    _compare(new, old)


@pytest.fixture(scope="module")
def three():
    return explore(_data(), "Y", engine="ols", config=ExploreConfig(max_depth=1, repeats=3, max_interaction_pairs=4))


def test_repeats_pool_folds_and_noise(three):
    f = three.folds
    assert sorted(f.repeat.unique()) == [0, 1, 2]
    inter = f[f.contrast == "interaction"]
    assert sorted(inter.repeat.unique()) == [0, 1, 2] and inter.fold.nunique() == 30
    assert f.fold.nunique() == 30 and (f.fold // 10 == f.repeat).all()
    for v in three.links.pass_rate:                              # a share of 30 folds
        assert abs(v * 30 - round(v * 30)) < 1e-9
    noise = f[f.noise & (f.contrast == "unique") & (f.source == "N")]
    assert noise.groupby("repeat").value.mean().nunique() == 3   # each repeat has its own noise column
    one = explore(_data(), "Y", engine="ols", config=ExploreConfig(max_depth=1, max_interaction_pairs=4))
    r0 = f[f.repeat == 0].drop(columns=["repeat", "bar", "passed"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r0, one.folds.drop(columns=["repeat", "bar", "passed"]), check_exact=True)
    rate = three.pairs.set_index(["a", "b"]).loc[("S1", "S2")]
    assert abs(rate.rate * 30 - round(rate.rate * 30)) < 1e-9
    assert "x 3 repeats" in three.summary()


def test_repeats_are_reproducible_and_validated(three):
    again = explore(_data(), "Y", engine="ols", config=ExploreConfig(max_depth=1, repeats=3, max_interaction_pairs=4))
    pd.testing.assert_frame_equal(again.links, three.links, check_exact=True)
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError):
            ExploreConfig(repeats=bad)
    d = _data()
    with pytest.raises(ValueError):
        assess_target(d, "Y", ("X1", "X3"), np.arange(len(d)) % 10, "ols", ExploreConfig(repeats=2))
