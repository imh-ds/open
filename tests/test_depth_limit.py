"""Hard depth cap: boundary behavior, honest truncation status and exports (software checks only)."""
import json

import numpy as np
import pandas as pd
import pytest

from opennet import explore_report as report
from opennet.explore import ExploreConfig, explore


def _chain(n=800, seed=2026):
    # X4 -> X1 -> Y <- X3 <- X2 ; X6..X10 noise. Depth 1 keeps X1 and X3, depth 2 keeps X4 and X2.
    r = np.random.default_rng(seed)
    d = {f"X{i}": r.normal(size=n) for i in (2, 4, 6, 7, 8)}
    d["X1"] = .5 * d["X4"] + r.normal(size=n)
    d["X3"] = .5 * d["X2"] + r.normal(size=n)
    d["Y"] = .4 * d["X1"] + .4 * d["X3"] + r.normal(size=n)
    return pd.DataFrame(d)


def _run(max_depth, data=None):
    return explore(_chain() if data is None else data, "Y", engine="ols",
                   config=ExploreConfig(folds=5, max_depth=max_depth, interactions=False))


@pytest.fixture(scope="module")
def uncapped():
    return _run(None)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "3"])
def test_invalid_caps_are_rejected(bad):
    with pytest.raises(ValueError, match="max_depth"):
        ExploreConfig(max_depth=bad)


def test_default_is_uncapped_and_reports_no_truncation(uncapped):
    assert ExploreConfig().max_depth is None
    assert uncapped.truncation == {} and "MAX_DEPTH" not in uncapped.stops.values()
    assert "Depth limit" not in uncapped.summary()


@pytest.mark.parametrize("cap", [1, 2])
def test_capped_run_matches_the_uncapped_prefix_and_fits_nothing_deeper(uncapped, cap):
    capped = _run(cap)
    assert int(capped.folds.depth.max()) == cap and int(capped.fit.depth.max()) == cap
    # the same earlier assessments, row for row
    prefix = uncapped.links[uncapped.links.depth <= cap].reset_index(drop=True)
    pd.testing.assert_frame_equal(capped.links.reset_index(drop=True), prefix)
    pd.testing.assert_frame_equal(
        capped.folds.reset_index(drop=True),
        uncapped.folds[uncapped.folds.depth <= cap].reset_index(drop=True))
    assert set(capped.nodes.variable) == set(uncapped.nodes[uncapped.nodes.depth <= cap].variable)


def test_cap_one_keeps_depth_one_and_marks_the_frontier_depth_limited():
    res = _run(1)
    assert set(res.nodes.variable) == {"Y", "X1", "X3"}
    assert res.stops == {"X1": "MAX_DEPTH", "X3": "MAX_DEPTH"}
    t = res.truncation
    assert t["max_depth"] == 1 and t["depth"] == 1 and sorted(t["frontier"]) == ["X1", "X3"]
    # the candidates not yet placed: everything except Y, X1 and X3
    assert sorted(t["unassessed_pool"]) == ["X2", "X4", "X6", "X7", "X8"]
    # nothing was fabricated for the unexecuted assessments
    assert set(res.links.target) == {"Y"}


def test_cap_reports_unassessed_not_dropped():
    res = _run(1)
    assert not ((res.links.target == "X1") | (res.links.target == "X3")).any()
    assert "not assessed" in res.summary() and "not a failed one" in res.summary()


def test_natural_end_before_a_cap_is_not_truncation(uncapped):
    """A cap that is never reached (cap 4 on a chain that ends at depth 2) reports nothing."""
    res = _run(4)
    assert res.truncation == {} and "MAX_DEPTH" not in res.stops.values()
    pd.testing.assert_frame_equal(res.links.reset_index(drop=True), uncapped.links.reset_index(drop=True))


def test_cap_four_stops_before_depth_five():
    """Root is depth 0: a cap of 4 keeps depth-4 nodes and fits no depth-5 targets."""
    r = np.random.default_rng(7)
    n = 1500
    data = pd.DataFrame({"X6": r.normal(size=n), "N": r.normal(size=n), "M": r.normal(size=n)})
    previous = "X6"
    for name in ("X5", "X4", "X3", "X2", "X1"):         # X6 -> X5 -> X4 -> X3 -> X2 -> X1 -> Y
        data[name] = .6 * data[previous] + r.normal(size=n)   # weak enough to avoid dependence groups
        previous = name
    data["Y"] = .8 * data["X1"] + r.normal(size=n)
    res = _run(4, data)
    depth = dict(zip(res.nodes.variable, res.nodes.depth))
    assert max(depth.values()) == 4 and depth["X4"] == 4
    assert "X5" not in depth and res.stops.get("X4") == "MAX_DEPTH"
    assert int(res.folds.depth.max()) == 4
    assert "X5" in res.truncation["unassessed_pool"]


def test_cap_with_empty_pool_is_not_called_truncation():
    r = np.random.default_rng(3)
    x = r.normal(size=600)
    data = pd.DataFrame({"X": x, "Y": x + r.normal(size=600)})
    res = _run(1, data)
    assert res.stops == {"X": "MAX_DEPTH"} and res.truncation == {}


def test_export_records_truncation_only_when_present(tmp_path, uncapped):
    capped = _run(1)
    report.write_explore(capped, tmp_path / "capped", plot=False)
    settings = json.loads((tmp_path / "capped" / "settings.json").read_text(encoding="utf-8"))
    assert settings["max_depth"] == 1 and settings["stops"] == {"X1": "MAX_DEPTH", "X3": "MAX_DEPTH"}
    assert settings["truncation"]["frontier"] == capped.truncation["frontier"]
    assert "Depth limit" in (tmp_path / "capped" / "summary.txt").read_text(encoding="utf-8")
    report.write_explore(uncapped, tmp_path / "full", plot=False)
    assert "truncation" not in json.loads((tmp_path / "full" / "settings.json").read_text(encoding="utf-8"))


def test_native_graph_legend_and_note_show_the_limit():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = report.plot_explore(_run(1))
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert any("depth-limited at max_depth = 1" in t for t in texts)
    assert any("not a failed retention" in t for t in texts)
    plt.close(fig)
    fig = report.plot_explore(_run(None))
    assert not any("depth-limited" in t.get_text() for t in fig.findobj(matplotlib.text.Text))
    plt.close(fig)
