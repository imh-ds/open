"""The regression-style effects table, the show_r2 toggle and edge-label placement beside a block."""
import re

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opennet import explore_report as report
from opennet.api import DiscoveryConfig, analyze, effects_table
from opennet.effects import EFFECT_COLUMNS
from opennet.explore import ExploreConfig, explore


def _data(n=2400, seed=3):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"P": r.normal(size=n), "N1": r.normal(size=n)})
    d["A"] = .7 * d.P + .7 * r.normal(size=n)
    d["B"] = .7 * d.P + .7 * r.normal(size=n)
    d["Y"] = .5 * d.A + .5 * d.B + r.normal(size=n)
    return d


@pytest.fixture(scope="module")
def split():
    return analyze(_data(), outcome="Y", config=DiscoveryConfig(folds=5, interactions=False,
                                                                blocks={"ab": ["A", "B"]}))


@pytest.fixture(scope="module")
def plain_split():
    return analyze(_data(), outcome="Y", config=DiscoveryConfig(folds=5, interactions=False))


def test_effects_table_has_one_row_per_drawn_link_with_effects_and_confirmation(plain_split):
    table = effects_table(plain_split)
    assert list(table.columns) == EFFECT_COLUMNS
    drawn = plain_split.exploration.links[plain_split.exploration.links.tier.isin(["kept", "flagged"])]
    assert len(table) == len(drawn)
    row = table[(table.predictor == "A") & (table.target == "Y")].iloc[0]
    link = drawn[(drawn.source == "A") & (drawn.target == "Y")].iloc[0]
    assert row.delta_r2 == pytest.approx(link.r2_median) and row.noise_bar == pytest.approx(link.noise_bar)
    assert row.folds_passed == "5/5" and row.tier == "kept"
    assert 0 <= row.p_adjusted <= 1 and row.confirmed in (True, False)
    assert row.target_r2_discovery > 0 and row.target_r2_confirmation > 0


def test_blocks_appear_as_one_row_with_no_group_test(split):
    table = effects_table(split)
    block = table[table.kind == "block"]
    assert list(block.predictor) == ["A+B"] and list(block.block) == ["ab"]
    assert block.p.isna().all() and block.confirmed.isna().all()
    assert block.confirmation_note.str.contains("not tested").all()
    members = table[table.kind == "unique"]
    assert set(zip(members.predictor, members.target)) >= {("P", "A"), ("P", "B")}
    everything = effects_table(split, include="all")
    assert len(everything) > len(table) and (everything.tier == "hidden").any()   # member diagnostics


def test_effects_table_for_a_plain_discovery_and_for_several_outcomes():
    res = explore(_data(), "Y", config=ExploreConfig(folds=5, interactions=False))
    table = effects_table(res)
    assert table.p.isna().all() and table.target_r2_confirmation.isna().all()
    assert (table.outcome == "Y").all()
    data = _data()
    data["Z"] = .6 * data.P + np.random.default_rng(1).normal(size=len(data))
    both = analyze(data, outcomes=["Y", "Z"], config=DiscoveryConfig(folds=5, interactions=False))
    combined = effects_table(both)
    assert set(combined.outcome) == {"Y", "Z"}
    with pytest.raises(ValueError, match="include"):
        effects_table(res, include="everything")


def test_effects_csv_is_exported(split, tmp_path):
    report.write_split(split, tmp_path, plot=False)
    saved = pd.read_csv(tmp_path / "effects.csv")
    assert list(saved.columns) == EFFECT_COLUMNS and len(saved) == len(effects_table(split))


def test_show_r2_false_removes_rings_labels_and_legend_entry(split):
    with_r2 = report.plot_explore(split.exploration)
    without = report.plot_explore(split.exploration, show_r2=False)
    labels = lambda fig: [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert any("R$^2$ (D)" in t for t in labels(with_r2))
    assert not any("R$^2$" in t or "ring: held-out" in t for t in labels(without))
    from matplotlib.patches import Wedge
    assert not [p for p in without.axes[0].patches if isinstance(p, Wedge)]
    plt.close(with_r2)
    plt.close(without)
    off = report.plot_explore(split.exploration, show_r2=False, edge_labels=None)    # effects only, no numbers
    assert not any(re.fullmatch(r"\.\d{3} \(\d+/\d+\)", t) for t in labels(off))
    plt.close(off)
    with pytest.raises(ValueError, match="show_r2"):
        report.GraphStyle(show_r2="no")


def test_edge_label_leaving_a_block_sits_beside_its_edge_not_on_the_block(split):
    from matplotlib.patches import FancyBboxPatch
    fig = report.plot_explore(split.exploration, show_r2=False)
    graph = fig.axes[0]
    box = next(p for p in graph.patches if isinstance(p, FancyBboxPatch))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    square = box.get_window_extent(renderer)
    label = next(t for t in graph.texts if re.fullmatch(r"\.\d{3} \(\d+/\d+\)", t.get_text())
                 and t.get_window_extent(renderer).y0 < square.y1 + 400)
    labels = [t for t in graph.texts if re.fullmatch(r"\.\d{3} \(\d+/\d+\)", t.get_text())]
    for text in labels:                                       # no edge label is drawn over the block node
        assert not text.get_window_extent(renderer).overlaps(square)
    assert label is not None
    plt.close(fig)
