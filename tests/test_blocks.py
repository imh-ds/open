"""A priori measurement blocks: joint assessment, block recursion and validation (software checks only).

Synthetic data only. These tests show how the code behaves; they do not calibrate the joint test or show
that group-level retention is reliable.
"""
import json

import numpy as np
import pandas as pd
import pytest

from opennet import explore_report as report
from opennet.explore import ExploreConfig, assess_target, explore

N = 1500
CFG = dict(folds=5, interactions=False)


def _data(seed=11, n=N):
    """P drives A and B (corr(A, B) ~ .5, below the automatic .70); Y depends on both; N1/N2 are noise."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"P": r.normal(size=n), "N1": r.normal(size=n), "N2": r.normal(size=n)})
    d["A"] = .7 * d.P + r.normal(size=n) * .7
    d["B"] = .7 * d.P + r.normal(size=n) * .7
    d["Y"] = .5 * d.A + .5 * d.B + r.normal(size=n)
    return d


def _run(blocks=None, data=None, **kw):
    cfg = ExploreConfig(**CFG, blocks=blocks, **kw)
    return explore(_data() if data is None else data, "Y", engine="ols", config=cfg)


@pytest.fixture(scope="module")
def declared():
    return _run({"ab": ["A", "B"]})


@pytest.fixture(scope="module")
def plain():
    return _run()


def _block_link(res, depth=1, target="Y"):
    return res.links[(res.links.origin == "declared") & (res.links.depth == depth) & (res.links.target == target)]


def test_moderately_correlated_block_is_assessed_though_automatic_grouping_does_not_trigger(declared, plain):
    assert not (plain.links.kind == "shared").any()              # corr(A, B) ~ .5 < dependence_pair
    row = _block_link(declared).iloc[0]
    assert row.kind == "shared" and row.block == "ab" and row.members == ("A", "B") and row.tier == "kept"


def test_block_promotes_members_together_with_joint_evidence_not_individual_support(declared):
    depth = dict(zip(declared.nodes.variable, declared.nodes.depth))
    assert depth["A"] == depth["B"] == 1
    members = declared.links[(declared.links.kind == "unique") & declared.links.source.isin(["A", "B"])
                             & (declared.links.target == "Y")]
    assert set(members.block) == {"ab"}
    # member scores are kept as diagnostics but never draw or promote on their own
    assert (members.tier != "kept").all()
    assert "joint evidence only" in declared.summary() and "not individually supported" in declared.summary()


def test_informative_and_uninformative_members_are_promoted_together_without_calling_both_supported():
    d = _data()
    d["Y"] = .6 * d.A + np.random.default_rng(1).normal(size=len(d))
    res = _run({"mix": ["A", "N1"]}, d)
    assert _block_link(res).iloc[0].tier == "kept"               # driven by A; joint test cannot tell
    depth = dict(zip(res.nodes.variable, res.nodes.depth))
    assert depth["A"] == depth["N1"] == 1
    n1 = res.links[(res.links.kind == "unique") & (res.links.source == "N1") & (res.links.target == "Y")].iloc[0]
    a = res.links[(res.links.kind == "unique") & (res.links.source == "A") & (res.links.target == "Y")].iloc[0]
    assert n1.fold_tier == "dropped" and a.fold_tier == "kept"   # diagnostics stay honest
    assert a.tier == "hidden"                                    # but the member is not placed by its own tier


def test_opposing_member_effects_are_assessed_jointly():
    d = _data()
    d["Y"] = .5 * d.A - .5 * d.B + np.random.default_rng(2).normal(size=len(d))
    res = _run({"ab": ["A", "B"]}, d)
    assert _block_link(res).iloc[0].tier == "kept"               # joint relevance gives no single sign


def test_failed_block_promotes_nothing_even_if_a_member_qualifies_alone():
    d = _data()
    d["Y"] = np.random.default_rng(3).normal(size=len(d))        # nothing predicts Y
    res = _run({"nn": ["N1", "N2"]}, d)
    assert _block_link(res).iloc[0].tier != "kept"
    assert set(res.nodes.variable) == {"Y"} and res.stops == {"Y": "NOTHING_KEPT"}


def test_declared_blocks_take_precedence_over_automatic_groups():
    d = _data()
    d["C"] = d.A + .3 * np.random.default_rng(4).normal(size=len(d))   # strongly correlated with A
    d["Y"] = d.Y + .4 * d.C
    res = _run({"ab": ["A", "B"]}, d)
    auto = res.links[(res.links.kind == "shared") & (res.links.origin != "declared")]
    assert not any({"A", "B"} & set(m) for m in auto.members)    # block members never join automatic groups
    assert not (set(res.pairs.a) | set(res.pairs.b)) & {"A", "B"}   # nor into automatic dependence pairs
    pure = _run(None, d)                                         # without the block, A ~ C is an automatic pair
    assert ("A", "C") in set(zip(pure.pairs.a, pure.pairs.b))


def test_block_traversal_has_no_within_block_chain_and_promotes_across_layers(declared):
    d = declared
    block = {"A", "B"}
    assert not ((d.links.source.isin(block)) & (d.links.target.isin(block))).any()
    deeper = d.links[(d.links.depth == 2) & d.links.target.isin(block)]
    assert len(deeper) and not deeper.source.isin(block).any()   # members are depth-2 targets, no member predictor
    assert set(d.folds[d.folds.depth == 2].source) & block == set()
    assert ("P", "A") in set(zip(d.links.source, d.links.target))


def test_no_block_behavior_is_unchanged(plain):
    for empty in (None, {}, ()):
        same = _run(empty)
        pd.testing.assert_frame_equal(same.links, plain.links)
        assert same.settings == plain.settings and "blocks" not in same.settings
    assert "block" not in plain.links and "origin" not in plain.links
    assert "Declared blocks" not in plain.summary()


@pytest.mark.parametrize("blocks, message", [
    ({"x": ["A"]}, "at least two"),
    ({"x": ["A", "A"]}, "duplicate members"),
    ({"x": ["A", "B"], "y": ["B", "N1"]}, "overlap"),
    ({"x": "AB"}, "sequence of column names"),
    ({"": ["A", "B"]}, "nonempty string"),
    ([("x", ["A", "B"]), ("x", ["N1", "N2"])], "duplicate block name"),
    ({"x": ["A", 3]}, "column-name strings"),
])
def test_invalid_declarations_are_rejected_with_clear_messages(blocks, message):
    with pytest.raises(ValueError, match=message):
        ExploreConfig(blocks=blocks)


@pytest.mark.parametrize("blocks, kwargs, message", [
    ({"x": ["A", "ghost"]}, {}, "candidate predictors"),
    ({"x": ["A", "Y"]}, {}, "candidate predictors"),
    ({"x": ["A", "B"]}, {"controls": ["A"]}, "controls and covariates must be distinct|candidate predictors"),
    ({"x": ["A", "N1"]}, {"covariates": ["N1"]}, "candidate predictors"),
])
def test_roles_are_checked_against_the_analysis(blocks, kwargs, message):
    with pytest.raises(ValueError, match=message):
        explore(_data(), "Y", engine="ols", config=ExploreConfig(**CFG, blocks=blocks), **kwargs)


def test_assessing_with_a_partly_available_block_is_an_error():
    d = _data()
    folds = (np.arange(len(d)) % 5,)
    with pytest.raises(ValueError, match="only partly"):
        assess_target(d, "Y", ("A", "N1"), folds, "ols", ExploreConfig(**CFG, blocks={"ab": ["A", "B"]}))


def test_export_and_graph_distinguish_the_block_from_individual_edges(declared, tmp_path):
    report.write_explore(declared, tmp_path, plot=False)
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert settings["blocks"] == {"ab": ["A", "B"]}
    links = pd.read_csv(tmp_path / "links.csv")
    assert {"block", "origin"} <= set(links.columns)
    assert links[links.origin == "declared"].shape[0] >= 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = report.plot_explore(declared)
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert any("declared block" in t for t in texts) and any("not confirmed as a group" in t for t in texts)
    plt.close(fig)


def test_blocks_flow_through_the_confirmation_workflow_without_group_confirmation():
    from opennet.api import analyze
    res = analyze(_data(n=2400), outcome="Y", config=ExploreConfig(folds=3, interactions=False,
                                                                    blocks={"ab": ["A", "B"]}))
    assert res.discovery.settings["blocks"] == {"ab": ["A", "B"]}
    shared = res.confirmation[res.confirmation.kind == "shared"]
    assert len(shared) and shared.note.str.contains("not tested").all() and shared.p.isna().all()


def _wide(k, shared_driver, seed=8, n=2400, interactions=False):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"Ctx": r.normal(size=n), "Noise": r.normal(size=n)})
    names = [f"D{i}" for i in range(1, k + 1)]
    for m in names:
        driver = d["Ctx"] if shared_driver else r.normal(size=n)
        if not shared_driver:
            d[f"Drv_{m}"] = driver
        d[m] = .6 * driver + .5 * r.normal(size=n)
    d["Y"] = sum(.3 * d[m] for m in names) + r.normal(size=n)
    cfg = ExploreConfig(folds=5, interactions=interactions, blocks={"big": names})
    return explore(d, "Y", engine="ols", config=cfg), names


@pytest.mark.parametrize("k, shared", [(2, True), (4, True), (6, True), (6, False)])
def test_large_blocks_draw_as_one_container_with_members_inside_and_no_member_edges(k, shared):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from opennet.explore_report import _block_layout, GraphStyle, _drawn
    res, names = _wide(k, shared)
    assert _block_link(res).iloc[0].tier == "kept"
    links = _drawn(res.links, True, False)
    pos, ring, candidate, junctions, shapes = _block_layout(res, links, GraphStyle(), "Y")
    g = shapes["big"]
    for m in names:                                           # every member sits inside the one container
        d = np.asarray(pos[m], float) - g["centre"]
        along = float(np.clip(d @ g["axis"], -g["half"], g["half"]))
        assert np.linalg.norm(d - along * g["axis"]) < g["w"] + 1e-6
    pts = np.array([pos[m] for m in names])                   # members never overlap each other
    gaps = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(k) * 9
    assert gaps.min() > .8
    fig = report.plot_explore(res)
    polys = [p for p in fig.axes[0].patches if isinstance(p, Polygon) and len(p.get_xy()) > 20]
    assert len(polys) == 1                                    # one container, not one band per member
    plt.close(fig)


def test_block_graph_with_interactions_and_flagged_links_still_draws():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res, _ = _wide(4, True, interactions=True)
    fig = report.plot_explore(res)
    plt.close(fig)
