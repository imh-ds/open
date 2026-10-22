"""Local node movement, distinct from label placement or global spacing."""
import numpy as np
import pandas as pd


def _links(*pairs):
    return pd.DataFrame([dict(depth=1, source=a, target=b, members=(a,), kind="unique", tier="kept")
                         for a, b in pairs])


def test_node_dodging_moves_obstructing_node_without_moving_quiet_nodes():
    from open.explore_report import GraphStyle, _dodge_node_positions, _point_segment

    positions = {"Y": (0., 0.), "A": (4., 0.), "B": (2., .2), "Quiet": (7., 5.)}
    moved, _ = _dodge_node_positions(positions, {}, _links(("A", "Y")), {"Y"}, {"Y", "A"},
                                    GraphStyle(), pinned={"Y", "A"})
    assert _point_segment(np.array(moved["B"]), np.array(moved["A"]), np.array(moved["Y"]))[0] > .55
    assert moved["Y"] == positions["Y"] and moved["A"] == positions["A"]
    assert moved["Quiet"] == positions["Quiet"]


def test_node_dodging_respects_fit_rings_and_is_not_uniform_scaling():
    from open.explore_report import GraphStyle, _dodge_node_positions

    positions = {"Y": (0., 0.), "A": (1.25, 0.), "B": (2.5, .4), "Quiet": (7., 5.)}
    args = (positions, {}, _links(("A", "Y"), ("B", "A")), {"Y"}, {"Y", "A"}, GraphStyle())
    moved, junctions = _dodge_node_positions(*args, pinned={"Y"})
    assert np.linalg.norm(np.array(moved["A"]) - moved["Y"]) >= 1.65
    assert moved["Y"] == positions["Y"]
    assert moved["Quiet"] == positions["Quiet"]
    assert moved["A"] != positions["A"]
    assert _dodge_node_positions(*args, pinned={"Y"}) == (moved, junctions)
    assert positions["A"] == (1.25, 0.)


def _edge_spec(source, target, text, key="e"):
    return dict(kind="edge", key=key, source=source, target=target, text=text, fontsize=8, lane=0.0)


def test_label_room_lengthens_an_edge_too_short_for_its_label():
    from open.explore_report import GraphStyle, _make_label_room, _point_segment

    positions = {"Y": (0., 0.), "A": (1.3, 0.), "Quiet": (0., 6.)}
    text = ".123 (10/10) ✓ (a long label)"
    moved, _, spots = _make_label_room(positions, {}, [_edge_spec("A", "Y", text)], _links(("A", "Y")),
                                       {"Y"}, {"Y"}, GraphStyle(), pinned={"Y"})
    assert moved["Y"] == positions["Y"] and moved["Quiet"] == positions["Quiet"]
    assert np.linalg.norm(np.array(moved["A"])) > 2.5          # long enough to hold the label
    assert _point_segment(np.array(spots["e"]), np.array(moved["A"]), np.array(moved["Y"]))[0] < 1e-6
    again = _make_label_room(positions, {}, [_edge_spec("A", "Y", text)], _links(("A", "Y")),
                             {"Y"}, {"Y"}, GraphStyle(), pinned={"Y"})
    assert again == (moved, {}, spots)


def test_label_room_spreads_edges_that_leave_a_node_together():
    from open.explore_report import GraphStyle, _make_label_room

    positions = {"Y": (0., 0.), "A": (3., 0.), "B": (6., .25)}
    specs = [_edge_spec("A", "Y", ".200 (5/5)", "a"), _edge_spec("B", "Y", ".150 (5/5)", "b")]
    moved, _, _ = _make_label_room(positions, {}, specs, _links(("A", "Y"), ("B", "Y")), {"Y"}, {"Y"},
                                   GraphStyle(), pinned={"Y"})
    angle = [np.arctan2(moved[v][1], moved[v][0]) for v in ("A", "B")]
    assert abs(angle[0] - angle[1]) > .3


def test_edge_labels_sit_on_their_own_edges():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from open.confirm import explore_and_confirm
    from open.explore import ExploreConfig
    from open.explore_report import _point_segment, plot_explore

    rng = np.random.default_rng(5)
    data = pd.DataFrame(rng.normal(size=(1000, 5)), columns=["A", "B", "C", "N1", "N2"])
    data["Y"] = .6 * data.A + .5 * data.B + .4 * data.C + rng.normal(size=1000)
    split = explore_and_confirm(data, "Y", config=ExploreConfig(max_depth=1, interactions=False), split_seed=1)
    figure = plot_explore(split.exploration, confirmation=split.confirmation)
    ax = figure.axes[0]
    figure.canvas.draw()
    lines = [line.get_transform().transform(line.get_xydata()) for line in ax.lines]
    for text in (t for t in ax.texts if "/10)" in t.get_text()):
        box = text.get_window_extent()
        c = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
        assert min(_point_segment(c, a, b)[0] for a, b in lines) < 2
    assert not any(isinstance(c, LineCollection) for c in ax.collections)    # no leader lines needed
    plt.close(figure)
