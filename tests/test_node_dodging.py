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
