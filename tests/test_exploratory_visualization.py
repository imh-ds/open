"""Saved exploratory plots separate frequency, strength and opportunity."""
from dataclasses import replace
import json

import pytest

from open.config import VisualizationConfig
from open.io import write_open_outputs
from open.visualization import load_visualization_data, plot_open_network
from test_exploration import _metadata, _result
from test_visualization import _bundle


def _export(tmp_path):
    directory = tmp_path / "new"
    write_open_outputs(_result(), directory, metadata=_metadata())
    return directory


def test_new_default_loads_observed_edges_below_consensus_threshold(tmp_path):
    data = load_visualization_data(_export(tmp_path))
    assert data.view == "exploratory"
    assert len(data.edges) == 2
    edge = next(e for e in data.edges if e.source == "b")
    assert (edge.count, edge.denominator, edge.eligible_count) == (1, 3, 2)
    assert edge.baseline_relative_median == .25
    assert not edge.consensus


def test_explicit_consensus_and_historical_fallback_are_labeled(tmp_path):
    assert load_visualization_data(_export(tmp_path), view="consensus").view == "consensus"
    old = load_visualization_data(_bundle(tmp_path / "old"))
    assert old.view == "consensus"
    with pytest.raises(ValueError, match="exploratory evidence"):
        load_visualization_data(tmp_path / "old", view="exploratory")


def test_boolean_opt_out_preserves_consensus_view(tmp_path):
    result = _result()
    result = replace(result, config=replace(result.config,
                                           visualization=VisualizationConfig(show_unstable=False)))
    write_open_outputs(result, tmp_path / "out", metadata=_metadata())
    assert load_visualization_data(tmp_path / "out").view == "consensus"
    assert load_visualization_data(tmp_path / "out", view="exploratory").view == "exploratory"


@pytest.mark.parametrize("field,value", [("denominator", 0), ("count", 4),
    ("eligible_count", 0), ("conditional_probability", .9), ("raw_median", None),
    ("baseline_relative_median", float("nan")), ("gain_relative_count", 0),
    ("gain_relative_median", None)])
def test_malformed_exploratory_counts_and_measurements_rejected(tmp_path, field, value):
    directory = _export(tmp_path)
    path = directory / "analysis_metadata.json"
    payload = json.loads(path.read_text())
    payload["exploration"]["edges"][1][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="analysis_metadata.json.*exploration"):
        load_visualization_data(directory)


def test_plot_encodes_strength_and_frequency_separately_and_filters_only_display(tmp_path):
    from matplotlib.patches import FancyArrowPatch
    data = load_visualization_data(_export(tmp_path))
    figure = plot_open_network(data)
    arrows = [p for p in figure.axes[0].patches if isinstance(p, FancyArrowPatch)]
    assert sorted(p.get_alpha() for p in arrows) == pytest.approx([.5, .75])
    assert sorted(p.get_linewidth() for p in arrows) == pytest.approx([2.4, 4.])
    evidence = "\n".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "1/3 overall" in evidence and "1/2 eligible" in evidence
    assert "exploratory" in figure.axes[0].get_title().lower()
    filtered = plot_open_network(data, min_frequency=.5)
    assert len([p for p in filtered.axes[0].patches if isinstance(p, FancyArrowPatch)]) == 1
    assert len(data.edges) == 2


@pytest.mark.parametrize("value", [-.1, 1.1, True, float("nan")])
def test_invalid_display_filter_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="min_frequency"):
        plot_open_network(load_visualization_data(_export(tmp_path)), min_frequency=value)


def test_one_weak_predictor_does_not_become_maximum_width(tmp_path):
    from matplotlib.patches import FancyArrowPatch
    data = load_visualization_data(_export(tmp_path))
    weak = replace(data.edges[0], baseline_relative_median=.01)
    figure = plot_open_network(replace(data, edges=(weak,)))
    arrow = next(p for p in figure.axes[0].patches if isinstance(p, FancyArrowPatch))
    assert arrow.get_linewidth() == pytest.approx(1.12)
    negative = replace(weak, baseline_relative_median=-.2, raw_median=-.4)
    figure = plot_open_network(replace(data, edges=(negative,)))
    arrow = next(p for p in figure.axes[0].patches if isinstance(p, FancyArrowPatch))
    assert arrow.get_linewidth() == .8
    assert any("-20.0%" in t.get_text() for t in figure.axes[1].texts)


def test_shared_group_draws_one_joint_arrow_without_unique_member_edges(tmp_path):
    from matplotlib.patches import FancyArrowPatch
    from open.stability_types import StabilityBatch, StabilityRunRecord
    from test_aggregation import _shared_group
    network = _shared_group(("a", "b"))
    result = _result(StabilityBatch(1, (StabilityRunRecord(0, 20, (0, 1), network=network),)))
    from open.exploration import build_exploration
    from open.exploratory_visualization import load_exploratory_payload
    legacy = load_visualization_data(_export(tmp_path), view="consensus")
    data = load_exploratory_payload(build_exploration(result, _metadata()),
                                    replace(legacy, completed_runs=1))
    assert len(data.groups) == 1 and not data.edges
    figure = plot_open_network(data)
    assert sum(isinstance(p, FancyArrowPatch) for p in figure.axes[0].patches) == 1
    assert len(figure.axes[0].lines) == 2
    assert any("selected joint fits" in t.get_text() for t in figure.axes[1].texts)
