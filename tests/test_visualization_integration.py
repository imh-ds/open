"""A saved P5.1 analysis can be reloaded and rendered without refitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from open.config import (
    AnalysisConfig,
    OpenConfig,
    PerformanceCVConfig,
    StabilityConfig,
    TuningCVConfig,
)
from open.io import write_open_outputs
from open.metadata import validate_metadata
from open.stability import fit_stable_open
from open.visualization import load_visualization_data, plot_open_network


def test_saved_bundle_matches_consensus_and_renders_png(tmp_path_factory):
    rng = np.random.default_rng(613)
    x = rng.normal(size=32)
    z = rng.normal(size=32)
    frame = pd.DataFrame({
        "x": x,
        "outcome": 2.0 * x + rng.normal(scale=0.15, size=32),
        "z": z,
    })
    metadata = validate_metadata(
        frame.columns,
        [
            {"name": "x", "role": "candidate", "type": "continuous"},
            {"name": "outcome", "role": "root", "type": "continuous"},
            {"name": "z", "role": "candidate", "type": "continuous"},
        ],
        "outcome",
    )
    config = OpenConfig(
        AnalysisConfig("outcome", 613, max_depth=1),
        performance_cv=PerformanceCVConfig(folds=2, repeats=1),
        tuning_cv=TuningCVConfig(folds=2),
        stability=StabilityConfig(fraction=0.8, n_resamples=2),
    )
    result = fit_stable_open(frame, metadata.root, tuple(frame.columns), config)
    output_dir = tmp_path_factory.mktemp("p5_2_smoke")
    paths = write_open_outputs(result, output_dir / "bundle", metadata=metadata)

    assert {path.name for path in paths} == {
        "nodes.csv", "edges.csv", "shared_groups.csv",
        "shared_comembership.csv", "local_targets.csv", "analysis_metadata.json",
    }
    assert all(path.is_file() for path in paths)

    loaded = load_visualization_data(output_dir / "bundle", view="consensus")
    consensus = result.consensus
    assert any(
        edge.source == "x" and edge.target == "outcome"
        and edge.relation.value == "PROXIMAL_UNIQUE"
        for edge in consensus.edges
    )
    assert loaded.root == metadata.root
    assert loaded.completed_runs == consensus.completed_runs == consensus.denominator
    assert len(loaded.nodes) == len(consensus.nodes) + 1  # export adds the root
    assert len(loaded.edges) == len(consensus.edges)
    assert len(loaded.groups) == len(consensus.shared_relations)
    assert len(loaded.shared_comembership) == len(consensus.shared_pairs)
    assert {node.name for node in loaded.nodes} == {
        metadata.root, *(node.name for node in consensus.nodes),
    }
    assert {
        (edge.source, edge.target, edge.relation):
        (edge.stability_numerator, edge.stability_denominator,
         edge.evaluation_numerator, edge.evaluation_denominator,
         edge.target_reached_numerator, edge.target_reached_denominator)
        for edge in loaded.edges
    } == {
        (edge.source, edge.target, edge.relation.value):
        (edge.frequency.numerator, edge.frequency.denominator,
         target.evaluated.numerator, target.evaluated.denominator,
         target.reached.numerator, target.reached.denominator)
        for edge in consensus.edges
        for target in consensus.targets if target.target == edge.target
    }
    assert {
        (group.target, group.members): (group.numerator, group.denominator)
        for group in loaded.groups
    } == {
        (group.target, group.members):
        (group.frequency.numerator, group.frequency.denominator)
        for group in consensus.shared_relations
    }
    assert {
        (pair.target, pair.node_a, pair.node_b):
        (pair.numerator, pair.denominator)
        for pair in loaded.shared_comembership
    } == {
        (pair.target, pair.member_a, pair.member_b):
        (pair.frequency.numerator, pair.frequency.denominator)
        for pair in consensus.shared_pairs
    }

    image_path = output_dir / "open-network.png"
    figure = plot_open_network(loaded, output_path=image_path)
    assert figure is not None
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
