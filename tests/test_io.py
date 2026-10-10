"""Six-file export preserves consensus values and the complete run ledger."""

from __future__ import annotations

import csv
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from open.aggregation import Frequency, MagnitudeSummary, aggregate_stability
from open.config import (AnalysisConfig, OpenConfig, PerformanceCVConfig,
                         StabilityConfig, TuningCVConfig)
from open.consensus import (
    ConsensusEdge, ConsensusNode, ConsensusOmission, ConsensusSharedPair,
    ConsensusSharedRelation, build_consensus,
)
from open.graph import NetworkRun, StopReason
from open.local_target import LocalClass, LocalResult
from open.metadata import MetadataValidation, validate_metadata
from open.relevance import RelevanceContrast, RelevanceResult
from open.stability import OPENResult, fit_stable_open
from open.stability_types import StabilityBatch, StabilityRunRecord


ROOT = "R,öot"
A = "A,α"


def _metadata() -> MetadataValidation:
    return MetadataValidation(ROOT, (A, "B", "C"), (
        {"name": ROOT, "display_label": "Root, Δ"},
        {"name": A, "display_label": "Alpha, é"},
        {"name": "B", "display_label": None},
        {"name": "C", "display_label": "C"},
    ))


def _local(baseline: float, full: float, *, low_gain: bool) -> LocalResult:
    gain = baseline - full
    relative = gain / baseline if baseline else None
    candidates = (A, "B", "C")
    broad = {name: RelevanceContrast(0.0, None if low_gain else 0.0)
             for name in candidates}
    unique = {name: RelevanceContrast(0.0, None if low_gain else 0.0)
              for name in candidates}
    relevance = RelevanceResult(gain, relative, "LOW_TOTAL_GAIN" if low_gain else None,
                                broad, unique, baseline, full)
    return LocalResult(low_gain, "LOW_TOTAL_GAIN" if low_gain else None, ROOT,
                       gain, relative, relevance, None, {}, {}, None, (), (), (),
                       {}, frozenset())


def _result(*, populated: bool = False) -> OPENResult:
    first = NetworkRun(ROOT, {ROOT: 0}, {}, {ROOT: _local(0.0, 0.0, low_gain=True)},
                       (), (), {ROOT: StopReason.LOW_TOTAL_GAIN}, 3)
    second = NetworkRun(ROOT, {ROOT: 0}, {}, {ROOT: _local(2.0, 1.0, low_gain=False)},
                        (), (), {ROOT: StopReason.NO_PROMOTION}, 3)
    if populated:
        second = replace(second, local_results={},
                         stop_reasons={ROOT: StopReason.NO_UNASSIGNED})
    preflight = NetworkRun(ROOT, {ROOT: 0}, {}, {}, (), (), {}, 3,
                           StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE)
    batch = StabilityBatch(4, (
        StabilityRunRecord(0, 41, (0, 2), network=first),
        StabilityRunRecord(1, 42, (1, 3), failure_type="RuntimeError",
                           failure_message="failed, é"),
        StabilityRunRecord(2, 43, (0, 1), network=second),
        StabilityRunRecord(3, 44, (2, 3), network=preflight,
                           failure_type="PreflightStop",
                           failure_message="INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE"),
    ))
    summary = aggregate_stability(batch)
    consensus = build_consensus(summary, .5)
    if populated:
        raw = MagnitudeSummary(2, .4, .3, 1, .2, .2)
        sparse = MagnitudeSummary(1, 0.0, 0.0, 0, None, None)
        broad_edge = MagnitudeSummary(2, .6, .5, 2, .3, .25)
        unique_edge = MagnitudeSummary(2, .4, .3, 2, .2, .1)
        consensus = replace(
            consensus,
            nodes=(ConsensusNode(A, 1, ((1, .5),), .5, Frequency(1, 2)),
                   ConsensusNode("B", 2, ((2, 1.0),), 1.0, Frequency(2, 2))),
            edges=(ConsensusEdge(A, ROOT, LocalClass.PROXIMAL_UNIQUE,
                                 Frequency(1, 2), .5, ("B",), broad_edge,
                                 unique_edge),),
            shared_relations=(
                ConsensusSharedRelation(ROOT, (A, "B"), Frequency(1, 2), (),
                                        Frequency(2, 2), Frequency(2, 2), raw),
                ConsensusSharedRelation(ROOT, (A, "B", "C"), Frequency(1, 2), (),
                                        Frequency(2, 2), Frequency(2, 2), sparse),
            ),
            shared_pairs=(ConsensusSharedPair(ROOT, A, "B", Frequency(2, 2)),
                          ConsensusSharedPair(ROOT, "B", "C", Frequency(1, 2))),
            omissions=(ConsensusOmission("node", ("C",), Frequency(0, 2),
                                         "stability 0 < pi_min 0.5"),),
        )
    return OPENResult(batch, summary, consensus, OpenConfig(AnalysisConfig(ROOT, 41)),
                      (A, "B", "C"), None, "Git unavailable", 1.25, None,
                      "tracemalloc was already active")


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_root_only_export_has_six_files_exact_headers_and_run_rows(tmp_path):
    from open.io import write_open_outputs

    paths = write_open_outputs(_result(), tmp_path / "parent" / "out",
                               metadata=_metadata())
    assert [path.name for path in paths] == [
        "nodes.csv", "edges.csv", "shared_groups.csv", "shared_comembership.csv",
        "local_targets.csv", "analysis_metadata.json",
    ]
    assert all(path.parent == tmp_path / "parent" / "out" for path in paths)
    headers = [
        ["node", "display_label", "depth", "selection_probability",
         "depth_probability_0", "depth_probability_1", "depth_probability_2",
         "depth_probability_3", "selection_numerator", "selection_denominator"],
        ["source", "target", "relation", "broad_raw", "broad_norm", "unique_raw",
         "unique_norm", "broad_raw_median", "broad_norm_median",
         "unique_raw_median", "unique_norm_median", "stability",
         "target_reached_probability", "target_reached_numerator",
         "target_reached_denominator", "evaluated_probability", "screened_by",
         "stability_numerator", "stability_denominator", "evaluated_numerator",
         "evaluated_denominator"],
        ["target", "group_id", "members", "group_raw", "group_raw_median",
         "raw_count", "group_norm", "group_norm_median", "normalized_count",
         "stability", "stability_numerator", "stability_denominator"],
        ["target", "node_a", "node_b", "shared_probability", "numerator", "denominator"],
        ["target", "depth", "baseline_risk", "full_risk", "total_gain",
         "relative_total_gain", "learner", "n_candidates", "stop_reason",
         "run_index", "run_seed", "n_observed_target_rows", "n_missing_target_rows"],
    ]
    tables = [_rows(path) for path in paths[:5]]
    assert [head for head, _ in tables] == headers
    assert tables[0][1] == [{**dict.fromkeys(headers[0], ""),
                             "node": ROOT, "display_label": "Root, Δ", "depth": "0",
                             "selection_probability": "1.0", "depth_probability_0": "1.0",
                             "depth_probability_1": "0.0", "depth_probability_2": "0.0",
                             "depth_probability_3": "0.0",
                             "selection_numerator": "2", "selection_denominator": "2"}]
    assert [rows for _, rows in tables[1:4]] == [[], [], []]
    assert [(row["run_index"], row["run_seed"], row["baseline_risk"],
             row["full_risk"], row["total_gain"], row["relative_total_gain"],
             row["learner"], row["n_candidates"], row["stop_reason"])
            for row in tables[4][1]] == [
                ("0", "41", "0.0", "0.0", "0.0", "", "elastic_net", "3", "LOW_TOTAL_GAIN"),
                ("2", "43", "2.0", "1.0", "1.0", "0.5", "elastic_net", "3",
                 "NO_PROMOTION"),
            ]
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    assert list(payload) == sorted([
        "package_version", "git_revision", "git_revision_unavailable_reason", "config",
        "seed", "learner_versions", "dependency_versions", "preprocessing", "loss",
        "thresholds", "resampling_design", "variable_metadata_sha256", "batch_counts",
        "run_ledger", "consensus_omissions", "diagnostics", "elapsed_seconds",
        "peak_traced_memory_bytes", "peak_memory_unavailable_reason",
        "metadata_availability", "measurement_overlap", "exploration",
    ])
    assert payload["batch_counts"] == {"requested_runs": 4, "attempted_runs": 4,
                                       "completed_runs": 2, "failed_runs": 2}
    assert [(row["run_index"], row["seed"], row["sampled_positions"], row["completed"],
             row["failure_type"], row["failure_message"], row["stop_reason"])
            for row in payload["run_ledger"]] == [
                (0, 41, [0, 2], True, None, None, None),
                (1, 42, [1, 3], False, "RuntimeError", "failed, é", None),
                (2, 43, [0, 1], True, None, None, None),
                (3, 44, [2, 3], False, "PreflightStop",
                 "INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
                 "INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE"),
            ]
    assert payload["peak_traced_memory_bytes"] is None


def test_populated_consensus_exports_counts_magnitudes_pairs_and_unicode(tmp_path):
    from open.io import write_open_outputs

    paths = write_open_outputs(_result(populated=True), tmp_path / "out",
                               metadata=_metadata())
    _, nodes = _rows(paths[0])
    assert [(row["node"], row["display_label"], row["depth"],
             row["selection_numerator"], row["selection_denominator"])
            for row in nodes] == [(ROOT, "Root, Δ", "0", "2", "2"),
                                  (A, "Alpha, é", "1", "1", "2"),
                                  ("B", "B", "2", "2", "2")]
    assert [tuple(row[f"depth_probability_{depth}"] for depth in range(4))
            for row in nodes] == [
                ("1.0", "0.0", "0.0", "0.0"),
                ("0.0", "0.5", "0.0", "0.0"),
                ("0.0", "0.0", "1.0", "0.0"),
            ]
    _, edges = _rows(paths[1])
    assert len(edges) == 1
    assert edges[0] == {
        "source": A, "target": ROOT, "relation": "PROXIMAL_UNIQUE",
        "broad_raw": "0.6", "broad_norm": "0.3", "unique_raw": "0.4",
        "unique_norm": "0.2", "broad_raw_median": "0.5",
        "broad_norm_median": "0.25", "unique_raw_median": "0.3",
        "unique_norm_median": "0.1", "stability": "0.5",
        "target_reached_probability": "1.0", "target_reached_numerator": "2",
        "target_reached_denominator": "2", "evaluated_probability": "0.5",
        "screened_by": '["B"]', "stability_numerator": "1",
        "stability_denominator": "2", "evaluated_numerator": "1",
        "evaluated_denominator": "2",
    }
    _, groups = _rows(paths[2])
    assert [(row["group_id"], row["members"], row["group_raw"],
             row["group_raw_median"], row["raw_count"], row["group_norm"],
             row["group_norm_median"], row["normalized_count"], row["stability"])
            for row in groups] == [
                ('["R,öot",["A,α","B"]]', '["A,α","B"]', "0.4", "0.3", "2",
                 "0.2", "0.2", "1", "0.5"),
                ('["R,öot",["A,α","B","C"]]', '["A,α","B","C"]', "0.0", "0.0",
                 "1", "", "", "0", "0.5"),
            ]
    assert [(row["stability_numerator"], row["stability_denominator"])
            for row in groups] == [("1", "2"), ("1", "2")]
    _, pairs = _rows(paths[3])
    assert [(row["target"], row["node_a"], row["node_b"], row["shared_probability"],
             row["numerator"], row["denominator"]) for row in pairs] == [
                (ROOT, A, "B", "1.0", "2", "2"),
                (ROOT, "B", "C", "0.5", "1", "2"),
            ]
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    assert payload["consensus_omissions"] == [{
        "structure": "node", "identity": ["C"],
        "frequency": {"numerator": 0, "denominator": 2},
        "reason": "stability 0 < pi_min 0.5",
    }]


def test_validated_measurement_overlap_survives_export_with_metadata_hash(tmp_path):
    from open.diagnostics import hash_variable_metadata
    from open.io import write_open_outputs

    metadata = validate_metadata(
        [ROOT, A, "B", "C"], [
            {"name": ROOT, "role": "root", "type": "continuous"},
            {"name": A, "role": "candidate", "type": "continuous",
             "scale_items": ["shared", "a"]},
            {"name": "B", "role": "candidate", "type": "continuous",
             "scale_items": ["shared", "b"]},
            {"name": "C", "role": "candidate", "type": "continuous",
             "scale_items": ["c"]},
        ], ROOT,
    )
    paths = write_open_outputs(_result(), tmp_path / "out", metadata=metadata)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    assert payload["measurement_overlap"] == {ROOT: False, A: True,
                                               "B": True, "C": False}
    assert payload["variable_metadata_sha256"] == hash_variable_metadata(metadata)


def test_public_fit_then_export_produces_readable_six_file_artifact(tmp_path):
    from open.io import write_open_outputs

    rng = np.random.default_rng(613)
    x = rng.normal(size=24)
    z = rng.normal(size=24)
    data = pd.DataFrame({"x": x,
                         "y": 1.5 * x + rng.normal(scale=.2, size=24),
                         "z": z})
    metadata = validate_metadata(
        data.columns, [
            {"name": "x", "role": "candidate", "type": "continuous"},
            {"name": "y", "role": "root", "type": "continuous"},
            {"name": "z", "role": "candidate", "type": "continuous"},
        ], "y",
    )
    config = OpenConfig(
        AnalysisConfig("y", 613, max_depth=1),
        performance_cv=PerformanceCVConfig(folds=2, repeats=1),
        tuning_cv=TuningCVConfig(folds=2),
        stability=StabilityConfig(fraction=.8, n_resamples=1),
    )
    result = fit_stable_open(data, metadata.root, tuple(data.columns), config)
    paths = write_open_outputs(result, tmp_path / "out", metadata=metadata)

    assert len(paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    for path in paths[:5]:
        headers, rows = _rows(path)
        assert headers
        assert isinstance(rows, list)
    payload = json.loads(paths[5].read_text(encoding="utf-8"))
    assert payload["batch_counts"]["completed_runs"] == 1
    assert payload["run_ledger"][0]["completed"] is True


def test_existing_destination_is_preserved_and_repeat_exports_match(tmp_path):
    from open.io import write_open_outputs

    destination = tmp_path / "out"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_open_outputs(_result(), destination, metadata=_metadata())
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out"]
    first = write_open_outputs(_result(populated=True), tmp_path / "one",
                               metadata=_metadata())
    second = write_open_outputs(_result(populated=True), tmp_path / "two",
                                metadata=_metadata())
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]


def test_serialization_failure_removes_staged_directory(tmp_path, monkeypatch):
    import open.io as io

    def fail_json(*args, **kwargs):
        raise OSError("injected JSON failure")

    monkeypatch.setattr(io.json, "dump", fail_json)
    with pytest.raises(OSError, match="injected JSON failure"):
        io.write_open_outputs(_result(), tmp_path / "parent" / "out",
                              metadata=_metadata())
    assert not (tmp_path / "parent" / "out").exists()
    assert list((tmp_path / "parent").iterdir()) == []


def test_destination_created_at_publication_boundary_is_preserved(tmp_path, monkeypatch):
    import open.io as io

    publish = io._publish_no_replace
    destination = tmp_path / "out"

    def collide(stage, target):
        assert target == destination
        target.mkdir()
        return publish(stage, target)

    monkeypatch.setattr(io, "_publish_no_replace", collide)
    with pytest.raises(FileExistsError):
        io.write_open_outputs(_result(), destination, metadata=_metadata())
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out"]
