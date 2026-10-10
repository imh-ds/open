"""Exploratory evidence preserves selection, opportunity and signed strength."""
from dataclasses import replace
import json

import pytest

from open.aggregation import aggregate_stability
from open.config import AnalysisConfig, OpenConfig, ThresholdConfig
from open.consensus import build_consensus
from open.graph import NetworkRun, StopReason
from open.io import write_open_outputs
from open.local_target import LocalClass, finalize_local_target
from open.metadata import validate_metadata
from open.relevance import compute_relevance
from open.stability import OPENResult
from open.stability_types import StabilityBatch, StabilityRunRecord
from test_aggregation import _chain, _local, _shared_group


def _batch():
    first = _chain()
    first = replace(first, local_results={
        t: replace(local, input_candidates=tuple(local.relevance.broad))
        for t, local in first.local_results.items()})
    negative = replace(_local("a", ("b",), kinds={"b": LocalClass.NEGLIGIBLE},
                              unique_raw={"b": -1.}), input_candidates=("b",))
    second = NetworkRun("y", {"y": 0, "a": 1}, {"a": ("y",)},
                        {"y": first.local_results["y"], "a": negative},
                        (first.core_edges[0],), (), {"a": StopReason.NO_PROMOTION}, 2)
    third = NetworkRun("y", {"y": 0}, {},
                       {"y": replace(_local("y", ("a", "b")), input_candidates=("a", "b"))},
                       (), (), {"y": StopReason.NO_PROMOTION}, 2)
    return StabilityBatch(3, tuple(StabilityRunRecord(i, 20+i, (0, 1), network=n)
                                  for i, n in enumerate((first, second, third))))


def _metadata():
    return validate_metadata(("y", "a", "b"), [
        {"name": n, "display_label": n.upper(), "type": "continuous",
         "role": "root" if n == "y" else "candidate"} for n in ("y", "a", "b")], "y")


def _result(batch=None):
    batch = batch or _batch()
    summary = aggregate_stability(batch)
    return OPENResult(batch, summary, build_consensus(summary, .7),
                      OpenConfig(AnalysisConfig("y", 20, max_depth=2)),
                      ("y", "a", "b"), None, "test", 0., None, "test")


def _payload(result=None):
    from open.exploration import build_exploration
    return build_exploration(result or _result(), _metadata())


def test_local_candidate_provenance_survives_low_gain_stop():
    relevance = compute_relevance(1., .99, broad_risks={"x": .99},
                                 ablated_risks={"x": 1.}, tau_gain=.02)
    local = finalize_local_target(target="y", candidates=("x",), relevance=relevance,
                                  classifications=None, grouping=None,
                                  thresholds=ThresholdConfig())
    assert local.input_candidates == ("x",)


def test_all_observed_edges_and_nodes_survive_consensus_omission():
    payload = _payload()
    assert {n["name"] for n in payload["nodes"]} == {"y", "a", "b"}
    edge = next(e for e in payload["edges"] if e["source"] == "b")
    assert (edge["count"], edge["denominator"], edge["eligible_count"]) == (1, 3, 2)
    assert edge["conditional_probability"] == .5
    assert edge["consensus"] is False
    assert edge["strength_count"] == 2
    assert edge["raw_median"] == .5
    assert edge["baseline_relative_median"] == .25


def test_unknown_legacy_eligibility_is_not_treated_as_nonselection():
    batch = _batch()
    second = batch.runs[1].network
    unknown = replace(second.local_results["a"], input_candidates=None,
                      classes={}, broad={}, unique={},
                      relevance=replace(second.local_results["a"].relevance, broad={}, unique={}))
    second = replace(second, local_results={**second.local_results, "a": unknown})
    batch = replace(batch, runs=(batch.runs[0], replace(batch.runs[1], network=second), batch.runs[2]))
    edge = next(e for e in _payload(_result(batch))["edges"] if e["source"] == "b")
    assert edge["eligible_count"] == 1
    assert edge["eligibility_unknown_count"] == 1
    assert edge["conditional_probability"] is None


def test_shared_relation_remains_joint_and_reports_measurement_scope():
    shared = _shared_group(("a", "b"))
    shared = replace(shared, local_results={"y": replace(shared.local_results["y"],
                                                        input_candidates=("a", "b"))})
    other = _batch().runs[2].network
    batch = StabilityBatch(2, tuple(StabilityRunRecord(i, i, (0, 1), network=n)
                                  for i, n in enumerate((shared, other))))
    payload = _payload(_result(batch))
    assert payload["edges"] == []
    group = payload["groups"][0]
    assert group["members"] == ["a", "b"]
    assert (group["count"], group["eligible_count"], group["strength_count"]) == (1, 2, 1)
    assert group["strength_scope"] == "selected joint ablations"


def test_export_adds_exploration_without_changing_consensus_csvs(tmp_path):
    result = _result()
    assert not result.consensus.edges
    paths = write_open_outputs(result, tmp_path / "bundle", metadata=_metadata())
    assert len(paths) == 6
    assert len((tmp_path / "bundle/edges.csv").read_text().splitlines()) == 1
    metadata = json.loads((tmp_path / "bundle/analysis_metadata.json").read_text())
    assert metadata["exploration"]["schema_version"] == 1
    assert len(metadata["exploration"]["edges"]) == 2


def test_gain_relative_median_has_its_own_measurement_count():
    batch = _batch()
    network = batch.runs[1].network
    local = network.local_results["a"]
    contrasts = {"b": replace(local.relevance.unique["b"], normalized=None)}
    local = replace(local, relevance=replace(local.relevance, unique=contrasts))
    network = replace(network, local_results={**network.local_results, "a": local})
    batch = replace(batch, runs=(batch.runs[0], replace(batch.runs[1], network=network), batch.runs[2]))
    edge = next(e for e in _payload(_result(batch))["edges"] if e["source"] == "b")
    assert edge["strength_count"] == edge["baseline_relative_count"] == 2
    assert edge["gain_relative_count"] == 1
    assert edge["gain_relative_median"] == batch.runs[0].network.local_results["a"].relevance.unique["b"].normalized
