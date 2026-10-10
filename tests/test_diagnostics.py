"""Export metadata retains fit provenance and exact diagnostic denominators."""

import hashlib
import json
from dataclasses import replace

import pandas as pd
import pytest

from open.aggregation import Frequency, aggregate_stability
from open.config import AnalysisConfig, OpenConfig
from open.consensus import ConsensusOmission, build_consensus
from open.diagnostics import build_analysis_metadata, hash_variable_metadata
from open.graph import NetworkRun, StopReason
from open.grouping import PairAblation, PairAblationResult, SharedComponent
from open.local_target import LocalResult
from open.metadata import MetadataValidation
from open.relevance import RelevanceResult
from open.risk import FoldRisk, RiskResult
import open.stability as stability
from open.stability import OPENResult, fit_stable_open
from open.stability_types import StabilityBatch, StabilityRunRecord


def _metadata() -> MetadataValidation:
    return MetadataValidation(
        "root", ("x", "y"),
        ({"name": "root", "role": "root", "display_label": "Röot"},
         {"name": "x", "role": "candidate", "display_label": "X"},
         {"name": "y", "role": "candidate", "display_label": "Y"}),
    )


def _result() -> OPENResult:
    folds = (FoldRisk(0, 0, ("x",), (), 1.0, None),
             FoldRisk(0, 1, ("x",), (), 1.0, None))
    risk = RiskResult("root", ("x", "y"), ("x",), "elastic_net", "mse", "id", 12,
                      folds, 1.0, 2, 3)
    pair = PairAblation("x", "y", ((0, 0), (0, 1)), 2, 2, 1.0, 1.0,
                        2.0, 1.0, 1.5, 1.0, 0.5, 0.5, risk)
    component = SharedComponent(("x", "y"), False, 0.2, 0.2, risk)
    local = LocalResult(
        False, None, "root", 1.0, 0.5,
        RelevanceResult(1.0, 0.5, None, {}, {}, 2.0, 1.0),
        PairAblationResult((), (pair,)), {}, {}, None, (component,), (), (), {},
        frozenset(), outer_model_fit_count=7, tuning_model_fit_count=11,
    )
    network = NetworkRun("root", {"root": 0}, {}, {"root": local}, (), (),
                         {"root": StopReason.NO_PROMOTION}, 3)
    preflight = NetworkRun("root", {"root": 0}, {}, {}, (), (), {}, 3,
                           StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE)
    batch = StabilityBatch(4, (
        StabilityRunRecord(0, 12, (0, 2, 4), network=network),
        StabilityRunRecord(1, 13, (1, 3), failure_type="RuntimeError",
                           failure_message="learner failed"),
        StabilityRunRecord(2, 14, (0, 1), network=preflight,
                           failure_type="PreflightStop",
                           failure_message="INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE"),
    ))
    summary = aggregate_stability(batch)
    consensus = build_consensus(summary, 0.7)
    consensus = replace(consensus, omissions=consensus.omissions + (
        ConsensusOmission("node", ("x",), Frequency(0, 1), "below pi_min"),
    ))
    return OPENResult(batch, summary, consensus, OpenConfig(AnalysisConfig("root", 12)),
                      ("x", "y"), None, "Git unavailable", 1.25, None,
                      "tracemalloc was already active")


def test_metadata_hash_is_canonical_over_keys_but_preserves_row_order():
    metadata = _metadata()
    expected = hashlib.sha256(json.dumps(metadata.variables, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    assert hash_variable_metadata(metadata) == expected
    reordered_keys = replace(metadata, variables=tuple(dict(reversed(list(row.items())))
                                                      for row in metadata.variables))
    assert hash_variable_metadata(reordered_keys) == expected
    reversed_rows = replace(metadata, variables=tuple(reversed(metadata.variables)))
    assert hash_variable_metadata(reversed_rows) != expected


def test_metadata_rejects_root_and_candidate_mismatch():
    result = _result()
    with pytest.raises(ValueError, match="root"):
        build_analysis_metadata(result, replace(_metadata(), root="other"))
    with pytest.raises(ValueError, match="candidate"):
        build_analysis_metadata(result, replace(_metadata(), candidates=("x",)))


def test_metadata_contains_versions_availability_ledger_omissions_and_counts():
    payload = build_analysis_metadata(_result(), _metadata())
    assert payload["package_version"] is None
    assert "fit time" in payload["metadata_availability"]["package_version"]
    assert "scikit-learn" in payload["dependency_versions"]
    assert payload["dependency_versions"]["scikit-learn"] is None
    assert "elastic_net" in payload["learner_versions"]
    assert payload["git_revision"] is None
    assert payload["metadata_availability"]["git_revision"] == "Git unavailable"
    assert payload["metadata_availability"]["peak_traced_memory_bytes"] == "tracemalloc was already active"
    assert payload["config"]["analysis"]["random_seed"] == payload["seed"] == 12
    assert payload["batch_counts"] == {"requested_runs": 4, "attempted_runs": 3,
                                        "completed_runs": 1, "failed_runs": 2}
    assert [(row["run_index"], row["seed"], row["sampled_positions"], row["completed"])
            for row in payload["run_ledger"]] == [
                (0, 12, [0, 2, 4], True), (1, 13, [1, 3], False),
                (2, 14, [0, 1], False)]
    assert payload["run_ledger"][1]["failure_message"] == "learner failed"
    assert payload["run_ledger"][2]["failure_type"] == "PreflightStop"
    assert {"structure": "node", "identity": ["x"],
            "frequency": {"numerator": 0, "denominator": 1},
            "reason": "below pi_min"} in payload["consensus_omissions"]
    assert payload["diagnostics"] == {
        "n_targets_evaluated": 1, "n_pair_ablations": 2,
        "n_group_ablations": 1, "n_group_ablation_folds": 2,
        "n_outer_model_fits": 7, "n_tuning_model_fits": 11,
        "n_model_fits": 18,
    }
    assert payload["elapsed_seconds"] == 1.25
    json.dumps(payload, allow_nan=False)


def test_missing_distribution_versions_have_explicit_availability(monkeypatch):
    source = _result()
    monkeypatch.setattr(stability, "run_stability_batch", lambda *args, **kwargs: source.batch)

    def unavailable_package(name):
        raise stability.distribution_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(stability.distribution_metadata, "version", unavailable_package)
    result = fit_stable_open(pd.DataFrame({"root": [1.0], "x": [2.0], "y": [3.0]}),
                             "root", ("x", "y"), source.config)
    payload = build_analysis_metadata(result, _metadata())
    assert payload["package_version"] is None
    assert "open-predictive-expansion" in payload["metadata_availability"]["package_version"]
    assert payload["dependency_versions"]["scikit-learn"] is None
    assert "scikit-learn" in payload["metadata_availability"]["dependency_versions.scikit-learn"]
    assert payload["learner_versions"]["elastic_net"] is None
    assert "scikit-learn" in payload["metadata_availability"]["learner_versions.elastic_net"]


def test_export_uses_versions_captured_when_fit_started(monkeypatch):
    source = _result()
    monkeypatch.setattr(stability, "run_stability_batch", lambda *args, **kwargs: source.batch)
    monkeypatch.setattr(stability, "read_git_revision", lambda: ("fit-revision", None))
    monkeypatch.setattr(stability.distribution_metadata, "version",
                        lambda name: f"fit-{name}")
    result = fit_stable_open(pd.DataFrame({"root": [1.0], "x": [2.0], "y": [3.0]}),
                             "root", ("x", "y"), source.config)
    monkeypatch.setattr(stability.distribution_metadata, "version",
                        lambda name: f"export-{name}")

    payload = build_analysis_metadata(result, _metadata())
    assert payload["package_version"] == "fit-open-predictive-expansion"
    assert payload["dependency_versions"]["scikit-learn"] == "fit-scikit-learn"
    assert payload["learner_versions"]["elastic_net"] == "fit-scikit-learn"
