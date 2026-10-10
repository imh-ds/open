"""Consensus thresholds, records, and public stability orchestration."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from open.aggregation import (
    CandidateClassFrequency,
    CandidateSummary,
    Frequency,
    MagnitudeSummary,
    NodeDepthFrequency,
    NodeSummary,
    PathSummary,
    ScreenedBySummary,
    SharedPairSummary,
    SharedRelationFrequency,
    StabilitySummary,
    TargetSummary,
)
from open.config import (
    AnalysisConfig,
    OpenConfig,
    PerformanceCVConfig,
    StabilityConfig,
    ThresholdConfig,
    TuningCVConfig,
)
from open.consensus import build_consensus
from open.graph import NetworkRun, StopReason
from open.local_target import LocalClass
from open.stability import StabilityBatchError, fit_stable_open


def _summary():
    den = 10
    mag = MagnitudeSummary(8, .4, .3, 7, .2, .15)
    candidate = CandidateSummary(
        "y", "x",
        (
            CandidateClassFrequency("y", "x", LocalClass.PROXIMAL_UNIQUE,
                                    Frequency(7, den)),
            CandidateClassFrequency("y", "x", LocalClass.NEGLIGIBLE,
                                    Frequency(3, den)),
        ),
        Frequency(8, den), mag, mag,
    )
    return StabilitySummary(
        12, 12, 10, 2, den,
        (
            NodeSummary("x", (NodeDepthFrequency(1, Frequency(4, den)),
                               NodeDepthFrequency(2, Frequency(4, den))),
                        Frequency(8, den)),
            NodeSummary("weak", (NodeDepthFrequency(1, Frequency(6, den)),),
                        Frequency(6, den)),
        ),
        (TargetSummary("y", Frequency(10, den), Frequency(8, den)),),
        (candidate,),
        (SharedRelationFrequency("y", ("a", "b"), Frequency(7, den), mag),),
        (SharedPairSummary("y", "a", "b", Frequency(8, den)),),
        (
            ScreenedBySummary("y", "w", "s", LocalClass.SCREENED_BY_SET,
                              Frequency(7, den)),
            ScreenedBySummary("y", "w", "t", LocalClass.SCREENED_BY_PARTNER,
                              Frequency(6, den)),
        ),
        (PathSummary(("x", "m", "y"), Frequency(7, den)),
         PathSummary(("a", "b", "y"), Frequency(6, den))),
    )


def test_consensus_includes_threshold_equality_and_uses_shallower_depth_tie():
    result = build_consensus(_summary(), .70)

    assert result.pi_min == .70
    assert result.denominator == 10
    assert (result.requested_runs, result.attempted_runs,
            result.completed_runs, result.failed_runs) == (12, 12, 10, 2)
    assert [(item.name, item.depth) for item in result.nodes] == [("x", 1)]
    assert result.targets[0].reached == Frequency(10, 10)
    assert result.targets[0].evaluated == Frequency(8, 10)
    assert result.nodes[0].depth_probabilities == ((1, .4), (2, .4))
    assert len(result.classes) == 1
    assert result.classes[0].relation is LocalClass.PROXIMAL_UNIQUE
    assert result.classes[0].target_evaluated == Frequency(8, 10)
    assert len(result.edges) == 1
    assert result.edges[0].evaluated_probability == .8
    assert result.edges[0].broad.raw_mean == .4
    assert result.shared_relations[0].members == ("a", "b")
    assert result.shared_relations[0].target_evaluated == Frequency(8, 10)
    assert result.shared_relations[0].magnitude.raw_count == 8
    assert result.shared_relations[0].magnitude.raw_mean == .4
    assert result.shared_relations[0].magnitude.raw_median == .3
    assert result.shared_relations[0].magnitude.normalized_count == 7
    assert result.shared_relations[0].magnitude.normalized_mean == .2
    assert result.shared_relations[0].magnitude.normalized_median == .15
    assert result.shared_pairs[0].frequency == Frequency(8, 10)
    assert len(result.screened_by) == 1
    assert result.screened_by[0].relation is LocalClass.SCREENED_BY_SET
    assert result.paths[0].nodes == ("x", "m", "y")
    assert result.intervals == ()

    omitted = {(item.structure, item.identity) for item in result.omissions}
    assert ("node", ("weak",)) in omitted
    assert ("class", ("y", "x", LocalClass.NEGLIGIBLE)) in omitted
    assert ("screened_by", ("y", "w", "t", LocalClass.SCREENED_BY_PARTNER)) in omitted
    assert ("path", (("a", "b", "y"),)) in omitted


def test_consensus_requires_completed_denominator_and_valid_threshold():
    with pytest.raises(ValueError, match="pi_min"):
        build_consensus(_summary(), 1.1)
    with pytest.raises(ValueError, match="completed run"):
        build_consensus(replace(_summary(), denominator=0))


def _config():
    return OpenConfig(
        analysis=AnalysisConfig("y", 17, max_depth=2),
        thresholds=ThresholdConfig(stability=.70),
        performance_cv=PerformanceCVConfig(folds=2, repeats=1),
        tuning_cv=TuningCVConfig(folds=2),
        stability=StabilityConfig(fraction=.8, n_resamples=2, final_resamples=2),
    )


def _small_data():
    rng = np.random.default_rng(52)
    x = rng.normal(size=14)
    z = rng.normal(size=14)
    return pd.DataFrame({"y": 1.5 * x + .2 * z, "x": x, "z": z})


def test_fit_stable_open_is_repeatable_and_saves_consensus():
    data, config = _small_data(), _config()
    first = fit_stable_open(data, "y", ("y", "x", "z"), config, n_resamples=2)
    second = fit_stable_open(data, "y", ("y", "x", "z"), config, n_resamples=2)

    assert [record.run_index for record in first.batch.runs] == [0, 1]
    assert [record.seed for record in first.batch.runs] == [17, 18]
    assert [record.sampled_positions for record in first.batch.runs] == [
        record.sampled_positions for record in second.batch.runs
    ]
    assert first.batch.completed_runs == 2
    assert first.stability.denominator == 2
    assert first.consensus.denominator == 2
    assert first.consensus == second.consensus
    assert isinstance(first.consensus.nodes, tuple)


def test_zero_completed_runs_raise_error_with_full_ledger(monkeypatch):
    from open.stability_types import StabilityBatch, StabilityRunRecord

    batch = StabilityBatch(2, (
        StabilityRunRecord(0, 17, (), failure_type="RuntimeError", failure_message="bad"),
        StabilityRunRecord(
            1, 18, (),
            network=NetworkRun("y", {"y": 0}, {}, {}, (), (), {}, 2,
                               StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE),
            failure_type="PreflightStop",
            failure_message=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE.value,
        ),
    ))
    monkeypatch.setattr("open.stability.run_stability_batch", lambda *args, **kwargs: batch)
    with pytest.raises(StabilityBatchError) as caught:
        fit_stable_open(_small_data(), "y", ("y", "x", "z"), _config())
    assert caught.value.batch is batch
    assert caught.value.batch.failed_runs == 2
    assert caught.value.batch.runs[1].failure_type == "PreflightStop"
    assert caught.value.batch.runs[1].network.stop_reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
