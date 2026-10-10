"""Tests for the immutable Phase 4 run ledger."""

import pytest

from open.graph import NetworkRun, StopReason
from open.local_target import LocalResult
from open.relevance import RelevanceResult
from open.stability_types import StabilityBatch, StabilityRunRecord


def _completed(index: int) -> StabilityRunRecord:
    result = LocalResult(
        stop=False,
        reason=None,
        target="root",
        total_gain=1.0,
        total_gain_relative=0.5,
        relevance=RelevanceResult(1.0, 0.5, None, {}, {}, 2.0, 1.0),
        pair_ablations=None,
        broad={},
        unique={},
        pair_classes=None,
        shared_components=(),
        retained_pairs=(),
        dependence_components=(),
        classes={},
        promoted=frozenset(),
    )
    network = NetworkRun(
        root="root",
        assigned_depth={"root": 0},
        parents={},
        local_results={"root": result},
        core_edges=(),
        shared_relations=(),
        stop_reasons={"root": StopReason.NO_PROMOTION},
        max_depth=3,
    )
    return StabilityRunRecord(index, 100 + index, (0, 2), network=network)


def _preflight_stopped(index: int) -> StabilityRunRecord:
    network = NetworkRun(
        root="root",
        assigned_depth={"root": 0},
        parents={},
        local_results={},
        core_edges=(),
        shared_relations=(),
        stop_reasons={},
        max_depth=3,
        stop_reason=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE,
    )
    return StabilityRunRecord(
        index,
        100 + index,
        (0,),
        network=network,
        failure_type="PreflightStop",
        failure_message=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE.value,
    )


def test_run_record_represents_completed_and_failed_attempts():
    completed = _completed(0)
    failed = StabilityRunRecord(
        1, 101, (1, 3), failure_type="RuntimeError", failure_message="fit failed"
    )

    assert completed.completed
    assert completed.network is not None
    assert not failed.completed
    assert failed.network is None
    assert failed.failure_type == "RuntimeError"


def test_preflight_stop_is_retained_but_not_counted_as_completed():
    run = _preflight_stopped(0)
    batch = StabilityBatch(1, (run,))

    assert run.network.stop_reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
    assert not run.completed
    assert batch.completed_runs == 0
    assert batch.failed_runs == 1
    assert not batch.complete


@pytest.mark.parametrize(
    "positions",
    [[0, 1], (1, 0), (0, 0), (-1,), (True,)],
)
def test_run_record_rejects_invalid_sample_positions(positions):
    with pytest.raises(ValueError):
        StabilityRunRecord(0, 10, positions, failure_type="Error", failure_message="x")


def test_batch_counts_failures_and_requires_every_planned_run_for_completion():
    failed = StabilityRunRecord(
        1, 101, (1, 3), failure_type="RuntimeError", failure_message="fit failed"
    )
    batch = StabilityBatch(3, (_completed(0), failed, _completed(2)))

    assert batch.requested_runs == 3
    assert batch.attempted_runs == 3
    assert batch.completed_runs == 2
    assert batch.failed_runs == 1
    assert not batch.complete
    assert StabilityBatch(2, (_completed(0), _completed(1))).complete


def test_batch_rejects_missing_or_reordered_planned_run_indices():
    with pytest.raises(ValueError, match="consecutive"):
        StabilityBatch(2, (_completed(1),))
    with pytest.raises(ValueError, match="consecutive"):
        StabilityBatch(2, (_completed(0), _completed(2)))


def test_run_record_requires_exactly_one_outcome():
    with pytest.raises(ValueError, match="requires"):
        StabilityRunRecord(0, 10, (), failure_type="Error")
    with pytest.raises(ValueError, match="cannot also"):
        StabilityRunRecord(0, 10, (), network=_completed(0).network,
                           failure_type="Error", failure_message="x")
