"""Whole-pipeline stability resampling and run-ledger behavior."""

import numpy as np
import pandas as pd
import pytest

import open.stability as stability
from open.config import AnalysisConfig, OpenConfig, StabilityConfig
from open.graph import NetworkRun, StopReason
from open.local_target import LocalResult
from open.relevance import RelevanceResult


def fixture(*, n=10, fraction=0.8, seed=17, n_resamples=3):
    data = pd.DataFrame({
        "y": np.arange(n, dtype=float),
        "x": np.arange(n, dtype=float) * 2,
    })
    config = OpenConfig(
        AnalysisConfig("y", seed),
        stability=StabilityConfig(fraction=fraction, n_resamples=n_resamples),
    )
    return data, config


def preflight_network(root="y"):
    return NetworkRun(
        root=root,
        assigned_depth={root: 0},
        parents={},
        local_results={},
        core_edges=(),
        shared_relations=(),
        stop_reasons={},
        max_depth=3,
        stop_reason=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE,
    )


def completed_network(root="y"):
    local = LocalResult(
        stop=False,
        reason=None,
        target=root,
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
    return NetworkRun(
        root=root,
        assigned_depth={root: 0},
        parents={},
        local_results={root: local},
        core_edges=(),
        shared_relations=(),
        stop_reasons={root: StopReason.NO_PROMOTION},
        max_depth=3,
    )


def test_seeded_subsamples_are_reproducible_and_use_exact_sorted_floor_size(monkeypatch):
    data, config = fixture(n=11, fraction=0.8, seed=10)
    calls = []

    def build(sample, root, variables, run_config):
        calls.append((tuple(sample.index), run_config.analysis.random_seed))
        return completed_network(root)

    monkeypatch.setattr(stability, "build_network_once", build)
    first = stability.run_stability_batch(data, "y", tuple(data.columns), config)
    first_calls = calls.copy()
    calls.clear()
    second = stability.run_stability_batch(data, "y", tuple(data.columns), config)

    assert first_calls == calls
    assert [run.seed for run in first.runs] == [10, 11, 12]
    assert [len(run.sampled_positions) for run in first.runs] == [8, 8, 8]
    assert all(tuple(sorted(run.sampled_positions)) == run.sampled_positions for run in first.runs)
    assert [seed for _, seed in first_calls] == [10, 11, 12]
    assert all(run.completed for run in first.runs)
    assert first.complete


def test_fractional_size_floors_and_one_row_minimum(monkeypatch):
    observed = []
    monkeypatch.setattr(
        stability,
        "build_network_once",
        lambda sample, root, variables, config: observed.append(len(sample)) or completed_network(root),
    )
    data, config = fixture(n=7, fraction=0.5, n_resamples=1)
    batch = stability.run_stability_batch(data, "y", tuple(data.columns), config)
    assert batch.runs[0].sampled_positions and len(batch.runs[0].sampled_positions) == 3
    assert observed == [3]

    one_row = data.iloc[:1]
    batch = stability.run_stability_batch(one_row, "y", tuple(data.columns), config)
    assert batch.runs[0].sampled_positions == (0,)
    assert observed[-1] == 1


def test_build_receives_full_pipeline_config_with_run_seed(monkeypatch):
    data, config = fixture(seed=51, n_resamples=2)
    received = []

    def build(sample, root, variables, run_config):
        received.append((sample, root, variables, run_config))
        return completed_network(root)

    monkeypatch.setattr(stability, "build_network_once", build)
    stability.run_stability_batch(data, "y", tuple(data.columns), config)

    assert [item[3].analysis.random_seed for item in received] == [51, 52]
    assert all(item[3] is not config for item in received)
    assert all(item[3].stability == config.stability for item in received)


def test_failure_is_logged_and_later_planned_runs_continue_without_retry(monkeypatch):
    data, config = fixture(n_resamples=4)
    attempted = []

    def build(sample, root, variables, run_config):
        attempted.append(run_config.analysis.random_seed)
        if run_config.analysis.random_seed == 18:
            raise RuntimeError("injected failure")
        return completed_network(root)

    monkeypatch.setattr(stability, "build_network_once", build)
    batch = stability.run_stability_batch(data, "y", tuple(data.columns), config)

    assert attempted == [17, 18, 19, 20]
    assert batch.requested_runs == 4
    assert batch.attempted_runs == 4
    assert batch.completed_runs == 3
    assert batch.failed_runs == 1
    assert [run.run_index for run in batch.runs] == [0, 1, 2, 3]
    failure = batch.runs[1]
    assert failure.failure_type == "RuntimeError"
    assert failure.failure_message == "injected failure"
    assert failure.sampled_positions
    assert not batch.complete


def test_preflight_stop_is_retained_as_failure_and_later_runs_continue(monkeypatch):
    data, config = fixture(n_resamples=3)
    attempted = []

    def build(sample, root, variables, run_config):
        attempted.append(run_config.analysis.random_seed)
        if run_config.analysis.random_seed == 18:
            return preflight_network(root)
        return completed_network(root)

    monkeypatch.setattr(stability, "build_network_once", build)
    batch = stability.run_stability_batch(data, "y", tuple(data.columns), config)

    assert attempted == [17, 18, 19]
    assert batch.attempted_runs == 3
    assert batch.completed_runs == 2
    assert batch.failed_runs == 1
    assert not batch.complete
    preflight = batch.runs[1]
    assert preflight.network is not None
    assert preflight.network.stop_reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
    assert not preflight.completed
    assert preflight.failure_type == "PreflightStop"
    assert preflight.failure_message == StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE.value


def test_empty_data_fails_before_any_run(monkeypatch):
    data, config = fixture(n=0)
    called = False

    def build(*args):
        nonlocal called
        called = True
        return completed_network()

    monkeypatch.setattr(stability, "build_network_once", build)
    with pytest.raises(ValueError, match="at least one row"):
        stability.run_stability_batch(data, "y", tuple(data.columns), config)
    assert not called


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_resample_override_must_be_positive_exact_integer(count):
    data, config = fixture()
    with pytest.raises(ValueError, match="n_resamples"):
        stability.run_stability_batch(data, "y", tuple(data.columns), config, n_resamples=count)


def test_default_uses_configured_development_resample_count(monkeypatch):
    data, config = fixture(n_resamples=2)
    monkeypatch.setattr(stability, "build_network_once", lambda sample, root, variables, cfg: completed_network(root))
    batch = stability.run_stability_batch(data, "y", tuple(data.columns), config)
    assert batch.requested_runs == 2
