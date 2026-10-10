"""Fit-time configuration and runtime provenance on OPEN results."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import open.stability as stability
from open.config import AnalysisConfig, OpenConfig, StabilityConfig


class FakeTracemalloc:
    def __init__(self, *, active=False):
        self.active = active
        self.starts = 0
        self.stops = 0

    def is_tracing(self):
        return self.active

    def start(self):
        self.active = True
        self.starts += 1

    def stop(self):
        self.active = False
        self.stops += 1

    def get_traced_memory(self):
        return 0, 42


def fit_inputs():
    data = pd.DataFrame({"y": np.arange(4, dtype=float), "x": np.arange(4, dtype=float)})
    config = OpenConfig(AnalysisConfig("y", 3), stability=StabilityConfig(n_resamples=1))
    variables = ("y", "x")
    batch = SimpleNamespace(completed_runs=1)
    return data, config, variables, batch


def mock_aggregation(monkeypatch, batch):
    summary = object()
    consensus = object()
    monkeypatch.setattr(stability, "run_stability_batch", lambda *args, **kwargs: batch)
    monkeypatch.setattr(stability, "aggregate_stability", lambda received: summary)
    monkeypatch.setattr(stability, "build_consensus", lambda received, threshold: consensus)
    return summary, consensus


def test_fit_retains_exact_configuration_variables_and_successful_git_revision(monkeypatch):
    data, config, variables, batch = fit_inputs()
    summary, consensus = mock_aggregation(monkeypatch, batch)
    expected_git_args = ["git", "rev-parse", "--verify", "HEAD"]

    def git_run(args, **kwargs):
        assert args == expected_git_args
        assert kwargs["cwd"] == Path(stability.__file__).resolve().parents[2]
        assert kwargs["timeout"] == 2
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(subprocess, "run", git_run)
    result = stability.fit_stable_open(data, "y", variables, config)

    assert result.config is config
    assert result.variables is variables
    assert result.batch is batch
    assert result.stability is summary
    assert result.consensus is consensus
    assert result.git_revision == "abc123"
    assert result.git_revision_unavailable_reason is None
    assert result.elapsed_seconds >= 0
    assert result.peak_traced_memory_bytes is None or result.peak_traced_memory_bytes >= 0
    if result.peak_traced_memory_bytes is None:
        assert result.peak_memory_unavailable_reason == "tracemalloc was already active"
    else:
        assert result.peak_memory_unavailable_reason is None


def test_fit_records_git_exit_status_reason(monkeypatch):
    data, config, variables, batch = fit_inputs()
    mock_aggregation(monkeypatch, batch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 128, stdout="", stderr="not a git repo"),
    )

    result = stability.fit_stable_open(data, "y", variables, config)

    assert result.git_revision is None
    assert "status 128" in result.git_revision_unavailable_reason


def test_fit_records_command_not_found_reason(monkeypatch):
    data, config, variables, batch = fit_inputs()
    mock_aggregation(monkeypatch, batch)

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git missing")

    monkeypatch.setattr(subprocess, "run", missing_git)
    result = stability.fit_stable_open(data, "y", variables, config)

    assert result.git_revision is None
    assert "not found" in result.git_revision_unavailable_reason.lower()


def test_fit_records_git_timeout_reason(monkeypatch):
    data, config, variables, batch = fit_inputs()
    mock_aggregation(monkeypatch, batch)

    def timed_out(args, **kwargs):
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired(args, timeout=2)

    monkeypatch.setattr(subprocess, "run", timed_out)
    result = stability.fit_stable_open(data, "y", variables, config)

    assert result.git_revision is None
    assert "timed out" in result.git_revision_unavailable_reason.lower()


def test_fit_does_not_take_ownership_of_active_tracemalloc(monkeypatch):
    data, config, variables, batch = fit_inputs()
    mock_aggregation(monkeypatch, batch)
    fake_tracemalloc = FakeTracemalloc(active=True)
    monkeypatch.setattr(stability, "tracemalloc", fake_tracemalloc)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="abc123", stderr=""),
    )
    result = stability.fit_stable_open(data, "y", variables, config)
    assert result.peak_traced_memory_bytes is None
    assert result.peak_memory_unavailable_reason == "tracemalloc was already active"
    assert fake_tracemalloc.active
    assert fake_tracemalloc.starts == 0
    assert fake_tracemalloc.stops == 0


def test_batch_exception_stops_tracer_started_by_fit(monkeypatch):
    data, config, variables, _ = fit_inputs()
    fake_tracemalloc = FakeTracemalloc()
    monkeypatch.setattr(stability, "tracemalloc", fake_tracemalloc)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="abc123", stderr=""),
    )

    def fail_batch(*args, **kwargs):
        assert fake_tracemalloc.is_tracing()
        raise RuntimeError("batch failed")

    monkeypatch.setattr(stability, "run_stability_batch", fail_batch)
    try:
        stability.fit_stable_open(data, "y", variables, config)
    except RuntimeError as exc:
        assert str(exc) == "batch failed"
    else:
        raise AssertionError("expected batch exception")
    assert not fake_tracemalloc.is_tracing()
    assert fake_tracemalloc.starts == 1
    assert fake_tracemalloc.stops == 1
