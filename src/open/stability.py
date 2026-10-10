"""Whole-pipeline resampling with an auditable per-run ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import metadata as distribution_metadata
from math import floor
from pathlib import Path
import subprocess
import time
import tracemalloc

import numpy as np
import pandas as pd

from open.config import OpenConfig
from open.aggregation import StabilitySummary, aggregate_stability
from open.consensus import ConsensusResult, build_consensus
from open.graph import StopReason
from open.recurse import build_network_once
from open.stability_types import StabilityBatch, StabilityRunRecord
from open.layer_types import LayeredOPENResult


@dataclass(frozen=True)
class OPENResult:
    """Saved output of the complete resampling, aggregation, and consensus path."""

    batch: StabilityBatch
    stability: StabilitySummary
    consensus: ConsensusResult
    config: OpenConfig
    variables: tuple[str, ...]
    git_revision: str | None
    git_revision_unavailable_reason: str | None
    elapsed_seconds: float
    peak_traced_memory_bytes: int | None
    peak_memory_unavailable_reason: str | None
    package_version: str | None = None
    dependency_versions: dict[str, str | None] | None = None
    learner_versions: dict[str, str | None] | None = None
    version_availability: dict[str, str] | None = None


def _capture_versions(config: OpenConfig) -> tuple[
    str | None, dict[str, str | None], dict[str, str | None], dict[str, str]
]:
    """Snapshot the distributions used by this fit, including missing ones."""
    availability: dict[str, str] = {}

    def version(distribution: str, field: str) -> str | None:
        try:
            return distribution_metadata.version(distribution)
        except distribution_metadata.PackageNotFoundError:
            availability[field] = f"distribution {distribution!r} is not installed"
            return None

    package_version = version("open-predictive-expansion", "package_version")
    dependency_versions = {
        name: version(distribution, f"dependency_versions.{name}")
        for name, distribution in (
            ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
            ("scikit-learn", "scikit-learn"), ("PyYAML", "PyYAML"),
        )
    }
    if "lightgbm" in config.learner.library:
        dependency_versions["lightgbm"] = version("lightgbm", "dependency_versions.lightgbm")
    learner_versions = {}
    for learner in config.learner.library:
        distribution = "lightgbm" if learner == "lightgbm" else "scikit-learn"
        learner_versions[learner] = dependency_versions[distribution]
        if learner_versions[learner] is None:
            availability[f"learner_versions.{learner}"] = availability[f"dependency_versions.{distribution}"]
    return package_version, dependency_versions, learner_versions, availability


def read_git_revision() -> tuple[str | None, str | None]:
    """Return the current checkout revision, or a bounded availability reason."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except FileNotFoundError:
        return None, "Git command not found"
    except subprocess.TimeoutExpired:
        return None, "Git revision command timed out after 2 seconds"
    except Exception as exc:
        reason = f"Git revision unavailable: {type(exc).__name__}: {exc}"
        return None, reason[:200]
    if completed.returncode == 0:
        return completed.stdout.strip(), None
    return None, f"Git revision command exited with status {completed.returncode}"[:200]


class StabilityBatchError(RuntimeError):
    """Raised when every planned stability run failed or stopped at preflight."""

    def __init__(self, batch: StabilityBatch):
        self.batch = batch
        super().__init__(
            f"no stability runs completed ({batch.failed_runs}/{batch.attempted_runs} attempts failed)"
        )


def fit_stable_open(
    data: pd.DataFrame,
    root: str,
    variables: tuple[str, ...],
    config: OpenConfig,
    *,
    n_resamples: int | None = None,
) -> OPENResult | LayeredOPENResult:
    """Fit the explicitly configured whole-pipeline or layer-consensus method."""
    if isinstance(config, OpenConfig) and config.stability.scope == "layer_consensus":
        from open.layer_stability import fit_layer_consensus
        return fit_layer_consensus(data, root, variables, config, n_resamples=n_resamples)
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    try:
        git_revision, git_reason = read_git_revision()
        (package_version, dependency_versions, learner_versions,
         version_availability) = _capture_versions(config)
        started = time.perf_counter()
        batch = run_stability_batch(data, root, variables, config,
                                    n_resamples=n_resamples)
        if batch.completed_runs == 0:
            raise StabilityBatchError(batch)
        summary = aggregate_stability(batch)
        consensus = build_consensus(summary, config.thresholds.stability)
        elapsed_seconds = time.perf_counter() - started
        if owns_tracing:
            peak_memory_bytes = tracemalloc.get_traced_memory()[1]
            memory_reason = None
        else:
            peak_memory_bytes = None
            memory_reason = "tracemalloc was already active"
        return OPENResult(
            batch=batch,
            stability=summary,
            consensus=consensus,
            config=config,
            variables=variables,
            git_revision=git_revision,
            git_revision_unavailable_reason=git_reason,
            elapsed_seconds=elapsed_seconds,
            peak_traced_memory_bytes=peak_memory_bytes,
            peak_memory_unavailable_reason=memory_reason,
            package_version=package_version,
            dependency_versions=dependency_versions,
            learner_versions=learner_versions,
            version_availability=version_availability,
        )
    finally:
        if owns_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()


def run_stability_batch(
    data: pd.DataFrame,
    root: str,
    variables: tuple[str, ...],
    config: OpenConfig,
    *,
    n_resamples: int | None = None,
) -> StabilityBatch:
    """Rebuild the full network once for each deterministic subsample.

    Planned run ``b`` uses seed ``analysis.random_seed + b`` both to select
    rows and to configure the nested network's CV splits. Each run gets one
    attempt. Ordinary exceptions are preserved in its ledger row, and later
    planned runs continue. A returned preflight stop is retained for
    diagnostics but counted as an unsuccessful attempt.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if len(data) == 0:
        raise ValueError("data must contain at least one row")
    if not isinstance(config, OpenConfig):
        raise TypeError("config must be an OpenConfig")
    if config.stability.scope != "whole_pipeline":
        raise NotImplementedError("run_stability_batch is whole-pipeline only; use the layer fitter")
    requested = config.stability.n_resamples if n_resamples is None else n_resamples
    if type(requested) is not int or requested <= 0:
        raise ValueError("n_resamples must be a positive integer")

    records: list[StabilityRunRecord] = []
    base_seed = config.analysis.random_seed
    n_rows = len(data)
    sample_size = max(1, floor(config.stability.fraction * n_rows))

    for run_index in range(requested):
        seed = base_seed + run_index
        positions: tuple[int, ...] = ()
        try:
            rng = np.random.default_rng(seed)
            positions = tuple(sorted(
                int(position)
                for position in rng.choice(n_rows, size=sample_size, replace=False)
            ))
            sampled_data = data.iloc[list(positions)]
            run_config = replace(
                config,
                analysis=replace(config.analysis, random_seed=seed),
            )
            network = build_network_once(sampled_data, root, variables, run_config)
        except Exception as exc:
            records.append(StabilityRunRecord(
                run_index=run_index,
                seed=seed,
                sampled_positions=positions,
                failure_type=type(exc).__name__,
                failure_message=str(exc),
            ))
        else:
            preflight_stopped = network.stop_reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
            records.append(StabilityRunRecord(
                run_index=run_index,
                seed=seed,
                sampled_positions=positions,
                network=network,
                failure_type="PreflightStop" if preflight_stopped else None,
                failure_message=network.stop_reason.value if preflight_stopped else None,
            ))

    return StabilityBatch(requested_runs=requested, runs=tuple(records))
