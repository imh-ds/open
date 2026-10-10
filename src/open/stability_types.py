"""Immutable shared records for Phase 4 run ledgers and summaries."""

from __future__ import annotations

from dataclasses import dataclass

from open.graph import NetworkRun


@dataclass(frozen=True)
class StabilityRunRecord:
    """One planned resample, including either its graph or its failure."""

    run_index: int
    seed: int
    sampled_positions: tuple[int, ...]
    network: NetworkRun | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if type(self.run_index) is not int or self.run_index < 0:
            raise ValueError("run_index must be a nonnegative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isinstance(self.sampled_positions, tuple):
            raise ValueError("sampled_positions must be a tuple")
        if any(type(position) is not int or position < 0 for position in self.sampled_positions):
            raise ValueError("sampled row positions must be nonnegative integers")
        if tuple(sorted(set(self.sampled_positions))) != self.sampled_positions:
            raise ValueError("sampled row positions must be unique and sorted")
        if self.network is not None:
            if not isinstance(self.network, NetworkRun):
                raise ValueError("network must be a NetworkRun")
            if self.network.stop_reason is None:
                if self.failure_type is not None or self.failure_message is not None:
                    raise ValueError("a completed run cannot also carry failure details")
            elif (self.network.stop_reason.value != "INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE"
                  or self.failure_type != "PreflightStop"
                  or self.failure_message != self.network.stop_reason.value):
                raise ValueError("preflight-stopped runs must be recorded as unsuccessful attempts")
        elif (type(self.failure_type) is not str or not self.failure_type.strip()
              or type(self.failure_message) is not str):
            raise ValueError("a failed run requires its exception type and message")

    @property
    def completed(self) -> bool:
        return self.network is not None and self.network.stop_reason is None


@dataclass(frozen=True)
class StabilityBatch:
    """Ordered ledger for all attempts in one requested stability batch."""

    requested_runs: int
    runs: tuple[StabilityRunRecord, ...]

    def __post_init__(self) -> None:
        if type(self.requested_runs) is not int or self.requested_runs <= 0:
            raise ValueError("requested_runs must be a positive integer")
        if not isinstance(self.runs, tuple):
            raise ValueError("runs must be a tuple")
        if len(self.runs) > self.requested_runs:
            raise ValueError("attempted runs cannot exceed requested runs")
        if any(not isinstance(run, StabilityRunRecord) for run in self.runs):
            raise ValueError("runs must contain StabilityRunRecord values")
        if tuple(run.run_index for run in self.runs) != tuple(range(len(self.runs))):
            raise ValueError("run indices must be consecutive from zero in planned order")

    @property
    def attempted_runs(self) -> int:
        return len(self.runs)

    @property
    def completed_runs(self) -> int:
        return sum(run.completed for run in self.runs)

    @property
    def failed_runs(self) -> int:
        return self.attempted_runs - self.completed_runs

    @property
    def complete(self) -> bool:
        """Whether every requested run was attempted and completed."""
        return self.attempted_runs == self.requested_runs and self.failed_runs == 0
