"""Pure risk-to-relevance arithmetic for the continuous/MSE path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping


@dataclass(frozen=True)
class RelevanceContrast:
    """A signed risk difference and its optional total-gain ratio."""

    raw: float
    normalized: float | None


@dataclass(frozen=True)
class RelevanceResult:
    """Target gain and predictor contrasts derived from held-out risks."""

    total_gain: float
    relative_gain: float | None
    stop_reason: Literal["LOW_TOTAL_GAIN"] | None
    broad: Mapping[str, RelevanceContrast]
    unique: Mapping[str, RelevanceContrast]
    baseline_risk: float
    full_risk: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.baseline_risk) or not math.isfinite(self.full_risk):
            raise ValueError("baseline_risk and full_risk must be finite")


def compute_relevance(
    baseline_risk: float,
    full_risk: float,
    *,
    broad_risks: Mapping[str, float],
    ablated_risks: Mapping[str, float],
    tau_gain: float,
) -> RelevanceResult:
    """Calculate signed broad/LOCO contrasts, normalizing only after the gain stop.

    ``broad_risks`` maps a predictor to its single-predictor risk;
    ``ablated_risks`` maps it to the full-model risk without that predictor.
    Both may be empty when only the target gain is available.
    """
    total_gain = baseline_risk - full_risk
    relative_gain = total_gain / baseline_risk if baseline_risk > 0 else None
    eligible = (
        relative_gain is not None
        and total_gain > 0
        and relative_gain >= tau_gain
    )
    denominator = total_gain if eligible else None

    broad = {
        name: RelevanceContrast(
            raw=baseline_risk - risk,
            normalized=(baseline_risk - risk) / denominator if denominator is not None else None,
        )
        for name, risk in broad_risks.items()
    }
    unique = {
        name: RelevanceContrast(
            raw=risk - full_risk,
            normalized=(risk - full_risk) / denominator if denominator is not None else None,
        )
        for name, risk in ablated_risks.items()
    }
    return RelevanceResult(
        total_gain=total_gain,
        relative_gain=relative_gain,
        stop_reason=None if eligible else "LOW_TOTAL_GAIN",
        broad=broad,
        unique=unique,
        baseline_risk=baseline_risk,
        full_risk=full_risk,
    )
