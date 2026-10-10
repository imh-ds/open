"""Training-fold Spearman scores for numeric scale-level candidate pairs.

Call ``score_dependence`` with training rows only. Missing values are filled
with medians estimated from those rows before rank correlation. Undefined
pairs remain in the returned diagnostics and are never candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


@dataclass(frozen=True)
class PairScore:
    """One canonically named predictor pair and its candidate decision."""

    a: str
    b: str
    score: float | None
    candidate: bool
    reason: str | None = None


def score_dependence(
    training_data: pd.DataFrame,
    candidates: Sequence[str],
    *,
    tau_dep: float,
) -> tuple[PairScore, ...]:
    """Score every named pair using only supplied training predictors.

    ``tau_dep`` screens pairs for later held-out group ablation; the score is
    not itself a predictive-relevance or inference statistic.
    """
    if not isinstance(training_data, pd.DataFrame):
        raise TypeError("training_data must be a pandas DataFrame")
    if (
        isinstance(tau_dep, bool)
        or not isinstance(tau_dep, (int, float))
        or not np.isfinite(tau_dep)
        or not 0 <= tau_dep <= 1
    ):
        raise ValueError("tau_dep must be a finite number in [0, 1]")
    names = tuple(candidates)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("candidate names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    if not training_data.columns.is_unique:
        raise ValueError("training columns must be unique")
    absent = set(names) - set(training_data.columns)
    if absent:
        raise ValueError(f"candidate columns are missing: {sorted(absent)}")

    prepared: dict[str, np.ndarray | None] = {}
    unavailable: dict[str, str] = {}
    for name in sorted(names):
        series = training_data[name]
        if series.isna().all():
            prepared[name] = None
            unavailable[name] = "all values missing"
            continue
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"candidate {name!r} must be numeric scale-level data")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"candidate {name!r} contains infinity")
        observed = values[~np.isnan(values)]
        values = np.where(np.isnan(values), np.median(observed), values)
        if len(values) < 2:
            unavailable[name] = "insufficient rows"
        elif np.all(values == values[0]):
            unavailable[name] = "constant column"
        prepared[name] = values

    scores: list[PairScore] = []
    for a, b in combinations(sorted(names), 2):
        reasons = (unavailable.get(a), unavailable.get(b))
        reason = "all values missing" if "all values missing" in reasons else next(
            (item for item in reasons if item is not None), None
        )
        if reason is not None:
            scores.append(PairScore(a, b, None, False, reason))
            continue
        ranks_a = rankdata(prepared[a])
        ranks_b = rankdata(prepared[b])
        if np.array_equal(ranks_a, ranks_b) or np.array_equal(
            ranks_a, len(ranks_a) + 1 - ranks_b
        ):
            score = 1.0
        else:
            score = float(abs(spearmanr(prepared[a], prepared[b]).statistic))
            if score >= 1.0:
                # Floating point correlation can round nonidentical ranks to 1.
                score = float(np.nextafter(1.0, 0.0))
        candidate = score >= tau_dep
        scores.append(PairScore(a, b, score, candidate))
    return tuple(scores)
