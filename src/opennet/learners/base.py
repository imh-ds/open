"""Learner-independent continuous predictor contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class Predictor(ABC):
    """A predictor that can tune, fit, predict, and create an unfitted clone."""

    engine_name: str
    # Size of the most recent tune() search; nominal size before first tuning.
    # Risk accounting must read this on the tuned clone, after tune() returns.
    tuning_grid_size: int
    # True when ``coefficients`` returns the signed linear effect of every input column.
    supports_coefficients: bool = False

    @abstractmethod
    def tune(self, X: Any, y: Any, cv: Any, loss: str) -> Mapping[str, Any]:
        """Select hyperparameters using only the supplied training-row folds."""

    @abstractmethod
    def fit(self, X: Any, y: Any, params: Mapping[str, Any] | None = None) -> "Predictor":
        """Fit the complete preprocessing and prediction pipeline."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict outcomes for rows not used to fit this instance."""

    @abstractmethod
    def clone(self) -> "Predictor":
        """Return a new unfitted predictor with the same configuration."""

    def coefficients(self):
        """Signed effect of each input column in the fitted model, or None when the model has none.

        Returns ``(standardized, raw)`` arrays aligned with the columns it was fitted on: the change in the
        outcome per standard deviation of the predictor, and per unit of the predictor. Only linear learners
        provide it; others return None rather than a sign taken from importance, correlation or a surrogate.
        """
        return None


def linear_coefficients(pipeline):
    """``(standardized, raw)`` coefficients of a fitted preprocessor + linear regressor pipeline."""
    import numpy as np
    standardized = np.asarray(pipeline["regressor"].coef_, dtype=float).ravel()
    scale = np.asarray(pipeline["preprocessor"]["scaler"].scale_, dtype=float)
    return standardized, standardized / np.where(scale > 0, scale, np.nan)
