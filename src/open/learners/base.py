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
