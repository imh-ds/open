"""Experimental: support vector regression (RBF kernel) with fold-local preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from open.config import PreprocessingConfig
from open.learners._grid import tune_grid
from open.learners.base import Predictor
from open.preprocessing import make_preprocessor

C_GRID = (0.3, 1.0, 3.0, 10.0)
# Kernel width as a multiple of 1 / (number of columns), sklearn's "scale" for standardized inputs.
GAMMA_GRID = (0.3, 1.0, 3.0)
EPSILON = 0.1                      # on the standardized outcome


class SVRPredictor(Predictor):
    """RBF support vector regression; C and the kernel width are tuned.

    The outcome is standardized inside each fit, so ``EPSILON`` and ``C`` mean
    the same at any outcome scale. The kernel width is stored as a multiple of
    1 / (columns in the tuned model) and turned into an absolute gamma there;
    refits that leave columns out keep that same absolute width.
    """

    engine_name = "svr"
    tuning_grid_size = len(C_GRID) * len(GAMMA_GRID)

    def __init__(self, preprocessing: PreprocessingConfig | None = None):
        if preprocessing is not None and not isinstance(preprocessing, PreprocessingConfig):
            raise TypeError("preprocessing must be a PreprocessingConfig or None")
        self.preprocessing_config = preprocessing or PreprocessingConfig()
        self._pipeline: Pipeline | None = None

    @staticmethod
    def _regressor(params: Mapping[str, Any]) -> TransformedTargetRegressor:
        svr = SVR(kernel="rbf", C=float(params["C"]), gamma=float(params["gamma"]), epsilon=EPSILON)
        return TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())

    def tune(self, X: Any, y: Any, cv: Any, loss: str = "mse") -> dict[str, float]:
        if loss not in {"mse", "neg_mean_squared_error"}:
            raise ValueError("SVR supports MSE loss only")
        self._pipeline = None
        n_columns = max(1, np.shape(X)[1])
        candidates = [{"C": c, "gamma": g / n_columns} for c in C_GRID for g in GAMMA_GRID]
        return tune_grid(X, y, cv, self.preprocessing_config, candidates, self._regressor)

    def fit(self, X: Any, y: Any, params: Mapping[str, Any] | None = None) -> "SVRPredictor":
        self._pipeline = None
        if not params:
            params = {"C": 1.0, "gamma": 1.0 / max(1, np.shape(X)[1])}
        if set(params) != {"C", "gamma"} or not (float(params["C"]) > 0 and float(params["gamma"]) > 0):
            raise ValueError("params must contain positive C and gamma")
        model = Pipeline([("preprocessor", make_preprocessor(self.preprocessing_config)),
                          ("regressor", self._regressor(params))])
        model.fit(X, np.asarray(y, dtype=float))
        self._pipeline = model
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("fit must be called before predict")
        return np.ravel(self._pipeline.predict(X))

    def clone(self) -> "SVRPredictor":
        return type(self)(self.preprocessing_config)
