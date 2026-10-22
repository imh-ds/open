"""Experimental: partial least squares regression with fold-local preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

from open.config import PreprocessingConfig
from open.learners._grid import tune_grid
from open.learners.base import Predictor
from open.preprocessing import make_preprocessor

COMPONENT_GRID = (1, 2, 3, 5, 8)


class PLSPredictor(Predictor):
    """PLS regression; the number of components is tuned on the training folds.

    Inputs are standardized by the shared preprocessing, so PLS does not scale
    them again. The component count is capped at the number of columns, so a
    refit that leaves columns out never asks for more components than exist.
    With as many components as columns, PLS equals OLS.
    """

    engine_name = "pls"
    tuning_grid_size = len(COMPONENT_GRID)

    def __init__(self, preprocessing: PreprocessingConfig | None = None):
        if preprocessing is not None and not isinstance(preprocessing, PreprocessingConfig):
            raise TypeError("preprocessing must be a PreprocessingConfig or None")
        self.preprocessing_config = preprocessing or PreprocessingConfig()
        self._pipeline: Pipeline | None = None

    @staticmethod
    def _regressor(params: Mapping[str, Any], n_columns: int) -> PLSRegression:
        return PLSRegression(n_components=max(1, min(int(params["n_components"]), n_columns)), scale=False)

    def tune(self, X: Any, y: Any, cv: Any, loss: str = "mse") -> dict[str, int]:
        if loss not in {"mse", "neg_mean_squared_error"}:
            raise ValueError("PLS supports MSE loss only")
        self._pipeline = None
        n_columns = np.shape(X)[1]
        grid = sorted({min(c, n_columns) for c in COMPONENT_GRID})
        self.tuning_grid_size = len(grid)
        return tune_grid(X, y, cv, self.preprocessing_config, [{"n_components": c} for c in grid],
                         lambda p: self._regressor(p, n_columns))

    def fit(self, X: Any, y: Any, params: Mapping[str, Any] | None = None) -> "PLSPredictor":
        self._pipeline = None
        params = dict(params) if params else {"n_components": 2}
        if set(params) != {"n_components"} or int(params["n_components"]) < 1:
            raise ValueError("params must contain a positive n_components")
        model = Pipeline([("preprocessor", make_preprocessor(self.preprocessing_config)),
                          ("regressor", self._regressor(params, np.shape(X)[1]))])
        model.fit(X, y)
        self._pipeline = model
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("fit must be called before predict")
        return np.ravel(self._pipeline.predict(X))

    def clone(self) -> "PLSPredictor":
        return type(self)(self.preprocessing_config)
