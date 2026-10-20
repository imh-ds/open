"""Continuous elastic-net reference predictor with fold-local preprocessing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from open.config import PreprocessingConfig
from open.learners.base import Predictor
from open.preprocessing import make_preprocessor


ALPHA_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
L1_RATIO_GRID = (0.1, 0.5, 0.9, 0.95, 1.0)
# Near-copy predictors can need more than 40,000 coordinate-descent steps.
# Keep the solver tolerance unchanged; allow both tuning and outer fits to finish.
MAX_ITER = 100000
_DEFAULT_PARAMS = {"alpha": 1.0, "l1_ratio": 0.5}


class ElasticNetPredictor(Predictor):
    """Elastic-net regression using median imputation and standardized inputs.

    The search pipeline contains both transformations and the estimator, so
    GridSearchCV fits fresh preprocessing state on each inner training fold.
    ``cv`` must be built from the outer-training rows passed as ``X`` and ``y``.
    """

    engine_name = "elastic_net"
    tuning_grid_size = len(ALPHA_GRID) * len(L1_RATIO_GRID)

    def __init__(self, preprocessing: PreprocessingConfig | None = None):
        if preprocessing is not None and not isinstance(preprocessing, PreprocessingConfig):
            raise TypeError("preprocessing must be a PreprocessingConfig or None")
        self.preprocessing_config = preprocessing or PreprocessingConfig()
        self._pipeline: Pipeline | None = None
        self._tuned_params: dict[str, float] | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has created a fitted prediction pipeline."""
        return self._pipeline is not None

    def _new_pipeline(self, params: Mapping[str, Any] | None = None) -> Pipeline:
        values = _validate_params(params or _DEFAULT_PARAMS)
        return Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(self.preprocessing_config)),
                (
                    "regressor",
                    ElasticNet(
                        alpha=values["alpha"],
                        l1_ratio=values["l1_ratio"],
                        max_iter=MAX_ITER,
                    ),
                ),
            ]
        )

    def tune(self, X: Any, y: Any, cv: Any, loss: str = "mse") -> dict[str, float]:
        """Tune on explicit inner splits derived from the supplied training rows."""
        if loss not in {"mse", "neg_mean_squared_error"}:
            raise ValueError("the continuous reference learner supports MSE loss only")
        splits = _materialize_cv(cv, X, y)
        self._pipeline = None
        self._tuned_params = None
        search = GridSearchCV(
            estimator=self._new_pipeline(),
            param_grid={
                "regressor__alpha": ALPHA_GRID,
                "regressor__l1_ratio": L1_RATIO_GRID,
            },
            scoring="neg_mean_squared_error",
            cv=splits,
            refit=False,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X, y)
        params = {
            "alpha": float(search.best_params_["regressor__alpha"]),
            "l1_ratio": float(search.best_params_["regressor__l1_ratio"]),
        }
        self._tuned_params = params
        return dict(params)

    def fit(self, X: Any, y: Any, params: Mapping[str, Any] | None = None) -> "ElasticNetPredictor":
        """Fit a fresh complete pipeline, optionally reusing selected parameters."""
        self._pipeline = None
        if params is not None:
            params = _validate_params(params)
            self._tuned_params = dict(params)
        selected = params if params is not None else (self._tuned_params or _DEFAULT_PARAMS)
        pipeline = self._new_pipeline(selected)
        pipeline.fit(X, y)
        self._pipeline = pipeline
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the fitted fold-local preprocessing and regressor."""
        if self._pipeline is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self._pipeline.predict(X))

    def clone(self) -> "ElasticNetPredictor":
        """Return an unfitted predictor with the same preprocessing settings."""
        return type(self)(preprocessing=self.preprocessing_config)


def _validate_params(params: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping containing alpha and l1_ratio")
    if set(params) != {"alpha", "l1_ratio"}:
        raise ValueError("params must contain exactly alpha and l1_ratio")
    alpha, l1_ratio = params["alpha"], params["l1_ratio"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a positive number")
    if (
        isinstance(l1_ratio, bool)
        or not isinstance(l1_ratio, (int, float))
        or not math.isfinite(l1_ratio)
        or not (0 <= l1_ratio <= 1)
    ):
        raise ValueError("l1_ratio must be a number in [0, 1]")
    return {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}


def _materialize_cv(cv: Any, X: Any, y: Any) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if hasattr(cv, "as_cv"):
        source: Iterable[Any] = cv.as_cv()
    elif callable(getattr(cv, "split", None)):
        source = cv.split(X, y)
    else:
        try:
            source = iter(cv)
        except TypeError as exc:
            raise ValueError("cv must provide explicit inner train/test splits") from exc
    materialized: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for split in source:
            if len(split) != 2:
                raise ValueError("each cv split must contain train and test indices")
            train_raw, test_raw = (np.asarray(indices) for indices in split)
            if (
                train_raw.ndim != 1
                or test_raw.ndim != 1
                or not np.issubdtype(train_raw.dtype, np.integer)
                or not np.issubdtype(test_raw.dtype, np.integer)
                or train_raw.dtype == np.dtype(bool)
                or test_raw.dtype == np.dtype(bool)
            ):
                raise ValueError("cv train and test indices must be one-dimensional integers")
            train, test = train_raw.astype(int, copy=False), test_raw.astype(int, copy=False)
            if train.size == 0 or test.size == 0:
                raise ValueError("cv train and test indices must be nonempty one-dimensional arrays")
            if np.unique(train).size != train.size or np.unique(test).size != test.size:
                raise ValueError("cv train and test indices must not contain duplicates")
            if np.intersect1d(train, test).size:
                raise ValueError("cv train/test indices must be disjoint")
            n_samples = len(X)
            if (
                np.any(train < 0)
                or np.any(test < 0)
                or np.any(train >= n_samples)
                or np.any(test >= n_samples)
            ):
                raise ValueError("cv indices are out of bounds for the supplied outer-training rows")
            materialized.append((train, test))
    except TypeError as exc:
        raise ValueError("cv must provide explicit inner train/test splits") from exc
    if not materialized:
        raise ValueError("cv must contain at least one inner train/test split")
    return tuple(materialized)
