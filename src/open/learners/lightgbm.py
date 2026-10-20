"""Deterministic LightGBM regressor with fold-local pipeline tuning."""
import math
from numbers import Integral, Real

import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline

from open.learners.base import Predictor
from open.learners.elastic_net import _materialize_cv
from open.preprocessing import make_preprocessor

GRID = {"num_leaves": [7, 15], "learning_rate": [.05, .1]}


class LightGBMPredictor(Predictor):
    engine_name = "lightgbm"

    def __init__(self, preprocessing=None, random_seed=0):
        if type(random_seed) is not int or not 0 <= random_seed < 2**32:
            raise ValueError("lightgbm random_seed must be an integer in [0, 2**32)")
        self.preprocessing_config = preprocessing
        self.random_seed = random_seed
        self.tuning_grid_size = len(ParameterGrid(GRID))
        self._pipeline = None
        self._tuned_params = None

    def _new_pipeline(self, params):
        if set(params) != set(GRID):
            raise ValueError("parameters must match the LightGBM tuning grid keys")
        for name, value in params.items():
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"invalid LightGBM parameter {name}")
        if not isinstance(params["num_leaves"], Integral):
            raise ValueError("num_leaves must be an integer")
        if params["learning_rate"] > 1:
            raise ValueError("learning_rate must be at most one")
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm engine unavailable; install 'open-predictive-expansion[lightgbm]'") from exc
        estimator = LGBMRegressor(n_estimators=100, min_child_samples=5, random_state=self.random_seed,
                                  n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True, **params)
        preprocessor = make_preprocessor(self.preprocessing_config)
        preprocessor.set_output(transform="pandas")
        return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])

    def tune(self, X, y, cv, loss="mse"):
        if loss not in ("mse", "neg_mean_squared_error"):
            raise ValueError("LightGBM supports MSE loss only")
        self._pipeline = None
        self._tuned_params = None
        if len(np.shape(X)) != 2:
            raise ValueError("X must be a two-dimensional predictor matrix")
        splits = _materialize_cv(cv, X, y)
        search = GridSearchCV(self._new_pipeline({k: v[0] for k, v in GRID.items()}),
                              {"regressor__" + k: v for k, v in GRID.items()},
                              cv=splits, scoring="neg_mean_squared_error", refit=False, n_jobs=1,
                              error_score="raise")
        search.fit(X, y)
        self._tuned_params = {k.removeprefix("regressor__"): v for k, v in search.best_params_.items()}
        return dict(self._tuned_params)

    def fit(self, X, y, params=None):
        self._pipeline = None
        selected = params if params is not None else self._tuned_params
        if selected is None:
            selected = {k: v[0] for k, v in GRID.items()}
        pipeline = self._new_pipeline(selected)
        pipeline.fit(X, y)
        self._pipeline = pipeline
        self._tuned_params = dict(selected)
        return self

    def predict(self, X):
        if self._pipeline is None:
            raise RuntimeError("fit must be called before predict")
        # Predict through the booster so LightGBM does not warn about feature names.
        transformed = self._pipeline["preprocessor"].transform(X)
        return np.asarray(self._pipeline["regressor"].booster_.predict(transformed, num_threads=1))

    def clone(self):
        return type(self)(self.preprocessing_config, self.random_seed)
