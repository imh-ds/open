"""Deterministic tree regressors with fold-local pipeline tuning."""
import math
from numbers import Integral, Real

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline

from open.learners.base import Predictor
from open.learners.elastic_net import _materialize_cv
from open.preprocessing import make_preprocessor

GRIDS = {
    "random_forest": {"min_samples_leaf": [2, 5], "max_features": [.7, 1.]},
    "gbm": {"max_depth": [2, 3], "learning_rate": [.05, .1]},
    "lightgbm": {"num_leaves": [7, 15], "learning_rate": [.05, .1]},
}


class TreePredictor(Predictor):
    def __init__(self, engine, preprocessing=None, random_seed=0):
        if engine not in GRIDS:
            raise ValueError("unknown tree engine")
        if type(random_seed) is not int or not 0 <= random_seed < 2**32:
            raise ValueError("tree random_seed must be an integer in [0, 2**32)")
        self.engine_name = engine
        self.preprocessing_config = preprocessing
        self.random_seed = random_seed
        self.tuning_grid_size = len(ParameterGrid(GRIDS[engine]))
        self._pipeline = None
        self._tuned_params = None

    def _new_pipeline(self, params):
        expected = GRIDS[self.engine_name]
        if set(params) != set(expected):
            raise ValueError("parameters must match the engine tuning grid keys")
        for name, value in params.items():
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"invalid tree parameter {name}")
            if name in ("min_samples_leaf", "max_depth", "num_leaves") and not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if name in ("max_features", "learning_rate") and value > 1:
                raise ValueError(f"{name} must be at most one")
        if self.engine_name == "random_forest":
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_seed, n_jobs=1, **params)
        elif self.engine_name == "gbm":
            estimator = GradientBoostingRegressor(n_estimators=100, random_state=self.random_seed, **params)
        else:
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("lightgbm engine unavailable; install 'open-predictive-expansion[lightgbm]'") from exc
            estimator = LGBMRegressor(n_estimators=100, min_child_samples=5,
                random_state=self.random_seed, n_jobs=1, verbosity=-1,
                deterministic=True, force_col_wise=True, **params)
        preprocessor = make_preprocessor(self.preprocessing_config)
        if self.engine_name == "lightgbm":
            preprocessor.set_output(transform="pandas")
        return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])

    def tune(self, X, y, cv, loss="mse"):
        if loss not in ("mse", "neg_mean_squared_error"):
            raise ValueError("tree engines support MSE loss only")
        self._pipeline = None
        self._tuned_params = None
        splits = _materialize_cv(cv, X, y)
        initial = {k:v[0] for k,v in GRIDS[self.engine_name].items()}
        search = GridSearchCV(self._new_pipeline(initial),
            {"regressor__"+k:v for k,v in GRIDS[self.engine_name].items()},
            cv=splits, scoring="neg_mean_squared_error", refit=False, n_jobs=1, error_score="raise")
        search.fit(X,y)
        self._tuned_params = {k.removeprefix("regressor__"):v for k,v in search.best_params_.items()}
        return dict(self._tuned_params)

    def fit(self, X, y, params=None):
        self._pipeline = None
        selected = params if params is not None else self._tuned_params
        if selected is None:
            selected = {k:v[0] for k,v in GRIDS[self.engine_name].items()}
        pipeline = self._new_pipeline(selected)
        pipeline.fit(X,y)
        self._pipeline = pipeline
        self._tuned_params = dict(selected)
        return self

    def predict(self, X):
        if self._pipeline is None:
            raise RuntimeError("fit must be called before predict")
        # LightGBM's sklearn wrapper otherwise warns about its internally assigned
        # feature names when receiving an ndarray from our preprocessing pipeline.
        transformed = self._pipeline["preprocessor"].transform(X)
        regressor = self._pipeline["regressor"]
        if self.engine_name == "lightgbm":
            return np.asarray(regressor.booster_.predict(transformed, num_threads=1))
        return np.asarray(regressor.predict(transformed))

    def clone(self):
        return type(self)(self.engine_name, self.preprocessing_config, self.random_seed)
