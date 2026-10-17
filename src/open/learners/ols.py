"""Unpenalized least-squares reference with fold-local preprocessing."""
from collections.abc import Mapping

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from open.config import OpenConfig, PreprocessingConfig
from open.learners.base import Predictor
from open.preprocessing import make_preprocessor


class OLSPredictor(Predictor):
    """OLS estimates every supplied coefficient; network pruning stays external."""

    engine_name = 'ols'
    tuning_grid_size = 0

    def __init__(self, preprocessing=None):
        if isinstance(preprocessing,OpenConfig):
            preprocessing=preprocessing.preprocessing
        if preprocessing is not None and not isinstance(preprocessing,PreprocessingConfig):
            raise TypeError('preprocessing must be a preprocessing configuration')
        self.preprocessing_config=preprocessing or PreprocessingConfig()
        self._pipeline=None

    def tune(self,X,y,cv,loss='mse'):
        if loss not in ('mse','neg_mean_squared_error'):
            raise ValueError('OLS supports MSE loss only')
        # No hyperparameter search or inner-fold model fitting is required.
        self._pipeline=None
        return {}

    def fit(self,X,y,params=None):
        self._pipeline=None
        if params is not None and (not isinstance(params,Mapping) or params):
            raise ValueError('OLS accepts an empty parameter mapping only')
        model=Pipeline([('preprocessor',make_preprocessor(self.preprocessing_config)),
                        ('regressor',LinearRegression())])
        model.fit(X,y)
        self._pipeline=model
        return self

    def predict(self,X):
        if self._pipeline is None:
            raise RuntimeError('fit must be called before predict')
        return np.asarray(self._pipeline.predict(X))

    def clone(self):
        return type(self)(self.preprocessing_config)
