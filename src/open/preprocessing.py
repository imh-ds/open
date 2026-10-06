"""Fold-local numeric imputation and scaling pipelines."""

from __future__ import annotations

from open.config import OpenConfig, PreprocessingConfig
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor(config: PreprocessingConfig | OpenConfig | None = None) -> Pipeline:
    """Create a fresh median-imputation and standardization pipeline.

    Fit the returned pipeline only on a training partition. In particular, do
    not fit it once on a complete dataset and reuse it across validation folds.
    The supported first path has numeric scale-level predictors only.
    """
    if isinstance(config, OpenConfig):
        config = config.preprocessing
    if config is not None and not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be a PreprocessingConfig, OpenConfig, or None")
    if config is not None:
        if config.missing != "median":
            raise ValueError("only median imputation is supported")
        if config.standardize_linear is not True:
            raise ValueError("linear predictors must be standardized in the first path")
        if config.one_hot_categorical is not False:
            raise ValueError("categorical predictors are unsupported in the first path")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
