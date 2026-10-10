"""Numerical regression for the near-copy predictor fixture."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from open.learners.elastic_net import ElasticNetPredictor


@pytest.mark.parametrize("rows", [slice(None), slice(40, 80)])
def test_near_copy_predictors_reach_optimizer_tolerance(rows):
    rng = np.random.default_rng(120)
    scale = rng.normal(size=80)
    y = scale + rng.normal(scale=.15, size=80)
    X = np.column_stack((scale, scale + rng.normal(scale=.01, size=80)))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        learner = ElasticNetPredictor().fit(
            X[rows], y[rows], {"alpha": .0001, "l1_ratio": .1}
        )
    regressor = learner._pipeline.named_steps["regressor"]
    assert regressor.dual_gap_ <= regressor.tol * np.var(y[rows])
    assert not [item for item in captured if issubclass(item.category, ConvergenceWarning)]
    assert np.isfinite(learner.predict(X)).all()
