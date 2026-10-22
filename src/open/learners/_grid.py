"""Grid tuning shared by the experimental engines, following the Elastic-Net rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import warnings
from typing import Any

import numpy as np
from sklearn.utils import _safe_indexing, indexable

from open.learners.elastic_net import _materialize_cv
from open.preprocessing import make_preprocessor


def tune_grid(X: Any, y: Any, cv: Any, preprocessing, candidates: Sequence[Mapping[str, Any]],
              make_regressor: Callable[[Mapping[str, Any]], Any]) -> dict[str, Any]:
    """Return the candidate with the best mean inner-fold MSE.

    Preprocessing is fitted once on each inner training fold and reused for
    every candidate. As in GridSearchCV, the first candidate wins a tie and
    NaN scores rank below valid ones.
    """
    splits = _materialize_cv(cv, X, y)
    X, y = indexable(X, y)
    scores = np.empty((len(candidates), len(splits)))
    for fold, (train, test) in enumerate(splits):
        ytrain, ytest = np.asarray(_safe_indexing(y, train)), np.asarray(_safe_indexing(y, test))
        preprocessor = make_preprocessor(preprocessing)
        Xtrain = preprocessor.fit_transform(_safe_indexing(X, train), ytrain)
        Xtest = preprocessor.transform(_safe_indexing(X, test))
        for index, candidate in enumerate(candidates):
            model = make_regressor(candidate).fit(Xtrain, ytrain)
            scores[index, fold] = -np.mean((np.ravel(model.predict(Xtest)) - ytest) ** 2)
    means = scores.mean(axis=1)
    if not np.isfinite(means).all():
        warnings.warn(f"One or more of the test scores are non-finite: {means}", UserWarning)
    return dict(candidates[int(np.argmax(np.where(np.isnan(means), -np.inf, means)))])
