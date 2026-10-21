"""Preprocessing reuse must preserve the original full-pipeline search."""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from open.explore import ExploreConfig, explore
from open.learners.elastic_net import ALPHA_GRID, L1_RATIO_GRID, ElasticNetPredictor
from open.learners import elastic_net as implementation


def _reference_search(model, X, y, cv):
    return GridSearchCV(model._new_pipeline(),
                          {"regressor__alpha": ALPHA_GRID, "regressor__l1_ratio": L1_RATIO_GRID},
                          cv=cv, scoring="neg_mean_squared_error", refit=False,
                          n_jobs=1, error_score="raise").fit(X, y)


def _reference(model, X, y, cv, loss="mse"):
    search = _reference_search(model, X, y, cv)
    return {k.removeprefix("regressor__"): float(v) for k, v in search.best_params_.items()}


def _data():
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(90, 4)), columns=list("abcd"), index=np.arange(90) + 500)
    return X, 2 * X.a.to_numpy() + rng.normal(size=90)


def test_preprocessing_is_fitted_once_per_split_without_skipping_grid_candidates(monkeypatch):
    X, y = _data()
    splits = tuple(KFold(3).split(X))
    original_imputer, original_fit = SimpleImputer.fit, ElasticNet.fit
    seen, estimator_fits = [], []

    def imputer_fit(self, data, y=None):
        seen.append(data.copy())
        return original_imputer(self, data, y)

    def estimator_fit(self, *args, **kwargs):
        estimator_fits.append((self.alpha, self.l1_ratio))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", imputer_fit)
    monkeypatch.setattr(ElasticNet, "fit", estimator_fit)
    model = ElasticNetPredictor()
    model.tune(X, y, iter(splits))
    assert len(seen) == len(splits)
    for actual, (train, _) in zip(seen, splits):
        pd.testing.assert_frame_equal(actual, X.iloc[train])
    assert len(estimator_fits) == len(splits) * len(ALPHA_GRID) * len(L1_RATIO_GRID)
    assert not model.is_fitted


def test_every_inner_fold_score_matches_original_search_exactly(monkeypatch):
    X, y = _data()
    X.iloc[::7, 1] = np.nan
    splits = tuple(KFold(4, shuffle=True, random_state=12).split(X))
    reference = _reference_search(ElasticNetPredictor(), X, y, splits)
    original_scorer = implementation.get_scorer
    observed = []

    def recording_scorer(name):
        scorer = original_scorer(name)

        def score(*args, **kwargs):
            value = scorer(*args, **kwargs)
            observed.append(value)
            return value
        return score

    monkeypatch.setattr(implementation, "get_scorer", recording_scorer)
    ElasticNetPredictor().tune(X, y, splits)
    actual = np.asarray(observed).reshape(len(splits), -1)
    for fold, scores in enumerate(actual):
        np.testing.assert_array_equal(scores, reference.cv_results_[f"split{fold}_test_score"])


@pytest.mark.filterwarnings("ignore:Objective did not converge")
@pytest.mark.parametrize("case", ["dataframe", "array32", "list", "missing", "empty_training_column", "tie"])
def test_tuning_parameters_and_predictions_match_original_pipeline_search(case):
    X, y = _data()
    splits = tuple(KFold(3, shuffle=True, random_state=7).split(X))
    if case == "missing":
        X.iloc[::4, 1] = np.nan
        X.iloc[:, -1] = np.nan
    elif case == "empty_training_column":
        X.iloc[splits[0][0], 0] = np.nan
    elif case == "tie":
        y = np.full(len(y), 2.)
    elif case == "array32":
        X = X.to_numpy(dtype=np.float32)
        y = y.astype(np.float32)
    elif case == "list":
        X, y = X.to_numpy().tolist(), y.tolist()
    before = X.copy(deep=True) if isinstance(X, pd.DataFrame) else np.array(X, copy=True)
    expected = _reference(ElasticNetPredictor(), X, y, splits)
    model = ElasticNetPredictor()
    actual = model.tune(X, y, splits)
    assert actual == expected
    if case == "tie":
        assert actual == {"alpha": ALPHA_GRID[0], "l1_ratio": L1_RATIO_GRID[0]}
    np.testing.assert_array_equal(model.fit(X, y).predict(X),
                                  ElasticNetPredictor().fit(X, y, expected).predict(X))
    if isinstance(X, pd.DataFrame):
        pd.testing.assert_frame_equal(X, before)
    else:
        np.testing.assert_array_equal(X, before)


def test_failed_search_clears_previous_fit_and_parameters():
    X, y = _data()
    model = ElasticNetPredictor().fit(X, y, {"alpha": .1, "l1_ratio": .5})
    y[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.tune(X, y, KFold(3))
    assert not model.is_fitted and model._tuned_params is None
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_repeated_exploration_with_controls_and_covariates_matches_original_search(monkeypatch):
    rng = np.random.default_rng(51)
    data = pd.DataFrame(rng.normal(size=(120, 5)), columns=["X", "N", "C", "V", "Y"])
    data.Y = 2 * data.X + data.V + data.Y
    data.loc[::9, "N"] = np.nan
    config = ExploreConfig(folds=3, tuning_folds=2, repeats=2, max_depth=1, interactions=False)
    actual = explore(data, "Y", engine="elastic_net", controls=["C"], covariates=["V"], config=config)
    monkeypatch.setattr(ElasticNetPredictor, "tune", _reference)
    expected = explore(data, "Y", engine="elastic_net", controls=["C"], covariates=["V"], config=config)
    for name in ("nodes", "links", "pairs", "folds", "bars", "fit"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
    assert actual.settings == expected.settings and actual.stops == expected.stops
