"""Interaction baseline tuning must match the original full-pipeline search."""
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

from open.explore import ExploreConfig, explore
from open import pair_validation as implementation
from open.pair_validation import PairValidationConfig, SplinePairFeatures, _estimator, _tune


def _data(n=91):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"), index=np.arange(n) + 600)
    y = X.a.to_numpy() + .4 * X.b.to_numpy() ** 2 + rng.normal(size=n)
    return X, y


def _reference_search(X, y, pair, config):
    splits = list(KFold(config.inner_folds, shuffle=True, random_state=config.seed).split(X))
    return GridSearchCV(_estimator(pair), {"curves__n_knots": list(config.knots),
                         "ridge__alpha": list(config.alphas)}, cv=splits,
                         scoring="neg_mean_squared_error", refit=True, error_score="raise", n_jobs=1).fit(X, y)


def _reference_tune(X, y, pair, config):
    search = _reference_search(X, y, pair, config)
    splits = list(KFold(config.inner_folds, shuffle=True, random_state=config.seed).split(X))
    return search.best_estimator_, dict(search.best_params_), [len(train) for train, _ in splits]


def test_preprocessing_work_is_reused_without_reducing_ridge_fits(monkeypatch):
    X, y = _data()
    config = PairValidationConfig(seed=3)
    original_imputer, original_curves, original_ridge = SimpleImputer.fit, SplinePairFeatures.fit, Ridge.fit
    imputer_data, curves_knots, ridge_alphas = [], [], []

    def imputer_fit(self, data, y=None):
        imputer_data.append(data.copy())
        return original_imputer(self, data, y)

    def curves_fit(self, data, y=None):
        curves_knots.append((len(data), self.n_knots))
        return original_curves(self, data, y)

    def ridge_fit(self, *args, **kwargs):
        ridge_alphas.append(self.alpha)
        return original_ridge(self, *args, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", imputer_fit)
    monkeypatch.setattr(SplinePairFeatures, "fit", curves_fit)
    monkeypatch.setattr(Ridge, "fit", ridge_fit)
    model, params, sizes = _tune(X, y, None, config)
    splits = list(KFold(config.inner_folds, shuffle=True, random_state=config.seed).split(X))
    assert len(imputer_data) == config.inner_folds + 1
    assert len(curves_knots) == config.inner_folds * len(config.knots) + 1
    assert len(ridge_alphas) == config.inner_folds * len(config.knots) * len(config.alphas) + 1
    for actual, (train, _) in zip(imputer_data[:-1], splits):
        pd.testing.assert_frame_equal(actual, X.iloc[train])
    pd.testing.assert_frame_equal(imputer_data[-1], X)
    assert sizes == [len(train) for train, _ in splits]
    assert model["curves"].n_knots == params["curves__n_knots"]
    assert model["ridge"].alpha == params["ridge__alpha"]


@pytest.mark.parametrize("pair", [None, (0, 1)])
@pytest.mark.parametrize("case", ["ordinary", "missing", "float32", "list", "tie"])
def test_selected_model_matches_original_pipeline_search_exactly(pair, case):
    X, y = _data()
    config = PairValidationConfig(seed=3, knots=(5, 3), alphas=(100., .1, 10.))
    if case == "missing":
        X.iloc[::5, 0] = np.nan
        X.iloc[:, -1] = np.nan
        train, _ = next(KFold(3, shuffle=True, random_state=3).split(X))
        X.iloc[train, 2] = np.nan
    elif case == "float32":
        X, y = X.to_numpy(dtype=np.float32), y.astype(np.float32)
    elif case == "list":
        X, y = X.to_numpy().tolist(), y.tolist()
    elif case == "tie":
        y = np.full(len(y), 2.)
    before = np.array(X, copy=True)
    expected = _reference_tune(X, y, pair, config)
    actual = _tune(X, y, pair, config)
    assert actual[1:] == expected[1:]
    np.testing.assert_array_equal(actual[0].predict(X), expected[0].predict(X))
    np.testing.assert_array_equal(actual[0]["ridge"].coef_, expected[0]["ridge"].coef_)
    np.testing.assert_array_equal(actual[0]["imputer"].statistics_, expected[0]["imputer"].statistics_)
    np.testing.assert_array_equal(np.asarray(X), before)
    if case == "tie":
        assert actual[1] == {"curves__n_knots": config.knots[0], "ridge__alpha": config.alphas[0]}


def test_all_candidate_fold_scores_match_original_search(monkeypatch):
    X, y = _data()
    X.iloc[::8, 0] = np.nan
    config = PairValidationConfig(seed=9)
    expected = _reference_search(X, y, (0, 1), config)
    original = implementation.get_scorer
    observed = []

    def recording_scorer(name):
        scorer = original(name)

        def score(*args, **kwargs):
            value = scorer(*args, **kwargs)
            observed.append(value)
            return value
        return score

    monkeypatch.setattr(implementation, "get_scorer", recording_scorer)
    _tune(X, y, (0, 1), config)
    for fold, actual in enumerate(np.asarray(observed).reshape(config.inner_folds, -1)):
        np.testing.assert_array_equal(actual, expected.cv_results_[f"split{fold}_test_score"])


def test_repeated_interaction_outputs_match_original_search_with_controls_and_covariates(monkeypatch):
    X, y = _data(180)
    X["Y"] = y + 1.5 * X.a * X.b
    X.iloc[::11, 2] = np.nan
    config = ExploreConfig(folds=3, tuning_folds=2, repeats=2, max_depth=1, max_interaction_pairs=2)
    actual = explore(X, "Y", controls=["d"], covariates=["c"], config=config)
    assert (actual.folds.contrast == "interaction").any()
    assert set(actual.folds[actual.folds.contrast == "interaction"].noise) == {False, True}
    monkeypatch.setattr(implementation, "_tune", _reference_tune)
    expected = explore(X, "Y", controls=["d"], covariates=["c"], config=config)
    for name in ("nodes", "links", "pairs", "folds", "bars", "fit"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
    assert actual.stops == expected.stops and actual.settings == expected.settings
