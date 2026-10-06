import json

import numpy as np
import pandas as pd
import pytest

from open.config import AnalysisConfig, OpenConfig, PerformanceCVConfig, PreprocessingConfig
from open.learners.elastic_net import ElasticNetPredictor
from open.splits import SplitPlan, make_split_plan
from open.preprocessing import make_preprocessor


def config(*, seed=19, folds=4, repeats=2):
    return OpenConfig(
        analysis=AnalysisConfig(root="y", random_seed=seed),
        performance_cv=PerformanceCVConfig(folds=folds, repeats=repeats),
    )


def test_repeated_kfold_plan_is_deterministic_disjoint_and_covers_each_row():
    data = pd.DataFrame({"x": np.arange(12), "y": np.arange(12) * 2})
    first = make_split_plan(data, config())
    second = make_split_plan(data, config())

    assert first.to_dict() == second.to_dict()
    assert len(first.splits) == 8
    held_out_counts = np.zeros(len(data), dtype=int)
    for split in first.splits:
        assert split.train_indices
        assert split.test_indices
        assert set(split.train_indices).isdisjoint(split.test_indices)
        held_out_counts[list(split.test_indices)] += 1
    assert held_out_counts.tolist() == [2] * len(data)


def test_split_plan_round_trip_json_and_file(tmp_path):
    data = pd.DataFrame({"x": np.arange(12)})
    plan = make_split_plan(data, config())

    restored = SplitPlan.from_json(plan.to_json())
    assert restored == plan
    path = tmp_path / "splits.json"
    plan.save(path)
    assert SplitPlan.load(path) == plan


def test_invalid_split_plan_version_bounds_and_overlap_are_rejected():
    plan = make_split_plan(pd.DataFrame({"x": np.arange(12)}), config())
    value = plan.to_dict()

    wrong_version = json.loads(json.dumps(value))
    wrong_version["version"] = 999
    with pytest.raises(ValueError, match="version"):
        SplitPlan.from_dict(wrong_version)

    out_of_bounds = json.loads(json.dumps(value))
    out_of_bounds["splits"][0]["test_indices"][0] = 12
    with pytest.raises(ValueError, match="bounds"):
        SplitPlan.from_dict(out_of_bounds)

    overlapping = json.loads(json.dumps(value))
    overlapping["splits"][0]["test_indices"][0] = overlapping["splits"][0]["train_indices"][0]
    with pytest.raises(ValueError, match="disjoint"):
        SplitPlan.from_dict(overlapping)

    empty = json.loads(json.dumps(value))
    empty["splits"][0]["train_indices"] = []
    with pytest.raises(ValueError, match="train partition must be nonempty"):
        SplitPlan.from_dict(empty)

    incomplete = json.loads(json.dumps(value))
    incomplete["splits"][0]["train_indices"].pop()
    with pytest.raises(ValueError, match="partition all sample positions"):
        SplitPlan.from_dict(incomplete)


def test_split_plan_rejects_empty_partitions_and_explicit_groups():
    with pytest.raises(ValueError, match="at least 2 rows"):
        make_split_plan(pd.DataFrame({"x": [1]}), config(folds=2, repeats=1))

    with pytest.raises(ValueError, match="grouped or repeated-measures"):
        make_split_plan(pd.DataFrame({"x": np.arange(12)}), config(), groups=np.repeat([1, 2], 6))

    negative_seed_config = OpenConfig(
        analysis=AnalysisConfig(root="y", random_seed=-1),
        performance_cv=PerformanceCVConfig(folds=4, repeats=2),
    )
    with pytest.raises(ValueError, match="scikit-learn range"):
        make_split_plan(pd.DataFrame({"x": np.arange(12)}), negative_seed_config)


def test_preprocessor_fit_uses_training_rows_only():
    train = np.array([[1.0], [np.nan], [3.0]])
    held_out = np.array([[1_000_000.0]])
    preprocessor = make_preprocessor(PreprocessingConfig())

    preprocessor.fit(train)
    training_median = preprocessor.named_steps["imputer"].statistics_.copy()
    training_mean = preprocessor.named_steps["scaler"].mean_.copy()
    preprocessor.transform(held_out)

    np.testing.assert_allclose(training_median, [2.0])
    np.testing.assert_allclose(training_mean, [2.0])
    np.testing.assert_allclose(preprocessor.named_steps["imputer"].statistics_, training_median)
    np.testing.assert_allclose(preprocessor.named_steps["scaler"].mean_, training_mean)


def test_preprocessor_preserves_a_feature_missing_in_the_training_fold():
    preprocessor = make_preprocessor(PreprocessingConfig())
    preprocessor.fit(np.array([[1.0, np.nan], [3.0, np.nan]]))

    transformed = preprocessor.transform(np.array([[2.0, 8.0]]))
    assert transformed.shape == (1, 2)
    np.testing.assert_allclose(preprocessor.named_steps["imputer"].statistics_, [2.0, 0.0])


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        ([0, 1, 2], [3, 3], "duplicates"),
        ([-1, 1, 2], [3], "out of bounds"),
        ([0, 1, 2], [5], "out of bounds"),
    ],
)
def test_elastic_net_rejects_invalid_inner_cv_indices(train, test, message):
    X = np.arange(10, dtype=float).reshape(5, 2)
    y = np.arange(5, dtype=float)
    cv = [(np.asarray(train), np.asarray(test))]

    with pytest.raises(ValueError, match=message):
        ElasticNetPredictor().tune(X, y, cv=cv)


def test_failed_refit_cannot_leave_a_stale_model_available():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.arange(6, dtype=float)
    learner = ElasticNetPredictor().fit(X, y)

    with pytest.raises(ValueError):
        learner.fit(X[:, 0], y)
    assert not learner.is_fitted
    with pytest.raises(RuntimeError, match="fit must be called"):
        learner.predict(X[:1])


@pytest.mark.parametrize(
    ("alpha", "l1_ratio"),
    [(np.nan, 0.5), (0.1, np.inf)],
)
def test_elastic_net_rejects_non_finite_parameters(alpha, l1_ratio):
    with pytest.raises(ValueError):
        ElasticNetPredictor().fit(
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            {"alpha": alpha, "l1_ratio": l1_ratio},
        )


def test_elastic_net_clone_tune_fit_and_predict():
    X = np.arange(30, dtype=float).reshape(15, 2)
    X[:, 1] = X[:, 0] ** 2
    y = 2.0 * X[:, 0] - 0.1 * X[:, 1]
    cv = [
        (np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), np.array([0, 1, 2, 13, 14])),
        (np.array([0, 1, 2, 8, 9, 10, 11, 12, 13, 14]), np.array([3, 4, 5, 6, 7])),
    ]
    learner = ElasticNetPredictor()
    clone = learner.clone()

    assert clone is not learner
    assert not clone.is_fitted
    params = learner.tune(X, y, cv=cv, loss="mse")
    assert set(params) == {"alpha", "l1_ratio"}
    assert params["alpha"] in {0.0001, 0.001, 0.01, 0.1, 1.0}
    assert params["l1_ratio"] in {0.1, 0.5, 0.9, 0.95, 1.0}

    learner.fit(X, y, params)
    prediction = learner.predict(X[:3])
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
