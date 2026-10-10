"""Held-out risk calculations and their pairing provenance."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from open.risk import evaluate_risk
from open.learners.elastic_net import ElasticNetPredictor
from open.splits import SplitFold, SplitPlan


def fixture_plan() -> SplitPlan:
    return SplitPlan(
        n_samples=5,
        folds=2,
        repeats=1,
        random_seed=17,
        splits=(
            SplitFold(0, 0, (3, 4), (0, 1, 2)),
            SplitFold(0, 1, (0, 1, 2), (3, 4)),
        ),
    )


def fixture_data() -> pd.DataFrame:
    return pd.DataFrame({"y": [0.0, 1.0, 2.0, 9.0, 11.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]})


def test_baseline_uses_each_training_mean_and_pools_heldout_losses():
    result = evaluate_risk(
        fixture_data(), target="y", candidate_set=("x",), predictors=(),
        split_plan=fixture_plan(),
    )

    assert result.learner is None
    assert result.loss == "mse"
    assert result.predictors == ()
    assert result.split_seed == 17
    assert [(o.row_position, o.observed, o.predicted, o.loss) for o in result.folds[0].observations] == [
        (0, 0.0, 10.0, 100.0), (1, 1.0, 10.0, 81.0), (2, 2.0, 10.0, 64.0),
    ]
    assert [(o.row_position, o.observed, o.predicted, o.loss) for o in result.folds[1].observations] == [
        (3, 9.0, 1.0, 64.0), (4, 11.0, 1.0, 100.0),
    ]
    assert result.folds[0].risk == pytest.approx(245 / 3)
    assert result.folds[1].risk == 82.0
    assert result.aggregate_risk == pytest.approx(409 / 5)
    assert result.outer_model_fit_count == 0
    assert result.tuning_model_fit_count == 0
    assert all(f.fitted_parameters is None for f in result.folds)
    with pytest.raises(FrozenInstanceError):
        result.aggregate_risk = 0


def test_full_and_ablated_models_reuse_fold_parameters_and_heldout_keys():
    data = pd.DataFrame({
        "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "z": [7.0, 0.0, 6.0, 1.0, 5.0, 2.0, 4.0, 3.0],
    })
    plan = SplitPlan(
        n_samples=8, folds=2, repeats=1, random_seed=17,
        splits=(
            SplitFold(0, 0, (4, 5, 6, 7), (0, 1, 2, 3)),
            SplitFold(0, 1, (0, 1, 2, 3), (4, 5, 6, 7)),
        ),
    )
    full = evaluate_risk(
        data, target="y", candidate_set=("x", "z"), predictors=("x", "z"),
        split_plan=plan, predictor=ElasticNetPredictor(), tuning_folds=2,
    )
    params = {(fold.repeat_index, fold.fold_index): fold.fitted_parameters for fold in full.folds}
    ablated = evaluate_risk(
        data, target="y", candidate_set=("x", "z"), predictors=("x",),
        split_plan=plan, predictor=ElasticNetPredictor(), fitted_parameters=params,
    )
    pair_subset = evaluate_risk(
        data, target="y", candidate_set=("x", "z"), predictors=("x",),
        split_plan=plan, predictor=ElasticNetPredictor(), fitted_parameters=params,
        selected_folds=((0, 1),),
    )
    baseline_subset = evaluate_risk(
        data, target="y", candidate_set=("x", "z"), predictors=(),
        split_plan=plan, selected_folds=((0, 0),),
    )

    def keys(result):
        return [
            (fold.repeat_index, fold.fold_index, item.row_position)
            for fold in result.folds for item in fold.observations
        ]

    assert full.learner == "elastic_net"
    assert full.candidate_set == ("x", "z")
    assert full.predictors == ("x", "z")
    assert full.outer_model_fit_count == 2
    assert full.tuning_model_fit_count == 100  # 25 candidates x 2 inner folds x 2 outer folds
    assert ablated.outer_model_fit_count == 2
    assert ablated.tuning_model_fit_count == 0
    assert all(set(f.fitted_parameters) == {"alpha", "l1_ratio"} for f in full.folds)
    assert [f.fitted_parameters for f in ablated.folds] == [f.fitted_parameters for f in full.folds]
    assert keys(full) == keys(ablated) == [(0, 0, i) for i in range(4)] + [(0, 1, i) for i in range(4, 8)]
    assert keys(baseline_subset) == keys(full)[:4]
    assert keys(pair_subset) == keys(full)[4:]
    assert pair_subset.outer_model_fit_count == 1
    assert pair_subset.tuning_model_fit_count == 0
    assert full.split_plan_id == ablated.split_plan_id == baseline_subset.split_plan_id == pair_subset.split_plan_id
    assert len(full.split_plan_id) == 64
    assert all(item.loss == pytest.approx((item.observed - item.predicted) ** 2)
               for fold in full.folds for item in fold.observations)


def test_empty_ablation_uses_baseline_even_with_a_predictor_instance():
    expected = evaluate_risk(
        fixture_data(), target="y", candidate_set=("x",), predictors=(),
        split_plan=fixture_plan(),
    )
    actual = evaluate_risk(
        fixture_data(), target="y", candidate_set=("x",), predictors=(),
        split_plan=fixture_plan(), predictor=ElasticNetPredictor(),
        selected_folds=((0, 0),),
    )

    assert actual.learner is None
    assert actual.predictors == ()
    assert actual.outer_model_fit_count == actual.tuning_model_fit_count == 0
    assert actual.split_plan_id == expected.split_plan_id
    assert actual.folds == expected.folds[:1]
    assert actual.aggregate_risk == pytest.approx(245 / 3)
