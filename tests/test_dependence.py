"""Checks for fold-local Spearman pair candidate generation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from open.dependence.spearman import score_dependence
from open.grouping import evaluate_pair_ablations
from open.learners.elastic_net import ElasticNetPredictor
from open.risk import evaluate_risk
from open.splits import SplitFold, SplitPlan


def pair_fixture():
    data = pd.DataFrame({
        "y": [0., 1., 2., 3., 4., 5.],
        "a": [0., 1., 2., 3., 4., 5.],
        "b": [0., 2., 1., 3., 4., 5.],
        "c": [0., 1., 2., 5., 3., 4.],
    })
    plan = SplitPlan(6, 2, 1, 17, (
        SplitFold(0, 0, (3, 4, 5), (0, 1, 2)),
        SplitFold(0, 1, (0, 1, 2), (3, 4, 5)),
    ))
    params = {(0, 0): {"alpha": .01, "l1_ratio": .5},
              (0, 1): {"alpha": .5, "l1_ratio": 1.0}}
    kwargs = dict(target="y", candidate_set=("a", "b", "c"), split_plan=plan)
    full = evaluate_risk(data, predictors=("a", "b", "c"),
                         predictor=ElasticNetPredictor(), fitted_parameters=params, **kwargs)
    baseline = evaluate_risk(data, predictors=(), **kwargs)
    return data, plan, full, baseline


def test_pair_ablation_uses_only_selected_folds_and_matched_losses():
    data, plan, full, baseline = pair_fixture()

    result = evaluate_pair_ablations(
        data, candidate_set=("a", "b", "c"), split_plan=plan,
        full=full, baseline=baseline, predictor=ElasticNetPredictor(), tau_dep=1.0,
    )

    assert len(result.fold_scores) == 2
    pair = next(item for item in result.pairs if (item.a, item.b) == ("a", "b"))
    assert pair.selected_folds == ((0, 0),)
    assert pair.selected_fold_count == pair.evaluated_fold_count == 1
    assert pair.selection_frequency == pair.evaluation_frequency == .5
    assert [(fold.repeat_index, fold.fold_index) for fold in pair.ablation.folds] == [(0, 0)]
    assert pair.ablation.outer_model_fit_count == 1
    assert pair.ablation.tuning_model_fit_count == 0
    assert full.folds[0].fitted_parameters == {"alpha": .01, "l1_ratio": .5}
    assert full.folds[1].fitted_parameters == {"alpha": .5, "l1_ratio": 1.0}
    assert pair.ablation.folds[0].fitted_parameters == full.folds[0].fitted_parameters
    base = np.mean([o.loss for o in baseline.folds[0].observations])
    full_risk = np.mean([o.loss for o in full.folds[0].observations])
    ablated = np.mean([o.loss for o in pair.ablation.folds[0].observations])
    assert pair.total_gain == pytest.approx(base - full_risk)
    assert pair.raw == pytest.approx(ablated - full_risk)
    assert pair.normalized == pytest.approx(pair.raw / pair.total_gain)


def test_pair_selection_ignores_its_own_heldout_predictors_and_target():
    data, plan, full, baseline = pair_fixture()
    original = evaluate_pair_ablations(
        data, candidate_set=("a", "b", "c"), split_plan=plan,
        full=full, baseline=baseline, predictor=ElasticNetPredictor(), tau_dep=1.0,
    )
    changed = data.copy()
    changed.loc[[0, 1, 2], ["a", "b", "c", "y"]] = [
        [100., 2., 9., -300.], [1., 100., -9., 500.], [50., 8., 1., 900.],
    ]
    kwargs = dict(target="y", candidate_set=("a", "b", "c"), split_plan=plan)
    params = {(0, 0): {"alpha": .01, "l1_ratio": .5},
              (0, 1): {"alpha": .01, "l1_ratio": .5}}
    changed_full = evaluate_risk(changed, predictors=("a", "b", "c"),
                                 predictor=ElasticNetPredictor(), fitted_parameters=params, **kwargs)
    changed_baseline = evaluate_risk(changed, predictors=(), **kwargs)
    altered = evaluate_pair_ablations(
        changed, candidate_set=("a", "b", "c"), split_plan=plan,
        full=changed_full, baseline=changed_baseline,
        predictor=ElasticNetPredictor(), tau_dep=1.0,
    )

    assert altered.fold_scores[0] == original.fold_scores[0]
    assert any(score.candidate for score in original.fold_scores[0].scores)


def test_nonpositive_pair_gain_has_no_normalized_relevance():
    data, plan, _, _ = pair_fixture()
    data["y"] = [0., 1., 2., 2., 1., 0.]
    kwargs = dict(target="y", candidate_set=("a", "b", "c"), split_plan=plan)
    params = {(0, 0): {"alpha": .01, "l1_ratio": .5},
              (0, 1): {"alpha": .01, "l1_ratio": .5}}
    full = evaluate_risk(data, predictors=("a", "b", "c"),
                         predictor=ElasticNetPredictor(), fitted_parameters=params, **kwargs)
    baseline = evaluate_risk(data, predictors=(), **kwargs)

    result = evaluate_pair_ablations(
        data, candidate_set=("a", "b", "c"), split_plan=plan,
        full=full, baseline=baseline, predictor=ElasticNetPredictor(), tau_dep=1.0,
    )

    assert result.pairs
    assert all(pair.total_gain <= 0 and pair.normalized is None for pair in result.pairs)


def test_rejects_unmatched_baseline_heldout_keys():
    data, plan, full, baseline = pair_fixture()
    first = baseline.folds[0]
    damaged = replace(baseline, folds=(replace(first, observations=(
        replace(first.observations[0], row_position=5), *first.observations[1:]
    )), baseline.folds[1]))

    with pytest.raises(ValueError, match="held-out keys"):
        evaluate_pair_ablations(
            data, candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=damaged, predictor=ElasticNetPredictor(), tau_dep=1.0,
        )


def test_selected_two_predictor_pair_uses_training_mean_ablation():
    data, plan, full, baseline = pair_fixture()
    candidates = ("a", "b")
    kwargs = dict(target="y", candidate_set=candidates, split_plan=plan)
    params = {(0, 0): {"alpha": .01, "l1_ratio": .5},
              (0, 1): {"alpha": .01, "l1_ratio": .5}}
    full = evaluate_risk(data, predictors=candidates,
                         predictor=ElasticNetPredictor(), fitted_parameters=params, **kwargs)
    baseline = evaluate_risk(data, predictors=(), **kwargs)

    result = evaluate_pair_ablations(
        data, candidate_set=candidates, split_plan=plan,
        full=full, baseline=baseline, predictor=ElasticNetPredictor(), tau_dep=1.0,
    )

    (pair,) = result.pairs
    assert pair.selected_folds == ((0, 0),)
    assert pair.ablation.predictors == ()
    assert pair.ablation.outer_model_fit_count == pair.ablation.tuning_model_fit_count == 0
    assert pair.ablated_risk == pytest.approx(pair.baseline_risk)


def test_target_column_is_required_even_if_no_pair_is_selected():
    data, plan, full, baseline = pair_fixture()
    without_target = data.drop(columns="y")
    one_candidate = ("a",)

    with pytest.raises(ValueError, match="target column"):
        evaluate_pair_ablations(
            without_target, candidate_set=one_candidate, split_plan=plan,
            full=replace(full, candidate_set=one_candidate, predictors=one_candidate),
            baseline=replace(baseline, candidate_set=one_candidate),
            predictor=ElasticNetPredictor(), tau_dep=1.0,
        )


def test_target_cannot_be_candidate_even_if_there_are_no_pairs():
    data, plan, full, baseline = pair_fixture()
    only_target = ("y",)

    with pytest.raises(ValueError, match="target.*candidate"):
        evaluate_pair_ablations(
            data, candidate_set=only_target, split_plan=plan,
            full=replace(full, candidate_set=only_target, predictors=only_target),
            baseline=replace(baseline, candidate_set=only_target),
            predictor=ElasticNetPredictor(), tau_dep=1.0,
        )


@pytest.mark.parametrize("stale_result", ("baseline", "full"))
def test_rejects_stale_observed_target_under_the_same_split_plan(stale_result):
    data, plan, full, baseline = pair_fixture()
    changed = data.copy()
    changed.loc[0, "y"] = 99.0
    kwargs = dict(target="y", candidate_set=("a", "b", "c"), split_plan=plan)
    params = {(fold.repeat_index, fold.fold_index): fold.fitted_parameters for fold in full.folds}
    changed_full = evaluate_risk(changed, predictors=("a", "b", "c"),
                                 predictor=ElasticNetPredictor(), fitted_parameters=params, **kwargs)
    changed_baseline = evaluate_risk(changed, predictors=(), **kwargs)
    if stale_result == "full":
        changed_full = full
    else:
        changed_baseline = baseline

    with pytest.raises(ValueError, match="observed target"):
        evaluate_pair_ablations(
            changed, candidate_set=("a", "b", "c"), split_plan=plan,
            full=changed_full, baseline=changed_baseline,
            predictor=ElasticNetPredictor(), tau_dep=1.0,
        )


def test_absolute_spearman_pairs_are_ordered_and_threshold_is_inclusive():
    training = pd.DataFrame(
        {
            "z": [5, 4, 3, 2, 1],
            "target": [100, 1, 100, 1, 100],
            "a": [1, 2, 3, 4, 5],
            "m": [1, 5, 2, 4, 3],
        }
    )

    scores = score_dependence(training, ("z", "m", "a"), tau_dep=1.0)

    assert [(item.a, item.b, item.score, item.candidate) for item in scores] == [
        ("a", "m", 0.3, False),
        ("a", "z", 1.0, True),
        ("m", "z", 0.3, False),
    ]
    assert scores == score_dependence(
        training.assign(target=-999)[["target", "m", "a", "z"]],
        ("a", "z", "m"),
        tau_dep=1.0,
    )


def test_missing_values_use_training_median_before_spearman():
    training = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1, 2, 3, 4]})

    (pair,) = score_dependence(training, ("a", "b"), tau_dep=0.95)

    assert pair.score == pytest.approx(0.9486832980505138)
    assert pair.candidate is False


def test_pair_just_below_threshold_is_not_selected():
    training = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1, 2, 3, 4]})

    (pair,) = score_dependence(training, ("a", "b"), tau_dep=0.9486832980505142)

    assert pair.score == pytest.approx(0.9486832980505138)
    assert pair.candidate is False


def test_rounded_nonperfect_correlation_does_not_meet_perfect_threshold():
    a = np.arange(500_000)
    b = a.copy()
    b[:2] = b[1::-1]

    (pair,) = score_dependence(pd.DataFrame({"a": a, "b": b}), ("a", "b"), tau_dep=1.0)

    assert pair.score < 1.0
    assert pair.candidate is False


def test_undefined_pairs_are_recorded_but_never_candidates():
    training = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "constant": [8.0, 8.0, 8.0],
         "missing": [None, None, None]}
    )

    scores = score_dependence(training, ("missing", "constant", "a"), tau_dep=0.0)

    assert [(item.a, item.b, item.score, item.candidate, item.reason) for item in scores] == [
        ("a", "constant", None, False, "constant column"),
        ("a", "missing", None, False, "all values missing"),
        ("constant", "missing", None, False, "all values missing"),
    ]
