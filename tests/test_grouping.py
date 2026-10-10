"""Shared-group validation and dependence-component screening contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from open.classify import PairRelation, classify_pairs
from open.config import ThresholdConfig
from open.grouping import (
    PairAblationResult,
    analyze_groups,
    evaluate_pair_ablations,
)
from open.learners.elastic_net import ElasticNetPredictor
from open.relevance import RelevanceContrast, RelevanceResult
from open.risk import ObservationLoss, evaluate_risk
from open.splits import SplitFold, SplitPlan


def fixture(candidates=("a", "b", "c")):
    n = 8
    x = np.arange(n, dtype=float)
    data = pd.DataFrame({"y": 2 * x + 1})
    for index, name in enumerate(candidates):
        data[name] = x + (index * (x % 2))
    plan = SplitPlan(
        n, 2, 1, 17,
        (SplitFold(0, 0, (4, 5, 6, 7), (0, 1, 2, 3)),
         SplitFold(0, 1, (0, 1, 2, 3), (4, 5, 6, 7))),
    )
    params = {(0, 0): {"alpha": .01, "l1_ratio": .5},
              (0, 1): {"alpha": .5, "l1_ratio": 1.0}}
    common = dict(target="y", candidate_set=candidates, split_plan=plan)
    predictor = ElasticNetPredictor()
    full = evaluate_risk(data, predictors=candidates, predictor=predictor,
                         fitted_parameters=params, **common)
    baseline = evaluate_risk(data, predictors=(), **common)
    pairs = evaluate_pair_ablations(
        data, candidate_set=candidates, split_plan=plan, full=full,
        baseline=baseline, predictor=predictor, tau_dep=0.0,
    )
    # Use explicit relevance scores so classification boundaries are fixed;
    # group-risk arithmetic itself remains generated from row-keyed risks.
    relevance = RelevanceResult(
        total_gain=baseline.aggregate_risk - full.aggregate_risk,
        relative_gain=(baseline.aggregate_risk - full.aggregate_risk) / baseline.aggregate_risk,
        stop_reason=None,
        broad={name: RelevanceContrast(.2, .2) for name in candidates},
        unique={name: RelevanceContrast(.01, .01) for name in candidates},
        baseline_risk=baseline.aggregate_risk,
        full_risk=full.aggregate_risk,
    )
    thresholds = ThresholdConfig(
        min_total_gain_relative=0, broad_relative=.1, unique_relative=.05,
        group_relative=0, dependence_pair=0,
    )
    classification = classify_pairs(
        {name: value.normalized for name, value in relevance.broad.items()},
        {name: value.normalized for name, value in relevance.unique.items()},
        {(pair.a, pair.b): pair.normalized for pair in pairs.pairs},
        thresholds,
    )
    return data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification


def _controlled_group_risk(full, predictors, risk, observed_values):
    """Return a correctly keyed risk result with a chosen mean loss."""
    folds = []
    for fold in full.folds:
        losses = tuple(
            ObservationLoss(
                row.row_position, row.observed,
                row.observed + np.sqrt(
                    risk + (index - (len(fold.observations) - 1) / 2) * 0.01
                ),
                risk + (index - (len(fold.observations) - 1) / 2) * 0.01,
            )
            for index, row in enumerate(fold.observations)
        )
        folds.append(replace(fold, predictors=predictors, observations=losses,
                             risk=float(np.mean([item.loss for item in losses]))))
    pooled = float(np.mean([row.loss for fold in folds for row in fold.observations]))
    return replace(full, predictors=predictors, folds=tuple(folds), aggregate_risk=pooled)


def _run(data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification,
         monkeypatch, group_risk):
    import open.grouping as grouping

    seen = []

    def controlled(data_arg, *, target, candidate_set, predictors, split_plan, predictor,
                   fitted_parameters=None, selected_folds=None, **kwargs):
        seen.append((predictors, fitted_parameters, selected_folds))
        return _controlled_group_risk(full, predictors, group_risk, data_arg[target].to_numpy())

    monkeypatch.setattr(grouping, "evaluate_risk", controlled)
    result = analyze_groups(
        data, target="y", candidate_set=tuple(name for name in data if name != "y"),
        split_plan=plan, full=full, baseline=baseline, pair_ablations=pairs,
        classifications=classification, relevance=relevance, predictor=predictor,
        thresholds=thresholds,
    )
    return result, seen


def test_valid_three_member_shared_group_uses_all_fold_matched_rows_and_full_parameters(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    controlled_group_risk = full.aggregate_risk + relevance.total_gain * .4
    result, calls = _run(
        *values[:8], classification, monkeypatch, controlled_group_risk,
    )

    (group,) = result.shared_components
    assert group.members == ("a", "b", "c")
    assert group.validated is True
    assert group.raw == pytest.approx(relevance.total_gain * .4)
    assert group.normalized == pytest.approx(.4)
    assert len(group.ablation.folds) == len(plan.splits)
    # This fixture ablates every predictor; its no-predictor fit needs no
    # learner parameters. A nonempty group ablation is checked below.
    assert calls[0][1] is None
    assert calls[0][2] is None
    assert {
        (fold.repeat_index, fold.fold_index, row.row_position)
        for fold in group.ablation.folds for row in fold.observations
    } == {
        (fold.repeat_index, fold.fold_index, row.row_position)
        for fold in full.folds for row in fold.observations
    }


def test_failed_merged_component_retains_overlapping_pair_specific_relations(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    # A group result below tau_group fails, while all supported constituent
    # shared edges remain separate relations carrying their original ablations.
    thresholds = replace(thresholds, group_relative=.1)
    result, _ = _run(
        data, plan, predictor, full, baseline, pairs, relevance, thresholds,
        classify_pairs(
            {name: value.normalized for name, value in relevance.broad.items()},
            {name: value.normalized for name, value in relevance.unique.items()},
            {(pair.a, pair.b): pair.normalized for pair in pairs.pairs}, thresholds,
        ), monkeypatch, full.aggregate_risk + relevance.total_gain * .05,
    )

    assert result.shared_components[0].validated is False
    assert [item.members for item in result.retained_pairs] == [("a", "b"), ("a", "c")]
    assert all(item.pair is next(
        pair for pair in pairs.pairs if (pair.a, pair.b) == item.members
    ) for item in result.retained_pairs)
    assert len({member for item in result.retained_pairs for member in item.members}) == 3


def test_only_supported_shared_edges_enter_shared_components(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, _ = values
    relevance = replace(
        relevance,
        unique={"a": RelevanceContrast(.01, .01), "b": RelevanceContrast(.01, .01),
                "c": RelevanceContrast(.2, .2)},
    )
    thresholds = replace(thresholds, group_relative=.1)
    relations = classify_pairs(
        {name: value.normalized for name, value in relevance.broad.items()},
        {name: value.normalized for name, value in relevance.unique.items()},
        {(pair.a, pair.b): pair.normalized for pair in pairs.pairs}, thresholds,
    )
    result, calls = _run(
        data, plan, predictor, full, baseline, pairs, relevance, thresholds,
        relations, monkeypatch, full.aggregate_risk + relevance.total_gain * .8,
    )

    assert [(item.a, item.b, item.relation) for item in relations.pairs] == [
        ("a", "b", PairRelation.PAIR_SHARED_REDUNDANT),
        ("a", "c", PairRelation.PARTNER_SCREENING),
        ("b", "c", PairRelation.NOT_GROUP_IMPORTANT),
    ]
    assert [component.members for component in result.shared_components] == [("a", "b")]
    assert calls[0][1] == {
        (fold.repeat_index, fold.fold_index): fold.fitted_parameters for fold in full.folds
    }


def test_dependence_components_screen_mixed_members_without_becoming_shared_edges(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    relevance = replace(
        relevance,
        broad={"a": RelevanceContrast(.2, .2), "b": RelevanceContrast(.2, .2),
               "c": RelevanceContrast(.2, .2)},
        unique={"a": RelevanceContrast(.2, .2), "b": RelevanceContrast(.01, .01),
                "c": RelevanceContrast(.01, .01)},
    )
    classification = classify_pairs(
        {name: item.normalized for name, item in relevance.broad.items()},
        {name: item.normalized for name, item in relevance.unique.items()},
        {(pair.a, pair.b): pair.normalized for pair in pairs.pairs}, thresholds,
    )
    result, _ = _run(
        data, plan, predictor, full, baseline, pairs, relevance, thresholds,
        classification, monkeypatch, full.aggregate_risk + relevance.total_gain * .4,
    )

    (component,) = result.dependence_components
    assert component.members == ("a", "b", "c")
    assert component.strong_members == ("a",)
    assert component.screened_by_set == (("b", ("a",)), ("c", ("a",)))
    assert all(edge.selected_folds == ((0, 0), (0, 1)) for edge in component.edges)
    assert all(edge.selected_fold_count == edge.evaluated_fold_count == 2 for edge in component.edges)
    assert result.shared_components == ()
    assert all(
        pair.relation is not PairRelation.PAIR_SHARED_REDUNDANT
        for pair in classification.pairs
    )


def test_weak_dependence_component_and_threshold_equality_are_preserved(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    relevance = replace(
        relevance,
        broad={"a": RelevanceContrast(.2, .2), "b": RelevanceContrast(.05, .05),
               "c": RelevanceContrast(.049, .049)},
        unique={"a": RelevanceContrast(.2, .2), "b": RelevanceContrast(.01, .01),
                "c": RelevanceContrast(.01, .01)},
    )
    thresholds = replace(thresholds, broad_relative=.05, group_relative=0.0)
    classification = classify_pairs(
        {name: item.normalized for name, item in relevance.broad.items()},
        {name: item.normalized for name, item in relevance.unique.items()},
        {(pair.a, pair.b): pair.normalized for pair in pairs.pairs}, thresholds,
    )
    result, _ = _run(
        data, plan, predictor, full, baseline, pairs, relevance, thresholds,
        classification, monkeypatch, full.aggregate_risk + relevance.total_gain,
    )

    (component,) = result.dependence_components
    assert component.screened_by_set == (("b", ("a",)),)
    assert component.negligible_members == ("c",)
    assert component.edges[0].selection_frequency == 1.0


def test_all_weak_dependence_component_has_no_set_screen_assignments(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    result, _ = _run(
        data, plan, predictor, full, baseline, pairs, relevance, thresholds,
        classification, monkeypatch, full.aggregate_risk + relevance.total_gain * .4,
    )

    (component,) = result.dependence_components
    assert component.strong_members == ()
    assert component.screened_by_set == ()
    assert component.negligible_members == ()


def test_rejects_candidate_flag_inconsistent_with_dependence_threshold():
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = fixture()
    first = pairs.fold_scores[0]
    changed = replace(first.scores[0], candidate=not first.scores[0].candidate)
    altered_pairs = replace(
        pairs,
        fold_scores=(replace(first, scores=(changed, *first.scores[1:])), *pairs.fold_scores[1:]),
    )

    with pytest.raises(ValueError, match="candidate decision does not match"):
        analyze_groups(
            data, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=baseline, pair_ablations=altered_pairs,
            classifications=classification, relevance=relevance, predictor=predictor,
            thresholds=thresholds,
        )


def test_rejects_dependence_score_outside_absolute_correlation_range():
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = fixture()
    first = pairs.fold_scores[0]
    altered_score = replace(first.scores[0], score=1.01)
    altered_pairs = replace(
        pairs,
        fold_scores=(replace(first, scores=(altered_score, *first.scores[1:])),
                     *pairs.fold_scores[1:]),
    )

    with pytest.raises(ValueError, match="scores must be finite numbers in"):
        analyze_groups(
            data, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=baseline, pair_ablations=altered_pairs,
            classifications=classification, relevance=relevance, predictor=predictor,
            thresholds=thresholds,
        )


def test_rejects_mismatched_risk_metadata_and_stale_observed_target(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    damaged = replace(full, target="other")
    with pytest.raises(ValueError, match="target, candidate set, or split plan"):
        analyze_groups(
            data, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=damaged, baseline=baseline, pair_ablations=pairs,
            classifications=classification, relevance=relevance, predictor=predictor,
            thresholds=thresholds,
        )

    changed = data.copy()
    changed.loc[0, "y"] += 1
    with pytest.raises(ValueError, match="observed target"):
        analyze_groups(
            changed, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=baseline, pair_ablations=pairs,
            classifications=classification, relevance=relevance, predictor=predictor,
            thresholds=thresholds,
        )


@pytest.mark.parametrize("invalid_score", [None, "not-a-number"])
def test_rejects_missing_or_malformed_normalized_relevance_scores(invalid_score):
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = fixture()
    malformed = replace(
        relevance,
        unique={
            "a": RelevanceContrast(.01, .01),
            "b": RelevanceContrast(.01, .01),
            "c": RelevanceContrast(.01, invalid_score),
        },
    )

    with pytest.raises(ValueError, match="finite and normalized"):
        analyze_groups(
            data, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=baseline, pair_ablations=pairs,
            classifications=classification, relevance=malformed, predictor=predictor,
            thresholds=thresholds,
        )


def test_rejects_forged_relative_gain_in_relevance_precondition():
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = fixture()
    forged = replace(relevance, relative_gain=.01)

    with pytest.raises(ValueError, match="relative gain does not match"):
        analyze_groups(
            data, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=baseline, pair_ablations=pairs,
            classifications=classification, relevance=forged, predictor=predictor,
            thresholds=thresholds,
        )


def test_two_candidate_pair_ablation_needs_no_fitted_parameters():
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = fixture(
        candidates=("a", "b")
    )

    result = analyze_groups(
        data, target="y", candidate_set=("a", "b"), split_plan=plan,
        full=full, baseline=baseline, pair_ablations=pairs,
        classifications=classification, relevance=relevance, predictor=predictor,
        thresholds=thresholds,
    )

    (pair,) = pairs.pairs
    assert pair.ablation.predictors == ()
    assert all(fold.fitted_parameters is None for fold in pair.ablation.folds)
    assert result.dependence_components


def test_rejects_group_row_loss_that_disagrees_with_observed_prediction_mse(monkeypatch):
    values = fixture()
    data, plan, predictor, full, baseline, pairs, relevance, thresholds, classification = values
    import open.grouping as grouping

    def forged_group_result(data_arg, *, target, candidate_set, predictors, split_plan,
                            predictor, fitted_parameters=None, selected_folds=None, **kwargs):
        result = _controlled_group_risk(
            full, predictors, full.aggregate_risk + relevance.total_gain * .4,
            data_arg[target].to_numpy(),
        )
        first_fold = result.folds[0]
        first_row = replace(first_fold.observations[0], loss=first_fold.observations[0].loss + .5)
        observations = (first_row, *first_fold.observations[1:])
        changed_fold = replace(
            first_fold, observations=observations,
            risk=float(np.mean([row.loss for row in observations])),
        )
        folds = (changed_fold, *result.folds[1:])
        aggregate = float(np.mean([row.loss for fold in folds for row in fold.observations]))
        return replace(result, folds=folds, aggregate_risk=aggregate)

    monkeypatch.setattr(grouping, "evaluate_risk", forged_group_result)
    with pytest.raises(ValueError, match="row-keyed loss does not match observed/predicted MSE"):
        analyze_groups(
            data, target="y", candidate_set=("a", "b", "c"), split_plan=plan,
            full=full, baseline=baseline, pair_ablations=pairs,
            classifications=classification, relevance=relevance, predictor=predictor,
            thresholds=thresholds,
        )
