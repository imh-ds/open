"""Hand-calculated relevance contrasts and low-gain behavior."""

import pandas as pd
import pytest

from open.grouping import evaluate_pair_ablations
from open.learners.elastic_net import ElasticNetPredictor
from open.relevance import compute_relevance
from open.risk import evaluate_risk
from open.splits import SplitFold, SplitPlan


def test_one_target_integration_produces_signed_relevance_and_pair_ledger():
    data = pd.DataFrame({
        "y": [0., 1., 2., 3., 4., 5.],
        "a": [0., 1., 2., 3., 4., 5.],
        "b": [0., 2., 1., 3., 4., 5.],
        "c": [0., 1., 2., 5., 3., 4.],
    })
    candidates = ("a", "b", "c")
    plan = SplitPlan(6, 2, 1, 17, (
        SplitFold(0, 0, (3, 4, 5), (0, 1, 2)),
        SplitFold(0, 1, (0, 1, 2), (3, 4, 5)),
    ))
    parameters = {(0, 0): {"alpha": .01, "l1_ratio": .5},
                  (0, 1): {"alpha": .1, "l1_ratio": .9}}
    predictor = ElasticNetPredictor()
    common = dict(data=data, target="y", candidate_set=candidates, split_plan=plan)
    baseline = evaluate_risk(predictors=(), **common)
    full = evaluate_risk(predictors=candidates, predictor=predictor,
                         fitted_parameters=parameters, **common)
    single_results = {
        name: evaluate_risk(predictors=(name,), predictor=predictor,
                            fitted_parameters=parameters, **common)
        for name in candidates
    }
    loco_results = {
        name: evaluate_risk(
            predictors=tuple(candidate for candidate in candidates if candidate != name),
            predictor=predictor, fitted_parameters=parameters, **common,
        )
        for name in candidates
    }
    broad_risks = {name: result.aggregate_risk for name, result in single_results.items()}
    loco_risks = {name: result.aggregate_risk for name, result in loco_results.items()}
    pair_ledger = evaluate_pair_ablations(
        data, candidate_set=candidates, split_plan=plan, full=full,
        baseline=baseline, predictor=predictor, tau_dep=1.0,
    )
    relevance = compute_relevance(
        baseline.aggregate_risk, full.aggregate_risk,
        broad_risks=broad_risks, ablated_risks=loco_risks, tau_gain=.02,
    )

    def row_keys(result):
        return tuple(sorted(
            (fold.repeat_index, fold.fold_index, item.row_position)
            for fold in result.folds for item in fold.observations
        ))

    expected_keys = row_keys(full)
    assert row_keys(baseline) == expected_keys
    assert all(row_keys(result) == expected_keys
               for result in (*single_results.values(), *loco_results.values()))
    assert relevance.stop_reason is None
    assert set(relevance.broad) == set(relevance.unique) == set(candidates)
    assert all(contrast.normalized is not None for contrast in relevance.broad.values())
    assert any(contrast.raw < 0 for contrast in relevance.unique.values())
    assert pair_ledger.pairs
    weak_member_pair = next(pair for pair in pair_ledger.pairs if (pair.a, pair.b) == ("a", "b"))
    assert relevance.unique["b"].normalized < 0.05
    assert weak_member_pair.selected_fold_count == 1
    assert all(pair.selected_fold_count == pair.evaluated_fold_count
               for pair in pair_ledger.pairs)
    assert all(pair.raw == pytest.approx(pair.ablated_risk - pair.full_risk)
               for pair in pair_ledger.pairs)
    for pair in pair_ledger.pairs:
        selected_keys = tuple(sorted(
            key for key in expected_keys if key[:2] in pair.selected_folds
        ))
        assert row_keys(pair.ablation) == selected_keys


def test_broad_and_unique_contrasts_retain_their_signs() -> None:
    result = compute_relevance(
        10.0,
        6.0,
        broad_risks={"a": 8.0, "b": 11.0},
        ablated_risks={"a": 7.0, "b": 5.0},
        tau_gain=0.02,
    )

    assert result.total_gain == 4.0
    assert result.baseline_risk == 10.0
    assert result.full_risk == 6.0
    assert result.relative_gain == 0.4
    assert result.stop_reason is None
    assert result.broad["a"].raw == 2.0
    assert result.broad["a"].normalized == 0.5
    assert result.broad["b"].raw == -1.0
    assert result.broad["b"].normalized == -0.25
    assert result.unique["a"].raw == 1.0
    assert result.unique["a"].normalized == 0.25
    assert result.unique["b"].raw == -1.0
    assert result.unique["b"].normalized == -0.25


def test_below_threshold_keeps_raw_contrasts_but_stops_before_normalization() -> None:
    result = compute_relevance(
        10.0,
        9.9,
        broad_risks={"a": 8.0},
        ablated_risks={"a": 9.8},
        tau_gain=0.02,
    )

    assert result.total_gain == pytest.approx(0.1)
    assert result.baseline_risk == 10.0
    assert result.full_risk == 9.9
    assert result.relative_gain == pytest.approx(0.01)
    assert result.stop_reason == "LOW_TOTAL_GAIN"
    assert result.broad["a"].raw == 2.0
    assert result.broad["a"].normalized is None
    assert result.unique["a"].raw == pytest.approx(-0.1)
    assert result.unique["a"].normalized is None


@pytest.mark.parametrize("baseline,full", [(0.0, 0.0), (-1.0, 2.0)])
def test_nonpositive_baseline_has_no_relative_gain(baseline: float, full: float) -> None:
    result = compute_relevance(
        baseline,
        full,
        broad_risks={"a": 1.0},
        ablated_risks={"a": 3.0},
        tau_gain=0.02,
    )

    assert result.total_gain == baseline - full
    assert result.baseline_risk == baseline
    assert result.full_risk == full
    assert result.relative_gain is None
    assert result.stop_reason == "LOW_TOTAL_GAIN"
    assert result.broad["a"].normalized is None
    assert result.unique["a"].normalized is None


@pytest.mark.parametrize("full", [10.0, 11.0])
def test_nonpositive_total_gain_stops_even_with_zero_threshold(full: float) -> None:
    result = compute_relevance(
        10.0, full, broad_risks={}, ablated_risks={}, tau_gain=0.0
    )

    assert result.total_gain == 10.0 - full
    assert result.stop_reason == "LOW_TOTAL_GAIN"


def test_gain_exactly_at_threshold_is_eligible() -> None:
    result = compute_relevance(
        100.0,
        98.0,
        broad_risks={"a": 99.0},
        ablated_risks={"a": 99.0},
        tau_gain=0.02,
    )

    assert result.stop_reason is None
    assert result.relative_gain == 0.02
    assert result.broad["a"].normalized == 0.5
    assert result.unique["a"].normalized == 0.5


@pytest.mark.parametrize("baseline,full", [(float("inf"), 1.0), (1.0, float("nan"))])
def test_nonfinite_source_risk_is_rejected(baseline: float, full: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_relevance(baseline, full, broad_risks={}, ablated_risks={}, tau_gain=.02)
