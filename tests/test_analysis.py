"""Composition tests for the one-target public analysis pipeline."""

from dataclasses import replace
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

from open.config import AnalysisConfig, OpenConfig, PerformanceCVConfig, TuningCVConfig
from open.learners.elastic_net import ElasticNetPredictor
from open.local_target import LocalClass
from open.risk import FoldRisk, ObservationLoss, RiskResult
from open.splits import make_split_plan


def analysis_module():
    assert importlib.util.find_spec("open.analysis") is not None, "one-target analysis module is missing"
    return importlib.import_module("open.analysis")


def fixture():
    rng = np.random.default_rng(41)
    a = rng.normal(size=48)
    b = rng.normal(size=48)
    data = pd.DataFrame({"y": 4 * a + rng.normal(scale=.1, size=48), "a": a, "b": b})
    config = OpenConfig(AnalysisConfig("y", 17),
                        performance_cv=PerformanceCVConfig(2, 1),
                        tuning_cv=TuningCVConfig(2))
    return data, config, make_split_plan(data, config)


def test_direct_predictor_real_elastic_net_path():
    data, config, plan = fixture()
    result = analysis_module().analyze_target(data, "y", ("a", "b"), config, plan)
    assert not result.stop
    assert result.classes["a"].kind is LocalClass.PROXIMAL_UNIQUE
    assert result.promoted == frozenset({"a"})
    assert result.relevance.broad["a"].raw > 0
    assert result.unique["a"] == result.relevance.unique["a"].normalized
    assert result.pair_ablations is not None


def test_redundant_shared_substitutes_and_matched_split_metadata(monkeypatch):
    data, config, plan = fixture()
    data["b"] = data["a"]
    monkeypatch.setattr(ElasticNetPredictor, "tune", lambda *args, **kwargs: {"alpha": .001, "l1_ratio": .1})
    module = analysis_module()
    import open.grouping as grouping
    source_evaluate = module.evaluate_risk
    risk_results = []

    def record_evaluation(*args, **kwargs):
        risk = source_evaluate(*args, **kwargs)
        risk_results.append(risk)
        return risk

    monkeypatch.setattr(module, "evaluate_risk", record_evaluation)
    monkeypatch.setattr(grouping, "evaluate_risk", record_evaluation)
    result = module.analyze_target(data, "y", ("b", "a"), config, plan)
    assert set(result.promoted) == {"a", "b"}
    assert all(value.kind is LocalClass.SHARED_GROUP_MEMBER for value in result.classes.values())
    group, = result.shared_components
    assert group.validated and group.members == ("a", "b")
    pair, = result.pair_ablations.pairs
    assert pair.selected_folds == tuple((s.repeat, s.fold) for s in plan.splits)
    assert pair.selected_fold_count == pair.evaluated_fold_count == 2
    assert pair.ablation.split_seed == plan.random_seed
    assert pair.ablation.split_plan_id == group.ablation.split_plan_id
    assert pair.raw == pytest.approx(pair.normalized * pair.total_gain)
    assert group.raw == pytest.approx(group.normalized * result.total_gain)
    assert group.ablation in risk_results
    assert result.outer_model_fit_count == sum(r.outer_model_fit_count for r in risk_results)
    assert result.tuning_model_fit_count == sum(r.tuning_model_fit_count for r in risk_results)


def controlled_risks(monkeypatch, module, data, plan, risks, results=None):
    """Inject row-keyed losses, leaving P1 arithmetic and all P2 stages real."""
    import hashlib
    import open.grouping as grouping
    calls = []

    def evaluate(data, *, target, candidate_set, predictors, split_plan, predictor=None,
                 fitted_parameters=None, selected_folds=None, tuning_folds=3):
        calls.append((predictors, fitted_parameters, selected_folds, tuning_folds))
        loss = risks[frozenset(predictors)]
        folds = []
        for split in split_plan.splits:
            key = (split.repeat, split.fold)
            if selected_folds is not None and key not in selected_folds:
                continue
            params = None if not predictors else (
                fitted_parameters[key] if fitted_parameters is not None else {"alpha": .01 + split.fold, "l1_ratio": .5})
            observations = tuple(ObservationLoss(row, float(data[target].iloc[row]),
                                                 float(data[target].iloc[row]) + np.sqrt(loss), loss)
                                 for row in split.test_indices)
            folds.append(FoldRisk(*key, predictors, observations, loss, params))
        result = RiskResult(target, candidate_set, predictors, "elastic_net" if predictors else None,
                          "mse", hashlib.sha256(split_plan.to_json(indent=None).encode()).hexdigest(),
                          split_plan.random_seed, tuple(folds), loss, len(folds) if predictors else 0,
                          len(folds) * 50 if predictors and fitted_parameters is None else 0)
        if results is not None:
            results.append(result)
        return result

    monkeypatch.setattr(module, "evaluate_risk", evaluate)
    monkeypatch.setattr(grouping, "evaluate_risk", evaluate)
    return calls


def test_signed_evidence_and_loco_parameter_reuse(monkeypatch):
    module = analysis_module()
    data, config, plan = fixture()
    config = replace(config, thresholds=replace(config.thresholds, dependence_pair=0))
    risk_results = []
    calls = controlled_risks(monkeypatch, module, data, plan,
                             {frozenset(): 10., frozenset({"a", "b"}): 2.,
                              frozenset({"a"}): 1., frozenset({"b"}): 12.}, risk_results)
    result = module.analyze_target(data, "y", ("a", "b"), config, plan)
    assert result.relevance.broad["b"].raw == -2
    assert result.broad["b"] == -.25
    assert result.relevance.unique["b"].raw == -1
    assert result.unique["b"] == -.125
    assert result.pair_ablations.pairs
    assert result.outer_model_fit_count == sum(r.outer_model_fit_count for r in risk_results)
    assert result.tuning_model_fit_count == sum(r.tuning_model_fit_count for r in risk_results)
    # Full model and single-predictor broad models tune independently;
    # nonempty LOCO fits reuse each outer fold's full-model parameters.
    reused = [params for predictors, params, selected, tuning in calls if predictors and params is not None]
    assert len(reused) == 2
    assert all(params == {(0, 0): {"alpha": .01, "l1_ratio": .5},
                          (0, 1): {"alpha": 1.01, "l1_ratio": .5}} for params in reused)
    assert all(tuning == 2 for predictors, params, selected, tuning in calls if predictors and params is None)


def test_low_gain_stops_before_pair_or_group_analysis(monkeypatch):
    module = analysis_module()
    data, config, plan = fixture()
    risk_results = []
    controlled_risks(monkeypatch, module, data, plan,
                     {frozenset(): 10., frozenset({"a", "b"}): 10.,
                      frozenset({"a"}): 11., frozenset({"b"}): 12.}, risk_results)
    def forbidden(*args, **kwargs):
        pytest.fail("stopped target reached P2")
    monkeypatch.setattr(module, "evaluate_pair_ablations", forbidden)
    monkeypatch.setattr(module, "analyze_groups", forbidden)
    result = module.analyze_target(data, "y", ("a", "b"), config, plan)
    assert result.stop and result.reason == "LOW_TOTAL_GAIN"
    assert result.pair_ablations is None and result.pair_classes is None
    assert not result.classes and not result.shared_components and not result.dependence_components
    assert result.relevance.broad["a"].raw == -1
    assert result.broad == {"a": None, "b": None}
    assert result.outer_model_fit_count == sum(r.outer_model_fit_count for r in risk_results)
    assert result.tuning_model_fit_count == sum(r.tuning_model_fit_count for r in risk_results)


@pytest.mark.parametrize("case", ["target_candidate", "duplicate", "empty", "missing", "rows", "seed", "folds", "repeats", "duplicate_columns", "nonnumeric", "boolean"])
def test_rejects_incompatible_roles_data_and_split_plan(case):
    module = analysis_module()
    data, config, plan = fixture()
    candidates = ("a", "b")
    if case == "target_candidate": candidates = ("a", "y")
    if case == "duplicate": candidates = ("a", "a")
    if case == "empty": candidates = ()
    if case == "missing": candidates = ("absent",)
    if case == "rows": data = data.iloc[:-1]
    if case == "seed": config = replace(config, analysis=AnalysisConfig("y", 18))
    if case == "folds": config = replace(config, performance_cv=PerformanceCVConfig(3, 1))
    if case == "repeats": config = replace(config, performance_cv=PerformanceCVConfig(2, 2))
    if case == "duplicate_columns": data.columns = ["y", "a", "a"]
    if case == "nonnumeric": data["a"] = "1"
    if case == "boolean": data["a"] = True
    with pytest.raises(ValueError):
        module.analyze_target(data, "y", candidates, config, plan)


def test_pairs_are_evaluated_only_on_selected_training_folds(monkeypatch):
    module = analysis_module()
    from open.splits import SplitFold, SplitPlan
    data = pd.DataFrame({"y": np.arange(12, dtype=float),
                         "a": np.tile(np.arange(6, dtype=float), 2),
                         "b": [0., 5., 1., 4., 2., 3., 0., 1., 2., 3., 4., 5.],
                         "c": np.tile([2., 0., 5., 1., 4., 3.], 2)})
    config = OpenConfig(AnalysisConfig("y", 17), performance_cv=PerformanceCVConfig(2, 1),
                        tuning_cv=TuningCVConfig(2))
    plan = SplitPlan(12, 2, 1, 17,
                     (SplitFold(0, 0, tuple(range(6, 12)), tuple(range(6))),
                      SplitFold(0, 1, tuple(range(6)), tuple(range(6, 12)))))
    risks = {frozenset(): 10., frozenset({"a", "b", "c"}): 2.,
             frozenset({"a"}): 4., frozenset({"b"}): 5., frozenset({"c"}): 6.,
             frozenset({"a", "b"}): 3., frozenset({"a", "c"}): 3.,
             frozenset({"b", "c"}): 3.}
    calls = controlled_risks(monkeypatch, module, data, plan, risks)
    result = module.analyze_target(data, "y", ("a", "b", "c"), config, plan)
    pair, = result.pair_ablations.pairs
    assert (pair.a, pair.b) == ("a", "b")
    assert pair.selected_folds == ((0, 0),)
    assert pair.selected_fold_count == pair.evaluated_fold_count == 1
    assert pair.selection_frequency == pair.evaluation_frequency == .5
    assert [fold.fold_index for fold in pair.ablation.folds] == [0]
    assert pair.ablation.folds[0].fitted_parameters == {"alpha": .01, "l1_ratio": .5}
    assert any(predictors == ("c",) and selected == ((0, 0),)
               for predictors, params, selected, tuning in calls)


@pytest.mark.parametrize("case", ["data", "config", "split_plan", "target_missing", "target_blank", "target_nan", "candidate_inf", "tuning_large"])
def test_invalid_inputs_fail_before_fitting(case, monkeypatch):
    module = analysis_module()
    data, config, plan = fixture()
    target = "y"
    if case == "data": data = data.to_numpy()
    if case == "config": config = None
    if case == "split_plan": plan = None
    if case == "target_missing": target = "absent"
    if case == "target_blank": target = " "
    if case == "target_nan": data.loc[0, "y"] = np.nan
    if case == "candidate_inf": data.loc[0, "a"] = np.inf
    if case == "tuning_large": config = replace(config, tuning_cv=TuningCVConfig(48))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid inputs reached learner fitting")
    monkeypatch.setattr(module, "evaluate_risk", forbidden)
    with pytest.raises((TypeError, ValueError)):
        module.analyze_target(data, target, ("a", "b"), config, plan)
