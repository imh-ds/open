"""Layer-synchronous network expansion contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import open.recurse as recurse
from open.classify import PairRelation
from open.config import (
    AnalysisConfig,
    OpenConfig,
    PerformanceCVConfig,
    ThresholdConfig,
    TuningCVConfig,
)
from open.graph import StopReason
from open.local_target import LocalClass, LocalResult, VariableResult
from open.relevance import RelevanceContrast, RelevanceResult
from open.splits import SplitPlan, make_split_plan


def fixture(n=12, *, folds=2, tuning=2, max_depth=3):
    data = pd.DataFrame({name: np.arange(n, dtype=float) + index
                         for index, name in enumerate(("y", "a", "b", "c", "d"))})
    config = OpenConfig(AnalysisConfig("y", 41, max_depth=max_depth),
                        performance_cv=PerformanceCVConfig(folds, 1),
                        tuning_cv=TuningCVConfig(tuning))
    return data, config


def result(target, candidates, promoted=(), *, stop=False):
    broad = {name: RelevanceContrast(.6, .3) for name in candidates}
    unique = {name: RelevanceContrast(.4, .2) for name in candidates}
    relevance = RelevanceResult(1.0, .5, "LOW_TOTAL_GAIN" if stop else None,
                                broad, unique, 2.0, 1.0)
    classes = {name: VariableResult(LocalClass.PROXIMAL_UNIQUE if name in promoted
                                   else LocalClass.NEGLIGIBLE)
               for name in candidates}
    return LocalResult(stop, "LOW_TOTAL_GAIN" if stop else None, target,
                       relevance.total_gain, relevance.relative_gain, relevance, None,
                       {name: .3 for name in candidates}, {name: .2 for name in candidates},
                       None, (), (), (), classes, frozenset(promoted))


def test_one_split_plan_is_shared_and_same_layer_candidates_are_snapshotted(monkeypatch):
    data, config = fixture()
    made = []
    plan = make_split_plan(data, config)
    monkeypatch.setattr(recurse, "make_split_plan", lambda *args: made.append(args) or plan)
    calls = []

    def analyze(data, target, candidates, config, split_plan):
        calls.append((target, candidates, split_plan))
        promos = {"y": ("a", "b"), "a": ("c",), "b": ("c",)}.get(target, ())
        return result(target, candidates, promos)

    monkeypatch.setattr(recurse, "analyze_target", analyze)
    run = recurse.build_network_once(data, "y", tuple(data.columns), config)

    assert len(made) == 1
    assert [target for target, _, _ in calls] == ["y", "a", "b", "c"]
    assert calls[0][1] == ("a", "b", "c", "d")
    assert calls[1][1] == calls[2][1] == ("c", "d")
    assert all(call[2] is plan for call in calls)
    assert run.assigned_depth == {"y": 0, "a": 1, "b": 1, "c": 2}
    assert run.parents["c"] == ("a", "b")
    assert tuple((edge.source, edge.target) for edge in run.core_edges) == (
        ("a", "y"), ("b", "y"), ("c", "a"), ("c", "b"))


def test_graph_assembly_is_order_invariant_under_reversed_variable_order(monkeypatch):
    data, config = fixture()
    plan = make_split_plan(data, config)
    monkeypatch.setattr(recurse, "make_split_plan", lambda *args: plan)
    execution_orders = []

    def analyze(data, target, candidates, config, split_plan):
        execution_orders.append(target)
        promotions = {"y": ("a", "b"), "a": ("c",), "b": ("c",)}.get(target, ())
        return result(target, candidates, promotions)

    monkeypatch.setattr(recurse, "analyze_target", analyze)
    forward = recurse.build_network_once(data, "y", ("y", "a", "b", "c", "d"), config)
    forward_order = tuple(execution_orders)
    execution_orders.clear()
    reversed_run = recurse.build_network_once(data, "y", ("d", "c", "b", "a", "y"), config)
    reversed_order = tuple(execution_orders)

    assert forward_order == ("y", "a", "b", "c")
    assert reversed_order == ("y", "b", "a", "c")
    assert forward.assigned_depth == reversed_run.assigned_depth
    assert forward.parents == reversed_run.parents
    assert forward.local_results == reversed_run.local_results
    assert forward.core_edges == reversed_run.core_edges
    assert forward.shared_relations == reversed_run.shared_relations
    assert forward.stop_reasons == reversed_run.stop_reasons


@pytest.mark.parametrize("n,folds,tuning", [(4, 5, 2), (4, 2, 3)])
def test_preflight_infeasibility_stops_before_split_or_target_evaluation(monkeypatch, n, folds, tuning):
    data, config = fixture(n, folds=folds, tuning=tuning)
    monkeypatch.setattr(recurse, "make_split_plan",
                        lambda *args: pytest.fail("infeasible run created a split"))
    monkeypatch.setattr(recurse, "analyze_target",
                        lambda *args: pytest.fail("infeasible run analyzed a target"))

    run = recurse.build_network_once(data, "y", tuple(data.columns), config)

    assert run.assigned_depth == {"y": 0}
    assert run.stop_reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
    assert not run.local_results and not run.stop_reasons


def test_preflight_exact_feasibility_boundary_makes_one_split(monkeypatch):
    data, config = fixture(5, folds=2, tuning=2)
    plan = make_split_plan(data, config)
    made = []
    monkeypatch.setattr(recurse, "make_split_plan", lambda *args: made.append(1) or plan)
    monkeypatch.setattr(recurse, "analyze_target", lambda data, target, candidates, config, split:
                        result(target, candidates))

    run = recurse.build_network_once(data, "y", ("y", "a"), config)

    assert made == [1]
    assert run.local_results["y"].target == "y"


@pytest.mark.parametrize("root,variables", [
    ("a", ("y", "a")), ("y", ("y", "y")), ("y", ("y", "missing")),
])
def test_invalid_root_and_variable_roles_rejected_before_split(monkeypatch, root, variables):
    data, config = fixture()
    monkeypatch.setattr(recurse, "make_split_plan", lambda *args: pytest.fail("split created"))
    with pytest.raises(ValueError):
        recurse.build_network_once(data, root, variables, config)


@pytest.mark.parametrize("case", ["text", "boolean", "candidate_infinity", "root_nan"])
def test_invalid_scale_columns_rejected_before_split_even_when_not_reached(monkeypatch, case):
    data, config = fixture()
    if case == "text":
        data["d"] = "x"
    elif case == "boolean":
        data["d"] = True
    elif case == "candidate_infinity":
        data.loc[0, "d"] = np.inf
    elif case == "root_nan":
        data.loc[0, "y"] = np.nan
    monkeypatch.setattr(recurse, "make_split_plan", lambda *args: pytest.fail("split created"))
    with pytest.raises(ValueError):
        recurse.build_network_once(data, "y", tuple(data.columns), config)


def test_candidate_nan_is_allowed_for_preprocessing(monkeypatch):
    data, config = fixture(max_depth=1)
    data.loc[0, "a"] = np.nan
    monkeypatch.setattr(recurse, "analyze_target",
                        lambda data, target, candidates, config, plan:
                        result(target, candidates))

    run = recurse.build_network_once(data, "y", ("y", "a"), config)

    assert run.stop_reasons["y"] is StopReason.NO_PROMOTION


def test_screened_variables_remain_available_and_low_gain_and_no_promotion_stops(monkeypatch):
    data, config = fixture(max_depth=2)
    calls = []

    def analyze(data, target, candidates, config, plan):
        calls.append((target, candidates))
        if target == "y":
            return result(target, candidates, ("a", "b"))
        if target == "a":
            return result(target, candidates, stop=True)
        return result(target, candidates)

    monkeypatch.setattr(recurse, "analyze_target", analyze)
    run = recurse.build_network_once(data, "y", tuple(data.columns), config)

    assert calls == [("y", ("a", "b", "c", "d")),
                     ("a", ("c", "d")), ("b", ("c", "d"))]
    assert run.stop_reasons["a"] is StopReason.LOW_TOTAL_GAIN
    assert run.stop_reasons["b"] is StopReason.NO_PROMOTION


def test_depth_boundary_assigns_promotions_and_stops_without_evaluating(monkeypatch):
    data, config = fixture(max_depth=2)
    calls = []

    def analyze(data, target, candidates, config, plan):
        calls.append(target)
        return result(target, candidates, ("a",) if target == "y" else ("b",))

    monkeypatch.setattr(recurse, "analyze_target", analyze)
    run = recurse.build_network_once(data, "y", ("y", "a", "b"), config)

    assert calls == ["y", "a"]
    assert run.assigned_depth == {"y": 0, "a": 1, "b": 2}
    assert run.stop_reasons["b"] is StopReason.MAX_DEPTH
    assert run.max_depth == 2


def test_default_depth_cap_remains_three():
    _, config = fixture()
    assert replace(config, analysis=AnalysisConfig("y", 41)).analysis.max_depth == 3


def test_empty_candidate_snapshot_marks_frontier_without_local_result(monkeypatch):
    data, config = fixture()
    plan_made = []
    monkeypatch.setattr(recurse, "make_split_plan",
                        lambda *args: plan_made.append(1) or make_split_plan(*args))
    monkeypatch.setattr(recurse, "analyze_target",
                        lambda *args: pytest.fail("empty candidate layer evaluated"))

    run = recurse.build_network_once(data, "y", ("y",), config)

    assert plan_made == [1]
    assert run.stop_reasons == {"y": StopReason.NO_UNASSIGNED}
    assert not run.local_results


def test_shared_components_and_retained_pairs_are_relations_without_graph_nodes(monkeypatch):
    from open.grouping import PairAblation, RetainedSharedPair, SharedComponent

    data, config = fixture(max_depth=1)
    component = SharedComponent(("a", "b"), True, .8, .4, None)
    pair_ablation = PairAblation("c", "d", ((0, 0),), 1, 1, 1., 1., 2., 1.,
                                 1.8, 1., .8, .4, None)
    retained = RetainedSharedPair("c", "d", ("c", "d"), pair_ablation)

    def analyze(data, target, candidates, config, plan):
        local = result(target, candidates, ("a", "b", "c", "d"))
        classes = {name: VariableResult(LocalClass.SHARED_GROUP_MEMBER)
                   for name in local.promoted}
        return replace(local, shared_components=(component,), retained_pairs=(retained,), classes=classes)

    monkeypatch.setattr(recurse, "analyze_target", analyze)
    run = recurse.build_network_once(data, "y", tuple(data.columns), config)

    assert {(relation.members, type(relation.provenance)) for relation in run.shared_relations} == {
        (("a", "b"), SharedComponent), (("c", "d"), RetainedSharedPair)}
    assert set(run.assigned_depth) == {"y", "a", "b", "c", "d"}


def test_real_one_target_pipeline_runs_through_network_entry_point():
    rng = np.random.default_rng(93)
    a = rng.normal(size=40)
    b = rng.normal(size=40)
    data = pd.DataFrame({"y": 3 * a + rng.normal(scale=.1, size=40), "a": a, "b": b})
    config = OpenConfig(AnalysisConfig("y", 9, max_depth=1),
                        performance_cv=PerformanceCVConfig(2, 1),
                        tuning_cv=TuningCVConfig(2))

    run = recurse.build_network_once(data, "y", ("y", "a", "b"), config)

    assert run.local_results["y"].target == "y"
    assert "a" in run.local_results["y"].promoted
    assert all(run.stop_reasons[name] is StopReason.MAX_DEPTH
               for name in run.local_results["y"].promoted)


def test_fixed_seed_mediated_chain_screening_and_depth_cap():
    rng = np.random.default_rng(271828)
    s = rng.normal(size=80)
    d = s + 0.7 * rng.normal(size=80)
    a = d + 0.3 * rng.normal(size=80)
    data = pd.DataFrame({"A": a, "D": d, "S": s})
    thresholds = ThresholdConfig(
        min_total_gain_relative=0.02,
        broad_relative=0.05,
        unique_relative=0.15,
        group_relative=0.10,
        dependence_pair=0.70,
        stability=0.70,
    )
    config = OpenConfig(
        AnalysisConfig("A", 27, max_depth=2),
        thresholds=thresholds,
        performance_cv=PerformanceCVConfig(folds=2, repeats=1),
        tuning_cv=TuningCVConfig(folds=2),
    )

    run = recurse.build_network_once(data, "A", ("A", "D", "S"), config)

    root_result = run.local_results["A"]
    assert root_result.classes["D"].kind is LocalClass.PROXIMAL_UNIQUE
    assert root_result.classes["S"].kind is LocalClass.SCREENED_BY_PARTNER
    assert root_result.classes["S"].screened_by_partner == ("D",)
    assert "D" in root_result.promoted
    assert "S" not in root_result.promoted
    assert root_result.classes["S"].shared_groups == ()
    ds_pairs = [pair for pair in root_result.pair_classes.pairs
                if {pair.a, pair.b} == {"D", "S"}]
    assert len(ds_pairs) == 1
    assert ds_pairs[0].relation is PairRelation.PARTNER_SCREENING
    assert ds_pairs[0].screened_member == "S"
    assert ds_pairs[0].screened_by == "D"
    assert all(pair.relation is not PairRelation.PAIR_SHARED_REDUNDANT
               for pair in root_result.pair_classes.pairs)

    d_result = run.local_results["D"]
    assert "S" in d_result.relevance.broad
    assert d_result.classes["S"].kind is LocalClass.PROXIMAL_UNIQUE
    assert d_result.promoted == frozenset({"S"})
    assert run.assigned_depth == {"A": 0, "D": 1, "S": 2}
    assert run.parents["D"] == ("A",)
    assert run.parents["S"] == ("D",)
    assert run.stop_reasons["S"] is StopReason.MAX_DEPTH
    assert "S" not in run.local_results

    assert {(edge.source, edge.target) for edge in run.core_edges} == {
        ("D", "A"), ("S", "D")
    }
    assert not run.shared_relations
    for edge in run.core_edges:
        local_result = run.local_results[edge.target]
        broad = local_result.relevance.broad[edge.source]
        unique = local_result.relevance.unique[edge.source]
        assert (edge.broad_raw, edge.broad_normalized) == (broad.raw, broad.normalized)
        assert (edge.unique_raw, edge.unique_normalized) == (unique.raw, unique.normalized)
