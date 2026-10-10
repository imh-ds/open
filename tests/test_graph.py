"""One-run graph boundaries preserve evidence and typed stop scope."""

from dataclasses import FrozenInstanceError, replace

import pytest

from open.graph import CoreEdge, NetworkRun, SharedRelation, StopReason
from open.grouping import PairAblation, RetainedSharedPair, SharedComponent
from open.local_target import LocalClass, LocalResult, VariableResult
from open.relevance import RelevanceContrast, RelevanceResult


def local(target="y", names=("a",), promoted=("a",), stopped=False):
    relevance = RelevanceResult(
        0.0 if stopped else 1.0, 0.0 if stopped else .5,
        "LOW_TOTAL_GAIN" if stopped else None,
        {n: RelevanceContrast(.6, .3) for n in names},
        {n: RelevanceContrast(.4, .2) for n in names},
        1.0 if stopped else 2.0, 1.0,
    )
    return LocalResult(
        stopped, relevance.stop_reason, target, relevance.total_gain,
        relevance.relative_gain, relevance, None,
        {n: .3 for n in names}, {n: .2 for n in names}, None, (), (), (),
        {} if stopped else {n: VariableResult(LocalClass.PROXIMAL_UNIQUE)
                            for n in names}, frozenset() if stopped else frozenset(promoted),
    )


def edge(source="a", target="y"):
    return CoreEdge(source, target, LocalClass.PROXIMAL_UNIQUE, .6, .3, .4, .2)


def network(**overrides):
    args = dict(root="y", assigned_depth={"y": 0, "a": 1},
                parents={"a": ("y",)}, local_results={"y": local()},
                core_edges=(edge(),), shared_relations=(),
                stop_reasons={"a": StopReason.MAX_DEPTH}, max_depth=1)
    args.update(overrides)
    return NetworkRun(**args)


def test_root_only_and_preflight_stopped_network():
    run = NetworkRun("y", {"y": 0}, {}, {}, (), (),
                     {"y": StopReason.NO_UNASSIGNED}, max_depth=1)
    assert tuple(run.assigned_depth.items()) == (("y", 0),)
    preflight = NetworkRun("y", {"y": 0}, {}, {}, (), (), {}, max_depth=1,
                           stop_reason=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE)
    assert not preflight.local_results
    assert preflight.stop_reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE


def test_edge_retains_signed_scores_and_predictor_to_target_direction():
    run = network()
    assert (run.core_edges[0].source, run.core_edges[0].target) == ("a", "y")
    assert run.core_edges[0].kind is LocalClass.PROXIMAL_UNIQUE
    signed = replace(edge(), broad_raw=-.6, broad_normalized=-.3)
    assert (signed.broad_raw, signed.broad_normalized,
            signed.unique_raw, signed.unique_normalized) == (-.6, -.3, .4, .2)
    with pytest.raises(ValueError):
        network(core_edges=(edge("y", "a"),))
    with pytest.raises(ValueError):
        replace(edge(), kind=LocalClass.SHARED_GROUP_MEMBER)


def test_shared_relation_uses_component_provenance_without_virtual_nodes():
    component = SharedComponent(("a", "b"), True, .8, .4, None)
    relation = SharedRelation("y", ("b", "a"), .8, .4, component)
    result = local(names=("a", "b"), promoted=("a", "b"))
    result = replace(result, shared_components=(component,),
                     classes={n: VariableResult(LocalClass.SHARED_GROUP_MEMBER)
                              for n in ("a", "b")})
    run = network(assigned_depth={"y": 0, "a": 1, "b": 1},
                  parents={"a": ("y",), "b": ("y",)},
                  local_results={"y": result}, core_edges=(),
                  shared_relations=(relation,),
                  stop_reasons={"a": StopReason.MAX_DEPTH, "b": StopReason.MAX_DEPTH})
    assert relation.members == ("a", "b")
    assert set(run.assigned_depth) == {"a", "b", "y"}
    with pytest.raises(ValueError):
        replace(relation, provenance=replace(component, validated=False))
    with pytest.raises(ValueError):
        replace(relation, raw=.9)


def test_retained_pair_provenance_and_duplicate_members():
    pair = PairAblation("a", "b", ((0, 0),), 1, 1, 1., 1., 2., 1., 1.8,
                        1., .8, .4, None)
    retained = RetainedSharedPair("a", "b", ("a", "b"), pair)
    relation = SharedRelation("y", ("b", "a"), .8, .4, retained)
    assert relation.provenance == retained
    with pytest.raises(ValueError):
        replace(relation, members=("a", "a"))
    with pytest.raises(ValueError):
        replace(relation, target="a")


def test_all_same_layer_parents_are_sorted_unique_and_preserved():
    run = network(
        assigned_depth={"y": 0, "b": 1, "a": 1, "c": 2},
        parents={"b": ("y",), "a": ("y",), "c": ("b", "a", "b")},
        local_results={"y": local(names=("a", "b"), promoted=("a", "b")),
                       "b": local("b", ("c",), ("c",)),
                       "a": local("a", ("c",), ("c",))},
        core_edges=(edge("c", "b"), edge("b"), edge("c", "a"), edge()),
        stop_reasons={"c": StopReason.MAX_DEPTH},
        max_depth=2,
    )
    assert run.parents["c"] == ("a", "b")
    assert tuple(run.assigned_depth) == ("a", "b", "c", "y")
    assert tuple((e.source, e.target) for e in run.core_edges) == (
        ("a", "y"), ("b", "y"), ("c", "a"), ("c", "b"))


@pytest.mark.parametrize("depths", [
    {"y": 1, "a": 2}, {"y": 0, "a": -1}, {"y": 0, "a": True},
    {"y": 0, "a": 1.0}, {"y": 0, "a": 0}, {"y": 0, "a": 2},
])
def test_invalid_depths_rejected(depths):
    with pytest.raises(ValueError):
        network(assigned_depth=depths)


@pytest.mark.parametrize("changes", [
    {"parents": {"a": ("missing",)}}, {"parents": {}},
    {"core_edges": (edge(), edge())},
    {"local_results": {"a": local()}},
    {"stop_reasons": {"missing": StopReason.MAX_DEPTH}},
    {"stop_reasons": {"a": "MAX_DEPTH"}},
    {"stop_reasons": {"y": StopReason.MAX_DEPTH}},
    {"stop_reasons": {"a": StopReason.LOW_TOTAL_GAIN}},
    {"stop_reasons": {"y": StopReason.NO_UNASSIGNED}},
    {"stop_reasons": {"a": StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE}},
    {"stop_reason": StopReason.NO_PROMOTION},
    {"stop_reason": StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE},
])
def test_incoherent_graph_or_stop_scope_rejected(changes):
    with pytest.raises(ValueError):
        network(**changes)


@pytest.mark.parametrize("reason, result", [
    (StopReason.LOW_TOTAL_GAIN, local(stopped=True)),
    (StopReason.NO_PROMOTION, local(promoted=())),
])
def test_evaluated_target_stop_reasons(reason, result):
    run = NetworkRun("y", {"y": 0}, {}, {"y": result}, (), (), {"y": reason},
                     max_depth=1)
    assert run.stop_reasons["y"] is reason
    with pytest.raises(ValueError):
        replace(run, stop_reasons={})
    with pytest.raises(ValueError):
        replace(run, stop_reasons={"y": StopReason.NO_PROMOTION
                                   if reason is StopReason.LOW_TOTAL_GAIN
                                   else StopReason.LOW_TOTAL_GAIN})


def test_owned_mappings_are_immutable_and_detached_from_inputs():
    depths = {"y": 0, "a": 1}
    result = local()
    run = network(assigned_depth=depths, local_results={"y": result})
    depths["a"] = 99
    result.relevance.broad["a"] = RelevanceContrast(99., 99.)
    assert run.assigned_depth["a"] == 1
    assert run.local_results["y"].relevance.broad["a"].raw == .6
    with pytest.raises(TypeError):
        run.parents["a"] = ()
    with pytest.raises(TypeError):
        run.local_results["y"].classes["a"] = VariableResult(None)
    with pytest.raises(FrozenInstanceError):
        run.root = "other"


@pytest.mark.parametrize("changes", [
    {"source": ""}, {"source": "y"}, {"broad_raw": float("nan")},
    {"unique_normalized": float("inf")}, {"unique_raw": True},
])
def test_invalid_edge_fields_rejected(changes):
    with pytest.raises(ValueError):
        replace(edge(), **changes)


def test_network_rejects_omitted_proximal_edge():
    with pytest.raises(ValueError):
        network(core_edges=())


def test_network_rejects_missing_shared_relation():
    component = SharedComponent(("a", "b"), True, .8, .4, None)
    result = replace(local(names=("a", "b"), promoted=("a", "b")),
                     shared_components=(component,),
                     classes={n: VariableResult(LocalClass.SHARED_GROUP_MEMBER)
                              for n in ("a", "b")})
    with pytest.raises(ValueError):
        network(assigned_depth={"y": 0, "a": 1, "b": 1},
                parents={"a": ("y",), "b": ("y",)},
                local_results={"y": result}, core_edges=(),
                stop_reasons={"a": StopReason.MAX_DEPTH, "b": StopReason.MAX_DEPTH})


def test_depth_boundary_requires_max_depth_stop():
    run = network()
    assert run.max_depth == 1
    with pytest.raises(ValueError):
        replace(run, stop_reasons={"a": StopReason.NO_UNASSIGNED})


def test_frontier_below_boundary_can_stop_for_empty_snapshot():
    run = network(max_depth=2, stop_reasons={"a": StopReason.NO_UNASSIGNED})
    assert run.stop_reasons["a"] is StopReason.NO_UNASSIGNED
    with pytest.raises(ValueError):
        replace(run, stop_reasons={"a": StopReason.MAX_DEPTH})


@pytest.mark.parametrize("cap", [0, -1, True, 1.0, None])
def test_max_depth_requires_positive_integer(cap):
    with pytest.raises(ValueError):
        network(max_depth=cap)


def test_assigned_depth_cannot_exceed_configured_cap():
    run = network(
        assigned_depth={"y": 0, "a": 1, "b": 2},
        parents={"a": ("y",), "b": ("a",)},
        local_results={"y": local(), "a": local("a", ("b",), ("b",))},
        core_edges=(edge(), edge("b", "a")),
        stop_reasons={"b": StopReason.MAX_DEPTH}, max_depth=2,
    )
    with pytest.raises(ValueError):
        replace(run, max_depth=1)


def test_evaluation_at_depth_boundary_is_rejected():
    with pytest.raises(ValueError):
        network(local_results={"y": local(), "a": local("a", ("b",), (), stopped=True)},
                stop_reasons={"a": StopReason.LOW_TOTAL_GAIN})
