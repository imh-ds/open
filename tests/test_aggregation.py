"""Exact-count checks for unconditional Phase 4 structural aggregation."""

from dataclasses import replace

import pytest

from open.aggregation import Frequency, aggregate_stability
from open.graph import CoreEdge, NetworkRun, SharedRelation, StopReason
from open.grouping import PairAblation, RetainedSharedPair, SharedComponent
from open.local_target import LocalClass, LocalResult, VariableResult
from open.relevance import RelevanceContrast, RelevanceResult
from open.stability_types import StabilityBatch, StabilityRunRecord


def _local(target, names, *, kinds=None, broad_raw=None, unique_raw=None,
           partner_screens=None, set_screens=None, shared=()):
    kinds = kinds or {}
    broad_raw = broad_raw or {}
    unique_raw = unique_raw or {}
    relevance = RelevanceResult(
        1.0, .5, None,
        {name: RelevanceContrast(broad_raw.get(name, 1.0), .2)
         for name in names},
        {name: RelevanceContrast(unique_raw.get(name, 2.0), .4)
         for name in names},
        2.0, 1.0,
    )
    classes = {
        name: VariableResult(
            kinds.get(name),
            screened_by_partner=tuple((partner_screens or {}).get(name, ())),
            screened_by_set=tuple((set_screens or {}).get(name, ())),
            shared_groups=tuple(group for group in shared if name in group),
        ) for name in names
    }
    promoted = frozenset(
        name for name, variable in classes.items()
        if variable.kind in (LocalClass.PROXIMAL_UNIQUE, LocalClass.SHARED_GROUP_MEMBER)
    )
    return LocalResult(
        False, None, target, 1., .5, relevance, None,
        {name: .2 for name in names}, {name: .4 for name in names}, None,
        (), (), (), classes, promoted,
    )


def _chain(*, broad=1., unique=2.):
    root_local = _local("y", ("a",), kinds={"a": LocalClass.PROXIMAL_UNIQUE},
                        broad_raw={"a": broad}, unique_raw={"a": unique})
    child_local = _local("a", ("b",), kinds={"b": LocalClass.PROXIMAL_UNIQUE})
    return NetworkRun(
        "y", {"y": 0, "a": 1, "b": 2}, {"a": ("y",), "b": ("a",)},
        {"y": root_local, "a": child_local},
        (CoreEdge("a", "y", LocalClass.PROXIMAL_UNIQUE, broad, .2,
                  unique, .4),
         CoreEdge("b", "a", LocalClass.PROXIMAL_UNIQUE, 1., .2, 2., .4)),
        (), {"b": StopReason.MAX_DEPTH}, 2,
    )


def _preflight():
    return NetworkRun("y", {"y": 0}, {}, {}, (), (), {}, 2,
                      StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE)


def _shared_group(members, target="y", *, retained=False, raw=.75, normalized=.3):
    members = tuple(sorted(members))
    if retained:
        assert len(members) == 2
        evidence = PairAblation(
            members[0], members[1], ((0, 0),), 1, 1, 1., 1.,
            2., 1., 1.8, 1., raw, normalized, None,
        )
        provenance = RetainedSharedPair(
            members[0], members[1], members, evidence
        )
    else:
        provenance = SharedComponent(members, True, raw, normalized, None)
    relation = SharedRelation(target, tuple(reversed(members)), raw,
                              provenance.pair.normalized if retained else normalized,
                              provenance)
    names = tuple(members)
    local = _local(
        target, names,
        kinds={name: LocalClass.SHARED_GROUP_MEMBER for name in names},
        shared=(tuple(sorted(names)),),
    )
    local = replace(
        local,
        shared_components=() if retained else (provenance,),
        retained_pairs=(provenance,) if retained else (),
    )
    return NetworkRun(
        target, {target: 0, **{name: 1 for name in names}},
        {name: (target,) for name in names}, {target: local}, (), (relation,),
        {name: StopReason.MAX_DEPTH for name in names}, 1,
    )


def _overlapping_shared_relations():
    first = SharedComponent(("a", "b", "c"), True, .75, .3, None)
    second = SharedComponent(("a", "b", "d"), True, .65, .25, None)
    groups = (first.members, second.members)
    names = ("a", "b", "c", "d")
    local = _local(
        "y", names,
        kinds={name: LocalClass.SHARED_GROUP_MEMBER for name in names},
        shared=groups,
    )
    local = replace(local, shared_components=(first, second))
    relations = (
        SharedRelation("y", first.members, first.raw, first.normalized, first),
        SharedRelation("y", second.members, second.raw, second.normalized, second),
    )
    return NetworkRun(
        "y",
        {"y": 0, **{name: 1 for name in names}},
        {name: ("y",) for name in names},
        {"y": local},
        (),
        relations,
        {name: StopReason.MAX_DEPTH for name in names},
        1,
    )


def _record(index, network=None, *, failed=False):
    if failed:
        return StabilityRunRecord(index, 20 + index, (), failure_type="RuntimeError",
                                  failure_message="injected")
    return StabilityRunRecord(index, 20 + index, (), network=network)


def _preflight_record(index):
    return StabilityRunRecord(
        index, 20 + index, (), network=_preflight(),
        failure_type="PreflightStop",
        failure_message=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE.value,
    )


def test_common_denominator_counts_only_completed_networks_and_tracks_target_states():
    batch = StabilityBatch(4, (
        _record(0, _chain()),
        _record(1, failed=True),
        _preflight_record(2),
        _record(3, _chain(broad=3., unique=4.)),
    ))
    summary = aggregate_stability(batch)

    assert (summary.requested_runs, summary.attempted_runs,
            summary.completed_runs, summary.failed_runs, summary.denominator) == (4, 4, 2, 2, 2)
    targets = {item.target: item for item in summary.targets}
    assert (targets["y"].reached.numerator, targets["y"].evaluated.numerator) == (2, 2)
    assert (targets["a"].reached.numerator, targets["a"].evaluated.numerator) == (2, 2)
    assert targets["y"].reached.probability == 1.
    assert targets["y"].evaluated.probability == 1.

    nodes = {item.name: item for item in summary.nodes}
    assert nodes["a"].selection == type(nodes["a"].selection)(2, 2)
    assert [(item.depth, item.frequency.numerator, item.frequency.denominator)
            for item in nodes["a"].depth_frequencies] == [(1, 2, 2)]

    candidate = next(item for item in summary.candidates
                     if (item.target, item.candidate) == ("y", "a"))
    assert candidate.target_evaluated == type(candidate.target_evaluated)(2, 2)
    assert candidate.broad.raw_count == candidate.unique.raw_count == 2
    assert candidate.broad.raw_mean == 2.
    assert candidate.unique.raw_median == 3.
    assert candidate.broad.normalized_count == 2
    assert candidate.broad.normalized_mean == pytest.approx(.2)
    assert [(item.relation, item.frequency.numerator, item.frequency.denominator)
            for item in candidate.class_frequencies] == [(LocalClass.PROXIMAL_UNIQUE, 2, 2)]

    paths = {item.nodes: item.frequency for item in summary.paths}
    assert paths[("b", "a", "y")] == type(candidate.target_evaluated)(2, 2)


def test_shared_pair_uses_target_canonical_members_and_no_group_identity():
    # The same pair occurs in two differently sized component records. Their
    # identity is the target and member pair, not an exact group or run ID.
    batch = StabilityBatch(3, (
        _record(0, _shared_group(("a", "b", "c"))),
        _record(1, _shared_group(("b", "a"))),
        _record(2, _shared_group(("a", "c"))),
    ))
    summary = aggregate_stability(batch)
    shared = {(row.target, row.member_a, row.member_b): row.frequency
              for row in summary.shared_pairs}

    assert shared[("y", "a", "b")] == type(shared[("y", "a", "b")])(2, 3)
    assert shared[("y", "a", "c")].numerator == 2
    assert shared[("y", "b", "c")].numerator == 1
    assert all(value.denominator == 3 for value in shared.values())


def test_pairwise_shared_comembership_counts_each_completed_run_once():
    # Overlapping groups can contain the same pair in one run. Co-membership
    # is a per-run indicator, not a count of the number of shared relations.
    summary = aggregate_stability(StabilityBatch(1, (
        _record(0, _overlapping_shared_relations()),
    )))
    shared = {(row.target, row.member_a, row.member_b): row.frequency
              for row in summary.shared_pairs}

    assert shared[("y", "a", "b")] == Frequency(1, 1)


def test_same_pair_at_different_targets_remains_distinct():
    summary = aggregate_stability(StabilityBatch(2, (
        _record(0, _shared_group(("a", "b"), target="y")),
        _record(1, _shared_group(("a", "b"), target="z")),
    )))
    shared = {(row.target, row.member_a, row.member_b): row.frequency
              for row in summary.shared_pairs}
    assert set(shared) == {("y", "a", "b"), ("z", "a", "b")}
    assert shared[("y", "a", "b")].probability == .5
    assert shared[("z", "a", "b")].probability == .5


def test_full_shared_relation_matches_canonical_members_not_provenance_identity():
    # These are equivalent groups represented by different provenance records,
    # analogous to different run-local group numbers for the same members.
    summary = aggregate_stability(StabilityBatch(2, (
        _record(0, _shared_group(("b", "a"))),
        _record(1, _shared_group(("a", "b"), retained=True)),
    )))
    assert len(summary.shared_relations) == 1
    relation = summary.shared_relations[0]
    assert (relation.target, relation.members) == ("y", ("a", "b"))
    assert (relation.frequency.numerator, relation.frequency.denominator,
            relation.frequency.probability) == (2, 2, 1.)
    assert relation.magnitude.raw_count == 2
    assert relation.magnitude.raw_mean == .75
    assert relation.magnitude.normalized_mean == .3


def test_shared_relation_magnitude_is_conditional_but_stability_denominator_is_unconditional():
    summary = aggregate_stability(StabilityBatch(3, (
        _record(0, _shared_group(("a", "b"), raw=.5, normalized=.2)),
        _record(1, _shared_group(("b", "a"), raw=1.5, normalized=None)),
        _record(2, _chain()),
    )))
    relation = summary.shared_relations[0]

    assert relation.frequency == type(relation.frequency)(2, 3)
    assert relation.magnitude.raw_count == 2
    assert relation.magnitude.raw_mean == 1.
    assert relation.magnitude.raw_median == 1.
    assert relation.magnitude.normalized_count == 1
    assert relation.magnitude.normalized_mean == .2
    assert relation.magnitude.normalized_median == .2


def test_screened_by_preserves_partner_and_set_relation_types():
    local = _local(
        "y", ("weak", "partner", "set"),
        kinds={"weak": LocalClass.SCREENED_BY_PARTNER,
               "partner": None,
               "set": None},
        partner_screens={"weak": ("partner",)},
        set_screens={"weak": ("set",)},
    )
    network = NetworkRun("y", {"y": 0}, {}, {"y": local}, (), (),
                         {"y": StopReason.NO_PROMOTION}, 1)
    summary = aggregate_stability(StabilityBatch(2, (
        _record(0, network), _preflight_record(1),
    )))
    rows = {(row.screened_member, row.screener, row.relation): row.frequency
            for row in summary.screened_by}
    assert rows[("weak", "partner", LocalClass.SCREENED_BY_PARTNER)].numerator == 1
    assert rows[("weak", "set", LocalClass.SCREENED_BY_SET)].numerator == 1
    assert all(value.denominator == 1 for value in rows.values())


def test_zero_completed_runs_have_a_clear_error():
    batch = StabilityBatch(2, (
        _record(0, failed=True), _preflight_record(1),
    ))
    with pytest.raises(ValueError, match="zero completed runs"):
        aggregate_stability(batch)
