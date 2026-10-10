from dataclasses import replace
import pytest

from open.layer_consensus import aggregate_target_assessments
from open.local_target import LocalClass, VariableResult
from layer_helpers import make_local, make_assessment, make_failed_assessment


def aggregate(locals_, p=.7):
    records = tuple(make_assessment(local, index=i) for i, local in enumerate(locals_))
    return aggregate_target_assessments(records, planned_count=len(records), pi_min=p)


def test_exact_threshold_qualifies_with_all_ten_opportunities():
    records = tuple(make_assessment(make_local("y", ("x",), unique=("x",) if b < 7 else ()), index=b)
                    for b in range(10))
    evidence = aggregate_target_assessments(records, planned_count=10, pi_min=.7)
    edge = next(r for r in evidence.relations if r.relation == "PROXIMAL_UNIQUE")
    assert edge.frequency.probability == .7 and edge.qualifies
    with pytest.raises(ValueError):
        aggregate_target_assessments(records[:-1], planned_count=10, pi_min=.7)
    evidence = aggregate([make_local("y", ("x",), unique=("x",) if b < 6 else ()) for b in range(10)])
    assert not evidence.relations[0].qualifies


def test_zero_threshold_does_not_invent_an_unobserved_edge():
    evidence = aggregate([make_local("y", ("x",))], p=0.)
    assert not evidence.relations
    assert evidence.candidates_evidence[0].promotion.selected == 0


def test_failed_assessment_preserves_planned_count_without_final_probability():
    records = (make_assessment(make_local("y", ("x",), unique=("x",))),
               make_failed_assessment("y", ("x",), index=1))
    evidence = aggregate_target_assessments(records, planned_count=2, pi_min=.7)
    edge = evidence.relations[0]
    assert not evidence.complete and not edge.qualifies
    assert edge.frequency.probability is None
    assert edge.frequency.provisional_probability == 1.


def test_low_gain_counts_as_a_completed_nonselection_and_preserves_scores():
    evidence = aggregate([make_local("y", ("x",), unique=("x",)),
                          make_local("y", ("x",), stop=True)])
    edge = evidence.relations[0]
    assert evidence.complete and edge.frequency.probability == .5
    assert edge.strength.raw_count == 2 and edge.strength.gain_relative_count == 1
    assert evidence.candidates_evidence[0].classes["LOW_TOTAL_GAIN"].selected == 1


def test_class_switching_does_not_turn_promotion_into_a_unique_arrow():
    locals_ = [make_local("y", ("a","b"), unique=("a",)) for _ in range(4)]
    locals_ += [make_local("y", ("a","b"), shared=(("a","b"),)) for _ in range(4)]
    locals_ += [make_local("y", ("a","b")) for _ in range(2)]
    evidence = aggregate(locals_)
    a = next(c for c in evidence.candidates_evidence if c.source == "a")
    assert a.promotion.probability == .8
    assert not any(r.qualifies for r in evidence.relations)


def test_changing_shared_sets_does_not_invent_a_larger_group():
    evidence = aggregate([make_local("y", ("a","b","c"), shared=(group,))
                          for group in (("a","b"), ("b","c")) * 5])
    assert {r.members for r in evidence.relations} == {("a","b"),("b","c")}
    assert not any(r.qualifies for r in evidence.relations)
    assert evidence.shared_comembership[("a","b")].selected == 5


def test_duplicate_component_and_pair_are_one_joint_selection_and_score():
    from test_aggregation import _shared_group
    pair = _shared_group(("a","b"), retained=True).shared_relations[0].provenance
    local = replace(make_local("y", ("a","b"), shared=(("a","b"),)), retained_pairs=(pair,))
    evidence = aggregate([local])
    group = evidence.relations[0]
    assert group.members == ("a","b") and group.frequency.selected == 1
    assert group.strength.raw_count == 1 and group.strength.raw_median == .3
    assert group.strength.scope == "selected joint ablations"
    assert evidence.shared_comembership[("a","b")].selected == 1


def test_screening_identities_and_targets_do_not_pool():
    first = make_local("y", ("a","b","c"), screened={"a":("b",)})
    second = make_local("y", ("a","b","c"), screened={"a":("c",)})
    evidence = aggregate([first, second])
    assert {r.screened_by for r in evidence.relations} == {("b",),("c",)}
    assert all(r.frequency.selected == 1 and not r.qualifies for r in evidence.relations)
    records = (make_assessment(first), make_assessment(replace(second, target="z"), index=1))
    with pytest.raises(ValueError):
        aggregate_target_assessments(records, planned_count=2, pi_min=.7)


def test_strength_medians_use_per_assessment_ratios_and_keep_negative_values():
    evidence = aggregate([make_local("y", ("x",), baseline=b, raw_unique={"x":u})
                          for b,u in ((1.,-.1),(2.,.2),(10.,.5))])
    strength = evidence.candidates_evidence[0].unique
    assert strength.raw_median == .2
    assert strength.baseline_relative_median == .05
    assert strength.raw_count == strength.baseline_relative_count == strength.gain_relative_count == 3


def test_zero_baseline_and_missing_gain_have_separate_availability_counts():
    evidence = aggregate([make_local("y", ("x",), baseline=0., stop=True),
                          make_local("y", ("x",), stop=True)])
    strength = evidence.candidates_evidence[0].unique
    assert (strength.raw_count, strength.baseline_relative_count, strength.gain_relative_count) == (2,1,0)
    assert strength.gain_relative_median is None


@pytest.mark.parametrize("p", [True, float("nan"), -.1, 1.1])
def test_threshold_validation(p):
    with pytest.raises(ValueError):
        aggregate([make_local("y", ("x",))], p=p)
