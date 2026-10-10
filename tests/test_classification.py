"""Exact pair-classification rules for normalized OPEN relevance scores."""

import pytest

from open.classify import PairRelation, classify_pairs
from open.config import ThresholdConfig


def classify(a_broad, b_broad, a_unique, b_unique, pair_group):
    return classify_pairs(
        {"a": a_broad, "b": b_broad},
        {"a": a_unique, "b": b_unique},
        {("a", "b"): pair_group},
        ThresholdConfig(broad_relative=.20, unique_relative=.05, group_relative=.10),
    )


def test_shared_group_important_pair_with_both_unique_scores_weak_is_redundant():
    result = classify(.3, .3, .01, .02, .40)

    assert result.pairs[0].relation is PairRelation.PAIR_SHARED_REDUNDANT
    assert result.pairs[0].group_important is True
    assert result.members["a"].proximal_unique is False
    assert result.members["b"].proximal_unique is False


def test_one_strong_member_screens_weak_member_when_its_broad_score_qualifies():
    result = classify(.3, .20, .25, .01, .40)

    assert result.pairs[0].relation is PairRelation.PARTNER_SCREENING
    assert result.members["a"].proximal_unique is True
    assert result.members["b"].screened_by == ("a",)


def test_two_strong_members_remain_proximal_without_redundant_pair_label():
    result = classify(.3, .3, .20, .18, .40)

    assert result.pairs[0].relation is PairRelation.BOTH_PROXIMAL
    assert result.pairs[0].relation is not PairRelation.PAIR_SHARED_REDUNDANT
    assert result.members["a"].proximal_unique is True
    assert result.members["b"].proximal_unique is True


def test_group_weak_pair_does_not_create_redundancy_or_partner_screening():
    result = classify(.9, .9, .01, .02, .099)

    assert result.pairs[0].relation is PairRelation.NOT_GROUP_IMPORTANT
    assert result.members["a"].screened_by == ()
    assert result.members["b"].screened_by == ()


def test_weak_member_below_broad_threshold_is_not_partner_screened():
    result = classify(.3, .199, .25, .01, .40)

    assert result.pairs[0].relation is PairRelation.GROUP_IMPORTANT
    assert result.members["a"].proximal_unique is True
    assert result.members["b"].screened_by == ()


def test_undefined_pair_group_score_is_preserved_and_cannot_be_important():
    result = classify(.3, .3, .01, .02, None)

    assert result.pairs[0].group_score is None
    assert result.pairs[0].group_important is None
    assert result.pairs[0].relation is PairRelation.UNDEFINED


def test_exact_threshold_equality_is_inclusive_for_all_three_scores():
    result = classify(.20, .20, .05, .01, .10)

    assert result.pairs[0].group_important is True
    assert result.pairs[0].relation is PairRelation.PARTNER_SCREENING
    assert result.members["a"].proximal_unique is True
    assert result.members["b"].screened_by == ("a",)


def test_undefined_target_relevance_is_not_interpreted_as_weak():
    result = classify(.9, .9, None, .01, .40)

    assert result.members["a"].proximal_unique is None
    assert result.pairs[0].relation is PairRelation.UNDETERMINED
    assert result.pairs[0].group_important is True
    assert result.members["b"].screened_by == ()


@pytest.mark.parametrize(
    "broad,unique,pair_group",
    [
        ({"a": .1}, {"a": .1, "b": .1}, {("a", "b"): .4}),
        ({"a": .1, "b": .1}, {"a": .1, "b": .1}, {("a", "missing"): .4}),
        ({"a": .1, "b": .1}, {"a": .1, "b": .1}, {("a", "a"): .4}),
        ({"a": .1, "b": .1}, {"a": .1, "b": .1}, {("b", "a"): .4, ("a", "b"): .4}),
    ],
)
def test_rejects_malformed_score_maps(broad, unique, pair_group):
    with pytest.raises(ValueError):
        classify_pairs(broad, unique, pair_group, ThresholdConfig())


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_rejects_nonfinite_or_boolean_scores(score):
    with pytest.raises(ValueError):
        classify(score, .1, .1, .01, .4)
