"""Final per-candidate classification and promotion contract."""

from dataclasses import replace

import pytest

from open.classify import (
    ClassificationResult,
    MemberClassification,
    PairClassification,
    PairRelation,
)
from open.config import ThresholdConfig
from open.grouping import (
    DependenceComponent,
    DependenceEdge,
    GroupingResult,
    PairAblation,
    PairAblationResult,
    RetainedSharedPair,
    SharedComponent,
)
from open.local_target import LocalClass, finalize_local_target
from open.relevance import RelevanceContrast, RelevanceResult


def _relevance(names, *, broad=None, unique=None, stopped=False):
    broad = broad or {}
    unique = unique or {}
    return RelevanceResult(
        total_gain=0.0 if stopped else 1.0,
        relative_gain=0.0 if stopped else .5,
        stop_reason="LOW_TOTAL_GAIN" if stopped else None,
        broad={
            name: RelevanceContrast(2 * broad.get(name, .01), broad.get(name, .01))
            for name in names
        },
        unique={
            name: RelevanceContrast(3 * unique.get(name, .01), unique.get(name, .01))
            for name in names
        },
        baseline_risk=1.0 if stopped else 2.0,
        full_risk=1.0,
    )


def _classifications(names, *, proximal=(), partner_screens=(), pairs=()):
    partner_map = {name: tuple(sorted(by)) for name, by in partner_screens}
    members = {
        name: MemberClassification(
            proximal_unique=(name in proximal),
            screened_by=partner_map.get(name, ()),
        )
        for name in names
    }
    return ClassificationResult(members, tuple(pairs))


def _grouping(*, shared=(), retained=(), dependence=()):
    return GroupingResult(tuple(shared), tuple(retained), tuple(dependence))


def _pair(a, b, relation=PairRelation.PAIR_SHARED_REDUNDANT, *, screened=None, by=None,
          score=.4):
    return PairClassification(
        a, b, score, score >= .2, relation,
        screened_member=screened, screened_by=by,
    )


def _shared(members, validated=True):
    return SharedComponent(tuple(members), validated, .4, .4, None)


def _retained(a, b, score=.4):
    pair = PairAblation(
        a, b, ((0, 0),), 1, 1, 1.0, 1.0, 2.0, 1.0, 1.0 + score,
        1.0, score, score, None,
    )
    return RetainedSharedPair(a, b, (a, b), pair)


def _edge(a, b):
    a, b = sorted((a, b))
    return DependenceEdge(a, b, ((0, 0),), 1, 1, 1.0, 1.0)


def _component(members, *, strong=(), screened=(), negligible=(), edges=()):
    return DependenceComponent(
        tuple(members), tuple(edges), tuple(strong), tuple(screened), tuple(negligible)
    )


def _run(names, relevance, classifications, grouping, thresholds=None,
         outer_model_fit_count=0, tuning_model_fit_count=0):
    return finalize_local_target(
        target="y", candidates=tuple(names), relevance=relevance,
        classifications=classifications, grouping=grouping,
        pair_ablations=(PairAblationResult(
            (), tuple(PairAblation(
                pair.a, pair.b, ((0, 0),), 1, 1, 1.0, 1.0, 2.0, 1.0,
                1.0 + (pair.group_score or 0), 1.0, pair.group_score,
                pair.group_score, None,
            ) for pair in classifications.pairs),
        ) if relevance.stop_reason is None else None),
        thresholds=thresholds or ThresholdConfig(
            min_total_gain_relative=.1, broad_relative=.1,
            unique_relative=.1, group_relative=.2,
        ),
        outer_model_fit_count=outer_model_fit_count,
        tuning_model_fit_count=tuning_model_fit_count,
    )


def test_direct_unique_predictor_is_the_only_promoted_candidate():
    names = ("signal", "noise")
    result = _run(
        names,
        _relevance(names, broad={"signal": .7, "noise": .01},
                   unique={"signal": .5, "noise": .01}),
        _classifications(names, proximal=("signal",)), _grouping(),
        outer_model_fit_count=7, tuning_model_fit_count=14,
    )

    assert result.classes["signal"].kind is LocalClass.PROXIMAL_UNIQUE
    assert result.classes["noise"].kind is LocalClass.NEGLIGIBLE
    assert result.promoted == frozenset({"signal"})
    assert result.relevance.broad["signal"].raw == pytest.approx(1.4)
    assert result.relevance.unique["signal"].raw == pytest.approx(1.5)
    assert (result.outer_model_fit_count, result.tuning_model_fit_count) == (7, 14)


def test_validated_shared_substitutes_are_classified_and_promoted():
    names = ("a", "b")
    result = _run(
        names, _relevance(names), _classifications(names, pairs=(_pair("a", "b"),)),
        _grouping(shared=(_shared(names),), dependence=(
            _component(names, edges=(_edge("a", "b"),)),
        )),
    )

    assert all(result.classes[name].kind is LocalClass.SHARED_GROUP_MEMBER for name in names)
    assert all(result.classes[name].shared_groups == (names,) for name in names)
    assert result.promoted == frozenset(names)


def test_failed_merged_group_retains_overlapping_supported_pair_relations():
    names = ("a", "b", "c")
    pairs = (_pair("a", "b"), _pair("a", "c"))
    grouping = _grouping(
        shared=(_shared(names, validated=False),),
        retained=(_retained("a", "b"), _retained("a", "c")),
        dependence=(_component(names, edges=(_edge("a", "b"), _edge("a", "c"))),),
    )
    result = _run(names, _relevance(names), _classifications(names, pairs=pairs), grouping)

    assert result.classes["a"].shared_groups == (("a", "b"), ("a", "c"))
    assert result.classes["b"].shared_groups == (("a", "b"),)
    assert result.classes["c"].shared_groups == (("a", "c"),)
    assert result.promoted == frozenset(names)


def test_incomplete_retained_pairs_cannot_drop_supported_member_promotion():
    names = ("a", "b", "c")
    result_classifications = _classifications(
        names, pairs=(_pair("a", "b"), _pair("a", "c")),
    )
    incomplete = _grouping(
        shared=(_shared(names, validated=False),),
        retained=(_retained("a", "b"),),
        dependence=(_component(names, edges=(_edge("a", "b"), _edge("a", "c"))),),
    )

    with pytest.raises(ValueError, match="every supported pair edge"):
        _run(names, _relevance(names), result_classifications, incomplete)


def test_partner_screen_preserves_partner_identity_and_does_not_promote():
    names = ("strong", "weak")
    result = _run(
        names,
        _relevance(names, broad={"strong": .8, "weak": .3},
                   unique={"strong": .4, "weak": .01}),
        _classifications(
            names, proximal=("strong",), partner_screens=(("weak", ("strong",)),),
            pairs=(_pair("strong", "weak", PairRelation.PARTNER_SCREENING,
                         screened="weak", by="strong"),),
        ), _grouping(dependence=(_component(
            names, strong=("strong",), screened=(("weak", ("strong",)),),
            edges=(_edge("strong", "weak"),),
        ),)),
    )

    assert result.classes["weak"].kind is LocalClass.SCREENED_BY_PARTNER
    assert result.classes["weak"].screened_by_partner == ("strong",)
    assert result.promoted == frozenset({"strong"})
    assert result.pair_ablations.pairs[0].raw == pytest.approx(.4)


def test_three_node_root_chain_keeps_distal_partner_screen_out_of_shared_groups():
    candidates = ("depression", "stress")
    result = finalize_local_target(
        target="anxiety",
        candidates=candidates,
        relevance=_relevance(
            candidates,
            broad={"depression": .8, "stress": .3},
            unique={"depression": .4, "stress": .01},
        ),
        classifications=_classifications(
            candidates,
            proximal=("depression",),
            partner_screens=(("stress", ("depression",)),),
            pairs=(_pair(
                "depression", "stress", PairRelation.PARTNER_SCREENING,
                screened="stress", by="depression",
            ),),
        ),
        grouping=_grouping(dependence=(
            _component(
                candidates,
                strong=("depression",),
                screened=(("stress", ("depression",)),),
                edges=(_edge("depression", "stress"),),
            ),
        )),
        pair_ablations=PairAblationResult(
            (), (_retained("depression", "stress").pair,),
        ),
        thresholds=ThresholdConfig(
            min_total_gain_relative=.1, broad_relative=.1,
            unique_relative=.1, group_relative=.2,
        ),
    )

    assert result.classes["depression"].kind is LocalClass.PROXIMAL_UNIQUE
    assert result.classes["stress"].kind is LocalClass.SCREENED_BY_PARTNER
    assert result.classes["stress"].screened_by_partner == ("depression",)
    assert result.classes["stress"].shared_groups == ()
    assert result.shared_components == ()
    assert result.retained_pairs == ()
    assert result.pair_classes.pairs[0].relation is PairRelation.PARTNER_SCREENING


def test_mixed_strong_weak_set_screening_is_contextual_and_not_promoted():
    names = ("strong", "weak")
    set_component = _component(
        names, strong=("strong",), screened=(("weak", ("strong",)),),
        edges=(_edge("strong", "weak"),),
    )
    result = _run(
        names, _relevance(names, broad={"strong": .6, "weak": .2},
                          unique={"strong": .4, "weak": .01}),
        _classifications(names, proximal=("strong",),
                         pairs=(_pair("strong", "weak", PairRelation.NOT_GROUP_IMPORTANT,
                                      score=.1),)),
        _grouping(dependence=(set_component,)),
    )

    assert result.classes["strong"].kind is LocalClass.PROXIMAL_UNIQUE
    assert result.classes["weak"].kind is LocalClass.SCREENED_BY_SET
    assert result.classes["weak"].screened_by_set == ("strong",)
    assert result.classes["weak"].dependence_components == (names,)
    assert result.promoted == frozenset({"strong"})


def test_dependence_edge_provenance_must_match_pair_ablation():
    names = ("a", "b")
    classifications = _classifications(
        names, pairs=(_pair("a", "b", PairRelation.NOT_GROUP_IMPORTANT, score=.1),),
    )
    edge = replace(_edge("a", "b"), selected_folds=((0, 1),))
    grouping = _grouping(dependence=(
        _component(names, edges=(edge,)),
    ))

    with pytest.raises(ValueError, match="pair-ablation provenance"):
        _run(names, _relevance(names), classifications, grouping)


def test_partner_screen_takes_precedence_over_set_screen_and_retains_both_provenances():
    names = ("strong", "weak", "other")
    relation = _pair("strong", "weak", PairRelation.PARTNER_SCREENING,
                     screened="weak", by="strong")
    component = _component(
        names, strong=("strong",), screened=(("weak", ("strong",)),),
        negligible=("other",),
        edges=(_edge("strong", "weak"), _edge("weak", "other")),
    )
    result = _run(
        names,
        _relevance(names, broad={"strong": .7, "weak": .3, "other": .01},
                   unique={"strong": .4, "weak": .01, "other": .01}),
        _classifications(names, proximal=("strong",),
                         partner_screens=(("weak", ("strong",)),),
                         pairs=(relation, _pair("other", "weak", PairRelation.NOT_GROUP_IMPORTANT,
                                                score=.1))),
        _grouping(dependence=(component,)),
    )

    assert result.classes["weak"].kind is LocalClass.SCREENED_BY_PARTNER
    assert result.classes["weak"].screened_by_partner == ("strong",)
    assert result.classes["weak"].screened_by_set == ("strong",)
    assert result.promoted == frozenset({"strong"})


def test_low_total_gain_stops_without_classes_or_promotions():
    names = ("a", "b")
    result = _run(
        names, _relevance(names, stopped=True), None, None,
        outer_model_fit_count=3, tuning_model_fit_count=8,
    )

    assert result.stop is True
    assert result.reason == "LOW_TOTAL_GAIN"
    assert result.classes == {}
    assert result.promoted == frozenset()
    assert result.pair_classes is None
    assert result.pair_ablations is None
    assert (result.outer_model_fit_count, result.tuning_model_fit_count) == (3, 8)


@pytest.mark.parametrize("outer,tuning", [(-1, 0), (0, -1), (1.0, 0), (0, True)])
def test_model_fit_counts_reject_invalid_values(outer, tuning):
    names = ("a",)
    with pytest.raises(ValueError, match="nonnegative integers"):
        _run(names, _relevance(names, stopped=True), None, None,
             outer_model_fit_count=outer, tuning_model_fit_count=tuning)


def test_low_total_gain_rejects_supplied_phase_two_artifacts():
    names = ("a",)
    with pytest.raises(ValueError, match="must not receive P2 artifacts"):
        finalize_local_target(
            target="y", candidates=names, relevance=_relevance(names, stopped=True),
            classifications=_classifications(names), grouping=_grouping(),
            pair_ablations=PairAblationResult((), ()),
            thresholds=ThresholdConfig(),
        )


def test_weak_nonqualifying_broad_score_is_negligible():
    names = ("weak",)
    result = _run(
        names, _relevance(names, broad={"weak": .099}, unique={"weak": .01}),
        _classifications(names), _grouping(),
    )

    assert result.classes["weak"].kind is LocalClass.NEGLIGIBLE
    assert result.promoted == frozenset()


def test_undefined_relevance_does_not_fabricate_class_or_promotion():
    names = ("unknown",)
    relevance = replace(
        _relevance(names),
        broad={"unknown": RelevanceContrast(0.0, None)},
        unique={"unknown": RelevanceContrast(0.0, None)},
    )
    result = _run(
        names, relevance,
        ClassificationResult({"unknown": MemberClassification(None)}, ()), _grouping(),
    )

    assert result.classes["unknown"].kind is None
    assert result.promoted == frozenset()


def test_candidate_mismatch_in_phase_one_artifact_is_rejected():
    with pytest.raises(ValueError, match="candidate"):
        _run(
            ("a",), _relevance(("a", "b")), _classifications(("a",)), _grouping(),
        )
