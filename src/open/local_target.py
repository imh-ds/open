"""Compose pair and grouping evidence into final one-run local classes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from open.classify import (
    ClassificationResult,
    MemberClassification,
    PairClassification,
    PairRelation,
    classify_pairs,
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
from open.relevance import RelevanceContrast, RelevanceResult


class LocalClass(str, Enum):
    """Final one-run variable classes; stability is assigned in a later phase."""

    PROXIMAL_UNIQUE = "PROXIMAL_UNIQUE"
    SHARED_GROUP_MEMBER = "SHARED_GROUP_MEMBER"
    SCREENED_BY_PARTNER = "SCREENED_BY_PARTNER"
    SCREENED_BY_SET = "SCREENED_BY_SET"
    SCREENED_GENERAL = "SCREENED_GENERAL"
    NEGLIGIBLE = "NEGLIGIBLE"


@dataclass(frozen=True)
class VariableResult:
    """One candidate's class and all applicable one-run provenance."""

    kind: LocalClass | None
    screened_by_partner: tuple[str, ...] = ()
    screened_by_set: tuple[str, ...] = ()
    shared_groups: tuple[tuple[str, ...], ...] = ()
    dependence_components: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class LocalResult:
    """Frozen composition of P1 relevance and P2 classification artifacts."""

    stop: bool
    reason: str | None
    target: str
    total_gain: float
    total_gain_relative: float | None
    relevance: RelevanceResult
    pair_ablations: PairAblationResult | None
    broad: Mapping[str, float | None]
    unique: Mapping[str, float | None]
    pair_classes: ClassificationResult | None
    shared_components: tuple[SharedComponent, ...]
    retained_pairs: tuple[RetainedSharedPair, ...]
    dependence_components: tuple[DependenceComponent, ...]
    classes: Mapping[str, VariableResult]
    promoted: frozenset[str]
    outer_model_fit_count: int = 0
    tuning_model_fit_count: int = 0
    n_observed_target_rows: int | None = None
    n_missing_target_rows: int | None = None
    input_candidates: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (type(self.outer_model_fit_count) is not int or self.outer_model_fit_count < 0
                or type(self.tuning_model_fit_count) is not int or self.tuning_model_fit_count < 0):
            raise ValueError("model fit counts must be nonnegative integers")
        coverage = (self.n_observed_target_rows, self.n_missing_target_rows)
        if coverage != (None, None) and any(type(n) is not int or n < 0 for n in coverage):
            raise ValueError("target row counts must both be nonnegative integers or both None")
        if self.input_candidates is not None:
            _candidate_names(self.target, self.input_candidates)


def finalize_local_target(
    *,
    target: str,
    candidates: tuple[str, ...],
    relevance: RelevanceResult,
    classifications: ClassificationResult | None,
    grouping: GroupingResult | None,
    thresholds: ThresholdConfig,
    pair_ablations: PairAblationResult | None = None,
    outer_model_fit_count: int = 0,
    tuning_model_fit_count: int = 0,
    n_observed_target_rows: int | None = None,
    n_missing_target_rows: int | None = None,
) -> LocalResult:
    """Apply §§16–17 precedence without recalculating normalized relevance."""
    names = _candidate_names(target, candidates)
    if not isinstance(relevance, RelevanceResult):
        raise ValueError("relevance must be a RelevanceResult")
    if not isinstance(thresholds, ThresholdConfig):
        raise ValueError("thresholds must be a ThresholdConfig")
    broad, unique = _relevance_scores(names, relevance, thresholds)

    if relevance.stop_reason is not None:
        if classifications is not None or grouping is not None or pair_ablations is not None:
            raise ValueError("stopped targets must not receive P2 artifacts")
        return LocalResult(
            True, relevance.stop_reason, target, relevance.total_gain,
            relevance.relative_gain, relevance, pair_ablations,
            MappingProxyType(broad), MappingProxyType(unique), classifications,
            (), (), (), MappingProxyType({}), frozenset(),
            outer_model_fit_count, tuning_model_fit_count,
            n_observed_target_rows, n_missing_target_rows, names,
        )

    if classifications is None or grouping is None or pair_ablations is None:
        raise ValueError("eligible targets require P2.1 classifications and P2.2 pair/group results")
    if not isinstance(classifications, ClassificationResult):
        raise ValueError("classifications must be a ClassificationResult")
    if not isinstance(grouping, GroupingResult):
        raise ValueError("grouping must be a GroupingResult")
    if not isinstance(pair_ablations, PairAblationResult):
        raise ValueError("pair_ablations must be a PairAblationResult")
    _validate_pair_classifications(names, broad, unique, classifications, thresholds)
    _validate_pair_ablations(classifications, pair_ablations)
    _validate_grouping(names, broad, unique, classifications, grouping, thresholds,
                       pair_ablations)

    groups_by_name: dict[str, list[tuple[str, ...]]] = {name: [] for name in names}
    for component in grouping.shared_components:
        if component.validated:
            for name in component.members:
                groups_by_name[name].append(component.members)
    for retained in grouping.retained_pairs:
        for name in retained.members:
            groups_by_name[name].append(retained.members)

    set_screens: dict[str, set[str]] = {name: set() for name in names}
    components_by_name: dict[str, list[tuple[str, ...]]] = {name: [] for name in names}
    for component in grouping.dependence_components:
        for name in component.members:
            components_by_name[name].append(component.members)
        for name, strong_members in component.screened_by_set:
            set_screens[name].update(strong_members)

    partner_screens = {
        name: tuple(classifications.members[name].screened_by)
        for name in names
    }
    results: dict[str, VariableResult] = {}
    promoted: set[str] = set()
    for name in names:
        u_j, b_j = unique[name], broad[name]
        is_proximal = u_j is not None and u_j >= thresholds.unique_relative
        groups = tuple(sorted(set(groups_by_name[name])))
        partner_by = partner_screens[name]
        set_by = tuple(sorted(set_screens[name]))
        if u_j is None or b_j is None:
            kind = None
        elif is_proximal:
            kind = LocalClass.PROXIMAL_UNIQUE
        elif groups:
            kind = LocalClass.SHARED_GROUP_MEMBER
        elif partner_by:
            kind = LocalClass.SCREENED_BY_PARTNER
        elif set_by:
            kind = LocalClass.SCREENED_BY_SET
        elif b_j >= thresholds.broad_relative:
            kind = LocalClass.SCREENED_GENERAL
        else:
            kind = LocalClass.NEGLIGIBLE

        results[name] = VariableResult(
            kind=kind,
            screened_by_partner=partner_by,
            screened_by_set=set_by,
            shared_groups=groups,
            dependence_components=tuple(sorted(set(components_by_name[name]))),
        )
        if kind is LocalClass.PROXIMAL_UNIQUE or kind is LocalClass.SHARED_GROUP_MEMBER:
            promoted.add(name)

    return LocalResult(
        False, None, target, relevance.total_gain, relevance.relative_gain,
        relevance, pair_ablations, MappingProxyType(broad), MappingProxyType(unique), classifications,
        grouping.shared_components, grouping.retained_pairs,
        grouping.dependence_components, MappingProxyType(results), frozenset(promoted),
        outer_model_fit_count, tuning_model_fit_count,
        n_observed_target_rows, n_missing_target_rows, names,
    )


def _candidate_names(target: str, candidates: tuple[str, ...]) -> tuple[str, ...]:
    if type(target) is not str or not target.strip():
        raise ValueError("target must be a nonempty variable name")
    if (not isinstance(candidates, tuple) or not candidates
            or any(type(name) is not str or not name.strip() for name in candidates)
            or len(set(candidates)) != len(candidates) or target in candidates):
        raise ValueError("candidates must be distinct nonempty names excluding target")
    return candidates


def _relevance_scores(
    names: tuple[str, ...], relevance: RelevanceResult, thresholds: ThresholdConfig
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    if not math.isfinite(relevance.total_gain):
        raise ValueError("relevance total gain must be finite")
    if (relevance.relative_gain is not None
            and (type(relevance.relative_gain) not in (int, float)
                 or not math.isfinite(relevance.relative_gain))):
        raise ValueError("relevance relative gain must be finite or None")
    if relevance.stop_reason not in (None, "LOW_TOTAL_GAIN"):
        raise ValueError("relevance has an unsupported stop reason")
    if (relevance.stop_reason == "LOW_TOTAL_GAIN"
            and relevance.total_gain > 0 and relevance.relative_gain is not None
            and relevance.relative_gain >= thresholds.min_total_gain_relative):
        raise ValueError("LOW_TOTAL_GAIN conflicts with the supplied gain threshold")
    scores: list[dict[str, float | None]] = []
    for label, contrasts in (("broad", relevance.broad), ("unique", relevance.unique)):
        if not isinstance(contrasts, Mapping) or set(contrasts) != set(names):
            raise ValueError(f"{label} relevance must match candidates exactly")
        current: dict[str, float | None] = {}
        for name in names:
            contrast = contrasts[name]
            if not isinstance(contrast, RelevanceContrast):
                raise ValueError(f"{label} relevance values must be RelevanceContrast")
            for value in (contrast.raw, contrast.normalized):
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value)
                ):
                    raise ValueError(f"{label} relevance values must be finite or None")
            current[name] = contrast.normalized
        scores.append(current)
    if relevance.stop_reason is None:
        if (relevance.total_gain <= 0 or relevance.relative_gain is None
                or relevance.relative_gain < thresholds.min_total_gain_relative):
            raise ValueError("eligible target relevance does not pass the total-gain stop")
    return scores[0], scores[1]


def _validate_pair_classifications(
    names: tuple[str, ...], broad: Mapping[str, float | None],
    unique: Mapping[str, float | None], result: ClassificationResult,
    thresholds: ThresholdConfig,
) -> None:
    if not isinstance(result.members, Mapping) or set(result.members) != set(names):
        raise ValueError("P2.1 member classifications must match candidates exactly")
    if any(not isinstance(value, MemberClassification) for value in result.members.values()):
        raise ValueError("P2.1 member values must be MemberClassification")
    pair_scores: dict[tuple[str, str], float | None] = {}
    for pair in result.pairs:
        if not isinstance(pair, PairClassification):
            raise ValueError("P2.1 pairs must be PairClassification values")
        key = frozenset((pair.a, pair.b))
        if key in {frozenset(existing) for existing in pair_scores}:
            raise ValueError("P2.1 pair classifications cannot repeat a pair")
        pair_scores[(pair.a, pair.b)] = pair.group_score
    try:
        expected = classify_pairs(broad, unique, pair_scores, thresholds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"P2.1 pair classifications are incompatible: {exc}") from exc
    if result != expected:
        raise ValueError("P2.1 classifications do not match relevance scores and thresholds")


def _validate_pair_ablations(
    classifications: ClassificationResult, result: PairAblationResult
) -> None:
    if (not isinstance(result.fold_scores, tuple) or not isinstance(result.pairs, tuple)
            or any(not isinstance(pair, PairAblation) for pair in result.pairs)):
        raise ValueError("P2.2 pairs must be PairAblation values")
    pair_map: dict[frozenset[str], PairAblation] = {}
    for pair in result.pairs:
        key = frozenset((pair.a, pair.b))
        if len(key) != 2 or key in pair_map:
            raise ValueError("P2.2 pair ablations must be unique pairs")
        if (type(pair.raw) not in (int, float) or not math.isfinite(pair.raw)
                or pair.normalized is not None and (
                    type(pair.normalized) not in (int, float)
                    or not math.isfinite(pair.normalized)
                )):
            raise ValueError("P2.2 pair raw/normalized scores must be finite or None")
        pair_map[key] = pair
    classified = {frozenset((pair.a, pair.b)): pair for pair in classifications.pairs}
    if set(pair_map) != set(classified):
        raise ValueError("P2.2 pair ablations must match P2.1 selected pair summaries")
    if any(pair.normalized != classified[key].group_score for key, pair in pair_map.items()):
        raise ValueError("P2.2 normalized pair scores disagree with P2.1 summaries")


def _validate_grouping(
    names: tuple[str, ...], broad: Mapping[str, float | None],
    unique: Mapping[str, float | None],
    classifications: ClassificationResult, grouping: GroupingResult,
    thresholds: ThresholdConfig, pair_ablations: PairAblationResult,
) -> None:
    if (not isinstance(grouping.shared_components, tuple)
            or not isinstance(grouping.retained_pairs, tuple)
            or not isinstance(grouping.dependence_components, tuple)
            or any(not isinstance(component, SharedComponent)
                   for component in grouping.shared_components)
            or any(not isinstance(component, DependenceComponent)
                   for component in grouping.dependence_components)):
        raise ValueError("P2.2 grouping artifacts have invalid result types")
    name_set = set(names)
    relation_by_pair = {
        frozenset((pair.a, pair.b)): pair
        for pair in classifications.pairs
    }
    ablation_by_pair = {
        frozenset((pair.a, pair.b)): pair for pair in pair_ablations.pairs
    }
    shared_seen: set[str] = set()
    covered_shared_pairs: set[frozenset[str]] = set()
    for component in grouping.shared_components:
        members = _component_members(component.members, name_set, "shared")
        if shared_seen.intersection(members):
            raise ValueError("shared components cannot overlap")
        shared_seen.update(members)
        if type(component.validated) is not bool:
            raise ValueError("shared component validation status must be boolean")
        if component.normalized is not None and (
            type(component.normalized) not in (int, float)
            or not math.isfinite(component.normalized)
        ):
            raise ValueError("shared component score must be finite or None")
        component_edges = [
            pair for pair in classifications.pairs
            if pair.relation is PairRelation.PAIR_SHARED_REDUNDANT
            and pair.a in members and pair.b in members
        ]
        if not _connected(members, component_edges):
            raise ValueError("shared component must be connected by supported shared pairs")
        covered_shared_pairs.update(frozenset((pair.a, pair.b)) for pair in component_edges)
        if component.validated and (
            component.normalized is None
            or component.normalized < thresholds.group_relative
            or any(unique[name] is None or unique[name] >= thresholds.unique_relative
                   for name in members)
        ):
            raise ValueError("validated shared component does not satisfy group rules")

    retained_seen: set[frozenset[str]] = set()
    for retained in grouping.retained_pairs:
        if not isinstance(retained, RetainedSharedPair):
            raise ValueError("retained shared pairs must use RetainedSharedPair")
        if not isinstance(retained.pair, PairAblation):
            raise ValueError("retained pair evidence must use PairAblation")
        members = _component_members(retained.members, name_set, "retained pair", size=2)
        key = frozenset(members)
        pair = relation_by_pair.get(key)
        component = next((item for item in grouping.shared_components
                          if set(item.members).issuperset(members)), None)
        if (key in retained_seen or pair is None
                or pair.relation is not PairRelation.PAIR_SHARED_REDUNDANT
            or pair.group_score != retained.pair.normalized
            or ablation_by_pair.get(key) != retained.pair
                or retained.a != members[0] or retained.b != members[1]
                or retained.pair.a != retained.a or retained.pair.b != retained.b
                or component is None or component.validated):
            raise ValueError("retained pair is incompatible with failed shared-component evidence")
        retained_seen.add(key)
    for component in grouping.shared_components:
        component_edges = {
            frozenset((pair.a, pair.b)) for pair in classifications.pairs
            if pair.relation is PairRelation.PAIR_SHARED_REDUNDANT
            and pair.a in component.members and pair.b in component.members
        }
        if component.validated and component_edges - covered_shared_pairs:
            raise ValueError("validated shared component omits a supported pair edge")
        if not component.validated and component_edges != {
            key for key in retained_seen
            if next(iter(key)) in component.members and key.issubset(component.members)
        }:
            raise ValueError("failed shared component must retain every supported pair edge")
    all_shared_pairs = {
        frozenset((pair.a, pair.b)) for pair in classifications.pairs
        if pair.relation is PairRelation.PAIR_SHARED_REDUNDANT
    }
    if all_shared_pairs != covered_shared_pairs:
        raise ValueError("shared pair classifications and grouping components disagree")

    dependence_seen: set[str] = set()
    dependence_edges: set[frozenset[str]] = set()
    for component in grouping.dependence_components:
        members = _component_members(component.members, name_set, "dependence")
        if dependence_seen.intersection(members):
            raise ValueError("dependence components cannot overlap")
        dependence_seen.update(members)
        if (not isinstance(component.edges, tuple) or not component.edges
                or any(not isinstance(edge, DependenceEdge) for edge in component.edges)):
            raise ValueError("dependence component must contain typed edges")
        if any(
            frozenset((edge.a, edge.b)) not in relation_by_pair
            or edge.a not in members or edge.b not in members
            for edge in component.edges
        ):
            raise ValueError("dependence component edges must match classified candidate pairs")
        component_edges = {frozenset((edge.a, edge.b)) for edge in component.edges}
        if len(component_edges) != len(component.edges) or component_edges & dependence_edges:
            raise ValueError("dependence edges must be unique across components")
        dependence_edges.update(component_edges)
        if not _dependence_connected(members, component.edges):
            raise ValueError("dependence component members must be connected by their edges")
        for edge in component.edges:
            pair_ablation = ablation_by_pair.get(frozenset((edge.a, edge.b)))
            if (edge.a >= edge.b or type(edge.selected_folds) is not tuple
                    or not edge.selected_folds
                    or len(set(edge.selected_folds)) != len(edge.selected_folds)
                    or edge.selected_fold_count != len(edge.selected_folds)
                    or edge.evaluated_fold_count != edge.selected_fold_count
                    or not _unit_frequency(edge.selection_frequency)
                    or edge.selection_frequency != edge.evaluation_frequency):
                raise ValueError("dependence edge fold provenance is inconsistent")
            if (pair_ablation is None
                    or edge.selected_folds != pair_ablation.selected_folds
                    or edge.selected_fold_count != pair_ablation.selected_fold_count
                    or edge.evaluated_fold_count != pair_ablation.evaluated_fold_count
                    or edge.selection_frequency != pair_ablation.selection_frequency
                    or edge.evaluation_frequency != pair_ablation.evaluation_frequency):
                raise ValueError("dependence edge disagrees with pair-ablation provenance")
        expected_strong = tuple(
            name for name in members
            if unique[name] is not None and unique[name] >= thresholds.unique_relative
        )
        expected_screens = tuple(
            (name, expected_strong) for name in members
            if expected_strong and name not in expected_strong
            and unique[name] is not None and broad[name] is not None
            and broad[name] >= thresholds.broad_relative
        )
        expected_negligible = tuple(
            name for name in members
            if expected_strong and name not in expected_strong
            and unique[name] is not None and broad[name] is not None
            and broad[name] < thresholds.broad_relative
        )
        if (component.strong_members != expected_strong
                or component.screened_by_set != expected_screens
                or component.negligible_members != expected_negligible):
            raise ValueError("dependence component classification disagrees with relevance")
    classified_pairs = {frozenset((pair.a, pair.b)) for pair in classifications.pairs}
    if dependence_edges != classified_pairs:
        raise ValueError("dependence components must cover every selected candidate pair")

def _component_members(
    members: tuple[str, ...], valid_names: set[str], label: str, *, size: int = 2
) -> tuple[str, ...]:
    if (not isinstance(members, tuple) or len(members) < size
            or any(type(name) is not str or name not in valid_names for name in members)
            or len(set(members)) != len(members)):
        raise ValueError(f"{label} component has invalid candidate members")
    return members


def _connected(members: tuple[str, ...], pairs: list[PairClassification]) -> bool:
    adjacency = {name: set() for name in members}
    for pair in pairs:
        adjacency[pair.a].add(pair.b)
        adjacency[pair.b].add(pair.a)
    seen: set[str] = set()
    pending = [members[0]]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(adjacency[current] - seen)
    return seen == set(members)


def _dependence_connected(members: tuple[str, ...], edges: tuple) -> bool:
    adjacency = {name: set() for name in members}
    for edge in edges:
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)
    seen: set[str] = set()
    pending = [members[0]]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(adjacency[current] - seen)
    return seen == set(members)


def _unit_frequency(value: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1
