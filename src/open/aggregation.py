"""Unconditional structural summaries across completed stability runs."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import TypeAlias

from open.graph import NetworkRun
from open.local_target import LocalClass
from open.stability_types import StabilityBatch


Relation: TypeAlias = LocalClass | None


@dataclass(frozen=True)
class Frequency:
    """A whole-batch count whose denominator is completed runs."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise ValueError("frequency counts must be integers")
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise ValueError("frequency requires 0 <= numerator <= positive denominator")

    @property
    def probability(self) -> float:
        return self.numerator / self.denominator


@dataclass(frozen=True)
class MagnitudeSummary:
    """Raw and normalized score summaries with their measurement counts."""

    raw_count: int
    raw_mean: float | None
    raw_median: float | None
    normalized_count: int
    normalized_mean: float | None
    normalized_median: float | None


@dataclass(frozen=True)
class NodeDepthFrequency:
    depth: int
    frequency: Frequency


@dataclass(frozen=True)
class NodeSummary:
    name: str
    depth_frequencies: tuple[NodeDepthFrequency, ...]
    selection: Frequency


@dataclass(frozen=True)
class TargetSummary:
    """Whether an analyzed target was reached and locally evaluated."""

    target: str
    reached: Frequency
    evaluated: Frequency


@dataclass(frozen=True)
class CandidateClassFrequency:
    target: str
    candidate: str
    relation: Relation
    frequency: Frequency


@dataclass(frozen=True)
class CandidateSummary:
    """Candidate classification probabilities and observed score magnitudes."""

    target: str
    candidate: str
    class_frequencies: tuple[CandidateClassFrequency, ...]
    target_evaluated: Frequency
    broad: MagnitudeSummary
    unique: MagnitudeSummary


@dataclass(frozen=True)
class SharedPairSummary:
    """Target-specific D-013 co-membership for a canonical member pair."""

    target: str
    member_a: str
    member_b: str
    frequency: Frequency


@dataclass(frozen=True)
class SharedRelationFrequency:
    """Target-specific frequency of a canonical full member tuple."""

    target: str
    members: tuple[str, ...]
    frequency: Frequency
    magnitude: MagnitudeSummary


@dataclass(frozen=True)
class ScreenedBySummary:
    target: str
    screened_member: str
    screener: str
    relation: LocalClass
    frequency: Frequency


@dataclass(frozen=True)
class PathSummary:
    """Empirical occurrence of a complete directed core-edge path."""

    nodes: tuple[str, ...]
    frequency: Frequency


@dataclass(frozen=True)
class StabilitySummary:
    """Typed, immutable P4.2 output consumed by consensus construction."""

    requested_runs: int
    attempted_runs: int
    completed_runs: int
    failed_runs: int
    denominator: int
    nodes: tuple[NodeSummary, ...]
    targets: tuple[TargetSummary, ...]
    candidates: tuple[CandidateSummary, ...]
    shared_relations: tuple[SharedRelationFrequency, ...]
    shared_pairs: tuple[SharedPairSummary, ...]
    screened_by: tuple[ScreenedBySummary, ...]
    paths: tuple[PathSummary, ...]


def aggregate_stability(batch: StabilityBatch) -> StabilitySummary:
    """Aggregate every completed NetworkRun using one common denominator.

    Failed attempts are retained in batch counts and never enter the structural
    denominator. A completed preflight-stopped NetworkRun does enter that
    denominator, with only its root reached and no targets evaluated.
    """
    if not isinstance(batch, StabilityBatch):
        raise ValueError("batch must be a StabilityBatch")
    completed = tuple(run.network for run in batch.runs if run.completed)
    denominator = len(completed)
    if denominator == 0:
        raise ValueError("cannot aggregate a stability batch with zero completed runs")

    depth_counts: dict[str, dict[int, int]] = {}
    target_reached: dict[str, int] = {}
    target_evaluated: dict[str, int] = {}
    class_counts: dict[tuple[str, str, Relation], int] = {}
    broad_values: dict[tuple[str, str], list[float | None]] = {}
    broad_normalized_values: dict[tuple[str, str], list[float | None]] = {}
    unique_values: dict[tuple[str, str], list[float | None]] = {}
    unique_normalized_values: dict[tuple[str, str], list[float | None]] = {}
    shared_counts: dict[tuple[str, str, str], int] = {}
    relation_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    relation_raw: dict[tuple[str, tuple[str, ...]], list[float | None]] = {}
    relation_normalized: dict[tuple[str, tuple[str, ...]], list[float | None]] = {}
    screening_counts: dict[tuple[str, str, str, LocalClass], int] = {}
    path_counts: dict[tuple[str, ...], int] = {}

    for network in completed:
        assert isinstance(network, NetworkRun)
        for name, depth in network.assigned_depth.items():
            if depth > 0:
                depths = depth_counts.setdefault(name, {})
                depths[depth] = depths.get(depth, 0) + 1
        for target in network.assigned_depth:
            target_reached[target] = target_reached.get(target, 0) + 1
        for target, result in network.local_results.items():
            target_evaluated[target] = target_evaluated.get(target, 0) + 1
            candidates = set(result.relevance.broad) | set(result.relevance.unique) | set(result.classes)
            for candidate in candidates:
                key = (target, candidate)
                broad = result.relevance.broad.get(candidate)
                unique = result.relevance.unique.get(candidate)
                broad_values.setdefault(key, []).append(
                    None if broad is None else broad.raw
                )
                unique_values.setdefault(key, []).append(
                    None if unique is None else unique.raw
                )
                broad_norm = result.relevance.broad.get(candidate)
                unique_norm = result.relevance.unique.get(candidate)
                if broad_norm is not None:
                    broad_normalized_values.setdefault(key, []).append(
                        broad_norm.normalized
                    )
                if unique_norm is not None:
                    unique_normalized_values.setdefault(key, []).append(
                        unique_norm.normalized
                    )
                variable = result.classes.get(candidate)
                relation = None if variable is None else variable.kind
                class_key = (target, candidate, relation)
                class_counts[class_key] = class_counts.get(class_key, 0) + 1
            for candidate, variable in result.classes.items():
                for screener in variable.screened_by_partner:
                    key = (target, candidate, screener, LocalClass.SCREENED_BY_PARTNER)
                    screening_counts[key] = screening_counts.get(key, 0) + 1
                for screener in variable.screened_by_set:
                    key = (target, candidate, screener, LocalClass.SCREENED_BY_SET)
                    screening_counts[key] = screening_counts.get(key, 0) + 1

        run_shared_pairs: set[tuple[str, str, str]] = set()
        for relation in network.shared_relations:
            members = tuple(sorted(relation.members))
            relation_key = (relation.target, members)
            relation_counts[relation_key] = relation_counts.get(relation_key, 0) + 1
            relation_raw.setdefault(relation_key, []).append(relation.raw)
            relation_normalized.setdefault(relation_key, []).append(relation.normalized)
            for index, member_a in enumerate(members):
                for member_b in members[index + 1:]:
                    key = (relation.target, member_a, member_b)
                    run_shared_pairs.add(key)

        # Co-membership is an indicator for this run, even when the pair
        # belongs to multiple overlapping validated shared relations.
        for key in run_shared_pairs:
            shared_counts[key] = shared_counts.get(key, 0) + 1

        for path in _core_paths(network):
            path_counts[path] = path_counts.get(path, 0) + 1

    nodes = tuple(
        NodeSummary(
            name=name,
            depth_frequencies=tuple(
                NodeDepthFrequency(depth, Frequency(count, denominator))
                for depth, count in sorted(depths.items())
            ),
            selection=Frequency(sum(depths.values()), denominator),
        )
        for name, depths in sorted(depth_counts.items())
    )
    targets = tuple(
        TargetSummary(
            target,
            Frequency(count, denominator),
            Frequency(target_evaluated.get(target, 0), denominator),
        )
        for target, count in sorted(target_reached.items())
    )

    candidates = []
    for target, candidate in sorted(
        broad_values.keys() | broad_normalized_values.keys()
        | unique_values.keys() | unique_normalized_values.keys()
    ):
        class_frequencies = tuple(
            CandidateClassFrequency(target, candidate, relation,
                                    Frequency(count, denominator))
            for (class_target, class_candidate, relation), count in sorted(
                class_counts.items(), key=lambda item: _relation_sort_key(item[0])
            )
            if class_target == target and class_candidate == candidate
        )
        candidates.append(CandidateSummary(
            target=target,
            candidate=candidate,
            class_frequencies=class_frequencies,
            target_evaluated=Frequency(target_evaluated.get(target, 0), denominator),
            broad=_magnitude(broad_values.get((target, candidate), []),
                             broad_normalized_values.get((target, candidate), [])),
            unique=_magnitude(unique_values.get((target, candidate), []),
                              unique_normalized_values.get((target, candidate), [])),
        ))

    return StabilitySummary(
        requested_runs=batch.requested_runs,
        attempted_runs=batch.attempted_runs,
        completed_runs=batch.completed_runs,
        failed_runs=batch.failed_runs,
        denominator=denominator,
        nodes=nodes,
        targets=targets,
        candidates=tuple(candidates),
        shared_relations=tuple(
            SharedRelationFrequency(
                target, members, Frequency(count, denominator),
                _magnitude(relation_raw[(target, members)],
                           relation_normalized[(target, members)]),
            )
            for (target, members), count in sorted(relation_counts.items())
        ),
        shared_pairs=tuple(
            SharedPairSummary(target, a, b, Frequency(count, denominator))
            for (target, a, b), count in sorted(shared_counts.items())
        ),
        screened_by=tuple(
            ScreenedBySummary(target, member, screener, relation,
                              Frequency(count, denominator))
            for (target, member, screener, relation), count in sorted(
                screening_counts.items(), key=lambda item: (
                    item[0][0], item[0][1], item[0][2], item[0][3].value
                )
            )
        ),
        paths=tuple(
            PathSummary(nodes, Frequency(count, denominator))
            for nodes, count in sorted(path_counts.items())
        ),
    )


def _relation_sort_key(item: tuple[str, str, Relation]) -> tuple[str, str, str]:
    target, candidate, relation = item
    return target, candidate, "" if relation is None else relation.value


def _magnitude(raw: list[float | None], normalized: list[float | None]) -> MagnitudeSummary:
    raw_values = [value for value in raw if value is not None]
    norm_values = [value for value in normalized if value is not None]
    return MagnitudeSummary(
        len(raw_values), mean(raw_values) if raw_values else None,
        median(raw_values) if raw_values else None,
        len(norm_values), mean(norm_values) if norm_values else None,
        median(norm_values) if norm_values else None,
    )


def _core_paths(network: NetworkRun) -> set[tuple[str, ...]]:
    """Return each directed path of two or more core edges present in a run."""
    children: dict[str, list[str]] = {}
    for edge in network.core_edges:
        children.setdefault(edge.source, []).append(edge.target)
    found: set[tuple[str, ...]] = set()

    def extend(path: tuple[str, ...]) -> None:
        for child in sorted(children.get(path[-1], ())):
            if child in path:
                continue
            longer = (*path, child)
            if len(longer) >= 3:
                found.add(longer)
            extend(longer)

    for source in sorted(children):
        extend((source,))
    return found
