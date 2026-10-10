"""Consensus records built from unconditional stability summaries."""

from __future__ import annotations

from dataclasses import dataclass

from open.aggregation import (
    Frequency,
    MagnitudeSummary,
    StabilitySummary,
)
from open.local_target import LocalClass


@dataclass(frozen=True)
class ConsensusNode:
    name: str
    depth: int
    depth_probabilities: tuple[tuple[int, float], ...]
    selection_probability: float
    frequency: Frequency


@dataclass(frozen=True)
class ConsensusClass:
    target: str
    candidate: str
    relation: LocalClass | None
    frequency: Frequency
    target_evaluated: Frequency
    broad: MagnitudeSummary
    unique: MagnitudeSummary


@dataclass(frozen=True)
class ConsensusEdge:
    """One proximal or screening relation; direction is candidate -> target."""

    source: str
    target: str
    relation: LocalClass
    frequency: Frequency
    evaluated_probability: float
    screened_by: tuple[str, ...]
    broad: MagnitudeSummary
    unique: MagnitudeSummary


@dataclass(frozen=True)
class ConsensusSharedRelation:
    target: str
    members: tuple[str, ...]
    frequency: Frequency
    pairwise_comembership: tuple[tuple[tuple[str, str], Frequency], ...]
    target_reached: Frequency
    target_evaluated: Frequency
    magnitude: MagnitudeSummary


@dataclass(frozen=True)
class ConsensusSharedPair:
    target: str
    member_a: str
    member_b: str
    frequency: Frequency


@dataclass(frozen=True)
class ConsensusScreenedBy:
    target: str
    screened_member: str
    screener: str
    relation: LocalClass
    frequency: Frequency


@dataclass(frozen=True)
class ConsensusPath:
    nodes: tuple[str, ...]
    frequency: Frequency


@dataclass(frozen=True)
class ConsensusTarget:
    target: str
    reached: Frequency
    evaluated: Frequency


@dataclass(frozen=True)
class ConsensusOmission:
    structure: str
    identity: tuple[object, ...]
    frequency: Frequency
    reason: str


@dataclass(frozen=True)
class ConsensusResult:
    pi_min: float
    denominator: int
    requested_runs: int
    attempted_runs: int
    completed_runs: int
    failed_runs: int
    nodes: tuple[ConsensusNode, ...]
    targets: tuple[ConsensusTarget, ...]
    classes: tuple[ConsensusClass, ...]
    edges: tuple[ConsensusEdge, ...]
    shared_relations: tuple[ConsensusSharedRelation, ...]
    screened_by: tuple[ConsensusScreenedBy, ...]
    paths: tuple[ConsensusPath, ...]
    omissions: tuple[ConsensusOmission, ...]
    shared_pairs: tuple[ConsensusSharedPair, ...]
    intervals: tuple = ()


def build_consensus(summary: StabilitySummary, pi_min: float = 0.70) -> ConsensusResult:
    """Select structures at or above ``pi_min``; equality is included.

    The P3 local result does not retain row-keyed held-out loss differences,
    so this phase emits no descriptive intervals. Intervals never participate
    in consensus inclusion.
    """
    if not isinstance(summary, StabilitySummary):
        raise TypeError("summary must be a StabilitySummary")
    if type(pi_min) not in (int, float) or not 0 <= pi_min <= 1:
        raise ValueError("pi_min must be in [0, 1]")
    if summary.denominator <= 0:
        raise ValueError("consensus requires at least one completed run")

    omissions: list[ConsensusOmission] = []
    nodes: list[ConsensusNode] = []
    for node in summary.nodes:
        frequency = node.selection
        if frequency.probability < pi_min:
            omissions.append(_omission("node", (node.name,), frequency, pi_min))
            continue
        depth_probs = tuple((item.depth, item.frequency.probability)
                            for item in node.depth_frequencies)
        # max probability, then shallower depth on a tie.
        depth = min(depth_probs, key=lambda item: (-item[1], item[0]))[0]
        nodes.append(ConsensusNode(node.name, depth, depth_probs,
                                   frequency.probability, frequency))

    classes: list[ConsensusClass] = []
    edges: list[ConsensusEdge] = []
    for candidate in summary.candidates:
        for item in candidate.class_frequencies:
            frequency = item.frequency
            if frequency.probability < pi_min:
                omissions.append(_omission(
                    "class", (candidate.target, candidate.candidate, item.relation),
                    frequency, pi_min,
                ))
                continue
            classes.append(ConsensusClass(
                candidate.target, candidate.candidate, item.relation, frequency,
                candidate.target_evaluated, candidate.broad, candidate.unique,
            ))
            if item.relation in (LocalClass.PROXIMAL_UNIQUE,
                                 LocalClass.SCREENED_BY_PARTNER,
                                 LocalClass.SCREENED_BY_SET):
                screeners = _screeners(summary, candidate.target, candidate.candidate,
                                       item.relation, pi_min)
                edges.append(ConsensusEdge(
                    candidate.candidate, candidate.target, item.relation, frequency,
                    candidate.target_evaluated.probability, screeners,
                    candidate.broad, candidate.unique,
                ))

    shared_relations: list[ConsensusSharedRelation] = []
    shared_pairs: list[ConsensusSharedPair] = []
    targets_by_name = {item.target: item for item in summary.targets}
    targets = tuple(ConsensusTarget(item.target, item.reached, item.evaluated)
                    for item in summary.targets)
    for relation in summary.shared_relations:
        frequency = relation.frequency
        if frequency.probability < pi_min:
            omissions.append(_omission("shared_relation",
                                       (relation.target, relation.members), frequency, pi_min))
            continue
        pair_probs = tuple(
            ((pair.member_a, pair.member_b), pair.frequency)
            for pair in summary.shared_pairs
            if pair.target == relation.target
            and pair.member_a in relation.members and pair.member_b in relation.members
        )
        shared_relations.append(ConsensusSharedRelation(
            relation.target, relation.members, frequency, pair_probs,
            targets_by_name[relation.target].reached,
            targets_by_name[relation.target].evaluated,
            relation.magnitude,
        ))
    for pair in summary.shared_pairs:
        if pair.frequency.probability >= pi_min:
            shared_pairs.append(ConsensusSharedPair(
                pair.target, pair.member_a, pair.member_b, pair.frequency,
            ))
        else:
            omissions.append(_omission("shared_pair",
                                       (pair.target, pair.member_a, pair.member_b),
                                       pair.frequency, pi_min))

    screened: list[ConsensusScreenedBy] = []
    for item in summary.screened_by:
        if item.frequency.probability >= pi_min:
            screened.append(ConsensusScreenedBy(
                item.target, item.screened_member, item.screener, item.relation,
                item.frequency,
            ))
        else:
            omissions.append(_omission(
                "screened_by", (item.target, item.screened_member, item.screener,
                                item.relation), item.frequency, pi_min,
            ))

    paths: list[ConsensusPath] = []
    for item in summary.paths:
        if item.frequency.probability >= pi_min:
            paths.append(ConsensusPath(item.nodes, item.frequency))
        else:
            omissions.append(_omission("path", (item.nodes,), item.frequency, pi_min))

    # Keep deterministic ordering and retain the target evaluation frequency
    # on each edge/class; no interval data are available in LocalResult.
    return ConsensusResult(
        pi_min=float(pi_min), denominator=summary.denominator,
        requested_runs=summary.requested_runs, attempted_runs=summary.attempted_runs,
        completed_runs=summary.completed_runs, failed_runs=summary.failed_runs,
        nodes=tuple(sorted(nodes, key=lambda row: row.name)),
        targets=tuple(sorted(targets, key=lambda row: row.target)),
        classes=tuple(sorted(classes, key=lambda row: (
            row.target, row.candidate, "" if row.relation is None else row.relation.value))),
        edges=tuple(sorted(edges, key=lambda row: (
            row.target, row.source, row.relation.value))),
        shared_relations=tuple(sorted(shared_relations,
                                      key=lambda row: (row.target, row.members))),
        shared_pairs=tuple(sorted(shared_pairs,
                                  key=lambda row: (row.target, row.member_a, row.member_b))),
        screened_by=tuple(sorted(screened, key=lambda row: (
            row.target, row.screened_member, row.screener, row.relation.value))),
        paths=tuple(sorted(paths, key=lambda row: row.nodes)),
        omissions=tuple(sorted(omissions, key=lambda row: (
            row.structure, repr(row.identity)))),
    )


def _screeners(summary, target, candidate, relation, pi_min):
    return tuple(sorted({item.screener for item in summary.screened_by
                         if item.target == target and item.screened_member == candidate
                         and item.relation is relation
                         and item.frequency.probability >= pi_min}))


def _omission(structure, identity, frequency, pi_min):
    return ConsensusOmission(structure, identity, frequency,
                             f"stability {frequency.probability:.6g} < pi_min {pi_min:.6g}")
