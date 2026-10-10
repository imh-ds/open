"""Immutable evidence for consensus established before recursive expansion."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

from open.config import OpenConfig
from open.graph import _freeze
from open.local_target import LocalResult
from open.splits import SplitPlan

LAYER_METHOD = "layer_consensus"
LAYER_SCHEMA = "open.layer_consensus"
LAYER_SCHEMA_VERSION = 1


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("expected an integer within bounds")


def _name(value):
    if type(value) is not str or not value.strip():
        raise ValueError("expected a nonempty name")


def _names(values, *, empty=True):
    if not isinstance(values, tuple) or (not empty and not values):
        raise ValueError("expected a tuple of names")
    for name in values:
        _name(name)
    if len(set(values)) != len(values):
        raise ValueError("names must be distinct")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite number")


def _detach(obj):
    for field in fields(obj):
        object.__setattr__(obj, field.name, _freeze(getattr(obj, field.name)))


@dataclass(frozen=True)
class AssessmentWarning:
    category: str
    message: str
    filename: str
    lineno: int

    def __post_init__(self):
        _name(self.category)
        if type(self.message) is not str or type(self.filename) is not str:
            raise ValueError("warning text must be strings")
        _integer(self.lineno)


@dataclass(frozen=True)
class PlannedSample:
    run_index: int
    seed: int
    sampled_positions: tuple[int, ...]
    split_plan: SplitPlan | None
    infeasibility_reason: str | None

    def __post_init__(self):
        _integer(self.run_index)
        _integer(self.seed)
        if self.seed >= 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not isinstance(self.sampled_positions, tuple) or not self.sampled_positions:
            raise ValueError("sampled positions must be a nonempty tuple")
        for position in self.sampled_positions:
            _integer(position)
        if tuple(sorted(set(self.sampled_positions))) != self.sampled_positions:
            raise ValueError("sampled positions must be sorted and unique")
        if self.split_plan is None:
            _name(self.infeasibility_reason)
        elif (not isinstance(self.split_plan, SplitPlan) or self.infeasibility_reason is not None
              or self.split_plan.n_samples != len(self.sampled_positions)
              or self.split_plan.random_seed != self.seed):
            raise ValueError("split plan must match sampled rows and seed")


@dataclass(frozen=True)
class LocalAssessment:
    depth: int
    target: str
    candidates: tuple[str, ...]
    sample: PlannedSample
    status: str
    local: LocalResult | None
    warnings: tuple[AssessmentWarning, ...]
    failure_type: str | None
    failure_message: str | None
    n_observed_target_rows: int | None
    n_missing_target_rows: int | None

    def __post_init__(self):
        _integer(self.depth)
        _name(self.target)
        _names(self.candidates, empty=False)
        if self.target in self.candidates or not isinstance(self.sample, PlannedSample):
            raise ValueError("assessment target and candidates conflict")
        if not isinstance(self.warnings, tuple) or any(not isinstance(w, AssessmentWarning) for w in self.warnings):
            raise ValueError("warnings must be a tuple of AssessmentWarning")
        coverage = (self.n_observed_target_rows, self.n_missing_target_rows)
        if coverage != (None, None):
            for count in coverage:
                _integer(count)
            if sum(coverage) != len(self.sample.sampled_positions):
                raise ValueError("outcome coverage must match sampled rows")
        if self.status == "completed":
            if (not isinstance(self.local, LocalResult) or self.local.target != self.target
                    or self.local.input_candidates != self.candidates
                    or self.failure_type is not None or self.failure_message is not None
                    or coverage == (None, None) or self.sample.split_plan is None
                    or coverage != (self.local.n_observed_target_rows, self.local.n_missing_target_rows)):
                raise ValueError("completed assessment must match its local evidence")
        elif self.status in ("failed", "infeasible"):
            if self.local is not None:
                raise ValueError("failed assessment cannot contain a local result")
            _name(self.failure_type)
            _name(self.failure_message)
            if len(self.failure_message) > 2000:
                raise ValueError("failure message exceeds 2000 characters")
        else:
            raise ValueError("unknown assessment status")
        _detach(self)


@dataclass(frozen=True)
class SelectionCount:
    selected: int
    planned: int
    completed: int

    def __post_init__(self):
        _integer(self.selected)
        _integer(self.planned, 1)
        _integer(self.completed)
        if not self.selected <= self.completed <= self.planned:
            raise ValueError("counts must satisfy selected <= completed <= planned")

    @property
    def probability(self):
        return self.selected / self.planned if self.completed == self.planned else None

    @property
    def provisional_probability(self):
        return self.selected / self.completed if self.completed else None


@dataclass(frozen=True)
class StrengthSummary:
    raw_count: int
    raw_median: float | None
    baseline_relative_count: int
    baseline_relative_median: float | None
    gain_relative_count: int
    gain_relative_median: float | None
    scope: str

    def __post_init__(self):
        for count, value in ((self.raw_count, self.raw_median),
                             (self.baseline_relative_count, self.baseline_relative_median),
                             (self.gain_relative_count, self.gain_relative_median)):
            _integer(count)
            if count == 0:
                if value is not None:
                    raise ValueError("unmeasured strength must be None")
            else:
                _number(value)
        if max(self.baseline_relative_count, self.gain_relative_count) > self.raw_count:
            raise ValueError("relative measurement count exceeds raw count")
        _name(self.scope)


@dataclass(frozen=True)
class CandidateEvidence:
    source: str
    target: str
    classes: Mapping[str, SelectionCount]
    promotion: SelectionCount
    broad: StrengthSummary
    unique: StrengthSummary

    def __post_init__(self):
        _name(self.source)
        _name(self.target)
        if self.source == self.target or not isinstance(self.classes, Mapping):
            raise ValueError("invalid candidate identity or classes")
        if not isinstance(self.promotion, SelectionCount):
            raise ValueError("invalid promotion count")
        for kind, count in self.classes.items():
            _name(kind)
            if (not isinstance(count, SelectionCount) or count.planned != self.promotion.planned
                    or count.completed != self.promotion.completed):
                raise ValueError("class counts must share assessment denominator")
        if sum(c.selected for c in self.classes.values()) != self.promotion.completed:
            raise ValueError("class counts must partition completed assessments")
        if not isinstance(self.broad, StrengthSummary) or not isinstance(self.unique, StrengthSummary):
            raise ValueError("invalid candidate strengths")
        _detach(self)


@dataclass(frozen=True)
class RelationEvidence:
    source: str | None
    target: str
    relation: str
    members: tuple[str, ...]
    screened_by: tuple[str, ...]
    frequency: SelectionCount
    strength: StrengthSummary
    qualifies: bool

    def __post_init__(self):
        _name(self.target)
        _names(self.members)
        _names(self.screened_by)
        if self.relation == "SHARED_GROUP":
            if self.source is not None or len(self.members) < 2 or self.members != tuple(sorted(self.members)) or self.screened_by:
                raise ValueError("joint relation requires canonical members")
        else:
            _name(self.source)
            if self.members or self.source == self.target or self.relation not in (
                "PROXIMAL_UNIQUE", "SCREENED_BY_PARTNER", "SCREENED_BY_SET", "SCREENED_GENERAL"):
                raise ValueError("invalid individual relation")
        if not isinstance(self.frequency, SelectionCount) or not isinstance(self.strength, StrengthSummary):
            raise ValueError("invalid relation evidence")
        if type(self.qualifies) is not bool:
            raise ValueError("qualifies must be boolean")
        if self.qualifies and (self.relation not in ("PROXIMAL_UNIQUE", "SHARED_GROUP")
                or self.frequency.probability is None or not self.frequency.selected):
            raise ValueError("only observed complete unique/joint relations can qualify")


@dataclass(frozen=True)
class TargetEvidence:
    depth: int
    target: str
    candidates: tuple[str, ...]
    assessments: tuple[LocalAssessment, ...]
    candidates_evidence: tuple[CandidateEvidence, ...]
    relations: tuple[RelationEvidence, ...]
    shared_comembership: Mapping[tuple[str, str], SelectionCount]

    def __post_init__(self):
        _integer(self.depth)
        _name(self.target)
        _names(self.candidates, empty=False)
        if not isinstance(self.assessments, tuple) or not self.assessments:
            raise ValueError("target evidence requires assessments")
        if sorted(a.sample.run_index for a in self.assessments) != list(range(len(self.assessments))):
            raise ValueError("target evidence needs each planned assessment exactly once")
        if any((a.depth, a.target, a.candidates) != (self.depth, self.target, self.candidates) for a in self.assessments):
            raise ValueError("target evidence contains different comparisons")
        if (len(self.candidates_evidence) != len(self.candidates)
                or {e.source for e in self.candidates_evidence} != set(self.candidates)):
            raise ValueError("candidate evidence must cover the frozen candidate set")
        for evidence in (*self.candidates_evidence, *self.relations):
            if evidence.target != self.target:
                raise ValueError("evidence target mismatch")
            count = evidence.promotion if isinstance(evidence, CandidateEvidence) else evidence.frequency
            if (count.planned, count.completed) != (len(self.assessments), self.completed_count):
                raise ValueError("evidence counts mismatch")
        identities = []
        for relation in self.relations:
            members = relation.members or (relation.source,)
            if not set((*members, *relation.screened_by)) <= set(self.candidates):
                raise ValueError("relation references foreign candidate")
            identities.append((relation.source, relation.relation, relation.members, relation.screened_by))
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate relation identity")
        for pair, count in self.shared_comembership.items():
            if (not isinstance(pair, tuple) or len(pair) != 2 or pair != tuple(sorted(set(pair)))
                    or not set(pair) <= set(self.candidates) or not isinstance(count, SelectionCount)
                    or (count.planned, count.completed) != (len(self.assessments), self.completed_count)):
                raise ValueError("invalid shared co-membership")
        _detach(self)

    @property
    def completed_count(self):
        return sum(a.status == "completed" for a in self.assessments)

    @property
    def complete(self):
        return self.completed_count == len(self.assessments)


@dataclass(frozen=True)
class ConsensusLayer:
    depth: int
    targets: tuple[str, ...]
    candidates: tuple[str, ...]
    evidence: tuple[TargetEvidence, ...]
    finalized: bool
    new_nodes: tuple[str, ...]
    promoted_parents: Mapping[str, tuple[str, ...]]

    def __post_init__(self):
        _integer(self.depth)
        _names(self.targets, empty=False)
        _names(self.candidates, empty=False)
        _names(self.new_nodes)
        if set(self.targets) & set(self.candidates):
            raise ValueError("layer targets cannot be candidates")
        if type(self.finalized) is not bool or self.finalized != all(e.complete for e in self.evidence):
            raise ValueError("only complete layers can finalize")
        if (len(self.evidence) != len(self.targets) or {e.target for e in self.evidence} != set(self.targets)
                or any(e.depth != self.depth or e.candidates != self.candidates for e in self.evidence)):
            raise ValueError("layer evidence must cover every frozen target")
        expected = {}
        if self.finalized:
            for target in self.evidence:
                for relation in target.relations:
                    if relation.qualifies:
                        for name in relation.members or (relation.source,):
                            expected.setdefault(name, set()).add(target.target)
        expected = {name: tuple(sorted(parents)) for name, parents in expected.items()}
        if set(self.new_nodes) != set(expected) or dict(self.promoted_parents) != expected:
            raise ValueError("promotions must be supported by finalized relations")
        _detach(self)


@dataclass(frozen=True)
class FitProvenance:
    git_revision: str | None
    git_revision_unavailable_reason: str | None
    elapsed_seconds: float
    peak_traced_memory_bytes: int | None
    peak_memory_unavailable_reason: str | None
    package_version: str | None
    dependency_versions: Mapping[str, str | None]
    learner_versions: Mapping[str, str | None]
    version_availability: Mapping[str, str]

    def __post_init__(self):
        _number(self.elapsed_seconds)
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed time must be nonnegative")
        if self.peak_traced_memory_bytes is not None:
            _integer(self.peak_traced_memory_bytes)
        _detach(self)


@dataclass(frozen=True)
class LayeredOPENResult:
    config: OpenConfig
    variables: tuple[str, ...]
    plans: tuple[PlannedSample, ...]
    layers: tuple[ConsensusLayer, ...]
    assigned_depth: Mapping[str, int]
    parents: Mapping[str, tuple[str, ...]]
    stop_reasons: Mapping[str, str]
    status: str
    provenance: FitProvenance

    def __post_init__(self):
        if not isinstance(self.config, OpenConfig) or self.config.stability.scope != LAYER_METHOD:
            raise ValueError("layer result requires layer_consensus configuration")
        _names(self.variables, empty=False)
        root = self.config.analysis.root
        if root not in self.variables or not isinstance(self.provenance, FitProvenance):
            raise ValueError("result root/provenance mismatch")
        if not self.plans or tuple(p.run_index for p in self.plans) != tuple(range(len(self.plans))):
            raise ValueError("result requires every planned sample")
        if any(p.seed != self.config.analysis.random_seed + p.run_index for p in self.plans):
            raise ValueError("sample seed does not match configured identity")
        depths, parents, frontier = {root: 0}, {}, {root}
        incomplete = False
        expected_stops = {}
        for depth, layer in enumerate(self.layers):
            if (incomplete or not frontier or layer.depth != depth or set(layer.targets) != frontier
                    or layer.candidates != tuple(n for n in self.variables if n not in depths)
                    or depth >= self.config.analysis.max_depth):
                raise ValueError("layer sequence conflicts with finalized prefix")
            for target in layer.evidence:
                if tuple(a.sample for a in target.assessments) != self.plans:
                    raise ValueError("assessment sample plans differ from the planned batch")
                for relation in target.relations:
                    expected = (relation.relation in ("PROXIMAL_UNIQUE", "SHARED_GROUP")
                                and relation.frequency.probability is not None and relation.frequency.selected > 0
                                and relation.frequency.probability >= self.config.thresholds.stability)
                    if relation.qualifies != expected:
                        raise ValueError("relation inclusion conflicts with threshold")
            incomplete = not layer.finalized
            for target in layer.evidence:
                if incomplete:
                    expected_stops[target.target] = "INCOMPLETE_LAYER"
                elif not any(relation.qualifies for relation in target.relations):
                    expected_stops[target.target] = "NO_CONSENSUS_PROMOTION"
            for name in layer.new_nodes:
                depths[name] = depth + 1
                parents[name] = layer.promoted_parents[name]
            frontier = set(layer.new_nodes)
        if dict(self.assigned_depth) != depths or dict(self.parents) != parents:
            raise ValueError("assigned nodes must match finalized relations")
        if self.status != ("INCOMPLETE_LAYER" if incomplete else "COMPLETE"):
            raise ValueError("result completeness does not match its layers")
        if (not incomplete and frontier and len(depths) < len(self.variables)
                and min(depths[n] for n in frontier) < self.config.analysis.max_depth):
            raise ValueError("complete result contains an unfinished recursive frontier")
        if not incomplete and frontier:
            reason = ("MAX_DEPTH" if min(depths[n] for n in frontier) >= self.config.analysis.max_depth
                      else "NO_UNASSIGNED")
            expected_stops.update((name, reason) for name in frontier)
        if dict(self.stop_reasons) != expected_stops:
            raise ValueError("target stop reasons contradict recursive evidence")
        _detach(self)

    @property
    def effective_resamples(self):
        return len(self.plans)
