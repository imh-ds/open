"""Immutable one-run graph records; this module does not expand a network."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from open.grouping import RetainedSharedPair, SharedComponent
from open.local_target import LocalClass, LocalResult


class StopReason(str, Enum):
    LOW_TOTAL_GAIN = "LOW_TOTAL_GAIN"
    NO_PROMOTION = "NO_PROMOTION"
    NO_UNASSIGNED = "NO_UNASSIGNED"
    MAX_DEPTH = "MAX_DEPTH"
    INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE = "INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE"


def _name(value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("variable names must be nonempty strings")


def _score(value: float | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("scores must be finite numbers (normalized scores may be None)")


def _freeze(value):
    """Detach nested result mappings as well as the graph's own containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _freeze(getattr(value, field.name))
                                 for field in fields(value)})
    return value


@dataclass(frozen=True)
class CoreEdge:
    """A promoted proximal predictor points toward its analyzed outcome."""

    source: str
    target: str
    kind: LocalClass
    broad_raw: float
    broad_normalized: float | None
    unique_raw: float
    unique_normalized: float | None

    def __post_init__(self) -> None:
        _name(self.source)
        _name(self.target)
        if self.source == self.target or self.kind is not LocalClass.PROXIMAL_UNIQUE:
            raise ValueError("core edges require distinct predictor/target and PROXIMAL_UNIQUE")
        _score(self.broad_raw)
        _score(self.unique_raw)
        _score(self.broad_normalized, optional=True)
        _score(self.unique_normalized, optional=True)


@dataclass(frozen=True)
class SharedRelation:
    """A visual relation with original component or D-003 pair provenance."""

    target: str
    members: tuple[str, ...]
    raw: float
    normalized: float | None
    provenance: SharedComponent | RetainedSharedPair

    def __post_init__(self) -> None:
        _name(self.target)
        if not isinstance(self.members, tuple) or len(self.members) < 2:
            raise ValueError("shared members must be a tuple with at least two variables")
        for name in self.members:
            _name(name)
        if len(set(self.members)) != len(self.members) or self.target in self.members:
            raise ValueError("shared members must be distinct and exclude the target")
        _score(self.raw)
        _score(self.normalized, optional=True)
        provenance = self.provenance
        if isinstance(provenance, SharedComponent):
            if provenance.validated is not True:
                raise ValueError("shared-component provenance must be validated")
            evidence = provenance
        elif isinstance(provenance, RetainedSharedPair):
            if (len(self.members) != 2 or provenance.a >= provenance.b
                    or provenance.members != (provenance.a, provenance.b)
                    or (provenance.pair.a, provenance.pair.b) != provenance.members):
                raise ValueError("retained pair provenance must identify a canonical pair")
            evidence = provenance.pair
        else:
            raise ValueError("shared provenance must be SharedComponent or RetainedSharedPair")
        if (tuple(sorted(self.members)) != tuple(sorted(provenance.members))
                or self.raw != evidence.raw or self.normalized != evidence.normalized):
            raise ValueError("shared relation members and scores must match provenance")
        object.__setattr__(self, "members", tuple(sorted(self.members)))
        object.__setattr__(self, "provenance", _freeze(provenance))


@dataclass(frozen=True)
class NetworkRun:
    """Completed one-run evidence, including assigned but unevaluated frontiers.

    ``parents`` maps each promoted child to all qualifying analyzed parents.
    The preflight ``stop_reason`` is separate from ``stop_reasons`` per target.
    ``max_depth`` is the positive configured cap; boundary nodes are not evaluated.
    Neither candidate-only variables nor virtual group nodes acquire a depth.
    """

    root: str
    assigned_depth: Mapping[str, int]
    parents: Mapping[str, tuple[str, ...]]
    local_results: Mapping[str, LocalResult]
    core_edges: tuple[CoreEdge, ...]
    shared_relations: tuple[SharedRelation, ...]
    stop_reasons: Mapping[str, StopReason]
    max_depth: int
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        _name(self.root)
        if type(self.max_depth) is not int or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        for mapping in (self.assigned_depth, self.parents, self.local_results, self.stop_reasons):
            if not isinstance(mapping, Mapping):
                raise ValueError("network containers must be mappings")
            for name in mapping:
                _name(name)
        depths = dict(self.assigned_depth)
        if (depths.get(self.root) != 0
                or any(type(d) is not int or d < 0 or d > self.max_depth
                       for d in depths.values())
                or any(d == 0 for n, d in depths.items() if n != self.root)):
            raise ValueError("root must have depth zero; other depths must be positive integers within max_depth")
        if self.stop_reason is not None:
            if self.stop_reason is not StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE:
                raise ValueError("only insufficient effective sample size is a run-level stop")
            if (depths != {self.root: 0} or self.parents or self.local_results
                    or self.core_edges or self.shared_relations or self.stop_reasons):
                raise ValueError("preflight stop must precede all target evaluations and promotions")
        parents = {}
        if set(self.parents) != set(depths) - {self.root}:
            raise ValueError("every promoted variable must have parents; root must not")
        for child, names in self.parents.items():
            if not isinstance(names, tuple) or not names:
                raise ValueError("parents must be nonempty tuples")
            for parent in names:
                _name(parent)
                if parent not in depths or depths[child] != depths[parent] + 1:
                    raise ValueError("parents must be assigned at the preceding layer")
            parents[child] = tuple(sorted(set(names)))
        for target, result in self.local_results.items():
            if (target not in depths or not isinstance(result, LocalResult)
                    or result.target != target or depths[target] == self.max_depth):
                raise ValueError("local results must match assigned analyzed targets")
            candidates = set(result.relevance.broad)
            if (candidates != set(result.relevance.unique) or target in candidates
                    or any(type(n) is not str or not n.strip() for n in candidates)
                    or any(n in depths and depths[n] <= depths[target] for n in candidates)
                    or not result.promoted.issubset(candidates)):
                raise ValueError("local candidate identities conflict with the target layer")
            actual = {child for child, names in parents.items() if target in names}
            if actual != set(result.promoted):
                raise ValueError("parent relations must preserve every local promotion")
        if any(parent not in self.local_results for names in parents.values() for parent in names):
            raise ValueError("parents must have actually evaluated local results")
        for target, reason in self.stop_reasons.items():
            if target not in depths or not isinstance(reason, StopReason):
                raise ValueError("per-target stops require assigned names and typed reasons")
            result = self.local_results.get(target)
            if reason in (StopReason.LOW_TOTAL_GAIN, StopReason.NO_PROMOTION):
                if result is None:
                    raise ValueError("evaluated stop reasons require a local result")
                expected = (StopReason.LOW_TOTAL_GAIN if result.stop
                            and result.reason == StopReason.LOW_TOTAL_GAIN.value
                            else StopReason.NO_PROMOTION if not result.stop and not result.promoted
                            else None)
                if reason is not expected:
                    raise ValueError("evaluated stop reason disagrees with local result")
            elif reason in (StopReason.NO_UNASSIGNED, StopReason.MAX_DEPTH,
                            StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE):
                if result is not None:
                    raise ValueError("unevaluated frontier stops must not have local results")
                if reason is StopReason.MAX_DEPTH and (
                    target == self.root or depths[target] != self.max_depth
                ):
                    raise ValueError("MAX_DEPTH requires a promoted node at the depth boundary")
                if reason is StopReason.NO_UNASSIGNED and depths[target] == self.max_depth:
                    raise ValueError("promoted nodes at max_depth must carry MAX_DEPTH")
                if reason is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE and (
                    target == self.root or depths[target] == self.max_depth
                ):
                    raise ValueError("target sample-size stops require a recursive target below max_depth")
            else:
                raise ValueError("unsupported per-target stop reason")
        for target, result in self.local_results.items():
            if result.stop:
                if (result.reason != StopReason.LOW_TOTAL_GAIN.value or result.promoted
                        or self.stop_reasons.get(target) is not StopReason.LOW_TOTAL_GAIN):
                    raise ValueError("stopped local results require LOW_TOTAL_GAIN and no promotions")
            elif not result.promoted:
                if self.stop_reasons.get(target) is not StopReason.NO_PROMOTION:
                    raise ValueError("empty evaluated promotions require NO_PROMOTION")
            elif target in self.stop_reasons:
                raise ValueError("promoting analyzed targets cannot also be stopped")
        if self.stop_reason is None and set(depths) != set(self.local_results) | set(self.stop_reasons):
            raise ValueError("every assigned target must be evaluated or have a frontier stop")
        if not isinstance(self.core_edges, tuple) or not isinstance(self.shared_relations, tuple):
            raise ValueError("edges and shared relations must be tuples")
        edge_ids = set()
        for edge in self.core_edges:
            if not isinstance(edge, CoreEdge):
                raise ValueError("core edges must be typed CoreEdge records")
            identity = (edge.source, edge.target)
            if identity in edge_ids:
                raise ValueError("core edge identities must be unique")
            edge_ids.add(identity)
            result = self.local_results.get(edge.target)
            if (result is None or edge.target not in parents.get(edge.source, ())
                    or edge.source not in result.classes
                    or result.classes[edge.source].kind is not LocalClass.PROXIMAL_UNIQUE):
                raise ValueError("core edge direction and class must match local promotion")
            broad, unique = result.relevance.broad[edge.source], result.relevance.unique[edge.source]
            if (edge.broad_raw, edge.broad_normalized, edge.unique_raw, edge.unique_normalized) != (
                broad.raw, broad.normalized, unique.raw, unique.normalized
            ):
                raise ValueError("core edge scores must preserve local relevance")
        expected_edges = {
            (source, target) for target, result in self.local_results.items()
            for source in result.promoted
            if source in result.classes
            and result.classes[source].kind is LocalClass.PROXIMAL_UNIQUE
        }
        if edge_ids != expected_edges:
            raise ValueError("core edges must include every proximal promotion")
        relation_ids = set()
        for relation in self.shared_relations:
            if not isinstance(relation, SharedRelation):
                raise ValueError("shared relations must be typed SharedRelation records")
            identity = (relation.target, relation.members)
            if identity in relation_ids:
                raise ValueError("shared relation identities must be unique")
            relation_ids.add(identity)
            result = self.local_results.get(relation.target)
            if (result is None
                    or any(relation.target not in parents.get(n, ()) for n in relation.members)
                    or relation.provenance not in result.shared_components + result.retained_pairs):
                raise ValueError("shared relation must preserve analyzed promotion provenance")
        expected_relations = {
            (target, tuple(sorted(item.members)))
            for target, result in self.local_results.items()
            for item in tuple(c for c in result.shared_components if c.validated)
            + result.retained_pairs
        }
        if relation_ids != expected_relations:
            raise ValueError("shared relations must include every validated component and retained pair")
        for field in ("assigned_depth", "local_results", "stop_reasons"):
            object.__setattr__(self, field, _freeze(getattr(self, field)))
        object.__setattr__(self, "parents", _freeze(parents))
        object.__setattr__(self, "core_edges", tuple(sorted(self.core_edges,
                           key=lambda e: (e.source, e.target))))
        object.__setattr__(self, "shared_relations", tuple(sorted(self.shared_relations,
                           key=lambda r: (r.target, r.members))))
