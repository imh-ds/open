"""Fold-local dependence selection and matched held-out pair ablations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from itertools import combinations
from typing import Mapping

import numpy as np
import pandas as pd

from open.classify import ClassificationResult, PairRelation, classify_pairs
from open.config import ThresholdConfig
from open.dependence.spearman import PairScore, score_dependence
from open.learners.base import Predictor
from open.relevance import RelevanceContrast, RelevanceResult
from open.risk import RiskResult, evaluate_risk
from open.splits import SplitPlan, observed_target_splits


@dataclass(frozen=True)
class FoldDependenceScores:
    repeat_index: int
    fold_index: int
    scores: tuple[PairScore, ...]


@dataclass(frozen=True)
class PairAblation:
    a: str
    b: str
    selected_folds: tuple[tuple[int, int], ...]
    selected_fold_count: int
    evaluated_fold_count: int
    selection_frequency: float
    evaluation_frequency: float
    baseline_risk: float
    full_risk: float
    ablated_risk: float
    total_gain: float
    raw: float
    normalized: float | None
    ablation: RiskResult | None


@dataclass(frozen=True)
class PairAblationResult:
    fold_scores: tuple[FoldDependenceScores, ...]
    pairs: tuple[PairAblation, ...]


@dataclass(frozen=True)
class SharedComponent:
    """All-fold ablation and validation decision for one shared-pair component."""

    members: tuple[str, ...]
    validated: bool
    raw: float
    normalized: float | None
    ablation: RiskResult | None


@dataclass(frozen=True)
class RetainedSharedPair:
    """A supported pair retained after its merged shared component failed."""

    a: str
    b: str
    members: tuple[str, str]
    pair: PairAblation


@dataclass(frozen=True)
class DependenceEdge:
    """Fold-selection provenance for an edge in the dependence graph."""

    a: str
    b: str
    selected_folds: tuple[tuple[int, int], ...]
    selected_fold_count: int
    evaluated_fold_count: int
    selection_frequency: float
    evaluation_frequency: float


@dataclass(frozen=True)
class DependenceComponent:
    """A dependence component used only for set-screening contexts."""

    members: tuple[str, ...]
    edges: tuple[DependenceEdge, ...]
    strong_members: tuple[str, ...]
    screened_by_set: tuple[tuple[str, tuple[str, ...]], ...]
    negligible_members: tuple[str, ...]


@dataclass(frozen=True)
class GroupingResult:
    """P2.2 results; pair retention is evidence for P2.3 to compose."""

    shared_components: tuple[SharedComponent, ...]
    retained_pairs: tuple[RetainedSharedPair, ...]
    dependence_components: tuple[DependenceComponent, ...]


def _losses_by_key(result: RiskResult) -> dict[tuple[int, int, int], float]:
    losses = {
        (fold.repeat_index, fold.fold_index, row.row_position): row.loss
        for fold in result.folds for row in fold.observations
    }
    if len(losses) != sum(len(fold.observations) for fold in result.folds):
        raise ValueError("risk result contains duplicate held-out keys")
    return losses


def _validate_result(result: RiskResult, target: str, candidates: tuple[str, ...],
                     split_id: str, seed: int, expected: set[tuple[int, int, int]],
                     observed_values: np.ndarray) -> dict[tuple[int, int, int], float]:
    if (result.target != target or result.candidate_set != candidates
            or result.split_plan_id != split_id or result.split_seed != seed or result.loss != "mse"):
        raise ValueError("risk result target, candidate set, or split plan does not match")
    losses = _losses_by_key(result)
    if set(losses) != expected:
        raise ValueError("risk result held-out keys do not match selected planned folds")
    for fold in result.folds:
        for row in fold.observations:
            if row.observed != float(observed_values[row.row_position]):
                raise ValueError("risk result observed target does not match supplied data")
            expected_loss = (row.observed - row.predicted) ** 2
            if not math.isclose(row.loss, expected_loss, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("row-keyed loss does not match observed/predicted MSE")
    return losses


def evaluate_pair_ablations(
    data: pd.DataFrame,
    *,
    candidate_set: tuple[str, ...],
    split_plan: SplitPlan,
    full: RiskResult,
    baseline: RiskResult,
    predictor: Predictor,
    tau_dep: float,
) -> PairAblationResult:
    """Select pairs on each training fold and evaluate only its held-out rows."""
    if not isinstance(data, pd.DataFrame) or len(data) != split_plan.n_samples:
        raise ValueError("data must be a DataFrame matching split_plan row count")
    if full.target not in data:
        raise ValueError(f"target column {full.target!r} is missing from data")
    if full.target in candidate_set:
        raise ValueError("target must be excluded from candidate_set")
    if full.predictors != candidate_set or baseline.predictors != ():
        raise ValueError("full and baseline predictors do not match the candidate set")
    observed_values = data[full.target].to_numpy(dtype=float, na_value=np.nan)
    target_splits = observed_target_splits(split_plan, observed_values)
    planned_keys = {
        (split.repeat, split.fold, row)
        for split in target_splits for row in split.test_indices
    }
    split_id = hashlib.sha256(split_plan.to_json(indent=None).encode("utf-8")).hexdigest()
    full_losses = _validate_result(full, full.target, candidate_set, split_id,
                                   split_plan.random_seed, planned_keys, observed_values)
    baseline_losses = _validate_result(baseline, full.target, candidate_set, split_id,
                                       split_plan.random_seed, planned_keys, observed_values)
    parameters = {(fold.repeat_index, fold.fold_index): fold.fitted_parameters for fold in full.folds}
    if any(parameters[key] is None for key in parameters):
        raise ValueError("full model lacks fitted parameters for an outer fold")

    fold_scores: list[FoldDependenceScores] = []
    selections: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for split in target_splits:
        scores = score_dependence(
            data.iloc[list(split.train_indices)][list(candidate_set)],
            candidate_set, tau_dep=tau_dep,
        )
        fold_scores.append(FoldDependenceScores(split.repeat, split.fold, scores))
        for score in scores:
            if score.candidate:
                selections.setdefault((score.a, score.b), []).append((split.repeat, split.fold))

    pair_results: list[PairAblation] = []
    total_folds = len(split_plan.splits)
    for (a, b), selected in sorted(selections.items()):
        selected_folds = tuple(selected)
        predictors = tuple(name for name in candidate_set if name not in {a, b})
        ablation = evaluate_risk(
            data, target=full.target, candidate_set=candidate_set, predictors=predictors,
            split_plan=split_plan, predictor=predictor,
            fitted_parameters=parameters if predictors else None,
            selected_folds=selected_folds,
        )
        expected = {
            (split.repeat, split.fold, row)
            for split in target_splits if (split.repeat, split.fold) in selected_folds
            for row in split.test_indices
        }
        ablation_losses = _validate_result(ablation, full.target, candidate_set, split_id,
                                            split_plan.random_seed, expected, observed_values)
        base_risk = float(np.mean([baseline_losses[key] for key in sorted(expected)]))
        full_risk = float(np.mean([full_losses[key] for key in sorted(expected)]))
        ablated_risk = float(np.mean([ablation_losses[key] for key in sorted(expected)]))
        gain = base_risk - full_risk
        raw = ablated_risk - full_risk
        count = len(selected_folds)
        pair_results.append(PairAblation(
            a, b, selected_folds, count, len(ablation.folds), count / total_folds,
            len(ablation.folds) / total_folds, base_risk, full_risk, ablated_risk,
            gain, raw, raw / gain if gain > 0 else None, ablation,
        ))
    return PairAblationResult(tuple(fold_scores), tuple(pair_results))


def analyze_groups(
    data: pd.DataFrame,
    *,
    target: str,
    candidate_set: tuple[str, ...],
    split_plan: SplitPlan,
    full: RiskResult,
    baseline: RiskResult,
    pair_ablations: PairAblationResult,
    classifications: ClassificationResult,
    relevance: RelevanceResult,
    predictor: Predictor,
    thresholds: ThresholdConfig,
) -> GroupingResult:
    """Validate shared components and separately derive dependence set screens.

    Shared components are ablated on every planned outer fold with the
    corresponding full-model parameters. Dependence components are the union
    of fold-local selected edges and are never treated as shared groups.
    """
    if not isinstance(data, pd.DataFrame) or len(data) != getattr(split_plan, "n_samples", None):
        raise ValueError("data must be a DataFrame matching split_plan row count")
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    if type(target) is not str or not target or target not in data:
        raise ValueError("target column is missing from data")
    if (not isinstance(candidate_set, tuple)
            or any(type(name) is not str or not name for name in candidate_set)
            or len(candidate_set) != len(set(candidate_set))):
        raise ValueError("candidate_set must be a tuple of unique variable names")
    if target in candidate_set:
        raise ValueError("target must be excluded from candidate_set")
    if any(name not in data.columns for name in candidate_set):
        raise ValueError("candidate_set contains an unknown column")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")
    if not isinstance(thresholds, ThresholdConfig):
        raise TypeError("thresholds must be a ThresholdConfig")
    if not isinstance(pair_ablations, PairAblationResult):
        raise TypeError("pair_ablations must be a PairAblationResult")
    if not isinstance(classifications, ClassificationResult):
        raise TypeError("classifications must be a ClassificationResult")
    if not isinstance(relevance, RelevanceResult):
        raise TypeError("relevance must be a RelevanceResult")

    planned_folds = tuple((split.repeat, split.fold) for split in split_plan.splits)
    planned_fold_set = set(planned_folds)
    observed_values = data[target].to_numpy(dtype=float, na_value=np.nan)
    target_splits = observed_target_splits(split_plan, observed_values)
    expected_keys = {
        (split.repeat, split.fold, row)
        for split in target_splits for row in split.test_indices
    }
    split_id = hashlib.sha256(split_plan.to_json(indent=None).encode("utf-8")).hexdigest()

    full_losses = _validate_result(
        full, target, candidate_set, split_id, split_plan.random_seed,
        expected_keys, observed_values,
    )
    baseline_losses = _validate_result(
        baseline, target, candidate_set, split_id, split_plan.random_seed,
        expected_keys, observed_values,
    )
    _validate_pooled_risks(full, full_losses)
    _validate_pooled_risks(baseline, baseline_losses)
    if full.predictors != candidate_set or baseline.predictors != ():
        raise ValueError("full and baseline predictors do not match candidate_set")
    parameters = {(fold.repeat_index, fold.fold_index): fold.fitted_parameters
                  for fold in full.folds}
    if set(parameters) != planned_fold_set or any(value is None for value in parameters.values()):
        raise ValueError("full model must retain fitted parameters for every planned fold")

    if relevance.stop_reason is not None or relevance.total_gain <= 0:
        raise ValueError("group analysis requires an eligible positive target gain")
    if (relevance.relative_gain is None
            or relevance.relative_gain < thresholds.min_total_gain_relative):
        raise ValueError("group analysis requires target gain at or above tau_gain")
    if baseline.aggregate_risk <= 0:
        raise ValueError("group analysis requires a positive baseline risk")
    expected_relative_gain = (
        baseline.aggregate_risk - full.aggregate_risk
    ) / baseline.aggregate_risk
    if not math.isclose(
        relevance.relative_gain, expected_relative_gain, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError("relevance relative gain does not match baseline and full risks")
    pooled_gain = float(np.mean(list(baseline_losses.values()))) - float(
        np.mean(list(full_losses.values()))
    )
    if not math.isclose(relevance.total_gain, pooled_gain, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("relevance total gain does not match full and baseline risks")
    if set(relevance.broad) != set(candidate_set) or set(relevance.unique) != set(candidate_set):
        raise ValueError("relevance scores must cover exactly candidate_set")
    for score_map in (relevance.broad, relevance.unique):
        for score in score_map.values():
            if (
                not isinstance(score, RelevanceContrast)
                or type(score.normalized) not in (int, float)
                or not math.isfinite(score.normalized)
                or type(score.raw) not in (int, float)
                or not math.isfinite(score.raw)
            ):
                raise ValueError("eligible relevance scores must be finite and normalized")

    pair_by_name: dict[tuple[str, str], PairAblation] = {}
    fold_selection: dict[tuple[str, str], list[tuple[int, int]]] = {}
    if len(pair_ablations.fold_scores) != len(planned_folds):
        raise ValueError("fold-local dependence scores must cover every planned fold")
    score_folds_seen: set[tuple[int, int]] = set()
    expected_pair_names = set(combinations(sorted(candidate_set), 2))
    for fold_scores in pair_ablations.fold_scores:
        fold_key = (fold_scores.repeat_index, fold_scores.fold_index)
        if fold_key not in planned_fold_set or fold_key in score_folds_seen:
            raise ValueError("fold-local dependence scores have invalid fold metadata")
        score_folds_seen.add(fold_key)
        score_names = [(score.a, score.b) for score in fold_scores.scores]
        if (len(score_names) != len(set(score_names))
                or set(score_names) != expected_pair_names):
            raise ValueError("fold-local dependence scores do not cover each candidate pair")
        for score in fold_scores.scores:
            if (score.a, score.b) != tuple(sorted((score.a, score.b))):
                raise ValueError("dependence pair names must be canonical")
            if type(score.candidate) is not bool:
                raise ValueError("dependence candidate decisions must be boolean")
            if score.score is not None and (
                type(score.score) not in (int, float)
                or not math.isfinite(score.score)
                or not 0 <= score.score <= 1
            ):
                raise ValueError("dependence scores must be finite numbers in [0, 1] or None")
            selected_by_threshold = (
                score.score is not None and score.score >= thresholds.dependence_pair
            )
            if score.candidate != selected_by_threshold:
                raise ValueError(
                    "dependence candidate decision does not match the inclusive tau_dep threshold"
                )
            if score.candidate:
                fold_selection.setdefault((score.a, score.b), []).append(fold_key)
    if score_folds_seen != planned_fold_set:
        raise ValueError("fold-local dependence scores do not match planned folds")

    for pair in pair_ablations.pairs:
        key = (pair.a, pair.b)
        if key != tuple(sorted(key)) or key in pair_by_name or key not in fold_selection:
            raise ValueError("pair ablations must be unique canonical selected pairs")
        selected = tuple(fold_selection[key])
        if pair.selected_folds != selected:
            raise ValueError("pair results disagree with fold-local selected edges")
        if (pair.selected_fold_count != len(selected)
                or pair.evaluated_fold_count != len(pair.ablation.folds)
                or pair.evaluated_fold_count != len(selected)):
            raise ValueError("pair selected/evaluated fold counts are inconsistent")
        if not math.isclose(pair.selection_frequency, len(selected) / len(planned_folds)):
            raise ValueError("pair selection frequency is inconsistent")
        if not math.isclose(pair.evaluation_frequency, len(selected) / len(planned_folds)):
            raise ValueError("pair evaluation frequency is inconsistent")
        remaining = tuple(name for name in candidate_set if name not in key)
        if pair.ablation.predictors != remaining:
            raise ValueError("pair ablation predictors do not match its pair")
        pair_keys = {
            (repeat, fold, row)
            for repeat, fold in selected
            for row in next(s for s in target_splits
                            if (s.repeat, s.fold) == (repeat, fold)).test_indices
        }
        pair_losses = _validate_result(
            pair.ablation, target, candidate_set, split_id, split_plan.random_seed,
            pair_keys, observed_values,
        )
        _validate_pooled_risks(pair.ablation, pair_losses)
        if remaining:
            for fold in pair.ablation.folds:
                fold_key = (fold.repeat_index, fold.fold_index)
                if fold.fitted_parameters != parameters[fold_key]:
                    raise ValueError("pair ablation did not reuse full-model fitted parameters")
        pair_base = float(np.mean([baseline_losses[item] for item in sorted(pair_keys)]))
        pair_full = float(np.mean([full_losses[item] for item in sorted(pair_keys)]))
        pair_ablated = float(np.mean([pair_losses[item] for item in sorted(pair_keys)]))
        pair_gain, pair_raw = pair_base - pair_full, pair_ablated - pair_full
        normalized = pair_raw / pair_gain if pair_gain > 0 else None
        if (not math.isclose(pair.baseline_risk, pair_base, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(pair.full_risk, pair_full, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(pair.ablated_risk, pair_ablated, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(pair.total_gain, pair_gain, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(pair.raw, pair_raw, rel_tol=1e-10, abs_tol=1e-12)
                or (normalized is None and pair.normalized is not None)
                or (normalized is not None and (
                    pair.normalized is None
                    or not math.isclose(pair.normalized, normalized, rel_tol=1e-10, abs_tol=1e-12)
                ))):
            raise ValueError("pair result risks or normalized score are inconsistent")
        pair_by_name[key] = pair

    if set(pair_by_name) != set(fold_selection):
        raise ValueError("pair results do not cover all fold-local selected edges")
    expected_classification = classify_pairs(
        {name: relevance.broad[name].normalized for name in candidate_set},
        {name: relevance.unique[name].normalized for name in candidate_set},
        {key: pair.normalized for key, pair in pair_by_name.items()},
        thresholds,
    )
    if classifications != expected_classification:
        raise ValueError("P2.1 classifications do not match relevance and pair results")

    shared_edges = sorted(
        (pair.a, pair.b) for pair in classifications.pairs
        if pair.relation is PairRelation.PAIR_SHARED_REDUNDANT
    )
    shared_components_members = _components(shared_edges)
    shared_components: list[SharedComponent] = []
    retained_pairs: list[RetainedSharedPair] = []
    for members in shared_components_members:
        predictors = tuple(name for name in candidate_set if name not in members)
        ablation = evaluate_risk(
            data, target=target, candidate_set=candidate_set, predictors=predictors,
            split_plan=split_plan, predictor=predictor,
            fitted_parameters=parameters if predictors else None,
        )
        group_losses = _validate_result(
            ablation, target, candidate_set, split_id, split_plan.random_seed,
            expected_keys, observed_values,
        )
        _validate_pooled_risks(ablation, group_losses)
        if predictors:
            for fold in ablation.folds:
                if fold.fitted_parameters != parameters[(fold.repeat_index, fold.fold_index)]:
                    raise ValueError("group ablation did not reuse full-model fitted parameters")
        raw = float(np.mean(list(group_losses.values()))) - float(np.mean(list(full_losses.values())))
        normalized = raw / relevance.total_gain if relevance.total_gain > 0 else None
        unique_weak = all(
            relevance.unique[name].normalized < thresholds.unique_relative for name in members
        )
        validated = (
            normalized is not None
            and normalized >= thresholds.group_relative
            and unique_weak
        )
        shared_components.append(SharedComponent(members, validated, raw, normalized, ablation))
        if not validated:
            for a, b in shared_edges:
                if a in members and b in members:
                    pair = pair_by_name[(a, b)]
                    retained_pairs.append(RetainedSharedPair(a, b, (a, b), pair))

    dependence_edges = tuple(sorted(fold_selection))
    dependence_components_members = _components(dependence_edges)
    dependence_components: list[DependenceComponent] = []
    for members in dependence_components_members:
        edges = tuple(
            DependenceEdge(
                a, b, tuple(fold_selection[(a, b)]), len(fold_selection[(a, b)]),
                pair_by_name[(a, b)].evaluated_fold_count,
                len(fold_selection[(a, b)]) / len(planned_folds),
                pair_by_name[(a, b)].evaluation_frequency,
            )
            for a, b in dependence_edges if a in members and b in members
        )
        strong = tuple(
            name for name in members
            if relevance.unique[name].normalized >= thresholds.unique_relative
        )
        screened = tuple(
            (name, strong) for name in members
            if strong and name not in strong
            and relevance.broad[name].normalized >= thresholds.broad_relative
        )
        negligible = tuple(
            name for name in members
            if strong and name not in strong
            and relevance.broad[name].normalized < thresholds.broad_relative
        )
        dependence_components.append(
            DependenceComponent(members, edges, strong, screened, negligible)
        )

    return GroupingResult(
        tuple(shared_components), tuple(retained_pairs), tuple(dependence_components)
    )


def _validate_pooled_risks(
    result: RiskResult, losses: Mapping[tuple[int, int, int], float]
) -> None:
    """Check stored fold/aggregate summaries against their row-keyed evidence."""
    for fold in result.folds:
        observed = float(np.mean([row.loss for row in fold.observations]))
        if not math.isclose(fold.risk, observed, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("risk result fold summary does not match row-keyed losses")
    pooled = float(np.mean(list(losses.values())))
    if not math.isclose(result.aggregate_risk, pooled, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("risk result aggregate does not match row-keyed losses")


def _components(edges: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> tuple[tuple[str, ...], ...]:
    """Return deterministic non-singleton connected components for canonical edges."""
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    components: list[tuple[str, ...]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack, members = [start], set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(adjacency[current] - members, reverse=True))
        unseen -= members
        if len(members) >= 2:
            components.append(tuple(sorted(members)))
    return tuple(sorted(components))
