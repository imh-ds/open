"""Deterministic local assessments for layer-wise consensus recursion."""
from dataclasses import replace
from math import floor
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd

from open.analysis import analyze_target
from open.config import OpenConfig
from open.layer_types import (AssessmentWarning, ConsensusLayer, FitProvenance,
                             LayeredOPENResult, LocalAssessment, PlannedSample, _integer, _names)
from open.layer_consensus import aggregate_target_assessments
from open.recurse import _preflight_infeasible, _validate_inputs
from open.splits import InsufficientTargetObservations, make_split_plan, observed_target_splits


def make_layer_sample_plans(data: pd.DataFrame, config: OpenConfig, *,
                           n_resamples: int | None = None) -> tuple[PlannedSample, ...]:
    if not isinstance(data, pd.DataFrame) or not len(data):
        raise ValueError("data must be a nonempty pandas DataFrame")
    if not isinstance(config, OpenConfig):
        raise TypeError("config must be an OpenConfig")
    budget = config.stability.n_resamples if n_resamples is None else n_resamples
    _integer(budget, 1)
    seed = config.analysis.random_seed
    if seed < 0 or seed + budget - 1 >= 2**32:
        raise ValueError("planned seeds must be in [0, 2**32)")
    size = max(1, floor(config.stability.fraction * len(data)))
    plans = []
    for b in range(budget):
        positions = tuple(sorted(int(i) for i in np.random.default_rng(seed+b).choice(
            len(data), size=size, replace=False)))
        run_config = replace(config, analysis=replace(config.analysis, random_seed=seed+b))
        infeasible = _preflight_infeasible(size, run_config)
        split = None if infeasible else make_split_plan(data.iloc[list(positions)], run_config)
        plans.append(PlannedSample(b, seed+b, positions, split,
                     "sample cannot support the requested outer/tuning folds" if infeasible else None))
    return tuple(plans)


def assess_layer_target(data: pd.DataFrame, *, depth: int, target: str,
                        candidates: tuple[str, ...], plans: tuple[PlannedSample, ...],
                        config: OpenConfig) -> tuple[LocalAssessment, ...]:
    _integer(depth)
    _names(candidates, empty=False)
    if target in candidates or target not in data or any(n not in data for n in candidates):
        raise ValueError("assessment target/candidates must be distinct data columns")
    if (not isinstance(plans, tuple) or not plans
            or tuple(p.run_index for p in plans) != tuple(range(len(plans)))):
        raise ValueError("assessment requires every planned sample")
    records = []
    for plan in plans:
        observed = missing = None
        local = failure_type = failure_message = None
        status = "completed"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                sampled = data.iloc[list(plan.sampled_positions)]
                observed = int(sampled[target].notna().sum())
                missing = len(sampled) - observed
                if plan.split_plan is None:
                    raise InsufficientTargetObservations(plan.infeasibility_reason)
                run_config = replace(config, analysis=replace(config.analysis, random_seed=plan.seed))
                observed_target_splits(plan.split_plan, sampled[target].to_numpy(dtype=float, na_value=np.nan),
                                       min_training_rows=config.tuning_cv.folds)
                local = analyze_target(sampled, target, candidates, run_config, plan.split_plan)
            except InsufficientTargetObservations as exc:
                status, failure_type, failure_message = "infeasible", type(exc).__name__, str(exc)[:2000] or type(exc).__name__
            except Exception as exc:
                status, failure_type, failure_message = "failed", type(exc).__name__, str(exc)[:2000] or type(exc).__name__
        messages = tuple(AssessmentWarning(w.category.__name__, str(w.message), w.filename, w.lineno)
                         for w in captured)
        try:
            record = LocalAssessment(depth, target, candidates, plan, status, local, messages,
                                     failure_type, failure_message, observed, missing)
        except (TypeError, ValueError) as exc:
            if status != "completed":
                raise
            record = LocalAssessment(depth, target, candidates, plan, "failed", None, messages,
                                     type(exc).__name__, str(exc)[:2000], observed, missing)
        records.append(record)
        for warning in captured:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return tuple(records)


def fit_layer_consensus(data: pd.DataFrame, root: str, variables: tuple[str, ...],
                        config: OpenConfig, *, n_resamples: int | None = None) -> LayeredOPENResult:
    """Finalize each complete consensus layer before scheduling the next one.

    Frequencies describe fixed comparisons conditional on the finalized prefix,
    rather than unconditional reconstruction of independently searched graphs.
    """
    _validate_inputs(data, root, variables, config)
    if config.stability.scope != "layer_consensus":
        raise ValueError("fit_layer_consensus requires layer_consensus scope")
    from open.stability import _capture_versions, read_git_revision
    revision, revision_reason = read_git_revision()
    package, dependencies, learners, availability = _capture_versions(config)
    started = time.perf_counter()
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    try:
        plans = make_layer_sample_plans(data, config, n_resamples=n_resamples)
        depths, parents, stops, layers = {root: 0}, {}, {}, []
        frontier, status = (root,), "COMPLETE"
        while frontier:
            depth = depths[frontier[0]]
            if depth >= config.analysis.max_depth:
                stops.update((target, "MAX_DEPTH") for target in frontier)
                break
            candidates = tuple(n for n in variables if n not in depths)
            if not candidates:
                stops.update((target, "NO_UNASSIGNED") for target in frontier)
                break
            evidence = tuple(aggregate_target_assessments(
                assess_layer_target(data, depth=depth, target=target, candidates=candidates,
                                    plans=plans, config=config),
                planned_count=len(plans), pi_min=config.thresholds.stability)
                for target in sorted(frontier))
            finalized = all(target.complete for target in evidence)
            next_parents = {}
            if finalized:
                for target in evidence:
                    retained = tuple(r for r in target.relations if r.qualifies)
                    if not retained:
                        stops[target.target] = "NO_CONSENSUS_PROMOTION"
                    for relation in retained:
                        for source in relation.members or (relation.source,):
                            next_parents.setdefault(source, set()).add(target.target)
            else:
                status = "INCOMPLETE_LAYER"
                stops.update((target, status) for target in frontier)
            next_parents = {source: tuple(sorted(names)) for source, names in sorted(next_parents.items())}
            next_layer = tuple(n for n in variables if n in next_parents)
            layers.append(ConsensusLayer(depth, tuple(sorted(frontier)), candidates, evidence,
                                          finalized, next_layer, next_parents))
            if not finalized:
                break
            for source in next_layer:
                depths[source] = depth + 1
                parents[source] = next_parents[source]
            frontier = next_layer
        if owns_tracing:
            peak, memory_reason = tracemalloc.get_traced_memory()[1], None
        else:
            peak, memory_reason = None, "tracemalloc was already active"
        provenance = FitProvenance(revision, revision_reason, time.perf_counter()-started,
            peak, memory_reason, package, dependencies, learners, availability)
        return LayeredOPENResult(config, variables, plans, tuple(layers), depths, parents,
                                  stops, status, provenance)
    finally:
        if owns_tracing:
            tracemalloc.stop()
