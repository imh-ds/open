"""Layer-synchronous construction of a one-run OPEN network."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from open.analysis import analyze_target
from open.config import OpenConfig
from open.graph import CoreEdge, NetworkRun, SharedRelation, StopReason
from open.local_target import LocalClass, LocalResult
from open.splits import (
    InsufficientTargetObservations, SplitPlan, make_split_plan, observed_target_splits,
)


def build_network_once(
    data: pd.DataFrame,
    root: str,
    all_variables: tuple[str, ...],
    config: OpenConfig,
) -> NetworkRun:
    """Expand the one-run network in synchronous depth layers.

    All targets at a depth receive the same snapshot of then-unassigned
    variables. Promotions are applied only after every target in that layer
    has finished, so same-layer targets can jointly parent the same child.
    """
    _validate_inputs(data, root, all_variables, config)
    if config.stability.scope != "whole_pipeline":
        raise NotImplementedError("build_network_once is whole-pipeline only; use the layer fitter")
    depths: dict[str, int] = {root: 0}
    parents: dict[str, tuple[str, ...]] = {}
    results: dict[str, LocalResult] = {}
    edges: list[CoreEdge] = []
    relations: list[SharedRelation] = []
    stops: dict[str, StopReason] = {}

    if _preflight_infeasible(len(data), config):
        return NetworkRun(
            root, depths, parents, results, (), (), {},
            max_depth=config.analysis.max_depth,
            stop_reason=StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE,
        )

    split_plan: SplitPlan = make_split_plan(data, config)
    layer = (root,)
    while layer:
        depth = depths[layer[0]]
        if depth >= config.analysis.max_depth:
            stops.update((target, StopReason.MAX_DEPTH) for target in layer)
            break

        snapshot = tuple(name for name in all_variables if name not in depths)
        if not snapshot:
            stops.update((target, StopReason.NO_UNASSIGNED) for target in layer)
            break

        layer_results: dict[str, LocalResult] = {}
        for target in layer:
            candidates = tuple(name for name in snapshot if name != target)
            try:
                observed_target_splits(
                    split_plan, data[target].to_numpy(dtype=float, na_value=np.nan),
                    min_training_rows=config.tuning_cv.folds,
                )
            except InsufficientTargetObservations:
                stops[target] = StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
                continue
            result = analyze_target(data, target, candidates, config, split_plan)
            layer_results[target] = result

        next_parents: dict[str, list[str]] = {}
        for target, result in layer_results.items():
            results[target] = result
            if result.stop:
                stops[target] = StopReason.LOW_TOTAL_GAIN
            elif not result.promoted:
                stops[target] = StopReason.NO_PROMOTION

            for name in sorted(result.promoted):
                next_parents.setdefault(name, []).append(target)
                variable = result.classes[name]
                if variable.kind is LocalClass.PROXIMAL_UNIQUE:
                    broad = result.relevance.broad[name]
                    unique = result.relevance.unique[name]
                    edges.append(CoreEdge(
                        name, target, LocalClass.PROXIMAL_UNIQUE,
                        broad.raw, broad.normalized, unique.raw, unique.normalized,
                    ))

            for component in result.shared_components:
                if component.validated:
                    relations.append(SharedRelation(
                        target, component.members, component.raw, component.normalized, component,
                    ))
            for pair in result.retained_pairs:
                relations.append(SharedRelation(
                    target, pair.members, pair.pair.raw, pair.pair.normalized, pair,
                ))

        new_layer: list[str] = []
        for name in all_variables:
            if name in next_parents and name not in depths:
                depths[name] = depth + 1
                parents[name] = tuple(next_parents[name])
                new_layer.append(name)
        if depth + 1 >= config.analysis.max_depth:
            stops.update((target, StopReason.MAX_DEPTH) for target in new_layer)
            break
        layer = tuple(new_layer)

    return NetworkRun(
        root, depths, parents, results, tuple(edges), tuple(relations), stops,
        max_depth=config.analysis.max_depth,
    )


def _preflight_infeasible(n_samples: int, config: OpenConfig) -> bool:
    folds = config.performance_cv.folds
    if n_samples < folds:
        return True
    smallest_test_fold = ceil(n_samples / folds)
    smallest_training_partition = n_samples - smallest_test_fold
    return smallest_training_partition < config.tuning_cv.folds


def _validate_inputs(
    data: pd.DataFrame,
    root: str,
    all_variables: tuple[str, ...],
    config: OpenConfig,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(config, OpenConfig):
        raise TypeError("config must be an OpenConfig")
    if type(root) is not str or not root.strip():
        raise ValueError("root must be a nonempty variable name")
    if root != config.analysis.root:
        raise ValueError("root must match config.analysis.root")
    if not isinstance(all_variables, tuple) or not all_variables:
        raise ValueError("all_variables must be a nonempty tuple")
    if (any(type(name) is not str or not name.strip() for name in all_variables)
            or len(set(all_variables)) != len(all_variables)):
        raise ValueError("all_variables must contain distinct nonempty variable names")
    if all_variables.count(root) != 1:
        raise ValueError("root must occur exactly once in all_variables")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")
    missing = tuple(name for name in all_variables if name not in data.columns)
    if missing:
        raise ValueError(f"all_variables contains missing data columns: {missing}")
    for name in all_variables:
        series = data[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"variable {name!r} must be numeric scale-level data")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (name == root and np.isnan(values).any()):
            raise ValueError("root must be finite and candidates must not contain infinity")
