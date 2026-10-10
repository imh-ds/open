"""Compose the supported continuous one-target OPEN analysis stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from open.classify import classify_pairs
from open.config import OpenConfig
from open.grouping import analyze_groups, evaluate_pair_ablations
from open.learners import make_predictor
from open.local_target import LocalResult, finalize_local_target
from open.relevance import compute_relevance
from open.risk import RiskResult, evaluate_risk
from open.splits import SplitPlan, observed_target_splits


def analyze_target(
    data: pd.DataFrame,
    target: str,
    candidates: tuple[str, ...],
    config: OpenConfig,
    split_plan: SplitPlan,
) -> LocalResult:
    """Analyze one target on the caller's shared positional split plan.

    Broad models tune independently; LOCO, pair, and group ablations reuse
    each full model's selected parameters. Signed contrasts and selected-fold
    pair evidence are preserved by the existing LocalResult contract. Missing
    recursive outcomes are excluded within the original folds, never imputed.
    """
    _validate_inputs(data, target, candidates, config, split_plan)
    predictor = make_predictor(config)
    common = dict(target=target, candidate_set=candidates, split_plan=split_plan,
                  predictor=predictor, tuning_folds=config.tuning_cv.folds)
    baseline = evaluate_risk(data, predictors=(), **common)
    full = evaluate_risk(data, predictors=candidates, **common)
    risk_results = [baseline, full]

    def record_risk(result: RiskResult) -> float:
        risk_results.append(result)
        return result.aggregate_risk

    def fit_counts() -> dict[str, int]:
        return dict(
            outer_model_fit_count=sum(result.outer_model_fit_count for result in risk_results),
            tuning_model_fit_count=sum(result.tuning_model_fit_count for result in risk_results),
            n_observed_target_rows=int(data[target].notna().sum()),
            n_missing_target_rows=int(data[target].isna().sum()),
        )

    parameters = {(fold.repeat_index, fold.fold_index): fold.fitted_parameters
                  for fold in full.folds}
    broad_risks = {
        name: record_risk(evaluate_risk(data, predictors=(name,), **common))
        for name in candidates
    }
    ablated_risks = {
        name: record_risk(evaluate_risk(
            data, predictors=tuple(other for other in candidates if other != name),
            fitted_parameters=parameters, **common,
        ))
        for name in candidates
    }
    relevance = compute_relevance(
        baseline.aggregate_risk, full.aggregate_risk, broad_risks=broad_risks,
        ablated_risks=ablated_risks, tau_gain=config.thresholds.min_total_gain_relative,
    )
    if relevance.stop_reason is not None:
        return finalize_local_target(
            target=target, candidates=candidates, relevance=relevance,
            classifications=None, grouping=None, thresholds=config.thresholds,
            **fit_counts(),
        )

    pairs = evaluate_pair_ablations(
        data, candidate_set=candidates, split_plan=split_plan, full=full,
        baseline=baseline, predictor=predictor, tau_dep=config.thresholds.dependence_pair,
    )
    risk_results.extend(pair.ablation for pair in pairs.pairs)
    classifications = classify_pairs(
        {name: relevance.broad[name].normalized for name in candidates},
        {name: relevance.unique[name].normalized for name in candidates},
        {(pair.a, pair.b): pair.normalized for pair in pairs.pairs}, config.thresholds,
    )
    grouping = analyze_groups(
        data, target=target, candidate_set=candidates, split_plan=split_plan,
        full=full, baseline=baseline, pair_ablations=pairs,
        classifications=classifications, relevance=relevance, predictor=predictor,
        thresholds=config.thresholds,
    )
    risk_results.extend(component.ablation for component in grouping.shared_components)
    return finalize_local_target(
        target=target, candidates=candidates, relevance=relevance,
        classifications=classifications, grouping=grouping,
        thresholds=config.thresholds, pair_ablations=pairs,
        **fit_counts(),
    )


def _validate_inputs(
    data: pd.DataFrame, target: str, candidates: tuple[str, ...],
    config: OpenConfig, split_plan: SplitPlan,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(config, OpenConfig):
        raise TypeError("config must be an OpenConfig")
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    if type(target) is not str or not target.strip():
        raise ValueError("target must be a nonempty variable name")
    if (not isinstance(candidates, tuple) or not candidates
            or any(type(name) is not str or not name.strip() for name in candidates)
            or len(set(candidates)) != len(candidates) or target in candidates):
        raise ValueError("candidates must be distinct nonempty names excluding target")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")
    if any(name not in data for name in (target, *candidates)):
        raise ValueError("target and candidates must name existing data columns")
    if len(data) != split_plan.n_samples:
        raise ValueError("data row count does not match split_plan")
    if (split_plan.random_seed != config.analysis.random_seed
            or split_plan.folds != config.performance_cv.folds
            or split_plan.repeats != config.performance_cv.repeats):
        raise ValueError("split_plan seed, folds, and repeats must match config")
    for name in (target, *candidates):
        series = data[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"variable {name!r} must be numeric scale-level data")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (
            name == target == config.analysis.root and np.isnan(values).any()
        ):
            raise ValueError("target must be finite and candidates must not contain infinity")
    if any(config.tuning_cv.folds > len(split.train_indices) for split in split_plan.splits):
        raise ValueError("tuning_cv.folds exceeds an outer training row count")
    if config.tuning_cv.folds < 2:
        raise ValueError("tuning_cv.folds must be at least 2")
    observed_target_splits(
        split_plan, data[target].to_numpy(dtype=float, na_value=np.nan),
        min_training_rows=config.tuning_cv.folds,
    )
