"""Held-out risk for continuous targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from open.learners.base import Predictor
from open.splits import SplitPlan, observed_target_splits


@dataclass(frozen=True)
class ObservationLoss:
    row_position: int
    observed: float
    predicted: float
    loss: float


@dataclass(frozen=True)
class FoldRisk:
    repeat_index: int
    fold_index: int
    predictors: tuple[str, ...]
    observations: tuple[ObservationLoss, ...]
    risk: float
    fitted_parameters: dict[str, object] | None


@dataclass(frozen=True)
class RiskResult:
    target: str
    candidate_set: tuple[str, ...]
    predictors: tuple[str, ...]
    learner: str | None
    loss: str
    split_plan_id: str
    split_seed: int
    folds: tuple[FoldRisk, ...]
    aggregate_risk: float
    outer_model_fit_count: int
    tuning_model_fit_count: int


def evaluate_risk(
    data: pd.DataFrame,
    *,
    target: str,
    candidate_set: tuple[str, ...],
    predictors: tuple[str, ...],
    split_plan: SplitPlan,
    predictor: Predictor | None = None,
    tuning_folds: int = 3,
    fitted_parameters: dict[tuple[int, int], dict[str, object]] | None = None,
    selected_folds: tuple[tuple[int, int], ...] | None = None,
) -> RiskResult:
    """Evaluate a model or the training-mean baseline on recorded outer folds."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    if len(data) != split_plan.n_samples:
        raise ValueError("data row count does not match split_plan")
    if target not in data:
        raise ValueError(f"unknown target {target!r}")
    if any(name not in data for name in candidate_set):
        raise ValueError("candidate_set contains an unknown column")
    if len(set(candidate_set)) != len(candidate_set):
        raise ValueError("candidate_set must not contain duplicates")
    if target in candidate_set or not set(predictors).issubset(candidate_set):
        raise ValueError("predictors must be a subset of candidate_set and exclude target")
    if len(set(predictors)) != len(predictors):
        raise ValueError("predictors must not contain duplicates")
    if predictor is None and predictors:
        raise ValueError("a predictor is required for a nonempty predictor set")
    if predictors and not isinstance(predictor, Predictor):
        raise TypeError("risk evaluation requires a Predictor")
    y = data[target].to_numpy(dtype=float, na_value=np.nan)
    selected = set(selected_folds) if selected_folds is not None else None
    planned = {(split.repeat, split.fold) for split in split_plan.splits}
    if selected is not None and (not selected or len(selected) != len(selected_folds) or not selected.issubset(planned)):
        raise ValueError("selected_folds must identify distinct planned folds")

    baseline = not predictors
    target_splits = observed_target_splits(
        split_plan, y,
        min_training_rows=tuning_folds if predictors and fitted_parameters is None else 1,
        selected_folds=selected,
    )
    outer_fits = 0
    tuning_fits = 0
    folds: list[FoldRisk] = []
    for split in target_splits:
        params: dict[str, object] | None = None
        if baseline:
            predicted = np.repeat(float(np.mean(y[list(split.train_indices)])), len(split.test_indices))
        else:
            assert predictor is not None
            model = predictor.clone()
            train_rows = list(split.train_indices)
            test_rows = list(split.test_indices)
            X_train = data.iloc[train_rows][list(predictors)]
            y_train = y[train_rows]
            key = (split.repeat, split.fold)
            if fitted_parameters is None:
                if type(tuning_folds) is not int or not 2 <= tuning_folds <= len(train_rows):
                    raise ValueError("tuning_folds must be between 2 and the outer training row count")
                inner_cv = KFold(n_splits=tuning_folds, shuffle=True, random_state=split_plan.random_seed)
                params = dict(model.tune(X_train, y_train, inner_cv, "mse"))
                tuning_fits += model.tuning_grid_size * tuning_folds
            else:
                if key not in fitted_parameters:
                    raise ValueError(f"missing fitted parameters for fold {key}")
                params = dict(fitted_parameters[key])
            model.fit(X_train, y_train, params)
            outer_fits += 1
            predicted = np.asarray(model.predict(data.iloc[test_rows][list(predictors)]), dtype=float)
            if predicted.shape != (len(test_rows),) or not np.isfinite(predicted).all():
                raise ValueError("predictor returned invalid held-out predictions")
        observations = tuple(
            ObservationLoss(int(row), float(y[row]), float(prediction), float((y[row] - prediction) ** 2))
            for row, prediction in zip(split.test_indices, predicted)
        )
        folds.append(FoldRisk(
            split.repeat, split.fold, tuple(predictors) if not baseline else (), observations,
            float(np.mean([item.loss for item in observations])), params,
        ))

    losses = [observation.loss for fold in folds for observation in fold.observations]
    split_id = hashlib.sha256(split_plan.to_json(indent=None).encode("utf-8")).hexdigest()
    return RiskResult(
        target=target,
        candidate_set=tuple(candidate_set),
        predictors=tuple(predictors) if not baseline else (),
        learner=predictor.engine_name if not baseline else None,
        loss="mse",
        split_plan_id=split_id,
        split_seed=split_plan.random_seed,
        folds=tuple(folds),
        aggregate_risk=float(np.mean(losses)),
        outer_model_fit_count=outer_fits,
        tuning_model_fit_count=tuning_fits,
    )
