"""Coordinated split, exploration, and confirmation for several outcomes."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from open.confirm import (
    SplitResult,
    _adjust,
    _confirmation_fit_with_shared_folds,
    _confirm_with_shared_folds,
    split_sample,
)
from open.explore import (ExploreConfig, _assessment_data_context, _explore_with_shared_context,
                          _fold_sets, _validate_model_columns)


class MultiOutcomeResult:
    """Separate confirmed networks and aggregate tables for multiple outcomes."""

    def __init__(self, by_outcome: dict[str, SplitResult], confirmation: pd.DataFrame,
                 fit: pd.DataFrame, settings: dict):
        self.by_outcome = dict(by_outcome)
        self.confirmation = confirmation.reset_index(drop=True)
        self.fit = fit.reset_index(drop=True)
        self.settings = dict(settings)

    def summary(self) -> str:
        s = self.settings
        lines = [
            f"Multi-outcome split (seed {s['split_seed']}): explored on {s['n_explore']} rows, "
            f"confirmed on {s['n_confirm']} rows; {s['correction']} correction, "
            f"scope {s['correction_scope']}, alpha {s['alpha']}.",
        ]
        for outcome, result in self.by_outcome.items():
            lines.extend((f"\nOutcome {outcome}:", result.summary()))
        return "\n".join(lines)


def _name_sequence(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of distinct column names")
    try:
        names = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of distinct column names") from exc
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError(f"{name} must contain nonempty string column names")
    if not allow_empty and not names:
        raise ValueError(f"{name} must include at least one column name")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must contain distinct column names")
    return names


def _label_table(table: pd.DataFrame, outcome: str) -> pd.DataFrame:
    labeled = table.copy().reset_index(drop=True)
    if "root_outcome" in labeled:
        labeled["root_outcome"] = outcome
    else:
        labeled.insert(0, "root_outcome", outcome)
    return labeled


def _apply_batch_correction(tables: dict[str, pd.DataFrame], correction: str,
                            correction_scope: str, alpha: float) -> None:
    if correction_scope == "all_outcomes":
        p = np.concatenate([
            table.loc[table.p.notna(), "p"].to_numpy(dtype=float)
            for table in tables.values()
        ]) if any(table.p.notna().any() for table in tables.values()) else np.array([], dtype=float)
        adjusted = _adjust(p, correction) if len(p) else p
        cursor = 0
        for table in tables.values():
            tested = table.p.notna()
            count = int(tested.sum())
            table["p_adjusted"] = np.nan
            if count:
                table.loc[tested, "p_adjusted"] = adjusted[cursor:cursor + count]
                cursor += count
    else:
        for table in tables.values():
            tested = table.p.notna()
            table["p_adjusted"] = np.nan
            if tested.any():
                table.loc[tested, "p_adjusted"] = _adjust(
                    table.loc[tested, "p"].to_numpy(dtype=float), correction
                )
    for table in tables.values():
        table["confirmed"] = table.p_adjusted < alpha
        table.attrs.update(correction=correction, correction_scope=correction_scope,
                           alpha=alpha, links_tested=int(table.p.notna().sum()))


def explore_and_confirm_outcomes(
    data: pd.DataFrame,
    outcomes: Sequence[str],
    predictors: Sequence[str] | None = None,
    engine: str = "ols",
    config: ExploreConfig | None = None,
    explore_fraction: float = 0.5,
    split_seed: int = 0,
    confirm_engine: str | None = None,
    correction: Literal["holm", "bh"] = "holm",
    correction_scope: Literal["all_outcomes", "per_outcome"] = "all_outcomes",
    alpha: float = 0.05,
    controls: Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
) -> MultiOutcomeResult:
    """Explore and confirm independent predictive networks for several outcomes.

    One row split and one exploration configuration are shared across outcomes.
    Each outcome keeps its own recursion and network. By default, confirmation
    p-values are adjusted across all outcomes; ``per_outcome`` limits adjustment
    to the links for each outcome.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    outcomes = _name_sequence(outcomes, "outcomes")
    unknown = [outcome for outcome in outcomes if outcome not in data.columns]
    if unknown:
        raise ValueError(f"unknown outcomes: {unknown}")
    for outcome in outcomes:
        values = data[outcome]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"outcome {outcome!r} must be numeric")
        if values.isna().any():
            raise ValueError(f"outcome {outcome!r} must have no missing values")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"outcome {outcome!r} contains infinity")

    controls = _name_sequence(controls, "controls", allow_empty=True) if controls is not None else ()
    covariates = _name_sequence(covariates, "covariates", allow_empty=True) if covariates is not None else ()
    if set(outcomes) & (set(controls) | set(covariates)):
        raise ValueError("selected outcomes cannot be controls or covariates")

    if predictors is None:
        predictor_names = tuple(
            name for name in data.columns
            if name not in set(outcomes) | set(controls) | set(covariates)
        )
    else:
        predictor_names = _name_sequence(predictors, "predictors", allow_empty=True)
        overlap = set(outcomes) & set(predictor_names)
        if overlap:
            raise ValueError(f"selected outcomes cannot also be predictors: {sorted(overlap)}")

    if correction not in ("holm", "bh"):
        raise ValueError("correction must be 'holm' or 'bh'")
    if correction_scope not in ("all_outcomes", "per_outcome"):
        raise ValueError("correction_scope must be 'all_outcomes' or 'per_outcome'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    explore_config = ExploreConfig() if config is None else config
    context_columns = tuple(dict.fromkeys((*outcomes, *predictor_names, *controls, *covariates)))
    _validate_model_columns(data, context_columns)
    if len(data) < 1000:
        warnings.warn(
            f"N = {len(data)}: splitting leaves small parts; with under about 1,000 rows, "
            "exploring on all data and confirming in a new sample keeps more power.",
            stacklevel=2,
        )

    # Positional row identities stay unique even if the caller's index is duplicated.
    rows = data.reset_index(drop=True)
    explore_part, confirm_part = split_sample(rows, explore_fraction, split_seed)
    fold_ids = _fold_sets(len(explore_part), explore_config)
    data_context = _assessment_data_context(explore_part, context_columns)
    assessment_cache = {}
    by_outcome: dict[str, SplitResult] = {}
    confirmation_tables: dict[str, pd.DataFrame] = {}
    fit_tables = []
    used_confirm_engines = []

    for outcome in outcomes:
        exploration = _explore_with_shared_context(
            explore_part, outcome, predictor_names, engine=engine, config=explore_config,
            controls=controls, covariates=covariates,
            fold_ids=fold_ids, assessment_cache=assessment_cache, data_context=data_context,
        )
        confirmation = _confirm_with_shared_folds(
            exploration, confirm_part, engine=confirm_engine, correction=correction, alpha=alpha,
            fold_seed=split_seed,
        )
        used = confirmation.engine.iloc[0] if len(confirmation) else (
            confirm_engine or ("lightgbm" if engine == "lightgbm" else "ols")
        )
        used_confirm_engines.append(used)
        fit = _confirmation_fit_with_shared_folds(
            exploration, confirm_part, seed=split_seed, fold_seed=split_seed,
        )
        settings = dict(
            split_seed=split_seed,
            explore_fraction=explore_fraction,
            n_explore=len(explore_part),
            n_confirm=len(confirm_part),
            explore_engine=engine,
            confirm_engine=used,
            include_flagged=False,
            correction=correction,
            correction_scope=correction_scope,
            alpha=alpha,
            outcome=outcome,
            outcomes=outcomes,
            predictors=predictor_names,
            controls=controls,
            covariates=covariates,
            explore_config=asdict(explore_config),
        )
        labeled_confirmation = _label_table(confirmation, outcome)
        labeled_fit = _label_table(fit, outcome)
        by_outcome[outcome] = SplitResult(exploration, labeled_confirmation, settings, labeled_fit)
        confirmation_tables[outcome] = labeled_confirmation
        fit_tables.append(labeled_fit)

    _apply_batch_correction(confirmation_tables, correction, correction_scope, alpha)
    for outcome, table in confirmation_tables.items():
        by_outcome[outcome].confirmation = table
        by_outcome[outcome].settings["correction_scope"] = correction_scope

    combined_confirmation = pd.concat(confirmation_tables.values(), ignore_index=True, sort=False)
    combined_confirmation.attrs.update(
        correction=correction, correction_scope=correction_scope, alpha=alpha,
        links_tested=int(combined_confirmation.p.notna().sum()),
    )
    combined_fit = pd.concat(fit_tables, ignore_index=True, sort=False)
    settings = dict(
        outcomes=outcomes,
        predictors=predictor_names,
        split_seed=split_seed,
        explore_fraction=explore_fraction,
        n_explore=len(explore_part),
        n_confirm=len(confirm_part),
        explore_engine=engine,
        confirm_engine=used_confirm_engines[0],
        correction=correction,
        correction_scope=correction_scope,
        alpha=alpha,
        controls=controls,
        covariates=covariates,
        explore_config=asdict(explore_config),
    )
    return MultiOutcomeResult(by_outcome, combined_confirmation, combined_fit, settings)
