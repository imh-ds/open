"""Public entry point for exploring, confirming, plotting and saving networks.

Short names are provided for common workflows; the established descriptive
names remain available. Plotting and LightGBM dependencies are loaded only
when their features are used.
"""
from collections.abc import Sequence

from open.confirm import (
    SplitResult,
    confirm as confirm_network,
    confirmation_fit as evaluate_confirmation_fit,
    explore_and_confirm,
    split_sample,
)
from open.explore import ExploreConfig, ExploreResult, explore as explore_network
from open.explore_report import (
    GraphStyle,
    plot_outcomes,
    save_multi_outcome_analysis,
    plot_explore as plot_network,
    write_explore as save_exploration,
    write_split as save_split_analysis,
)
from open.multi_outcome import MultiOutcomeResult, explore_and_confirm_outcomes


# Concise names for the public API; the descriptive names above remain available.
explore = explore_network
confirm = confirm_network
evaluate_fit = evaluate_confirmation_fit
save_split = save_split_analysis
save_batch = save_multi_outcome_analysis


def analyze(
    data,
    *,
    outcome: str | None = None,
    outcomes: Sequence[str] | None = None,
    predictors=None,
    engine: str = "ols",
    config: ExploreConfig | None = None,
    explore_fraction: float = 0.5,
    split_seed: int = 0,
    confirm_engine: str | None = None,
    correction: str = "holm",
    alpha: float = 0.05,
    controls=None,
    covariates=None,
) -> SplitResult | MultiOutcomeResult:
    """Explore and confirm one or more outcomes with the matching workflow.

    Pass exactly one of ``outcome`` or ``outcomes``. A string outcome or a
    one-item sequence runs the single-outcome workflow; two or more names run
    the coordinated multi-outcome workflow.

    Args:
        data: Input observations as a pandas DataFrame.
        outcome: Name of the single outcome to analyze.
        outcomes: Sequence of outcome names. A one-item sequence is treated as
            a single-outcome analysis.
        predictors: Candidate predictors, or ``None`` to use eligible columns.
        engine: Exploratory learner name. Defaults to ``"ols"``.
        config: Exploration settings, including folds, recursion depth and seed.
        explore_fraction: Fraction of rows assigned to exploration.
        split_seed: Seed for the row split and workflow-level random streams.
        confirm_engine: Optional learner used for link confirmation.
        correction: Multiple-testing correction method, ``"holm"`` or ``"bh"``.
        alpha: Confirmation significance threshold.
        controls: Variables adjusted for in every model but not tested.
        covariates: Variables tested in models but never used as recursion targets.

    Returns:
        ``SplitResult`` for one outcome or ``MultiOutcomeResult`` for several.

    Use ``explore_and_confirm`` or ``explore_and_confirm_outcomes`` directly
    for workflow-specific options such as ``include_flagged`` or
    ``correction_scope``.
    """
    if (outcome is None) == (outcomes is None):
        raise ValueError("pass exactly one of 'outcome' or 'outcomes'")

    common = dict(
        predictors=predictors,
        engine=engine,
        config=config,
        explore_fraction=explore_fraction,
        split_seed=split_seed,
        confirm_engine=confirm_engine,
        correction=correction,
        alpha=alpha,
        controls=controls,
        covariates=covariates,
    )
    if outcome is not None:
        return explore_and_confirm(data, outcome=outcome, **common)
    if isinstance(outcomes, str):
        raise TypeError("'outcomes' must be a sequence of outcome names, not a string")
    if len(outcomes) == 1:
        return explore_and_confirm(data, outcome=outcomes[0], **common)
    return explore_and_confirm_outcomes(data, outcomes=outcomes, **common)


analyse = analyze

__all__ = (
    "ExploreConfig", "ExploreResult", "GraphStyle", "SplitResult",
    "explore", "confirm", "evaluate_fit", "analyze", "analyse",
    "save_split", "save_batch",
    "confirm_network", "evaluate_confirmation_fit", "explore_and_confirm",
    "explore_and_confirm_outcomes", "MultiOutcomeResult",
    "plot_outcomes", "save_multi_outcome_analysis",
    "explore_network", "plot_network", "save_exploration", "save_split_analysis",
    "split_sample",
)
