"""Public entry point for exploring, confirming, plotting and saving networks.

Workflow names are direct aliases: arguments, return types and behavior are
identical to the established module APIs, which remain supported. Plotting and
LightGBM dependencies are loaded only when their features are used.
"""
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
    plot_explore as plot_network,
    write_explore as save_exploration,
    write_split as save_split_analysis,
)

__all__ = (
    "ExploreConfig", "ExploreResult", "GraphStyle", "SplitResult",
    "confirm_network", "evaluate_confirmation_fit", "explore_and_confirm",
    "explore_network", "plot_network", "save_exploration", "save_split_analysis",
    "split_sample",
)
