"""Validated configuration for the first supported OPEN analysis path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


def _choice(name: str, value: Any, expected: Any) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r} in the first supported path")


def _positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _unit_interval(name: str, value: Any, *, include_zero: bool = True) -> None:
    if type(value) not in (int, float) or not (0 <= value <= 1):
        raise ValueError(f"{name} must be a number in [0, 1]")
    if not include_zero and value == 0:
        raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class AnalysisConfig:
    root: str
    random_seed: int
    max_depth: int = 3

    def __post_init__(self) -> None:
        if type(self.root) is not str or not self.root.strip():
            raise ValueError("analysis.root must name exactly one root")
        if type(self.random_seed) is not int:
            raise ValueError("analysis.random_seed must be an explicit integer")
        _positive_int("analysis.max_depth", self.max_depth)


@dataclass(frozen=True)
class OutcomeConfig:
    type: str = "continuous"
    loss: str = "mse"

    def __post_init__(self) -> None:
        _choice("outcome.type", self.type, "continuous")
        _choice("outcome.loss", self.loss, "mse")


@dataclass(frozen=True)
class LearnerConfig:
    mode: str = "fixed"
    library: tuple[str, ...] = ("elastic_net",)
    retune_ablation: bool = False

    def __post_init__(self) -> None:
        _choice("learner.mode", self.mode, "fixed")
        if (not isinstance(self.library, tuple) or len(self.library) != 1
                or self.library[0] not in ("elastic_net", "random_forest", "gbm", "lightgbm")):
            raise ValueError("learner.library must contain exactly one supported engine: elastic_net, random_forest, gbm, lightgbm")
        _choice("learner.retune_ablation", self.retune_ablation, False)


@dataclass(frozen=True)
class PreprocessingConfig:
    missing: str = "median"
    standardize_linear: bool = True
    one_hot_categorical: bool = False

    def __post_init__(self) -> None:
        _choice("preprocessing.missing", self.missing, "median")
        _choice("preprocessing.standardize_linear", self.standardize_linear, True)
        _choice("preprocessing.one_hot_categorical", self.one_hot_categorical, False)


@dataclass(frozen=True)
class DependenceConfig:
    method: str = "spearman"
    absolute: bool = True

    def __post_init__(self) -> None:
        _choice("dependence.method", self.method, "spearman")
        _choice("dependence.absolute", self.absolute, True)


@dataclass(frozen=True)
class ThresholdConfig:
    min_total_gain_relative: float = 0.02
    broad_relative: float = 0.05
    unique_relative: float = 0.05
    group_relative: float = 0.10
    dependence_pair: float = 0.70
    stability: float = 0.70

    def __post_init__(self) -> None:
        for field in fields(self):
            _unit_interval(f"thresholds.{field.name}", getattr(self, field.name))


@dataclass(frozen=True)
class PerformanceCVConfig:
    folds: int = 5
    repeats: int = 2
    grouped: bool = False

    def __post_init__(self) -> None:
        _positive_int("performance_cv.folds", self.folds)
        if self.folds < 2:
            raise ValueError("performance_cv.folds must be at least 2")
        _positive_int("performance_cv.repeats", self.repeats)
        _choice("performance_cv.grouped", self.grouped, False)


@dataclass(frozen=True)
class TuningCVConfig:
    folds: int = 3

    def __post_init__(self) -> None:
        _positive_int("tuning_cv.folds", self.folds)
        if self.folds < 2:
            raise ValueError("tuning_cv.folds must be at least 2")


@dataclass(frozen=True)
class StabilityConfig:
    method: str = "subsample"
    fraction: float = 0.80
    n_resamples: int = 100
    final_resamples: int = 500
    scope: str = "whole_pipeline"

    def __post_init__(self) -> None:
        if type(self.scope) is not str or self.scope not in ("whole_pipeline", "layer_consensus"):
            raise ValueError("stability.scope must be whole_pipeline or layer_consensus")
        _choice("stability.method", self.method, "subsample")
        _unit_interval("stability.fraction", self.fraction, include_zero=False)
        _positive_int("stability.n_resamples", self.n_resamples)
        _positive_int("stability.final_resamples", self.final_resamples)


@dataclass(frozen=True)
class VisualizationConfig:
    show_screened: bool = True
    show_unstable: bool = True
    distance_mode: str = "visual_strength"
    edge_width: str = "unique"
    edge_opacity: str = "stability"

    def __post_init__(self) -> None:
        _choice("visualization.show_screened", self.show_screened, True)
        if type(self.show_unstable) is not bool:
            raise ValueError("visualization.show_unstable must be boolean")
        _choice("visualization.distance_mode", self.distance_mode, "visual_strength")
        _choice("visualization.edge_width", self.edge_width, "unique")
        _choice("visualization.edge_opacity", self.edge_opacity, "stability")


@dataclass(frozen=True)
class OpenConfig:
    analysis: AnalysisConfig
    outcome: OutcomeConfig = OutcomeConfig()
    learner: LearnerConfig = LearnerConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    dependence: DependenceConfig = DependenceConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    performance_cv: PerformanceCVConfig = PerformanceCVConfig()
    tuning_cv: TuningCVConfig = TuningCVConfig()
    stability: StabilityConfig = StabilityConfig()
    visualization: VisualizationConfig = VisualizationConfig()


_SECTIONS = {
    "analysis": AnalysisConfig,
    "outcome": OutcomeConfig,
    "learner": LearnerConfig,
    "preprocessing": PreprocessingConfig,
    "dependence": DependenceConfig,
    "thresholds": ThresholdConfig,
    "performance_cv": PerformanceCVConfig,
    "tuning_cv": TuningCVConfig,
    "stability": StabilityConfig,
    "visualization": VisualizationConfig,
}


def parse_config(data: Mapping[str, Any]) -> OpenConfig:
    """Parse the section names in spec section 40, rejecting deferred options."""
    if not isinstance(data, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    if "analysis" not in data:
        raise ValueError("analysis section is required with root and random_seed")
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        values = data.get(name, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping")
        extra = set(values) - {field.name for field in fields(cls)}
        if extra:
            raise ValueError(f"unknown {name} fields: {sorted(extra)}")
        values = dict(values)
        if name == "learner" and "library" in values:
            library = values["library"]
            if not isinstance(library, list) or any(type(item) is not str for item in library):
                raise ValueError("learner.library must be a list of learner names")
            values["library"] = tuple(library)
        try:
            sections[name] = cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid {name} configuration: {exc}") from exc
    return OpenConfig(**sections)


def load_config(path: str | Path) -> OpenConfig:
    """Read a YAML file with PyYAML's safe loader and validate it."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return parse_config(yaml.safe_load(stream))


def dump_config(config: OpenConfig, path: str | Path) -> None:
    """Write a validated configuration as portable YAML."""
    if not isinstance(config, OpenConfig):
        raise TypeError("config must be an OpenConfig")
    data = asdict(config)
    data["learner"]["library"] = list(config.learner.library)
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
