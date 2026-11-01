"""Validated preprocessing settings shared by every engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _choice(name: str, value: Any, expected: Any) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} must be {expected!r} in the supported path")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fold-local median imputation and standardization of numeric predictors."""

    missing: str = "median"
    standardize_linear: bool = True
    one_hot_categorical: bool = False

    def __post_init__(self) -> None:
        _choice("preprocessing.missing", self.missing, "median")
        _choice("preprocessing.standardize_linear", self.standardize_linear, True)
        _choice("preprocessing.one_hot_categorical", self.one_hot_categorical, False)
