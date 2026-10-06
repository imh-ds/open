"""Versioned, reproducible positional cross-validation split plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import RepeatedKFold

from open.config import OpenConfig


SPLIT_PLAN_VERSION = 1
_SPLIT_PLAN_SCHEMA = "open.split_plan"


def _is_int(value: Any) -> bool:
    return type(value) is int


@dataclass(frozen=True)
class SplitFold:
    """One realized train/test partition, using zero-based row positions."""

    repeat: int
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_int(self.repeat) or self.repeat < 0:
            raise ValueError("split repeat must be a non-negative integer")
        if not _is_int(self.fold) or self.fold < 0:
            raise ValueError("split fold must be a non-negative integer")
        for label, indices in (("train", self.train_indices), ("test", self.test_indices)):
            if not isinstance(indices, tuple) or any(not _is_int(i) for i in indices):
                raise ValueError(f"{label}_indices must be a tuple of integers")
            if not indices:
                raise ValueError(f"{label} partition must be nonempty")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{label}_indices must not contain duplicates")
        if set(self.train_indices).intersection(self.test_indices):
            raise ValueError("train and test indices must be disjoint")


@dataclass(frozen=True)
class SplitPlan:
    """A persisted set of realized repeated K-fold splits for one row order."""

    n_samples: int
    folds: int
    repeats: int
    random_seed: int
    splits: tuple[SplitFold, ...]
    version: int = SPLIT_PLAN_VERSION
    scheme: str = "repeated_k_fold"

    def __post_init__(self) -> None:
        if not _is_int(self.version) or self.version != SPLIT_PLAN_VERSION:
            raise ValueError(f"unsupported split plan version {self.version!r}")
        if self.scheme != "repeated_k_fold":
            raise ValueError(f"unsupported split scheme {self.scheme!r}")
        if not _is_int(self.n_samples) or self.n_samples < 2:
            raise ValueError("n_samples must be an integer of at least 2")
        if not _is_int(self.folds) or self.folds < 2:
            raise ValueError("folds must be an integer of at least 2")
        if self.folds > self.n_samples:
            raise ValueError("folds cannot exceed n_samples")
        if not _is_int(self.repeats) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not _is_int(self.random_seed):
            raise ValueError("random_seed must be an integer")
        if not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be in the scikit-learn range [0, 2**32 - 1]")
        if not isinstance(self.splits, tuple) or any(not isinstance(split, SplitFold) for split in self.splits):
            raise ValueError("splits must be a tuple of SplitFold values")
        if len(self.splits) != self.folds * self.repeats:
            raise ValueError("split count must equal folds multiplied by repeats")

        seen: set[tuple[int, int]] = set()
        test_counts = [[0] * self.n_samples for _ in range(self.repeats)]
        for split in self.splits:
            key = (split.repeat, split.fold)
            if split.repeat >= self.repeats or split.fold >= self.folds:
                raise ValueError("split repeat/fold metadata is out of bounds")
            if key in seen:
                raise ValueError("split repeat/fold metadata must be unique")
            seen.add(key)
            for label, indices in (("train", split.train_indices), ("test", split.test_indices)):
                if any(index < 0 or index >= self.n_samples for index in indices):
                    raise ValueError(f"{label} index is out of bounds for n_samples={self.n_samples}")
            if set(split.train_indices).union(split.test_indices) != set(range(self.n_samples)):
                raise ValueError("each train/test split must partition all sample positions")
            for index in split.test_indices:
                test_counts[split.repeat][index] += 1
        expected = {(repeat, fold) for repeat in range(self.repeats) for fold in range(self.folds)}
        if seen != expected:
            raise ValueError("split plan must contain every repeat/fold pair")
        for counts in test_counts:
            if any(count != 1 for count in counts):
                raise ValueError("test folds must cover every row exactly once per repeat")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with all realized positions."""
        return {
            "schema": _SPLIT_PLAN_SCHEMA,
            "version": self.version,
            "scheme": self.scheme,
            "n_samples": self.n_samples,
            "folds": self.folds,
            "repeats": self.repeats,
            "random_seed": self.random_seed,
            "index_semantics": "zero-based positional indices in the original input row order",
            "splits": [
                {
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "train_indices": list(split.train_indices),
                    "test_indices": list(split.test_indices),
                }
                for split in self.splits
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete split plan as JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, value: Any) -> "SplitPlan":
        """Parse and validate a JSON-compatible split plan mapping."""
        if not isinstance(value, dict):
            raise ValueError("split plan must be a JSON object")
        required = {
            "schema", "version", "scheme", "n_samples", "folds", "repeats",
            "random_seed", "index_semantics", "splits",
        }
        if set(value) != required:
            missing, extra = required - set(value), set(value) - required
            raise ValueError(f"invalid split plan fields (missing={sorted(missing)}, unknown={sorted(extra)})")
        if value["schema"] != _SPLIT_PLAN_SCHEMA:
            raise ValueError(f"unsupported split plan schema {value['schema']!r}")
        if not _is_int(value["version"]) or value["version"] != SPLIT_PLAN_VERSION:
            raise ValueError(f"unsupported split plan version {value['version']!r}")
        if value["index_semantics"] != "zero-based positional indices in the original input row order":
            raise ValueError("unsupported split plan index semantics")
        if not isinstance(value["splits"], list):
            raise ValueError("splits must be a JSON array")
        parsed_splits: list[SplitFold] = []
        for index, split in enumerate(value["splits"]):
            if not isinstance(split, dict) or set(split) != {"repeat", "fold", "train_indices", "test_indices"}:
                raise ValueError(f"split {index} has invalid fields")
            for field in ("train_indices", "test_indices"):
                if not isinstance(split[field], list):
                    raise ValueError(f"split {index} {field} must be a JSON array")
            parsed_splits.append(
                SplitFold(
                    repeat=split["repeat"],
                    fold=split["fold"],
                    train_indices=tuple(split["train_indices"]),
                    test_indices=tuple(split["test_indices"]),
                )
            )
        return cls(
            n_samples=value["n_samples"],
            folds=value["folds"],
            repeats=value["repeats"],
            random_seed=value["random_seed"],
            splits=tuple(parsed_splits),
            version=value["version"],
            scheme=value["scheme"],
        )

    @classmethod
    def from_json(cls, value: str) -> "SplitPlan":
        """Deserialize and validate a JSON split plan."""
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid split plan JSON: {exc}") from exc
        return cls.from_dict(parsed)

    def save(self, path: str | Path) -> None:
        """Write the plan as UTF-8 JSON."""
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SplitPlan":
        """Read and validate a persisted JSON split plan."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def as_cv(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Return immutable-in-practice index arrays for estimator CV APIs."""
        return tuple(
            (np.asarray(split.train_indices, dtype=int), np.asarray(split.test_indices, dtype=int))
            for split in self.splits
        )


def _row_count(data: Any) -> int:
    try:
        rows = len(data)
    except TypeError as exc:
        raise ValueError("data must provide a finite row count") from exc
    if not _is_int(rows) or rows < 0:
        raise ValueError("data must provide a finite row count")
    return rows


def make_split_plan(data: Any, config: OpenConfig, *, groups: Iterable[Any] | None = None) -> SplitPlan:
    """Realize deterministic repeated K-fold splits for independent rows.

    ``groups`` exists to make an unsupported grouped/repeated-measures request
    fail explicitly instead of silently treating dependent rows as independent.
    The returned indices are positional and apply to ``data`` in its current
    row order.
    """
    if not isinstance(config, OpenConfig):
        raise TypeError("config must be an OpenConfig")
    if groups is not None or config.performance_cv.grouped:
        raise ValueError("grouped or repeated-measures splits are unsupported in the first path")
    n_samples = _row_count(data)
    folds = config.performance_cv.folds
    repeats = config.performance_cv.repeats
    seed = config.analysis.random_seed
    if n_samples < 2:
        raise ValueError("at least 2 rows are required for cross-validation")
    if folds > n_samples:
        raise ValueError(f"performance_cv.folds={folds} exceeds the {n_samples} available rows")
    if not 0 <= seed < 2**32:
        raise ValueError("analysis.random_seed must be in the scikit-learn range [0, 2**32 - 1]")

    splitter = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    positions = np.arange(n_samples).reshape(-1, 1)
    realized: list[SplitFold] = []
    for split_index, (train, test) in enumerate(splitter.split(positions)):
        realized.append(
            SplitFold(
                repeat=split_index // folds,
                fold=split_index % folds,
                train_indices=tuple(int(i) for i in train),
                test_indices=tuple(int(i) for i in test),
            )
        )
    return SplitPlan(
        n_samples=n_samples,
        folds=folds,
        repeats=repeats,
        random_seed=seed,
        splits=tuple(realized),
    )
