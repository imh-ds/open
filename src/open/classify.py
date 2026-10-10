"""Exact pair decisions from normalized broad, unique, and group scores."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from open.config import ThresholdConfig


class PairRelation(str, Enum):
    """Pair-level result used by the grouping and local-target stages."""

    PAIR_SHARED_REDUNDANT = "pair_shared_redundant"
    PARTNER_SCREENING = "partner_screening"
    BOTH_PROXIMAL = "both_proximal"
    GROUP_IMPORTANT = "group_important"
    NOT_GROUP_IMPORTANT = "not_group_important"
    UNDEFINED = "undefined"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class MemberClassification:
    """Individual unique status and any qualifying partner screens."""

    proximal_unique: bool | None
    screened_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairClassification:
    """Classification evidence for one pair, preserving undefined scores."""

    a: str
    b: str
    group_score: float | None
    group_important: bool | None
    relation: PairRelation
    screened_member: str | None = None
    screened_by: str | None = None


@dataclass(frozen=True)
class ClassificationResult:
    """Pair decisions and per-member unique/partner-screening summaries."""

    members: Mapping[str, MemberClassification]
    pairs: tuple[PairClassification, ...]


def _score_map(name: str, values: Mapping[str, float | None]) -> dict[str, float | None]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, float | None] = {}
    for variable, score in values.items():
        if type(variable) is not str or not variable:
            raise ValueError(f"{name} keys must be nonempty variable names")
        if score is not None and (
            type(score) not in (int, float) or not math.isfinite(score)
        ):
            raise ValueError(f"{name} scores must be finite numbers or None")
        result[variable] = float(score) if score is not None else None
    return result


def classify_pairs(
    broad: Mapping[str, float | None],
    unique: Mapping[str, float | None],
    pair_group: Mapping[tuple[str, str], float | None],
    thresholds: ThresholdConfig,
) -> ClassificationResult:
    """Apply the exact §13 pair rules without rounding normalized scores.

    ``None`` denotes an undefined normalized score. Undefined unique scores
    remain unknown instead of being treated as weak; an undefined pair group
    score cannot satisfy its threshold.
    """
    broad_scores = _score_map("broad", broad)
    unique_scores = _score_map("unique", unique)
    if broad_scores.keys() != unique_scores.keys():
        raise ValueError("broad and unique must contain the same variable names")
    if not isinstance(pair_group, Mapping):
        raise ValueError("pair_group must be a mapping")
    if not isinstance(thresholds, ThresholdConfig):
        raise ValueError("thresholds must be a ThresholdConfig")

    names = set(broad_scores)
    pairs: list[PairClassification] = []
    seen: set[frozenset[str]] = set()
    screened_by: dict[str, set[str]] = {name: set() for name in names}
    for pair, group_score in pair_group.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(type(name) is not str or not name for name in pair)
            or pair[0] == pair[1]
            or any(name not in names for name in pair)
        ):
            raise ValueError("pair_group keys must be pairs of distinct scored variable names")
        identity = frozenset(pair)
        if identity in seen:
            raise ValueError("pair_group cannot contain duplicate or reversed pairs")
        seen.add(identity)
        if group_score is not None and (
            type(group_score) not in (int, float) or not math.isfinite(group_score)
        ):
            raise ValueError("pair_group scores must be finite numbers or None")
        score = float(group_score) if group_score is not None else None

        if score is None:
            pairs.append(PairClassification(*pair, None, None, PairRelation.UNDEFINED))
            continue
        group_important = score >= thresholds.group_relative
        if not group_important:
            pairs.append(PairClassification(
                *pair, score, False, PairRelation.NOT_GROUP_IMPORTANT
            ))
            continue

        unique_a, unique_b = unique_scores[pair[0]], unique_scores[pair[1]]
        if unique_a is None or unique_b is None:
            pairs.append(PairClassification(
                *pair, score, True, PairRelation.UNDETERMINED
            ))
            continue
        strong_a = unique_a >= thresholds.unique_relative
        strong_b = unique_b >= thresholds.unique_relative
        if not strong_a and not strong_b:
            pairs.append(PairClassification(
                *pair, score, True, PairRelation.PAIR_SHARED_REDUNDANT
            ))
        elif strong_a and strong_b:
            pairs.append(PairClassification(
                *pair, score, True, PairRelation.BOTH_PROXIMAL
            ))
        else:
            strong_name, weak_name = pair if strong_a else (pair[1], pair[0])
            weak_broad = broad_scores[weak_name]
            if weak_broad is not None and weak_broad >= thresholds.broad_relative:
                screened_by[weak_name].add(strong_name)
                pairs.append(PairClassification(
                    *pair, score, True, PairRelation.PARTNER_SCREENING,
                    screened_member=weak_name, screened_by=strong_name,
                ))
            else:
                pairs.append(PairClassification(
                    *pair, score, True, PairRelation.GROUP_IMPORTANT
                ))

    members = {
        name: MemberClassification(
            proximal_unique=(
                None if unique_scores[name] is None
                else unique_scores[name] >= thresholds.unique_relative
            ),
            screened_by=tuple(sorted(screened_by[name])),
        )
        for name in broad_scores
    }
    return ClassificationResult(MappingProxyType(members), tuple(pairs))
