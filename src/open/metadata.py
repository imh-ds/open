"""Validate metadata for OPEN's cross-sectional numeric scale-level input path.

Call ``validate_metadata(columns, metadata, root, *, mode='cross_sectional',
prospective=False)`` with an ordered sequence of column names and one metadata
mapping per variable. The result contains the root name, ordered predictor names,
and normalized metadata records. This module does not fit models or split data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MetadataValidation:
    """Validated root, candidates, and normalized per-variable metadata."""

    root: str
    candidates: tuple[str, ...]
    variables: tuple[dict[str, Any], ...]


def validate_metadata(
    columns: Iterable[str],
    metadata: Iterable[Mapping[str, Any]],
    root: str,
    *,
    mode: str = "cross_sectional",
    prospective: bool = False,
) -> MetadataValidation:
    """Validate columns and metadata for cross-sectional continuous-target use.

    Metadata fields follow spec section 3: name, role, type, construct,
    scale_items, time_index, exclude_reason, and theory_group. Numeric
    continuous and ordinal predictors are eligible. ``display_label`` is
    retained when provided. ``measurement_overlap`` is added to every record
    participating in any shared scale-item overlap.
    """
    names = list(columns)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate columns are not supported: {duplicates}")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("root must be a non-empty column name")
    if mode in {"item", "item_level", "item-level"}:
        raise ValueError("item-level analysis is unsupported; provide scale-level data")
    if mode != "cross_sectional":
        raise ValueError(f"unsupported analysis mode {mode!r}; expected 'cross_sectional'")

    records: dict[str, dict[str, Any]] = {}
    for item in metadata:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError("each metadata record must include a string 'name'")
        name = item["name"]
        if name in records:
            raise ValueError(f"duplicate metadata record for {name!r}")
        records[name] = {
            key: item.get(key)
            for key in ("name", "display_label", "role", "type", "construct", "scale_items",
                        "time_index", "exclude_reason", "theory_group",
                        "fold_label", "subject_id", "target_derived", "derived_from")
        }
    absent = [name for name in names if name not in records]
    extra = [name for name in records if name not in names]
    if absent or extra:
        raise ValueError(f"metadata/column mismatch (missing metadata={absent}, unknown metadata={extra})")
    if root not in names:
        raise ValueError(f"root {root!r} is not present in columns")

    root_record = records[root]
    if root_record["role"] not in (None, "root"):
        raise ValueError(f"root {root!r} must have role 'root'")
    if root_record["type"] != "continuous":
        raise ValueError(f"root {root!r} must have type 'continuous'")
    root_record["role"] = "root"
    root_items = set(root_record["scale_items"] or ())

    item_sets = {name: set(record["scale_items"] or ()) for name, record in records.items()}
    for name, record in records.items():
        record["measurement_overlap"] = any(
            other != name and item_sets[name].intersection(other_items)
            for other, other_items in item_sets.items()
        )
        record["_target_item"] = name in root_items

    candidates: list[str] = []
    for name in names:
        if name == root:
            continue
        record = records[name]
        role, kind = record["role"], record["type"]
        if prospective and record["time_index"] is not None and root_record["time_index"] is not None:
            if record["time_index"] > root_record["time_index"]:
                raise ValueError(f"post-outcome variable {name!r} is not allowed in prospective mode")
        if role in {"id", "time", "exclude"} or record["exclude_reason"]:
            continue
        if record.get("fold_label") or role in {"fold", "fold_label"}:
            continue
        if record.get("subject_id") or role in {"subject_id", "row_id"}:
            continue
        if record.get("target_derived") or record.get("derived_from") == root:
            continue
        if record["_target_item"]:
            continue
        if kind not in {"continuous", "ordinal"}:
            if role == "candidate" or role is None:
                raise ValueError(f"predictor {name!r} has unsupported type {kind!r}; expected 'continuous' or 'ordinal'")
            continue
        if role not in (None, "candidate"):
            raise ValueError(f"predictor {name!r} has unsupported role {role!r}")
        candidates.append(name)
    visible = tuple(
        {key: value for key, value in records[name].items()
         if key not in {"_target_item", "fold_label", "subject_id", "target_derived", "derived_from"}}
        for name in names
    )
    return MetadataValidation(root, tuple(candidates), visible)

