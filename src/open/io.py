"""Deterministic, staged export of one completed OPEN result."""

from __future__ import annotations

import csv
import ctypes
import errno
import json
import os
from pathlib import Path
import sys
import tempfile

from open.diagnostics import build_analysis_metadata
from open.metadata import MetadataValidation
from open.stability import OPENResult
from open.layer_types import LayeredOPENResult


_FILES = (
    "nodes.csv", "edges.csv", "shared_groups.csv", "shared_comembership.csv",
    "local_targets.csv", "analysis_metadata.json",
)
_EDGE_FIELDS = (
    "source", "target", "relation", "broad_raw", "broad_norm", "unique_raw",
    "unique_norm", "broad_raw_median", "broad_norm_median",
    "unique_raw_median", "unique_norm_median", "stability",
    "target_reached_probability", "target_reached_numerator",
    "target_reached_denominator", "evaluated_probability", "screened_by",
    "stability_numerator", "stability_denominator", "evaluated_numerator",
    "evaluated_denominator",
)
_GROUP_FIELDS = (
    "target", "group_id", "members", "group_raw", "group_raw_median",
    "raw_count", "group_norm", "group_norm_median", "normalized_count",
    "stability", "stability_numerator", "stability_denominator",
)
_PAIR_FIELDS = ("target", "node_a", "node_b", "shared_probability", "numerator", "denominator")
_LOCAL_FIELDS = (
    "target", "depth", "baseline_risk", "full_risk", "total_gain",
    "relative_total_gain", "learner", "n_candidates", "stop_reason",
    "run_index", "run_seed",
    "n_observed_target_rows", "n_missing_target_rows",
)


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _nodes(result: OPENResult, metadata: MetadataValidation):
    max_depth = result.config.analysis.max_depth
    fields = ("node", "display_label", "depth", "selection_probability",
              *(f"depth_probability_{depth}" for depth in range(max_depth + 1)),
              "selection_numerator", "selection_denominator")
    labels = {item["name"]: item.get("display_label") or item["name"]
              for item in metadata.variables}
    root = result.config.analysis.root
    denominator = result.consensus.denominator
    depth_zeros = {f"depth_probability_{depth}": 0.0
                   for depth in range(max_depth + 1)}
    rows = [{"node": root, "display_label": labels.get(root, root), "depth": 0,
             "selection_probability": 1.0, **depth_zeros,
             "depth_probability_0": 1.0,
             "selection_numerator": denominator, "selection_denominator": denominator}]
    for node in sorted(result.consensus.nodes, key=lambda item: item.name):
        if node.name == root:
            continue
        row = {"node": node.name, "display_label": labels.get(node.name, node.name),
               "depth": node.depth, "selection_probability": node.selection_probability,
               **depth_zeros,
               "selection_numerator": node.frequency.numerator,
               "selection_denominator": node.frequency.denominator}
        row.update({f"depth_probability_{depth}": probability
                    for depth, probability in node.depth_probabilities})
        rows.append(row)
    return fields, rows


def _edges(result: OPENResult):
    targets = {item.target: item for item in result.consensus.targets}
    rows = []
    for edge in sorted(result.consensus.edges,
                       key=lambda item: (item.target, item.source, item.relation.value)):
        target = targets[edge.target]
        rows.append({
            "source": edge.source, "target": edge.target, "relation": edge.relation.value,
            "broad_raw": edge.broad.raw_mean, "broad_norm": edge.broad.normalized_mean,
            "unique_raw": edge.unique.raw_mean, "unique_norm": edge.unique.normalized_mean,
            "broad_raw_median": edge.broad.raw_median,
            "broad_norm_median": edge.broad.normalized_median,
            "unique_raw_median": edge.unique.raw_median,
            "unique_norm_median": edge.unique.normalized_median,
            "stability": edge.frequency.probability,
            "target_reached_probability": target.reached.probability,
            "target_reached_numerator": target.reached.numerator,
            "target_reached_denominator": target.reached.denominator,
            "evaluated_probability": edge.evaluated_probability,
            "screened_by": edge.screened_by,
            "stability_numerator": edge.frequency.numerator,
            "stability_denominator": edge.frequency.denominator,
            "evaluated_numerator": target.evaluated.numerator,
            "evaluated_denominator": target.evaluated.denominator,
        })
    return _EDGE_FIELDS, rows


def _groups(result: OPENResult):
    rows = []
    for relation in sorted(result.consensus.shared_relations,
                           key=lambda item: (item.target, item.members)):
        magnitude = relation.magnitude
        rows.append({
            "target": relation.target,
            "group_id": _compact([relation.target, list(relation.members)]),
            "members": relation.members,
            "group_raw": magnitude.raw_mean,
            "group_raw_median": magnitude.raw_median,
            "raw_count": magnitude.raw_count,
            "group_norm": magnitude.normalized_mean,
            "group_norm_median": magnitude.normalized_median,
            "normalized_count": magnitude.normalized_count,
            "stability": relation.frequency.probability,
            "stability_numerator": relation.frequency.numerator,
            "stability_denominator": relation.frequency.denominator,
        })
    return _GROUP_FIELDS, rows


def _pairs(result: OPENResult):
    rows = [{
        "target": pair.target, "node_a": pair.member_a, "node_b": pair.member_b,
        "shared_probability": pair.frequency.probability,
        "numerator": pair.frequency.numerator,
        "denominator": pair.frequency.denominator,
    } for pair in sorted(result.consensus.shared_pairs,
                         key=lambda item: (item.target, item.member_a, item.member_b))]
    return _PAIR_FIELDS, rows


def _local_targets(result: OPENResult):
    rows = []
    learner = result.config.learner.library[0]
    for run in result.batch.runs:
        if not run.completed:
            continue
        network = run.network
        for target, local in sorted(network.local_results.items()):
            rows.append({
                "target": target, "depth": network.assigned_depth[target],
                "baseline_risk": local.relevance.baseline_risk,
                "full_risk": local.relevance.full_risk,
                "total_gain": local.total_gain,
                "relative_total_gain": local.total_gain_relative,
                "learner": learner, "n_candidates": len(local.relevance.broad),
                "stop_reason": local.reason or (
                    network.stop_reasons[target].value
                    if target in network.stop_reasons else None),
                "run_index": run.run_index, "run_seed": run.seed,
                "n_observed_target_rows": local.n_observed_target_rows,
                "n_missing_target_rows": local.n_missing_target_rows,
            })
    return _LOCAL_FIELDS, rows


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: (_compact(value) if isinstance(value, (list, tuple))
                                    else "" if value is None else value)
                             for field, value in row.items()})


def _publish_no_replace(stage: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination is absent."""
    if os.name == "nt":
        # Windows os.rename raises FileExistsError when destination exists.
        stage.rename(destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise NotImplementedError("atomic no-replace rename is unavailable") from exc
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint)
        arguments = (-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as exc:
            raise NotImplementedError("atomic no-replace rename is unavailable") from exc
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        arguments = (os.fsencode(stage), os.fsencode(destination), 0x00000004)
    else:
        raise NotImplementedError("atomic no-replace rename is unavailable")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise OSError(error, os.strerror(error), str(destination))


def write_open_outputs(
    result: OPENResult | LayeredOPENResult, output_dir: Path, *, metadata: MetadataValidation,
) -> tuple[Path, ...]:
    """Publish method-specific files atomically at a new destination directory."""
    if isinstance(result, LayeredOPENResult):
        from open.layer_io import write_layer_outputs
        return write_layer_outputs(result, output_dir, metadata=metadata)
    if not isinstance(result, OPENResult):
        raise TypeError("result must be an OPENResult")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.",
                                     dir=destination.parent) as temporary:
        stage = Path(temporary)
        for name, (fields, rows) in zip(_FILES[:5], (
            _nodes(result, metadata), _edges(result), _groups(result),
            _pairs(result), _local_targets(result),
        )):
            _write_csv(stage / name, fields, rows)
        payload = build_analysis_metadata(result, metadata)
        with (stage / _FILES[5]).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            handle.write("\n")
        if destination.exists():
            raise FileExistsError(destination)
        _publish_no_replace(stage, destination)
    return tuple(destination / name for name in _FILES)
