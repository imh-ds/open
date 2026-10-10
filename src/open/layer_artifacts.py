"""Portable layer evidence and deterministic audit tables; never fits models."""
from dataclasses import fields, is_dataclass
from enum import Enum
from collections.abc import Mapping
import csv
import hashlib
import io
import json
import math
import types
import typing

import numpy as np

from open.config import parse_config
from open.diagnostics import hash_variable_metadata
from open.layer_consensus import aggregate_target_assessments, selected_joint_relations
from open.layer_types import (AssessmentWarning, ConsensusLayer, FitProvenance, LayeredOPENResult,
    LocalAssessment, PlannedSample, LAYER_METHOD, LAYER_SCHEMA, LAYER_SCHEMA_VERSION, _integer)
from open.local_target import LocalResult
from open.metadata import validate_metadata
from open.splits import SplitPlan

FILES = ("analysis_metadata.json", "layer_evidence.json", "assessment_ledger.json",
         "nodes.csv", "edges.csv", "shared_groups.csv", "local_targets.csv")
STRENGTH_FIELDS = ("raw_count", "raw_median", "baseline_relative_count", "baseline_relative_median",
                   "gain_relative_count", "gain_relative_median", "strength_scope")
COUNT_FIELDS = ("selected_count", "planned_count", "completed_count", "selection_frequency")
TABLE_FIELDS = {
    "nodes.csv": ("node", "display_label", "depth", "measurement_overlap"),
    "edges.csv": ("source", "target", "target_depth", "relation", *COUNT_FIELDS, *STRENGTH_FIELDS),
    "shared_groups.csv": ("target", "target_depth", "members", *COUNT_FIELDS, *STRENGTH_FIELDS),
    "local_targets.csv": ("depth", "target", "run_index", "seed", "status", "candidates", "baseline_risk",
        "full_risk", "total_gain", "relative_total_gain", "n_observed_target_rows", "n_missing_target_rows",
        "outer_model_fit_count", "tuning_model_fit_count", "failure_type", "failure_message"),
}


def portable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return portable(value.item())
    if is_dataclass(value):
        return {field.name: portable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON mapping keys must be strings")
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [portable(item) for item in items]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("unsupported or nonfinite artifact value")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode(cls, value):
    """Decode only trusted, statically declared types; no artifact-controlled imports."""
    origin, args = typing.get_origin(cls), typing.get_args(cls)
    if cls in (object, typing.Any):
        return value
    if origin in (typing.Union, types.UnionType):
        for option in args:
            try:
                return decode(option, value)
            except (ValueError, TypeError, KeyError):
                pass
        raise ValueError("value does not match declared optional/union type")
    if cls is type(None):
        if value is not None: raise ValueError("expected null")
        return None
    if origin is typing.Literal:
        if value not in args: raise ValueError("invalid literal")
        return value
    if is_dataclass(cls):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError(f"invalid {cls.__name__} fields")
        hints = typing.get_type_hints(cls)
        return cls(**{key: decode(hints[key], item) for key, item in value.items()})
    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls(value)
    if origin in (dict, Mapping, typing.Mapping):
        if not isinstance(value, dict): raise ValueError("expected mapping")
        return {decode(args[0], key): decode(args[1], item) for key, item in value.items()}
    if origin in (tuple, list, frozenset, set):
        if not isinstance(value, list): raise ValueError("expected array")
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            if len(args) != len(value): raise ValueError("tuple length mismatch")
            return tuple(decode(t, item) for t, item in zip(args, value))
        converted = [decode(args[0], item) for item in value]
        return origin(converted)
    if cls is float:
        if type(value) not in (int, float) or not math.isfinite(value): raise ValueError("expected finite number")
        return float(value)
    if type(value) is not cls:
        raise ValueError(f"expected {cls}")
    return value


def target_payload(target):
    payload = {field.name: portable(getattr(target, field.name)) for field in fields(target)
               if field.name not in ("assessments", "shared_comembership")}
    payload["assessments"] = [[a.depth, a.target, a.sample.run_index] for a in target.assessments]
    payload["shared_comembership"] = [{"members": list(pair), "frequency": portable(count)}
                                      for pair, count in target.shared_comembership.items()]
    return payload


def build_payloads(result, metadata):
    root = result.config.analysis.root
    if (metadata.root != root or result.variables.count(root) != 1
            or tuple(n for n in result.variables if n != root) != metadata.candidates):
        raise ValueError("metadata does not match result variables/root")
    meta = {"schema": LAYER_SCHEMA, "schema_version": LAYER_SCHEMA_VERSION,
        "method": LAYER_METHOD, "status": result.status, "config": portable(result.config),
        "planned_resamples_per_target": result.effective_resamples, "variables": list(result.variables),
        "root": root, "max_depth": result.config.analysis.max_depth,
        "assigned_depth": portable(result.assigned_depth), "parents": portable(result.parents),
        "stop_reasons": portable(result.stop_reasons), "variable_metadata": portable(metadata.variables),
        "variable_metadata_sha256": hash_variable_metadata(metadata), "plans": [],
        "provenance": portable(result.provenance), "interpretation": "layer_conditional",
        "path_frequency_available": False}
    for plan in result.plans:
        row = portable(plan)
        row["split_plan"] = plan.split_plan.to_dict() if plan.split_plan is not None else None
        meta["plans"].append(row)
    layers, ledger = [], []
    for layer in result.layers:
        row = {field.name: portable(getattr(layer, field.name)) for field in fields(layer) if field.name != "evidence"}
        row["evidence"] = [target_payload(t) for t in layer.evidence]
        layers.append(row)
        for target in layer.evidence:
            for assessment in target.assessments:
                record = {field.name: portable(getattr(assessment, field.name)) for field in fields(assessment)
                          if field.name != "sample"}
                record["run_index"] = assessment.sample.run_index
                ledger.append(record)
    return {"analysis_metadata.json": meta,
        "layer_evidence.json": {"schema": "open.layer_evidence", "schema_version": 1, "layers": layers},
        "assessment_ledger.json": {"schema": "open.layer_assessments", "schema_version": 1, "assessments": ledger}}


def table_texts(result, metadata):
    variables = {row["name"]: row for row in metadata.variables}
    rows = {name: [] for name in TABLE_FIELDS}
    for name, depth in sorted(result.assigned_depth.items(), key=lambda item: (item[1], item[0])):
        variable = variables[name]
        rows["nodes.csv"].append(dict(node=name, display_label=variable.get("display_label") or name,
            depth=depth, measurement_overlap=bool(variable.get("measurement_overlap", False))))
    for layer in result.layers:
        for target in layer.evidence:
            if layer.finalized:
                for relation in target.relations:
                    if not relation.qualifies:
                        continue
                    counts, strength = relation.frequency, relation.strength
                    row = {"target": relation.target, "target_depth": layer.depth,
                        "selected_count": counts.selected, "planned_count": counts.planned,
                        "completed_count": counts.completed, "selection_frequency": counts.probability,
                        **{f.name: getattr(strength, f.name) for f in fields(strength) if f.name != "scope"},
                        "strength_scope": strength.scope}
                    if relation.relation == "SHARED_GROUP":
                        row["members"] = relation.members
                        rows["shared_groups.csv"].append(row)
                    else:
                        row.update(source=relation.source, relation=relation.relation)
                        rows["edges.csv"].append(row)
            for assessment in target.assessments:
                local = assessment.local
                rows["local_targets.csv"].append(dict(depth=layer.depth, target=target.target,
                    run_index=assessment.sample.run_index, seed=assessment.sample.seed, status=assessment.status,
                    candidates=target.candidates, baseline_risk=local.relevance.baseline_risk if local else None,
                    full_risk=local.relevance.full_risk if local else None, total_gain=local.total_gain if local else None,
                    relative_total_gain=local.total_gain_relative if local else None,
                    n_observed_target_rows=assessment.n_observed_target_rows,
                    n_missing_target_rows=assessment.n_missing_target_rows,
                    outer_model_fit_count=local.outer_model_fit_count if local else None,
                    tuning_model_fit_count=local.tuning_model_fit_count if local else None,
                    failure_type=assessment.failure_type, failure_message=assessment.failure_message))
    texts = {}
    for name, data in rows.items():
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS[name], lineterminator="\n")
        writer.writeheader()
        for row in data:
            writer.writerow({key: canonical(value) if isinstance(value, tuple) else
                             "" if value is None else value for key, value in row.items()})
        texts[name] = handle.getvalue()
    return texts


def validate_local(local, sample, config):
    """Check saved risk arithmetic and scored provenance without refitting."""
    relevance = local.relevance
    def same(a, b):
        if a is None or b is None:
            return a is b
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
    gain = relevance.baseline_risk - relevance.full_risk
    relative = gain / relevance.baseline_risk if relevance.baseline_risk > 0 else None
    if not all((same(local.total_gain, gain), same(relevance.total_gain, gain),
                same(local.total_gain_relative, relative), same(relevance.relative_gain, relative))):
        raise ValueError("saved gain arithmetic is inconsistent")
    eligible = gain > 0 and relative is not None and relative >= config.thresholds.min_total_gain_relative
    if local.stop != (not eligible) or local.reason != relevance.stop_reason:
        raise ValueError("saved total-gain stop is inconsistent")
    for name, contrasts in (("broad", relevance.broad), ("unique", relevance.unique)):
        for source, contrast in contrasts.items():
            expected = contrast.raw / gain if eligible else None
            if not same(contrast.normalized, expected) or not same(getattr(local, name)[source], expected):
                raise ValueError("saved normalized contrast is inconsistent")
    expected_promoted = set()
    for name, variable in local.classes.items():
        unique = relevance.unique[name].normalized
        proximal = unique is not None and unique >= config.thresholds.unique_relative
        if (variable.kind is not None and variable.kind.value == "PROXIMAL_UNIQUE") != proximal:
            raise ValueError("saved unique class contradicts scored threshold")
        if variable.kind is not None and variable.kind.value in ("PROXIMAL_UNIQUE", "SHARED_GROUP_MEMBER"):
            expected_promoted.add(name)
    if expected_promoted != set(local.promoted):
        raise ValueError("saved promoted members contradict classes")
    for members, (raw, normalized, provenance) in selected_joint_relations(local).items():
        if not set(members) <= set(local.promoted):
            raise ValueError("joint relation must preserve every promoted member")
        denominator = provenance.pair.total_gain if hasattr(provenance, "pair") else gain
        if not same(normalized, raw / denominator if denominator > 0 else None):
            raise ValueError("saved joint score normalization is inconsistent")
        if normalized is None or normalized < config.thresholds.group_relative:
            raise ValueError("selected joint relation fails the group threshold")
    risks = []
    declared_splits = {(s.repeat, s.fold): s for s in sample.split_plan.splits}
    pair_by_members = {}
    if local.pair_ablations is not None:
        planned = sample.split_plan.folds * sample.split_plan.repeats
        fold_ids = {(s.repeat, s.fold) for s in sample.split_plan.splits}
        for pair in local.pair_ablations.pairs:
            pair_by_members[tuple(sorted((pair.a, pair.b)))] = pair
            _integer(pair.selected_fold_count)
            _integer(pair.evaluated_fold_count)
            if (len(set(pair.selected_folds)) != pair.selected_fold_count
                    or pair.evaluated_fold_count != pair.selected_fold_count
                    or not set(pair.selected_folds) <= fold_ids
                    or not same(pair.selection_frequency, pair.selected_fold_count/planned)
                    or not same(pair.evaluation_frequency, pair.evaluated_fold_count/planned)
                    or not same(pair.raw, pair.ablated_risk-pair.full_risk)
                    or not same(pair.total_gain, pair.baseline_risk-pair.full_risk)
                    or not same(pair.normalized, pair.raw/pair.total_gain if pair.total_gain > 0 else None)):
                raise ValueError("saved pair provenance is inconsistent")
            if pair.ablation is not None:
                risks.append((pair.ablation, (pair.a, pair.b), set(pair.selected_folds), pair.ablated_risk))
    for pair in local.retained_pairs:
        if pair_by_members.get(pair.members) != pair.pair:
            raise ValueError("retained joint pair differs from its saved pair ablation")
    for component in local.shared_components:
        if component.ablation is not None:
            risks.append((component.ablation, component.members, set(declared_splits),
                          relevance.full_risk + component.raw))
    split_id = hashlib.sha256(sample.split_plan.to_json(indent=None).encode("utf-8")).hexdigest()
    for risk, removed, expected_folds, expected_risk in risks:
        expected_predictors = tuple(n for n in local.input_candidates if n not in removed)
        if (risk.target != local.target or risk.candidate_set != local.input_candidates
                or risk.split_seed != sample.seed or risk.split_plan_id != split_id or risk.loss != "mse"
                or risk.predictors != expected_predictors or not same(risk.aggregate_risk, expected_risk)):
            raise ValueError("saved ablation references a different comparison/split")
        losses = []
        keys = set()
        seen_folds = set()
        for fold in risk.folds:
            identity = (fold.repeat_index, fold.fold_index)
            if (identity not in expected_folds or identity in seen_folds
                    or fold.predictors != expected_predictors):
                raise ValueError("invalid ablation fold/predictor identity")
            seen_folds.add(identity)
            test_positions = set(declared_splits[identity].test_indices)
            for row in fold.observations:
                _integer(row.row_position)
                key = (fold.repeat_index, fold.fold_index, row.row_position)
                if key in keys or row.row_position not in test_positions:
                    raise ValueError("invalid held-out observation identity")
                keys.add(key)
                if not same(row.loss, (row.observed-row.predicted)**2):
                    raise ValueError("saved MSE does not match predictions")
                losses.append(row.loss)
            if not fold.observations or not same(fold.risk, sum(r.loss for r in fold.observations)/len(fold.observations)):
                raise ValueError("saved fold risk is inconsistent")
        if seen_folds != expected_folds or not losses or not same(risk.aggregate_risk, sum(losses)/len(losses)):
            raise ValueError("saved pooled risk is inconsistent")


def reconstruct_payloads(meta, layers_payload, ledger_payload):
    """Recount saved local snapshots into typed evidence, not a newly fitted model."""
    if (meta.get("schema") != LAYER_SCHEMA or type(meta.get("schema_version")) is not int
            or meta["schema_version"] != LAYER_SCHEMA_VERSION or meta.get("method") != LAYER_METHOD
            or meta.get("interpretation") != "layer_conditional" or meta.get("path_frequency_available") is not False):
        raise ValueError("unsupported layer method/schema/interpretation")
    for payload, schema, key in ((layers_payload,"open.layer_evidence","layers"),
                                 (ledger_payload,"open.layer_assessments","assessments")):
        if (set(payload) != {"schema","schema_version",key} or payload["schema"] != schema
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
            raise ValueError("unsupported evidence schema")
    config = parse_config(meta["config"])
    plans = tuple(PlannedSample(p["run_index"], p["seed"], tuple(p["sampled_positions"]),
                    SplitPlan.from_dict(p["split_plan"]) if p["split_plan"] is not None else None,
                    p["infeasibility_reason"]) for p in meta["plans"])
    if type(meta["planned_resamples_per_target"]) is not int or meta["planned_resamples_per_target"] != len(plans):
        raise ValueError("effective sample budget mismatch")
    metadata = validate_metadata([v["name"] for v in meta["variable_metadata"]],
                                 meta["variable_metadata"], meta["root"])
    records = {}
    for entry in ledger_payload["assessments"]:
        values = dict(entry)
        index = values.pop("run_index")
        _integer(index)
        if index >= len(plans): raise ValueError("foreign planned sample")
        values["sample"] = portable(plans[index])
        # The declared type uses SplitPlan fields, while the metadata plan has a versioned schema.
        record = decode(LocalAssessment, values)
        if record.local is not None:
            validate_local(record.local, record.sample, config)
        key = (record.depth, record.target, index)
        if key in records: raise ValueError("duplicate assessment record")
        records[key] = record
    layers, used = [], set()
    for row in layers_payload["layers"]:
        targets = []
        for target in row["evidence"]:
            refs = [tuple(ref) for ref in target["assessments"]]
            if any(ref in used or ref not in records for ref in refs):
                raise ValueError("missing or reused assessment reference")
            used.update(refs)
            targets.append(aggregate_target_assessments(tuple(records[ref] for ref in refs),
                planned_count=len(plans), pi_min=config.thresholds.stability))
        layers.append(ConsensusLayer(row["depth"], tuple(row["targets"]), tuple(row["candidates"]),
            tuple(targets), row["finalized"], tuple(row["new_nodes"]),
            {name:tuple(parents) for name,parents in row["promoted_parents"].items()}))
    if used != set(records): raise ValueError("unreferenced assessment record")
    result = LayeredOPENResult(config, tuple(meta["variables"]), plans, tuple(layers),
        meta["assigned_depth"], {n:tuple(v) for n,v in meta["parents"].items()}, meta["stop_reasons"],
        meta["status"], decode(FitProvenance, meta["provenance"]))
    expected = build_payloads(result, metadata)
    for name, payload in ((FILES[0],meta),(FILES[1],layers_payload),(FILES[2],ledger_payload)):
        if canonical(expected[name]) != canonical(payload):
            raise ValueError(f"{name}: saved evidence does not match recounted assessments")
    return result, metadata
