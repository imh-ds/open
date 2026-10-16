"""Versioned, no-fit audit helpers for tree-engine recursive signal studies.

This module adds the RF singleton search policy without changing either of the
historical recursive auditors. Case archive replay is delegated to those
auditors; these helpers enforce this study's policy and matched-chain outcome.
"""

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

import multiengine_recursive_audit as legacy
import optimized_rf_recursive_audit as rf_singleton
from open.learners.trees import effective_grid
from sklearn.model_selection import ParameterGrid


POLICY = "tree-signal-v1"
SCHEMA = "open.tree_recursive_signal.v1"
ENGINES = ("random_forest", "gbm", "lightgbm")


def expected_tuning_fits(engine, n_features, outer_folds, inner_folds, *, tuned=True):
    """Compute expected tuning fits from the frozen engine grid and fold counts."""
    if engine not in ENGINES:
        raise ValueError("unsupported audit engine")
    for name, value in (("n_features", n_features), ("outer_folds", outer_folds),
                        ("inner_folds", inner_folds)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(tuned) is not bool:
        raise ValueError("tuned must be boolean")
    if not tuned:
        return 0
    candidate_count = len(ParameterGrid(effective_grid(engine, n_features)))
    return outer_folds * candidate_count * inner_folds


def verify_tuning_count(engine, risk, config, tuned):
    """Validate a saved risk's tuning count against this engine's actual grid."""
    n_features = len(risk["predictors"])
    expected = expected_tuning_fits(
        engine, n_features, len(risk["folds"]), config.tuning_cv.folds, tuned=tuned
    )
    if risk["tuning_model_fit_count"] != expected:
        raise ValueError("saved tuning count differs from tree signal policy")
    if engine == "random_forest":
        return rf_singleton.verify_tuning(risk, config, tuned)
    return legacy.verify_tuning(risk, config, tuned)


def has_false_findings(network):
    """Return whether a finalized network contains a node, edge or group."""
    if not isinstance(network, Mapping):
        raise ValueError("network summary must be a mapping")
    for key in ("nodes", "edges", "groups"):
        value = network.get(key, ())
        if value:
            return True
    return False


def _row_case_id(row):
    if not isinstance(row, Mapping):
        raise ValueError("case audit row must be a mapping")
    case_id = row.get("case_id")
    if case_id is None and isinstance(row.get("spec"), Mapping):
        case_id = row["spec"].get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case audit row has no case identity")
    return case_id


def audit_cell(expected_case_ids: Sequence[str], case_results: Sequence[Mapping[str, object]]):
    """Retain planned identities and represent absent cases as explicit unknowns."""
    if len(expected_case_ids) != len(set(expected_case_ids)):
        raise ValueError("duplicate expected case identity")
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_case_ids):
        raise ValueError("invalid expected case identity")
    expected = set(expected_case_ids)
    found = {}
    for row in case_results:
        case_id = _row_case_id(row)
        if case_id not in expected:
            raise ValueError("foreign case identity")
        if case_id in found:
            raise ValueError("duplicate case audit identity")
        found[case_id] = dict(row)
    return [found.get(case_id, {
        "case_id": case_id,
        "status": "missing",
        "fit_cost_unknown": True,
        "audit_error": "no case audit row was received",
    }) for case_id in expected_case_ids]


def audit_case(spec, store, manifest, *, data=None, config=None):
    """Audit one case archive without fitting any model.

    `spec`, `store`, `data`, and `config` use the established multi-engine
    archive interfaces. The caller must pass the exact frozen study manifest.
    """
    if manifest.get("schema") != SCHEMA or manifest.get("grid_policy") != POLICY:
        raise ValueError("tree signal manifest policy/schema differs")
    engine = spec.get("engine")
    if engine not in ENGINES or engine not in manifest.get("engines", ()):
        raise ValueError("case engine is not in frozen manifest")
    if engine == "random_forest":
        if manifest.get("grids", {}).get(engine) != rf_singleton.NOMINAL_GRID:
            raise ValueError("tree signal RF grid differs from singleton audit policy")
        # The receipt seals the original tree-null manifest. Keep it intact and
        # select the RF verifier explicitly; the historical entrypoint continues
        # to require its own rf-singleton-v1 manifest label.
        with rf_singleton._policy_scope():
            row = legacy.audit_case(spec, store, manifest, data=data, config=config)
    else:
        row = legacy.audit_case(spec, store, manifest, data=data, config=config)
    row["case_id"] = spec["id"]
    row["engine"] = engine
    row["sample_size"] = spec["n"]
    row["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    location = Path(store.location)
    if location.is_file():
        row["archive_sha256"] = hashlib.sha256(location.read_bytes()).hexdigest()
        row["archive_bytes"] = location.stat().st_size
    if row.get("status") == "complete":
        import recursive_preview_design as design
        row["structure"] = design.structure(spec, row["edges"], row["groups"], row["assigned_depth"])
        row["false_network"] = bool(row["structure"]["incorrect"])
        row["exact_network"] = bool(row["structure"]["exact"])
    else:
        row["false_network"] = None
        row["exact_network"] = None
        row["fit_cost_unknown"] = True
    return row
