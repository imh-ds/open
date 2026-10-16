import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))


def test_rf_adapter_preserves_original_manifest_and_enables_singleton_audit(tmp_path, monkeypatch):
    import tree_recursive_qualification_audit as audit

    manifest = {"schema": audit.SCHEMA, "grid_policy": audit.POLICY,
                "engines": ["random_forest"],
                "grids": {"random_forest": audit.rf_singleton.NOMINAL_GRID}}
    original = copy.deepcopy(manifest)
    calls = []

    def saved_case(spec, store, passed_manifest, **kwargs):
        assert passed_manifest is manifest  # receipt binds its exact original digest
        assert audit.legacy.verify_tuning is audit.rf_singleton.verify_tuning
        calls.append(spec["id"])
        return {"status": "complete", "edges": [], "groups": [], "assigned_depth": {"Y": 0}}

    monkeypatch.setattr(audit.legacy, "audit_case", saved_case)
    prior = audit.legacy.verify_tuning
    row = audit.audit_case({"id": "rf-case", "engine": "random_forest", "n": 80},
                          SimpleNamespace(location=tmp_path), manifest)
    assert calls == ["rf-case"]
    assert row["false_network"] is False
    assert manifest == original
    assert audit.legacy.verify_tuning is prior
    with pytest.raises(ValueError, match="explicit grid policy"):
        audit.rf_singleton.audit_case({}, None, manifest)


@pytest.mark.parametrize("change", ["wrapper_policy", "rf_grid"])
def test_rf_adapter_rejects_unknown_policy_or_grid_before_replay(tmp_path, monkeypatch, change):
    import tree_recursive_qualification_audit as audit

    manifest = {"schema": audit.SCHEMA, "grid_policy": audit.POLICY,
                "engines": ["random_forest"],
                "grids": {"random_forest": copy.deepcopy(audit.rf_singleton.NOMINAL_GRID)}}
    if change == "wrapper_policy":
        manifest["grid_policy"] = "unrecognized"
    else:
        manifest["grids"]["random_forest"]["min_samples_leaf"] = [99]
    monkeypatch.setattr(audit.legacy, "audit_case", lambda *a, **k: pytest.fail("replayed unrecognized policy"))
    with pytest.raises(ValueError, match="policy|grid"):
        audit.audit_case({"id": "rf-case", "engine": "random_forest", "n": 80},
                         SimpleNamespace(location=tmp_path), manifest)


def test_engine_search_policy_has_expected_tuning_fit_counts():
    import tree_recursive_qualification_audit as audit

    # Ten outer folds (5-fold CV repeated twice), three inner folds.
    assert audit.expected_tuning_fits("random_forest", 1, 10, 3) == 60
    assert audit.expected_tuning_fits("random_forest", 2, 10, 3) == 120
    assert audit.expected_tuning_fits("gbm", 1, 10, 3) == 120
    assert audit.expected_tuning_fits("gbm", 4, 10, 3) == 120
    assert audit.expected_tuning_fits("lightgbm", 1, 10, 3) == 120
    assert audit.expected_tuning_fits("lightgbm", 4, 10, 3) == 120
    assert audit.expected_tuning_fits("gbm", 2, 10, 3, tuned=False) == 0


@pytest.mark.parametrize("finding", ["edge", "group", "node"])
def test_null_endpoint_counts_replayed_network_without_legacy_structure(tmp_path, monkeypatch, finding):
    import tree_recursive_qualification_audit as audit

    saved = {"status": "complete", "structure": None, "edges": [], "groups": [],
             "assigned_depth": {"Y": 0}}
    if finding == "edge":
        saved["edges"] = [["X1", "Y"]]
    elif finding == "group":
        saved["groups"] = [["Y", ["X1", "X2"]]]
    else:
        saved["assigned_depth"]["X1"] = 1
    monkeypatch.setattr(audit.legacy, "audit_case", lambda *a, **k: saved)
    row = audit.audit_case({"id": "gbm-case", "engine": "gbm", "n": 80},
                          SimpleNamespace(location=tmp_path),
                          {"schema": audit.SCHEMA, "grid_policy": audit.POLICY, "engines": ["gbm"]})
    assert row["false_network"] is True


@pytest.mark.parametrize(
    "network,expected",
    [
        ({"nodes": [], "edges": [], "groups": []}, False),
        ({"nodes": ["X"], "edges": [], "groups": []}, True),
        ({"nodes": [], "edges": [["X", "Y"]], "groups": []}, True),
        ({"nodes": [], "edges": [], "groups": [["Y", ["X1", "X2"]]]}, True),
    ],
)
def test_empty_truth_counts_any_extra_node_edge_or_group(network, expected):
    import tree_recursive_qualification_audit as audit

    assert audit.has_false_findings(network) is expected


def test_audit_cell_keeps_missing_cases_and_rejects_duplicate_ids():
    import tree_recursive_qualification_audit as audit

    rows = audit.audit_cell(["case-a", "case-b"], [
        {"case_id": "case-a", "status": "complete"},
    ])
    assert [row["case_id"] for row in rows] == ["case-a", "case-b"]
    assert rows[1]["status"] == "missing"
    assert rows[1]["fit_cost_unknown"] is True

    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_cell(["case-a"], [
            {"case_id": "case-a", "status": "complete"},
            {"case_id": "case-a", "status": "complete"},
        ])
