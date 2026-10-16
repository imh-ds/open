import hashlib
import json
import sys
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))


def test_release_bundles_are_split_deterministic_and_hash_verifiable(tmp_path):
    import tree_recursive_null_evidence as evidence

    inputs = []
    for i in range(3):
        path = tmp_path / f"case-{i}.zip"
        path.write_bytes(bytes([i]) * 240)
        inputs.append({
            "case_id": f"gbm-n80-r{i + 1:02}",
            "archive": path,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    assets = evidence.build_release_bundles(inputs, tmp_path / "bundles", max_asset_bytes=850)
    assert len(assets) == 2
    assert all(asset["bytes"] <= 850 for asset in assets)
    for asset in assets:
        assert evidence.verify_release_bundle(asset["path"], asset["members"])


def test_release_bundle_rejects_changed_member_and_single_oversize_case(tmp_path):
    import tree_recursive_null_evidence as evidence

    source = tmp_path / "raw.zip"
    source.write_bytes(b"x" * 800)
    record = {"case_id": "rf-n80-r01", "archive": source,
              "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="exceeds"):
        evidence.build_release_bundles([record], tmp_path / "too-small", max_asset_bytes=500)


def test_asset_inventory_preserves_relative_bundle_path(tmp_path):
    import tree_recursive_null_evidence as evidence

    asset_path = tmp_path / "bundles" / "cell.zip"
    asset_path.parent.mkdir()
    asset_path.write_bytes(b"asset")
    inventory = evidence.write_inventory(tmp_path / "archive-inventory.json", [{
        "name": asset_path.name,
        "path": asset_path,
        "bytes": 5,
        "sha256": hashlib.sha256(b"asset").hexdigest(),
        "members": {},
    }])
    assert inventory[0]["path"] == "bundles/cell.zip"


def test_case_soft_deadline_reserves_setup_and_evidence_upload_time():
    import tree_recursive_null_actions as actions

    assert actions.soft_compute_seconds(60, job_started_at=100, now=300) == 2800
    assert actions.soft_compute_seconds(25, job_started_at=100, now=1200) == 0


def test_deadline_case_receipt_seals_available_inputs_without_fitting(tmp_path):
    import tree_recursive_null_actions as actions
    import tree_recursive_null_study as study

    manifest = study.build_manifest()
    spec = manifest["cases"][0]
    case_dir = tmp_path / "case"
    actions._write_incomplete_case(spec, manifest, case_dir, "deadline")
    receipt = json.loads((case_dir / "receipt.json").read_text())
    assert receipt["status"] == "incomplete"
    assert receipt["files"] == {
        path.relative_to(case_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in case_dir.rglob("*") if path.is_file() and path.name != "receipt.json"
    }


@pytest.mark.parametrize(("field", "value"), [("case_cap_minutes", 70), ("attempt", True)])
def test_case_authorization_rejects_tampered_cap_or_boolean_attempt(tmp_path, monkeypatch, field, value):
    import tree_recursive_null_actions as actions
    import tree_recursive_null_study as study

    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    manifest = study.build_manifest()
    authorization = study.prepare_wave(80, manifest, tmp_path / "authorization",
                                       source_sha="a" * 40, run_id="123")
    case_id = manifest["cases"][0]["id"]
    control_path = authorization / "cases" / case_id / "control.json"
    control = json.loads(control_path.read_text())
    control[field] = value
    control_path.write_text(json.dumps(control))
    monkeypatch.setattr(actions, "checked_manifest", lambda *a, **k: (manifest, actions.study._digest(manifest)))
    monkeypatch.setattr(actions.runner, "run_case", lambda *a, **k: pytest.fail("fit despite cap mismatch"))
    with pytest.raises(ValueError, match="authorization"):
        actions.run_case(case_id, authorization, tmp_path / "out", job_started_at=time.time() + 100,
                         expected_sample_size=80)


def test_malformed_case_control_is_retained_as_unknown(tmp_path):
    import tree_recursive_null_actions as actions

    expected = {
        "case_id": "null-gbm-n80-r01",
        "engine": "gbm",
        "sample_size": 80,
        "repetition": 1,
        "study_id": "tree-null-study",
        "manifest_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "attempt": 1,
    }
    artifact = tmp_path / "artifact"
    (artifact / "raw").mkdir(parents=True)
    (artifact / "control.json").write_text("[]")
    (artifact / "raw" / "case.zip").write_bytes(b"preserved-but-untrusted")
    row = actions.audit_case_artifact(expected, artifact, {}, "b" * 40, "123")
    assert row["status"] == "audit_failed"
    assert row["fit_cost_unknown"] is True
    assert row["false_network"] is None


def test_transport_audit_rejects_duplicate_workflow_job_names():
    import tree_recursive_null_actions as actions
    import tree_recursive_null_study as study

    manifest = study.build_manifest()
    records = [{"name": "prepare", "steps": [{"name": "Upload authorization", "conclusion": "success"}]},
               {"name": "prepare", "steps": [{"name": "Upload authorization", "conclusion": "success"}]}]
    records.extend({"name": f"case N={size} ({case['id']})",
                    "steps": [{"name": "Upload case evidence", "conclusion": "success"}]}
                   for size in (80, 200) for case in study.cases_for_n(manifest, size))

    def api(path):
        if path == "actions/runs/123":
            return {"head_sha": "a" * 40, "run_attempt": 1,
                    "event": "workflow_dispatch", "path": actions.WORKFLOW}
        page = int(path.rsplit("=", 1)[1])
        offset = (page - 1) * 100
        return {"jobs": records[offset:offset + 100]}

    result = actions.verify_transport("123", "a" * 40, manifest, api=api)
    assert result["verified"] is False
    assert "duplicate workflow job names" in result["errors"]


def test_workflow_has_two_independent_waves_caps_and_scoped_write_permission():
    workflow_text = (Path(__file__).parents[1] / ".github/workflows/tree-recursive-qualification.yml").read_text()
    workflow = yaml.safe_load(workflow_text)
    assert "python -I examples/tree_recursive_null_actions.py" not in workflow_text
    assert workflow["permissions"]["contents"] == "read"
    for wave in ("case-n80", "case-n200"):
        assert workflow["jobs"][wave]["strategy"]["fail-fast"] is False
        assert workflow["jobs"][wave]["strategy"]["max-parallel"] == 10
        assert workflow["jobs"][wave]["timeout-minutes"] == "${{ matrix.case_cap_minutes }}"
    assert workflow["jobs"]["publish"]["permissions"]["contents"] == "write"
    assert workflow["jobs"]["prepare"]["timeout-minutes"] == 30
    assert workflow["jobs"]["aggregate"]["timeout-minutes"] == 180
    assert workflow["jobs"]["publish"]["timeout-minutes"] == 45
    assert workflow["jobs"]["aggregate"]["if"] == "always() && needs.prepare.result == 'success'"
    source_checkout_jobs = ("prepare", "case-n80", "case-n200", "aggregate")
    import tree_recursive_null_study as study
    for job_name in source_checkout_jobs:
        checkout = next(step for step in workflow["jobs"][job_name]["steps"]
                        if step.get("uses", "").startswith("actions/checkout@"))
        sparse = checkout["with"]["sparse-checkout"]
        for source in study.SOURCE_FILES:
            if source.startswith("src/open/"):
                assert "src/**" in sparse
            else:
                assert source in sparse
    register = workflow["jobs"]["register"]
    assert not any(step.get("uses", "").startswith("actions/checkout@") for step in register["steps"])
    assert "no scientific fits" in register["steps"][0]["run"].lower()
