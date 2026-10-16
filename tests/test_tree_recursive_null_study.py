import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    import open.layer_stability as stability

    monkeypatch.setattr(stability, "assess_layer_target", lambda *a, **k: pytest.fail("scientific fit"))


def test_manifest_has_fresh_paired_cases_and_reproducible_inputs():
    import tree_recursive_null_study as study

    manifest = study.build_manifest()
    assert manifest["python_version"] == "3.11.9"
    assert "python" not in manifest["runtime"]
    assert {
        "examples/tree_recursive_null_actions.py",
        "examples/tree_recursive_null_evidence.py",
        "examples/recursive_preview_actions.py",
        "examples/qualification_artifacts.py",
        "examples/tree_recursive_null_validation_protocol.md",
        ".github/workflows/tree-recursive-qualification.yml",
    } <= set(manifest["sources"])
    for n in (80, 200):
        cases = study.cases_for_n(manifest, n)
        assert len(cases) == 165
        assert len({case["id"] for case in cases}) == 165
        assert {case["engine"] for case in cases} == set(study.ENGINES)
        assert {case["repetition"] for case in cases} == set(range(1, 56))
        hashes = {}
        for case in cases:
            hashes.setdefault(case["repetition"], set()).add(manifest["data_sha256"][case["id"]])
        assert all(len(values) == 1 for values in hashes.values())
    assert not set(case["id"] for case in study.cases_for_n(manifest, 80)) & {
        case["id"] for case in study.cases_for_n(manifest, 200)
    }


def test_prepare_wave_writes_165_verified_case_controls_without_fitting(tmp_path):
    import tree_recursive_null_study as study

    manifest = study.build_manifest()
    prepared = study.prepare_wave(80, manifest, tmp_path / "wave", source_sha="a" * 40, run_id="123")
    matrix = json.loads((prepared / "matrix.json").read_text())
    controls = list((prepared / "cases").glob("*/control.json"))
    assert len(matrix["include"]) == 165
    assert all("case_cap_minutes" in row for row in matrix["include"])
    assert len(controls) == 165
    first = json.loads(controls[0].read_text())
    data = controls[0].parent / "data.csv"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == first["data_sha256"]
    assert first["source_sha"] == "a" * 40
    assert first["attempt"] == 1


def test_manifest_validation_rejects_mutated_case_or_runtime():
    import tree_recursive_null_study as study

    manifest = study.build_manifest()
    altered = json.loads(json.dumps(manifest))
    altered["runtime"]["numpy"] = "latest"
    with pytest.raises(ValueError, match="manifest"):
        study.validate_manifest(altered)


def test_frozen_manifest_write_and_readback_are_deterministic(tmp_path):
    import tree_recursive_null_study as study

    target = tmp_path / "tree-recursive-null.json"
    digest = study.write_manifest(target)
    manifest = json.loads(target.read_text())
    assert digest == study.validate_manifest(manifest)
    assert manifest["runner_minute_ceiling"] == 20_000
    assert manifest["unallocated_runner_minutes"] == 1_319
