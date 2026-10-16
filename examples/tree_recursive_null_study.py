"""Prepare immutable paired null cases for RF, GBM and LightGBM validation.

Preparation generates data/configuration controls only. It never invokes the
analysis engine or fits a model.
"""

from dataclasses import replace
import hashlib
import json
import re
from pathlib import Path

import recursive_preview_design as design
from open.layer_artifacts import portable
from open.learners.trees import GRIDS


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "open.tree_recursive_null.v1"
STUDY_ID = "tree-recursive-null-2026-10-15"
ENGINES = ("random_forest", "gbm", "lightgbm")
SAMPLE_SIZES = (80, 200)
REPETITIONS = tuple(range(1, 56))
POLICY = "tree-null-v1"
CASE_CAPS = {
    "random_forest": {80: 60, 200: 100},
    "gbm": {80: 45, 200: 70},
    "lightgbm": {80: 25, 200: 35},
}
RUNTIME = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.11.2",
    "scikit-learn": "1.3.0",
    "PyYAML": "6.0.2",
    "matplotlib": "3.9.2",
    "lightgbm": "4.6.0",
    "setuptools": "75.1.0",
    "wheel": "0.45.1",
}
SOURCE_FILES = (
    "configs/default.yaml",
    "pyproject.toml",
    "examples/recursive_preview_design.py",
    "examples/recursive_preview_runner.py",
    "examples/recursive_preview_actions.py",
    "examples/recursive_preview_equivalence.py",
    "examples/recursive_preview_audit.py",
    "examples/qualification_artifacts.py",
    "examples/multiengine_recursive_audit.py",
    "examples/optimized_rf_recursive_audit.py",
    "examples/tree_recursive_qualification_audit.py",
    "examples/qualify_tree_recursive_validation.py",
    "examples/tree_recursive_null_study.py",
    "examples/tree_recursive_null_evidence.py",
    "examples/tree_recursive_null_actions.py",
    ".github/workflows/tree-recursive-qualification.yml",
    "examples/tree_recursive_null_validation_protocol.md",
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(portable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _text_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _seeds(sample_size, repetition):
    return (
        2_600_000_000 + sample_size * 100 + repetition,
        2_700_000_000 + sample_size * 100 + repetition,
    )


def case_specs():
    specs = []
    for sample_size in SAMPLE_SIZES:
        for repetition in REPETITIONS:
            generator_seed, analysis_seed = _seeds(sample_size, repetition)
            for engine in ENGINES:
                specs.append({
                    "id": f"null-{engine}-n{sample_size}-r{repetition:02d}",
                    "scenario": "noise",
                    "n": sample_size,
                    "sample_size": sample_size,
                    "repetition": repetition,
                    "engine": engine,
                    "generator_seed": generator_seed,
                    "analysis_seed": analysis_seed,
                })
    return specs


def cases_for_n(manifest, sample_size):
    if sample_size not in SAMPLE_SIZES:
        raise ValueError("sample size must be 80 or 200")
    return [case for case in manifest["cases"] if case["sample_size"] == sample_size]


def settings(spec):
    config = design.settings(spec)
    return replace(config, learner=replace(config.learner, library=(spec["engine"],)))


def _source_paths():
    src_files = sorted(path for path in (ROOT / "src/open").rglob("*.py") if path.is_file())
    explicit = [ROOT / relative for relative in SOURCE_FILES]
    return sorted({path for path in (*src_files, *explicit) if path.is_file()})


def build_manifest():
    cases = case_specs()
    configurations = {case["id"]: portable(settings(case)) for case in cases}
    data_hashes = {
        case["id"]: hashlib.sha256(
            design.generate(case).to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest()
        for case in cases
    }
    sources = {
        path.relative_to(ROOT).as_posix(): _text_hash(path)
        for path in _source_paths()
    }
    return {
        "schema": SCHEMA,
        "study_id": STUDY_ID,
        "stage": "null-only",
        "engines": list(ENGINES),
        "sample_sizes": list(SAMPLE_SIZES),
        "repetitions": list(REPETITIONS),
        "cases": cases,
        "python_version": "3.11.9",
        "runtime": dict(RUNTIME),
        "sources": sources,
        "configurations": configurations,
        "data_sha256": data_hashes,
        "grids": {engine: portable(GRIDS[engine]) for engine in ENGINES},
        "grid_policy": POLICY,
        "analysis": {
            "resamples": 100,
            "subsample_fraction": 0.8,
            "performance_cv": {"folds": 5, "repeats": 2},
            "tuning_cv_folds": 3,
            "trees": 100,
            "workers_per_case": 4,
            "numerical_threads_per_worker": 1,
        },
        "case_minutes": {
            engine: {str(n): minutes for n, minutes in CASE_CAPS[engine].items()}
            for engine in ENGINES
        },
        "max_parallel_per_wave": 10,
        "case_job_count_per_wave": 165,
        "planned_cases_per_cell": 55,
        "reserved_runner_minutes": 18_681,
        "runner_minute_ceiling": 20_000,
        "unallocated_runner_minutes": 1_319,
        "scientific_fits_performed_during_preparation": 0,
    }


def validate_manifest(value):
    if not isinstance(value, dict) or value != build_manifest():
        raise ValueError("manifest differs from frozen source/design/settings/runtime")
    return _digest(value)


def write_manifest(path):
    """Write the current source/data/runtime manifest without fitting models."""
    value = build_manifest()
    write_json(path, value)
    return _digest(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(value), sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def prepare_wave(sample_size, manifest, output_dir, *, source_sha, run_id):
    """Write one deterministic 165-case wave with paired data and no model fits."""
    validate_manifest(manifest)
    if sample_size not in SAMPLE_SIZES:
        raise ValueError("sample size must be 80 or 200")
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source SHA must be a full lowercase Git SHA")
    if not isinstance(run_id, str) or not run_id.isdecimal():
        raise ValueError("GitHub Actions run ID must be decimal text")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_sha = _digest(manifest)
    cases = cases_for_n(manifest, sample_size)
    if len(cases) != 165:
        raise ValueError("wave must contain exactly 165 frozen cases")
    includes = []
    for spec in cases:
        case_dir = output_dir / "cases" / spec["id"]
        case_dir.mkdir(parents=True, exist_ok=False)
        data = design.generate(spec)
        csv = data.to_csv(index=False, lineterminator="\n").encode()
        config = portable(settings(spec))
        data_sha = hashlib.sha256(csv).hexdigest()
        if data_sha != manifest["data_sha256"][spec["id"]]:
            raise ValueError("prepared data differs from frozen input hash")
        (case_dir / "data.csv").write_bytes(csv)
        write_json(case_dir / "config.json", config)
        control = {
            "schema": SCHEMA,
            "study_id": manifest["study_id"],
            "case_id": spec["id"],
            "spec": spec,
            "engine": spec["engine"],
            "sample_size": sample_size,
            "repetition": spec["repetition"],
            "attempt": 1,
            "source_sha": source_sha,
            "run_id": run_id,
            "manifest_sha256": manifest_sha,
            "data_sha256": data_sha,
            "config_sha256": _digest(config),
            "case_cap_minutes": CASE_CAPS[spec["engine"]][sample_size],
            "fit_cost_unknown": True,
        }
        write_json(case_dir / "control.json", control)
        includes.append({"id": spec["id"], "engine": spec["engine"], "sample_size": sample_size,
                         "case_cap_minutes": CASE_CAPS[spec["engine"]][sample_size]})
    result = {
        "schema": SCHEMA,
        "study_id": manifest["study_id"],
        "sample_size": sample_size,
        "manifest_sha256": manifest_sha,
        "source_sha": source_sha,
        "run_id": run_id,
        "attempt": 1,
        "planned": len(cases),
        "scientific_fits": 0,
        "matrix": {"include": includes},
    }
    write_json(output_dir / "wave.json", result)
    write_json(output_dir / "matrix.json", result["matrix"])
    return output_dir
