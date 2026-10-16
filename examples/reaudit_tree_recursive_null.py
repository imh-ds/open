"""Recheck sealed RF archives after audit-only repairs; fitting is prohibited."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import qualify_tree_recursive_validation as gate
import recursive_preview_design as design
import recursive_preview_runner as runner
import tree_recursive_null_evidence as evidence
import tree_recursive_null_study as study
import tree_recursive_qualification_audit as audit

RUN = "38846677048"
SOURCE = "279c380c637ad7b0420df9d93f1e680480f2cf3e"
DIGEST = "3a31a744ed6c38b7b497c5fb6e010ac715b7e597d3a1666f5c21448875c6dccb"
ORIGINAL = ROOT / "examples/results/2026-10-16-multiengine-null-validation"
ADAPTER = "examples/tree_recursive_qualification_audit.py"


def forbidden(*args, **kwargs):
    raise RuntimeError("model fitting is prohibited during saved-evidence audit")


def forbid_fitting():
    import open.layer_stability as stability
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    runner.NATIVE_ASSESS = runner.run_case = stability.assess_layer_target = forbidden
    estimators = [RandomForestRegressor, GradientBoostingRegressor, ElasticNet]
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        pass  # Optional engine; check_inputs still requires it for a real replay.
    else:
        estimators.append(LGBMRegressor)
    for estimator in estimators:
        estimator.fit = forbidden


forbid_fitting()  # Also installed on import in each Windows-spawned worker.


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def check_inputs():
    manifest = load(ORIGINAL / "study-manifest.json")
    original = load(ORIGINAL / "study-report.json")
    if design.digest(manifest) != DIGEST or original["manifest_sha256"] != DIGEST:
        raise ValueError("original frozen manifest digest differs")
    if original["source_sha"] != SOURCE or str(original["run_id"]) != RUN or not original["transport_verified"]:
        raise ValueError("original producer identity/transport differs")
    if platform.python_version() != manifest["python_version"]:
        raise ValueError("audit requires original pinned Python runtime")
    if {n: importlib.metadata.version(n) for n in manifest["runtime"]} != manifest["runtime"]:
        raise ValueError("audit requires original pinned package runtime")
    for relative, expected in manifest["sources"].items():
        if relative == ADAPTER:
            old = subprocess.check_output(["git", "show", SOURCE + ":" + relative], cwd=ROOT)
            if hashlib.sha256(old.replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError("original audit adapter provenance differs")
        elif design.text_hash(ROOT / relative) != expected:
            raise ValueError("producer science/source changed: " + relative)
    return manifest, original


def check_one(task):
    spec, archive, manifest, context, cache = task
    case_id = spec["id"]
    path = Path(cache) / (case_id + ".json")
    binding = dict(context, case_id=case_id, archive_sha256=evidence.sha256_file(archive))
    if path.exists():
        saved = load(path)
        if saved["context"] != binding or saved["row_sha256"] != design.digest(saved["row"]):
            raise ValueError("cached audit identity differs")
        return saved["row"]
    store = audit.legacy.Store(archive, case_id)
    try:
        receipt = store.json("receipt.json")
        if (receipt.get("github_run"), receipt.get("github_sha"), receipt.get("github_attempt")) != (RUN, SOURCE, "1"):
            raise ValueError("original receipt producer identity differs")
        data = design.generate(spec)
        if hashlib.sha256(data.to_csv(index=False, lineterminator="\n").encode()).hexdigest() != manifest["data_sha256"][case_id]:
            raise ValueError("regenerated data differs from sealed input")
        row = audit.audit_case(spec, store, manifest, data=data, config=study.settings(spec))
        if row["status"] != "complete" or row["fit_cost_unknown"] or row["archive_sha256"] != binding["archive_sha256"]:
            raise ValueError("case not fully verified")
        row.update(repetition=spec["repetition"], study_id=manifest["study_id"],
                   source_sha=SOURCE, attempt=1)
    finally:
        store.close()
    write(path, {"context": binding, "row": row, "row_sha256": design.digest(row)})
    return row


def corrected_report(original, manifest, rf_rows):
    report = copy.deepcopy(original)
    for i, cell in enumerate(report["cells"]):
        if cell["engine"] == "random_forest":
            actual = [row for row in rf_rows if row["sample_size"] == cell["sample_size"]]
        else:
            # These network rows already passed independent checkpoint replay.
            # Reclassify their endpoint from the saved reconstructed network.
            actual = copy.deepcopy(cell["cases"])
            for row in actual:
                if row["status"] != "complete" or row["fit_cost_unknown"]:
                    raise ValueError("prior non-RF result was not independently audited")
                row["false_network"] = bool(row["edges"] or row["groups"] or
                                            any(depth > 0 for depth in row["assigned_depth"].values()))
        report["cells"][i] = gate.summarize_cell(cell["cases"], actual)
    report["engine_eligibility"] = {
        engine: all(cell["passed"] for cell in report["cells"] if cell["engine"] == engine)
        for engine in study.ENGINES
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("use 1 to 8 independent audit workers")
    manifest, original = check_inputs()
    context = {"producer_run": RUN, "producer_source": SOURCE, "manifest_sha256": DIGEST,
               "adapter_sha256": design.text_hash(ROOT / ADAPTER),
               "replay_sha256": design.text_hash(Path(__file__)),
               "original_report_sha256": evidence.sha256_file(ORIGINAL / "study-report.json"),
               "audit_source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tasks = []
    release = load(ORIGINAL / "release-api.json")
    for bundle in load(ORIGINAL / "archive-inventory.json"):
        if "random_forest" not in bundle["name"]:
            continue
        path = args.bundles / bundle["name"]
        remote = next(a for a in release["assets"] if a["name"] == bundle["name"])
        if (remote["digest"] != "sha256:" + bundle["sha256"] or remote["size"] != bundle["bytes"]
                or evidence.sha256_file(path) != bundle["sha256"]):
            raise ValueError("RF bundle hash/size differs from published inventory")
        evidence.verify_release_bundle(path, bundle["members"])
        with zipfile.ZipFile(path) as archive:
            for name, member in bundle["members"].items():
                spec = next(s for s in manifest["cases"] if s["id"] == member["case_id"])
                if spec["engine"] != "random_forest" or name != "cases/" + spec["id"] + ".zip":
                    raise ValueError("foreign bundle member")
                target = raw / (spec["id"] + ".zip")
                target.write_bytes(archive.read(name))
                tasks.append((spec, target, manifest, context, args.out / "cache"))
    if len(tasks) != 110 or len({t[0]["id"] for t in tasks}) != 110:
        raise ValueError("all 110 original RF case archives are required")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(check_one, task) for task in tasks]):
            row = future.result()
            rows.append(row)
            print(f"verified {len(rows)}/110 {row['case_id']} false_network={row['false_network']}", flush=True)
    report = corrected_report(original, manifest, rows)
    report["audit_correction"] = dict(context, scientific_fits=0, rf_archives_verified=110,
                                       non_rf_endpoints_reclassified=220,
                                       original_report_retained=True, auditor_runtime=manifest["runtime"])
    gate.write_report(report, args.out / "study-report-corrected.json", args.out / "interpretation-corrected.md")
    print(json.dumps({"eligibility": report["engine_eligibility"], "scientific_fits": 0}), flush=True)


if __name__ == "__main__":
    main()
