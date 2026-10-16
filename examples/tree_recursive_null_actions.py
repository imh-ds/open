"""GitHub Actions entrypoints for the tree-engine recursive null study."""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile

import pandas as pd

import qualify_tree_recursive_validation as gate
import recursive_preview_design as design
import recursive_preview_runner as runner
import tree_recursive_null_evidence as evidence
import tree_recursive_null_study as study
import tree_recursive_qualification_audit as audit


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/tree-recursive-qualification.yml"
MANIFEST = ROOT / "examples/study_manifests/tree-recursive-null.json"
ARTIFACT_PREFIX = "tree-null-case-"
RELEASE_TAG_PREFIX = "tree-recursive-null-2026-10-15"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def first_attempt():
    if (os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
            or not os.environ.get("GITHUB_RUN_ID")
            or not os.environ.get("GITHUB_SHA")):
        raise ValueError("requires first Actions attempt with run/source identity; no automatic retries")


def soft_compute_seconds(cap_minutes, *, job_started_at, now=None):
    """Reserve ten minutes inside each job cap for setup, packing and upload."""
    if type(cap_minutes) is not int or cap_minutes <= 0:
        raise ValueError("case cap must be a positive integer number of minutes")
    started = float(job_started_at)
    current = time.time() if now is None else float(now)
    elapsed = max(0, int(current - started))
    return max(0, cap_minutes * 60 - elapsed - 600)


def verify_sources(manifest):
    for relative, expected in manifest["sources"].items():
        path = ROOT / relative
        if not path.is_file() or design.text_hash(path) != expected:
            raise ValueError("frozen study source hash mismatch: " + relative)


def runtime_check(manifest):
    if platform.python_version() != manifest["python_version"]:
        raise ValueError("Python runtime differs from frozen manifest")
    actual = {name: importlib.metadata.version(name) for name in manifest["runtime"]}
    expected = dict(manifest["runtime"])
    if actual != expected:
        raise ValueError("installed package runtime differs from frozen manifest")
    verify_sources(manifest)


def checked_manifest(path=MANIFEST, *, check_runtime=False):
    value = load_json(path)
    digest = study.validate_manifest(value)
    if check_runtime:
        runtime_check(value)
    return value, digest


def _event_is_dispatch():
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("scientific workflow is permitted only by workflow_dispatch")


def prepare(output_dir, manifest_path=MANIFEST):
    first_attempt()
    _event_is_dispatch()
    value, _ = checked_manifest(manifest_path, check_runtime=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", value)
    result = {}
    for sample_size in study.SAMPLE_SIZES:
        wave = study.prepare_wave(
            sample_size, value, output / f"wave-n{sample_size}",
            source_sha=os.environ["GITHUB_SHA"], run_id=os.environ["GITHUB_RUN_ID"],
        )
        matrix = load_json(wave / "matrix.json")
        result[f"matrix{sample_size}"] = json.dumps(matrix, sort_keys=True, separators=(",", ":"))
    return result


def _write_incomplete_case(spec, manifest, case_dir, message):
    case_dir.mkdir(parents=True, exist_ok=False)
    data = design.generate(spec)
    config = study.settings(spec)
    data.to_csv(case_dir / "data.csv", index=False, lineterminator="\n")
    runner.write_json(case_dir / "config.json", runner.portable(config))
    observed_runtime = {}
    for name in manifest["runtime"]:
        try:
            observed_runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed_runtime[name] = None
    receipt = {
        "spec": spec,
        "config": runner.portable(config),
        "context": runner.case_context(spec, data, config, manifest),
        "manifest": design.digest(manifest),
        "status": "incomplete",
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run": os.environ.get("GITHUB_RUN_ID"),
        "github_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python": platform.python_version(),
        "module": str(runner.package.__file__),
        "runtime": observed_runtime,
        "workers": 4,
        "fit_cost_unknown": True,
        "failure": message[:2000],
        "files": runner.file_hashes(case_dir),
    }
    runner.write_json(case_dir / "receipt.json", receipt)


def run_case(case_id, authorization_dir, output_dir, *, manifest_path=MANIFEST,
             job_started_at, expected_sample_size):
    first_attempt()
    _event_is_dispatch()
    manifest, manifest_digest = checked_manifest(manifest_path, check_runtime=True)
    authorization_dir, output_dir = Path(authorization_dir), Path(output_dir)
    controls = [p for p in authorization_dir.rglob("control.json") if p.parent.name == case_id]
    if len(controls) != 1:
        raise ValueError("case authorization is missing or duplicated")
    control = load_json(controls[0])
    spec = control.get("spec")
    expected = [row for row in study.cases_for_n(manifest, expected_sample_size) if row["id"] == case_id]
    if (len(expected) != 1 or spec != expected[0] or control.get("schema") != study.SCHEMA
            or control.get("manifest_sha256") != manifest_digest
            or control.get("source_sha") != os.environ["GITHUB_SHA"]
            or control.get("run_id") != os.environ["GITHUB_RUN_ID"]
            or type(control.get("attempt")) is not int or control["attempt"] != 1
            or type(control.get("sample_size")) is not int
            or control.get("sample_size") != expected_sample_size
            or type(control.get("case_cap_minutes")) is not int
            or control.get("case_cap_minutes") != study.CASE_CAPS[spec["engine"]][expected_sample_size]):
        raise ValueError("case authorization identity/source/attempt differs")
    data_path = controls[0].parent / "data.csv"
    if evidence.sha256_file(data_path) != control.get("data_sha256"):
        raise ValueError("authorized data hash differs")
    data = pd.read_csv(data_path, float_precision="round_trip")
    if evidence.sha256_file(data_path) != manifest["data_sha256"][case_id]:
        raise ValueError("prepared input differs from frozen manifest")
    config = study.settings(spec)
    config_path = controls[0].parent / "config.json"
    if design.digest(load_json(config_path)) != design.digest(runner.portable(config)):
        raise ValueError("prepared configuration differs from frozen settings")

    output_dir.mkdir(parents=True, exist_ok=False)
    case_dir = output_dir / "cases" / case_id
    cap_seconds = control["case_cap_minutes"] * 60
    elapsed = max(0, int(time.time() - float(job_started_at)))
    soft_seconds = soft_compute_seconds(control["case_cap_minutes"], job_started_at=job_started_at)
    failure = None
    try:
        if soft_seconds <= 0:
            _write_incomplete_case(spec, manifest, case_dir, "case cap reached before fit step; no assessment started")
            failure = "case cap reached before fit step"
        else:
            runner.run_case(spec, case_dir, manifest, data=data, config=config, seconds=soft_seconds)
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)[:2000]
    status = "complete" if failure is None else "incomplete"
    packed = evidence.pack_case(case_dir, case_id, output_dir / "raw" / "case.zip")
    control.update({
        "status": status,
        "fit_cost_unknown": failure is not None,
        "failure": failure,
        "case_cap_minutes": cap_seconds // 60,
        "job_elapsed_seconds_before_fit": elapsed,
        "soft_compute_seconds": max(0, soft_seconds),
        "archive_sha256": packed["sha256"],
        "archive_bytes": packed["bytes"],
    })
    write_json(output_dir / "control.json", control)
    return control


def _unknown_result(expected, reason):
    return {
        **expected,
        "status": "audit_failed",
        "fit_cost_unknown": True,
        "false_network": None,
        "archive_sha256": None,
        "failure": reason[:2000],
    }


def audit_case_artifact(expected, artifact_dir, manifest, source_sha, run_id):
    artifact_dir = Path(artifact_dir)
    control_path = artifact_dir / "control.json"
    archive_path = artifact_dir / "raw" / "case.zip"
    if not control_path.is_file() or not archive_path.is_file():
        return _unknown_result(expected, "case artifact or raw archive is missing")
    try:
        control = load_json(control_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _unknown_result(expected, f"case control cannot be read: {exc}")
    if not isinstance(control, dict):
        return _unknown_result(expected, "case control must be a JSON object")
    if any(type(control.get(field)) is not int for field in ("sample_size", "repetition", "attempt")):
        return _unknown_result(expected, "case control numeric identity must use integer values")
    for field in ("case_id", "engine", "sample_size", "repetition", "study_id",
                  "manifest_sha256", "source_sha", "attempt"):
        if control.get(field) != expected[field]:
            return _unknown_result(expected, f"case control {field} differs")
    if control.get("run_id") != run_id or source_sha != expected["source_sha"]:
        return _unknown_result(expected, "case run/source provenance differs")
    if control.get("status") != "complete" or control.get("fit_cost_unknown") is not False:
        return _unknown_result(expected, control.get("failure") or "case fit cost is unknown")
    if evidence.sha256_file(archive_path) != control.get("archive_sha256"):
        return _unknown_result(expected, "raw archive SHA differs from control")
    try:
        with zipfile.ZipFile(archive_path) as archive_file:
            if (archive_file.testzip() is not None
                    or any(not name.startswith(f"cases/{expected['case_id']}/")
                           for name in archive_file.namelist())):
                raise ValueError("raw archive CRC/path differs")
        store = audit.legacy.Store(archive_path, expected["case_id"])
        try:
            receipt = store.json("receipt.json")
            if (receipt.get("github_run") != run_id or receipt.get("github_sha") != source_sha
                    or receipt.get("github_attempt") != "1"):
                raise ValueError("case receipt run/source/attempt differs")
            spec = next(row for row in manifest["cases"] if row["id"] == expected["case_id"])
            data = design.generate(spec)
            if hashlib.sha256(data.to_csv(index=False, lineterminator="\n").encode()).hexdigest() != manifest["data_sha256"][spec["id"]]:
                raise ValueError("recomputed input data hash differs")
            result = audit.audit_case(spec, store, manifest, data=data,
                                      config=study.settings(spec))
        finally:
            store.close()
    except Exception as exc:
        return _unknown_result(expected, type(exc).__name__ + ": " + str(exc))
    result.update({
        "case_id": expected["case_id"],
        "engine": expected["engine"],
        "sample_size": expected["sample_size"],
        "repetition": expected["repetition"],
        "study_id": expected["study_id"],
        "manifest_sha256": expected["manifest_sha256"],
        "source_sha": source_sha,
        "attempt": 1,
        "status": "complete",
        "fit_cost_unknown": False,
        "archive_sha256": control["archive_sha256"],
        "false_network": result["false_network"],
    })
    return result


def _github_api(path):
    from recursive_preview_actions import github_api

    return github_api(path)


def verify_transport(run_id, source_sha, manifest, *, api=_github_api):
    state = api(f"actions/runs/{run_id}")
    errors = []
    if (state.get("head_sha") != source_sha or state.get("run_attempt") != 1
            or state.get("event") != "workflow_dispatch" or state.get("path") != WORKFLOW):
        errors.append("workflow run/source/attempt differs")
    records, page = [], 1
    while True:
        payload = api(f"actions/runs/{run_id}/jobs?per_page=100&page={page}")
        records.extend(payload.get("jobs", []))
        if len(payload.get("jobs", [])) < 100:
            break
        page += 1
    expected = {"prepare"}
    expected.update(f"case N=80 ({case['id']})" for case in study.cases_for_n(manifest, 80))
    expected.update(f"case N=200 ({case['id']})" for case in study.cases_for_n(manifest, 200))
    found = {job.get("name"): job for job in records}
    if len(found) != len(records):
        errors.append("duplicate workflow job names")
    missing_jobs = expected - set(found)
    unexpected_cases = {
        name for name in found
        if name.startswith("case N=") and name not in expected
    }
    if missing_jobs or unexpected_cases:
        errors.append("workflow job matrix does not match frozen 330-case plan")
    for name in expected & set(found):
        job = found[name]
        required_step = "Upload authorization" if name == "prepare" else "Upload case evidence"
        uploads = [step for step in job.get("steps", []) if step.get("name") == required_step]
        if len(uploads) != 1 or uploads[0].get("conclusion") != "success":
            errors.append(f"required artifact upload missing or failed: {name}")
    return {"verified": not errors, "run_id": str(run_id), "source_sha": source_sha,
            "planned_case_jobs": 330, "observed_jobs": len(records), "errors": errors}


def aggregate(download_root, output_dir, manifest_path=MANIFEST):
    first_attempt()
    _event_is_dispatch()
    manifest, manifest_digest = checked_manifest(manifest_path, check_runtime=True)
    source_sha, run_id = os.environ["GITHUB_SHA"], os.environ["GITHUB_RUN_ID"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    names = [ARTIFACT_PREFIX + case["id"] for case in manifest["cases"]]
    transport_errors = []
    try:
        import qualification_artifacts

        qualification_artifacts.download_artifacts(run_id, names, download_root, allow_missing=True)
    except Exception as exc:
        transport_errors.append(type(exc).__name__ + ": " + str(exc)[:2000])
    try:
        transport = verify_transport(run_id, source_sha, manifest)
    except Exception as exc:
        transport = {"verified": False, "errors": [type(exc).__name__ + ": " + str(exc)[:2000]]}
    transport["download_errors"] = transport_errors
    if transport_errors:
        transport["verified"] = False

    rows_by_n = {80: [], 200: []}
    archive_records = defaultdict(list)
    for spec in manifest["cases"]:
        expected = {
            "case_id": spec["id"],
            "engine": spec["engine"],
            "sample_size": spec["sample_size"],
            "repetition": spec["repetition"],
            "study_id": manifest["study_id"],
            "manifest_sha256": manifest_digest,
            "source_sha": source_sha,
            "attempt": 1,
        }
        artifact = Path(download_root) / (ARTIFACT_PREFIX + spec["id"])
        result = audit_case_artifact(expected, artifact, manifest, source_sha, run_id)
        rows_by_n[spec["sample_size"]].append(result)
        audit_path = output_dir / "case-audits" / f"{spec['id']}.json"
        write_json(audit_path, result)
        archive_path = artifact / "raw" / "case.zip"
        if archive_path.is_file():
            archive_records[(spec["engine"], spec["sample_size"])].append({
                "case_id": spec["id"], "archive": archive_path,
                "sha256": evidence.sha256_file(archive_path),
            })

    waves = []
    for sample_size in (80, 200):
        cells = []
        for engine in study.ENGINES:
            expected = [
                {
                    "case_id": case["id"], "engine": engine, "sample_size": sample_size,
                    "repetition": case["repetition"], "study_id": manifest["study_id"],
                    "manifest_sha256": manifest_digest, "source_sha": source_sha, "attempt": 1,
                }
                for case in study.cases_for_n(manifest, sample_size) if case["engine"] == engine
            ]
            audited = [row for row in rows_by_n[sample_size] if row["engine"] == engine]
            cells.append(gate.summarize_cell(expected, audited))
        waves.append({
            "sample_size": sample_size,
            "wave_id": f"n{sample_size}",
            "manifest_sha256": manifest_digest,
            "study_id": manifest["study_id"],
            "source_sha": source_sha,
            "run_id": run_id,
            "attempt": 1,
            "transport_verified": transport["verified"],
            "cells": cells,
        })
    try:
        if transport["verified"]:
            qualification = gate.qualify_study(waves[0], waves[1])
        else:
            qualification = {
                "schema": study.SCHEMA,
                "study_id": manifest["study_id"],
                "source_sha": source_sha,
                "transport_verified": False,
                "cells": sorted(waves[0]["cells"] + waves[1]["cells"],
                                key=lambda cell: (cell["engine"], cell["sample_size"])),
                "engine_eligibility": {engine: False for engine in study.ENGINES},
                "blocker": "Actions artifact transport was not fully verified",
            }
    except Exception as exc:
        qualification = {
            "schema": study.SCHEMA,
            "study_id": manifest["study_id"],
            "source_sha": source_sha,
            "transport_verified": False,
            "cells": sorted(waves[0]["cells"] + waves[1]["cells"],
                            key=lambda cell: (cell["engine"], cell["sample_size"])),
            "engine_eligibility": {engine: False for engine in study.ENGINES},
            "blocker": type(exc).__name__ + ": " + str(exc)[:2000],
        }
    qualification["transport"] = transport
    qualification["run_id"] = run_id
    qualification["attempt"] = 1
    qualification["manifest_sha256"] = manifest_digest
    gate.write_report(qualification, output_dir / "study-report.json", output_dir / "interpretation.md")

    publication_dir = output_dir / "publication"
    publication_dir.mkdir()
    shutil.copy2(manifest_path, publication_dir / "study-manifest.json")
    shutil.copy2(ROOT / "examples/tree_recursive_null_validation_protocol.md",
                 publication_dir / "study-protocol.md")
    shutil.copy2(output_dir / "study-report.json", publication_dir / "study-report.json")
    shutil.copy2(output_dir / "interpretation.md", publication_dir / "interpretation.md")
    asset_records = []
    for sample_size in (80, 200):
        for engine in study.ENGINES:
            records = archive_records.get((engine, sample_size), [])
            if records:
                asset_records.extend(evidence.build_release_bundles(
                    records, publication_dir / f"bundles-{engine}-n{sample_size}",
                    cell_name=f"tree-null-{engine}-n{sample_size}",
                ))
    inventory = evidence.write_inventory(publication_dir / "archive-inventory.json", asset_records)
    write_json(publication_dir / "publication.json", {
        "study_id": manifest["study_id"],
        "run_id": run_id,
        "source_sha": source_sha,
        "transport_verified": transport["verified"],
        "assets": inventory,
    })
    return qualification


def publish(publication_dir, *, run=subprocess.run):
    first_attempt()
    _event_is_dispatch()
    publication_dir = Path(publication_dir)
    publication = load_json(publication_dir / "publication.json")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repository:
        raise ValueError("release publication requires scoped GitHub credentials")
    assets = [publication_dir / row["path"] for row in publication["assets"]]
    assets.extend(publication_dir / name for name in (
        "study-manifest.json", "study-protocol.md", "study-report.json", "interpretation.md",
        "archive-inventory.json", "publication.json"
    ))
    if any(not path.is_file() for path in assets):
        raise ValueError("release publication asset is missing")
    environment = dict(os.environ, GH_TOKEN=token)
    release_tag = f"{RELEASE_TAG_PREFIX}-{os.environ['GITHUB_RUN_ID']}"
    view = run(["gh", "release", "view", release_tag, "--repo", repository],
               capture_output=True, text=True, env=environment)
    if view.returncode != 0:
        notes = publication_dir / "release-notes.md"
        notes.write_text(
            "Durable raw evidence and independently audited reports for the tree-engine recursive null study.\n"
            "Eligibility is limited to the declared null population and requires a separate signal-stage review.\n",
            encoding="utf-8", newline="\n",
        )
        run(["gh", "release", "create", release_tag, "--repo", repository,
             "--title", "Tree recursive null validation", "--notes-file", str(notes),
             "--prerelease", "--target", os.environ["GITHUB_SHA"]], check=True, env=environment)
    run(["gh", "release", "upload", release_tag, *map(str, assets),
         "--repo", repository, "--clobber"], check=True, env=environment)
    download = publication_dir / "uploaded-assets"
    download.mkdir(exist_ok=False)
    for asset in assets:
        run(["gh", "release", "download", release_tag, "--repo", repository,
             "--pattern", asset.name, "--dir", str(download)], check=True, env=environment)
        evidence.verify_uploaded_asset(asset, download / asset.name)
    return {"tag": release_tag, "verified_assets": len(assets), "verified": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--out", type=Path, default=MANIFEST)
    check = sub.add_parser("check")
    check.add_argument("--manifest", type=Path, default=MANIFEST)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, default=MANIFEST)
    prep.add_argument("--out", type=Path, required=True)
    case = sub.add_parser("case")
    case.add_argument("--id", required=True)
    case.add_argument("--sample-size", type=int, choices=(80, 200), required=True)
    case.add_argument("--authorization", type=Path, required=True)
    case.add_argument("--manifest", type=Path, default=MANIFEST)
    case.add_argument("--out", type=Path, required=True)
    case.add_argument("--job-started-at", type=float, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--manifest", type=Path, default=MANIFEST)
    agg.add_argument("--downloads", type=Path, required=True)
    agg.add_argument("--out", type=Path, required=True)
    pub = sub.add_parser("publish")
    pub.add_argument("--publication", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps({"manifest_sha256": study.write_manifest(args.out), "fit_count": 0}))
    elif args.command == "check":
        value, digest = checked_manifest(args.manifest, check_runtime=True)
        print(json.dumps({"manifest_sha256": digest, "cases": len(value["cases"]), "fit_count": 0}))
    elif args.command == "prepare":
        print(json.dumps(prepare(args.out, args.manifest), separators=(",", ":")))
    elif args.command == "case":
        result = run_case(args.id, args.authorization, args.out, manifest_path=args.manifest,
                          job_started_at=args.job_started_at, expected_sample_size=args.sample_size)
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "complete":
            raise SystemExit("case incomplete; retained evidence and stopped without retry")
    elif args.command == "aggregate":
        result = aggregate(args.downloads, args.out, args.manifest)
        print(json.dumps({"transport_verified": result["transport_verified"],
                          "engine_eligibility": result["engine_eligibility"]}, sort_keys=True))
    elif args.command == "publish":
        print(json.dumps(publish(args.publication), sort_keys=True))


if __name__ == "__main__":
    main()
