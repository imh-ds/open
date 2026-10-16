"""GitHub Actions entrypoints for the tree-engine recursive signal study."""

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

import qualify_tree_recursive_signal as gate
import recursive_preview_design as design
import recursive_preview_runner as runner
import tree_recursive_null_evidence as evidence
import tree_recursive_signal_study as study
import tree_recursive_signal_audit as audit


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/tree-recursive-signal-qualification.yml"
MANIFEST = ROOT / "examples/study_manifests/tree-recursive-signal.json"
ARTIFACT_PREFIX = "tree-signal-case-"
RELEASE_TAG_PREFIX = "tree-recursive-signal-2026-10-16"


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


def expected_identity(spec, manifest, digest, source):
    return {"case_id": spec["id"], "engine": spec["engine"], "sample_size": spec["sample_size"],
            "repetition": spec["repetition"], "study_id": manifest["study_id"],
            "manifest_sha256": digest, "source_sha": source, "attempt": 1}


def aggregate(download_root, output_dir, manifest_path=MANIFEST, *, engine, sample_size):
    """Audit one 55-case cell, keeping expanded evidence bounded per runner."""
    first_attempt()
    _event_is_dispatch()
    manifest, digest = checked_manifest(manifest_path, check_runtime=True)
    source, run_id = os.environ["GITHUB_SHA"], os.environ["GITHUB_RUN_ID"]
    specs = [s for s in study.cases_for_n(manifest, sample_size) if s["engine"] == engine]
    if len(specs) != 55:
        raise ValueError("cell must contain exactly 55 original cases")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    import qualification_artifacts
    errors = []
    try:
        qualification_artifacts.download_artifacts(run_id, [ARTIFACT_PREFIX+s["id"] for s in specs],
                                                   download_root, allow_missing=True)
    except Exception as exc:
        errors.append(type(exc).__name__ + ": " + str(exc)[:2000])
    try:
        transport = verify_transport(run_id, source, manifest)
    except Exception as exc:
        transport = {"verified": False, "errors": [str(exc)]}
    transport["download_errors"] = errors
    transport["verified"] = transport["verified"] and not errors
    rows, archives = [], []
    for spec in specs:
        artifact = Path(download_root) / (ARTIFACT_PREFIX + spec["id"])
        row = audit_case_artifact(expected_identity(spec, manifest, digest, source), artifact,
                                  manifest, source, run_id)
        rows.append(row)
        write_json(out / "case-audits" / (spec["id"] + ".json"), row)
        raw = artifact / "raw/case.zip"
        if raw.is_file():
            archives.append({"case_id": spec["id"], "archive": raw,
                             "sha256": evidence.sha256_file(raw)})
    cell = gate.summarize_cell([expected_identity(s, manifest, digest, source) for s in specs], rows)
    document = {"run_id": run_id, "source_sha": source, "manifest_sha256": digest,
                "attempt": 1, "transport": transport, "cell": cell}
    write_json(out / "cell-report.json", document)
    pub = out / "publication"
    pub.mkdir()
    assets = evidence.build_release_bundles(archives, pub / "bundles",
                                            cell_name=f"tree-signal-{engine}-n{sample_size}") if archives else []
    inventory = evidence.write_inventory(pub / "archive-inventory.json", assets)
    write_json(pub / "publication.json", {"run_id": run_id, "source_sha": source,
               "manifest_sha256": digest, "engine": engine, "sample_size": sample_size, "assets": inventory})
    return document


def combine(download_root, output_dir):
    first_attempt()
    _event_is_dispatch()
    manifest, digest = checked_manifest(check_runtime=True)
    import qualification_artifacts
    names = [f"tree-signal-report-{engine}-n{n}" for n in study.SAMPLE_SIZES for engine in study.ENGINES]
    qualification_artifacts.download_artifacts(os.environ["GITHUB_RUN_ID"], names, download_root)
    documents = [load_json(Path(download_root)/name/"cell-report.json") for name in names]
    for document in documents:
        if (str(document["run_id"]) != os.environ["GITHUB_RUN_ID"]
                or document["source_sha"] != os.environ["GITHUB_SHA"]
                or document["manifest_sha256"] != digest or type(document["attempt"]) is not int
                or document["attempt"] != 1):
            raise ValueError("cell audit producer identity differs")
        cell = document["cell"]
        specs = [s for s in manifest["cases"] if s["engine"] == cell["engine"]
                 and s["sample_size"] == cell["sample_size"]]
        rebuilt = gate.summarize_cell([expected_identity(s, manifest, digest, os.environ["GITHUB_SHA"])
                                      for s in specs], cell["cases"])
        if rebuilt != cell:
            raise ValueError("cell summary differs from its audited cases")
    if {(d["cell"]["engine"],d["cell"]["sample_size"]) for d in documents} != {
            (engine,n) for engine in study.ENGINES for n in study.SAMPLE_SIZES}:
        raise ValueError("combined cells differ from six-cell frozen design")
    transport = all(d["transport"]["verified"] for d in documents)
    waves = [{"sample_size": n, "wave_id": f"n{n}", "run_id": os.environ["GITHUB_RUN_ID"],
              "manifest_sha256": digest, "study_id": manifest["study_id"],
              "source_sha": os.environ["GITHUB_SHA"], "attempt": 1, "transport_verified": True,
              "cells": [d["cell"] for d in documents if d["cell"]["sample_size"] == n]}
             for n in study.SAMPLE_SIZES]
    result = gate.qualify_study(*waves)
    result.update(run_id=os.environ["GITHUB_RUN_ID"], attempt=1, manifest_sha256=digest,
                  transport_verified=transport, cell_transports=[d["transport"] for d in documents])
    if not transport:
        result["engine_eligibility"] = {engine: False for engine in study.ENGINES}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    gate.write_report(result, out/"study-report.json", out/"interpretation.md")
    shutil.copy2(MANIFEST, out/"study-manifest.json")
    shutil.copy2(ROOT/"examples/tree_recursive_signal_validation_protocol.md", out/"study-protocol.md")
    return result


def publish(publication_dir):
    """Publish one cell at a time; never retain all six raw bundles on one runner."""
    first_attempt()
    _event_is_dispatch()
    manifest, digest = checked_manifest(check_runtime=True)
    out = Path(publication_dir)
    run_id, source = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"]
    repo = os.environ["GITHUB_REPOSITORY"]
    tag = RELEASE_TAG_PREFIX + "-" + run_id
    import qualification_artifacts
    report = load_json(out/"study-report.json")
    if report["source_sha"] != source or report["manifest_sha256"] != digest or report["run_id"] != run_id:
        raise ValueError("combined report publication identity differs")
    notes = out/"release-notes.md"
    notes.write_text("Raw matched-chain evidence and independently audited recovery/error results. "
                     "Interpretation is limited to the declared linear Gaussian chain at N80/200.\n", encoding="utf-8")
    subprocess.run(["gh","release","create",tag,"--repo",repo,"--title","Tree recursive matched-chain validation",
                    "--prerelease","--target",source,"--notes-file",str(notes)],check=True)
    verified = []

    def upload_and_verify(path, name=None):
        name = name or path.name
        if path.name != name:
            renamed = out/"named-assets"/name
            renamed.parent.mkdir(exist_ok=True)
            shutil.copy2(path,renamed)
            path = renamed
        subprocess.run(["gh","release","upload",tag,str(path),"--repo",repo],check=True)
        remote = out/"remote-verified"/name
        remote.parent.mkdir(exist_ok=True)
        subprocess.run(["gh","release","download",tag,"--repo",repo,"--pattern",name,
                        "--dir",str(remote.parent)],check=True)
        evidence.verify_uploaded_asset(path,remote)
        verified.append({"name": name,"sha256": evidence.sha256_file(path)})
        remote.unlink()

    for name in ("study-report.json","interpretation.md","study-manifest.json","study-protocol.md"):
        upload_and_verify(out/name)
    for n in study.SAMPLE_SIZES:
        for engine in study.ENGINES:
            name = f"tree-signal-publication-{engine}-n{n}"
            cell_root = out/"cells"/name
            qualification_artifacts.download_artifacts(run_id,[name],cell_root)
            folder = cell_root/name
            record = load_json(folder/"publication.json")
            if (record["run_id"] != run_id or record["source_sha"] != source
                    or record["manifest_sha256"] != digest or record["engine"] != engine
                    or record["sample_size"] != n):
                raise ValueError("raw publication identity differs")
            cell = next(c for c in report["cells"] if c["engine"] == engine and c["sample_size"] == n)
            required = {r["case_id"]: r["archive_sha256"] for r in cell["cases"]
                        if r["status"] == "complete" and r["fit_cost_unknown"] is False}
            allowed = {s["id"] for s in manifest["cases"] if s["engine"] == engine and s["sample_size"] == n}
            bundled = {}
            for asset in record["assets"]:
                for member_name, member in asset["members"].items():
                    case_id = member["case_id"]
                    if case_id not in allowed or case_id in bundled or member_name != f"cases/{case_id}.zip":
                        raise ValueError("raw archive coverage has foreign or duplicate cases")
                    bundled[case_id] = member["sha256"]
            if any(bundled.get(case_id) != sha for case_id,sha in required.items()):
                raise ValueError("verified case archive is absent or differs in publication")
            for asset in record["assets"]:
                path = (folder/asset["path"]).resolve()
                path.relative_to(folder.resolve())
                if evidence.sha256_file(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                    raise ValueError("raw publication bundle changed")
                evidence.verify_release_bundle(path,asset["members"])
                upload_and_verify(path)
                path.unlink()  # verified remote copy plus original Actions artifact retained
            upload_and_verify(folder/"archive-inventory.json",f"inventory-{engine}-n{n}.json")
    write_json(out/"publication-verification.json",{"tag":tag,"verified_assets":verified,"verified":True})
    upload_and_verify(out/"publication-verification.json")
    return {"tag":tag,"verified":True,"asset_count":len(verified)}


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
    agg.add_argument("--engine", choices=study.ENGINES, required=True)
    agg.add_argument("--sample-size", type=int, choices=(80, 200), required=True)
    combine_parser = sub.add_parser("combine")
    combine_parser.add_argument("--downloads", type=Path, required=True)
    combine_parser.add_argument("--out", type=Path, required=True)
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
        result = aggregate(args.downloads, args.out, args.manifest, engine=args.engine, sample_size=args.sample_size)
        print(json.dumps({"transport_verified": result["transport"]["verified"],
                          "cell_passed": result["cell"]["passed"]}, sort_keys=True))
    elif args.command == "combine":
        print(json.dumps(combine(args.downloads, args.out)))
    elif args.command == "publish":
        print(json.dumps(publish(args.publication), sort_keys=True))


if __name__ == "__main__":
    main()
