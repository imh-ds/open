"""Read frozen N200 evidence to diagnose nuisance links; never fit an estimator.

The exported rows are exploratory descriptions of an already-seen validation
set, not a new validation or a calibrated replacement selection rule.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "examples/results/2026-10-16-multiengine-signal-validation"
RUN = "38891293628"
SOURCE = "e1dc2fa5cf25611ff2ab54499594f6121298bfd7"
MANIFEST = "afbff171ed5bae801c366530028c4fd24bac3f87adb44c73bd904ed18be32c53"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def equal(actual, expected, name):
    if not np.isclose(actual, expected, rtol=1e-12, atol=1e-14):
        raise ValueError("saved arithmetic differs: " + name)


def main():
    import zipfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = load(EVIDENCE / "study-manifest.json")
    report = load(EVIDENCE / "study-report.json")
    if digest(manifest) != MANIFEST or report["source_sha"] != SOURCE:
        raise ValueError("frozen producer identity differs")
    known = {row["case_id"]: row for cell in report["cells"] if cell["sample_size"] == 200
             for row in cell["cases"]}
    records, sampling, warning_counts, fingerprints = [], [], Counter(), []
    completed = []
    for inventory in sorted(EVIDENCE.glob("inventory-*-n200.json"),
                            key=lambda p: ("random_forest" not in p.name, p.name)):
        for bundle_record in load(inventory):
            bundle_path = args.bundles / bundle_record["name"]
            if sha_file(bundle_path) != bundle_record["sha256"]:
                raise ValueError("outer bundle checksum differs")
            fingerprints.append({"name":bundle_path.name,"sha256":bundle_record["sha256"]})
            with zipfile.ZipFile(bundle_path) as bundle:
                if set(bundle.namelist()) != set(bundle_record["members"]):
                    raise ValueError("outer bundle coverage differs")
                for member_name, member in bundle_record["members"].items():
                    case_id = member["case_id"]
                    raw = bundle.read(member_name)
                    raw_sha = hashlib.sha256(raw).hexdigest()
                    if raw_sha != member["sha256"] or raw_sha != known[case_id]["archive_sha256"]:
                        raise ValueError("case archive checksum differs")
                    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                        prefix = "cases/" + case_id + "/"
                        receipt = json.loads(archive.read(prefix+"receipt.json"))
                        if (receipt["github_run"],receipt["github_sha"],receipt["github_attempt"],receipt["manifest"]) != (RUN,SOURCE,"1",MANIFEST):
                            raise ValueError("case receipt producer identity differs")
                        spec = receipt["spec"]
                        if receipt["status"] != "complete" or receipt["fit_cost_unknown"] or spec["id"] != case_id:
                            raise ValueError("case not fully completed")

                        def read_sealed(relative):
                            data = archive.read(prefix+relative)
                            if hashlib.sha256(data).hexdigest() != receipt["files"][relative]:
                                raise ValueError("sealed evidence differs: " + relative)
                            return data

                        network = json.loads(gzip.decompress(read_sealed("network.json.gz")))
                        data_bytes = read_sealed("data.csv")
                        if hashlib.sha256(data_bytes).hexdigest() != manifest["data_sha256"][case_id]:
                            raise ValueError("saved data differs from frozen input")
                        data = pd.read_csv(io.BytesIO(data_bytes),float_precision="round_trip")
                        layer = next(layer for layer in network["layer_evidence.json"]["layers"] if layer["depth"] == 1)
                        target = next(target for target in layer["evidence"] if target["target"] == "X2")
                        if target["candidates"] != ["X1","Z1","Z2"]:
                            raise ValueError("N200 target comparison is not matched")
                        assessments = [a for a in network["assessment_ledger.json"]["assessments"]
                                       if a["depth"] == 1 and a["target"] == "X2"]
                        if len(assessments) != 100 or any(a["status"] != "completed" for a in assessments):
                            raise ValueError("target assessment coverage differs")
                        for assessment in assessments:
                            warning_counts.update(w["category"]+": "+w["message"] for w in assessment["warnings"])
                        for candidate in target["candidates_evidence"]:
                            source = candidate["source"]
                            proximal = sum(a["local"]["classes"].get(source,{}).get("kind") == "PROXIMAL_UNIQUE"
                                           for a in assessments)
                            if proximal != candidate["classes"].get("PROXIMAL_UNIQUE",{}).get("selected",0):
                                raise ValueError("saved promotion numerator differs")
                            unique_baseline = [a["local"]["relevance"]["unique"][source]["raw"] /
                                               a["local"]["relevance"]["baseline_risk"] for a in assessments]
                            unique_gain = [a["local"]["relevance"]["unique"][source]["normalized"] for a in assessments]
                            broad_baseline = [a["local"]["relevance"]["broad"][source]["raw"] /
                                              a["local"]["relevance"]["baseline_risk"] for a in assessments]
                            equal(float(np.median(unique_baseline)),candidate["unique"]["baseline_relative_median"],case_id)
                            equal(float(np.median(broad_baseline)),candidate["broad"]["baseline_relative_median"],case_id)
                            residual = data["X2"]-.6*data["X1"]  # known generator, no fitted coefficients
                            rows = {"case_id":case_id,"engine":spec["engine"],"repetition":spec["repetition"],
                                    "source":source,"target":"X2","population_truth":source=="X1",
                                    "selected":[source,"X2"] in known[case_id]["edges"],
                                    "retention":proximal/100,
                                    "median_unique_gain_fraction":float(np.median([v for v in unique_gain if v is not None])),
                                    "median_unique_baseline_fraction":float(np.median(unique_baseline)),
                                    "median_broad_baseline_fraction":float(np.median(broad_baseline)),
                                    "median_full_gain_baseline_fraction":float(np.median([a["local"]["total_gain_relative"] for a in assessments])),
                                    "sample_r_target":float(data[source].corr(data["X2"])),
                                    "sample_r_known_error":float(data[source].corr(residual)),
                                    "sample_rank_r_known_error":float(data[source].rank().corr(residual.rank())),
                                    "raw_archive_sha256":raw_sha}
                            if rows["selected"] != (proximal>=70):
                                raise ValueError("final target edge differs from original 70% rule")
                            records.append(rows)
                        if spec["engine"] == "random_forest":
                            contexts = defaultdict(list)
                            for assessment in assessments:
                                index = assessment["run_index"]
                                checkpoint = json.loads(gzip.decompress(read_sealed(f"checkpoints/d1-X2/{index:03}.json.gz")))
                                signature = checkpoint.pop("digest")
                                if digest(checkpoint) != signature:
                                    raise ValueError("checkpoint digest differs")
                                risks = [entry["risk"] for entry in checkpoint["risks"] if entry["origin"] == "analysis"]
                                full = next(r for r in risks if r["predictors"] == ["X1","Z1","Z2"])
                                for source in ["Z1","Z2"]:
                                    reduced = next(r for r in risks if r["predictors"] == [c for c in full["predictors"] if c != source])
                                    if reduced["split_plan_id"] != full["split_plan_id"]:
                                        raise ValueError("drop comparison split changed")
                                    equal(reduced["aggregate_risk"]-full["aggregate_risk"],assessment["local"]["relevance"]["unique"][source]["raw"],"drop contrast")
                                    for f,g in zip(full["folds"],reduced["folds"],strict=True):
                                        if (f["repeat_index"],f["fold_index"],f["fitted_parameters"]) != (g["repeat_index"],g["fold_index"],g["fitted_parameters"]):
                                            raise ValueError("drop comparison parameters/fold changed")
                                        fraction = f["fitted_parameters"]["max_features"]
                                        full_count, drop_count = max(1,int(fraction*3)), max(1,int(fraction*2))
                                        difference = g["risk"]-f["risk"]
                                        contexts[(source,fraction,full_count,drop_count)].append(difference)
                            for (source,fraction,full_count,drop_count),differences in contexts.items():
                                sampling.append({"case_id":case_id,"source":source,"max_features":fraction,
                                    "full_features_per_split":full_count,"drop_features_per_split":drop_count,
                                    "nominal_true_predictor_offer_probability_full":full_count/3,
                                    "nominal_true_predictor_offer_probability_drop":drop_count/2,
                                    "saved_fold_contexts":len(differences),"mean_saved_drop_loss_increase":float(np.mean(differences)),
                                    "fraction_saved_drop_worse":float(np.mean(np.array(differences)>0)),
                                    "selected":[source,"X2"] in known[case_id]["edges"]})
                    completed.append(case_id)
                    print(f"read {len(completed)}/165 {case_id}",flush=True)
    if len(completed) != 165 or set(completed) != set(known):
        raise ValueError("not all original N200 cases were examined")
    frame = pd.DataFrame(records)
    frame.to_csv(args.out/"target-comparisons.csv",index=False)
    pd.DataFrame(sampling).to_csv(args.out/"rf-feature-sampling.csv",index=False)
    summaries=[]
    for (engine,truth,selected),group in frame.groupby(["engine","population_truth","selected"]):
        summaries.append({"engine":engine,"population_truth":bool(truth),"selected":bool(selected),"count":len(group),
                          **{column:{"min":float(group[column].min()),"median":float(group[column].median()),"max":float(group[column].max())}
                             for column in ["retention","median_unique_baseline_fraction","median_unique_gain_fraction",
                                            "median_broad_baseline_fraction","sample_r_known_error"]}})
    snapshots=[]
    for cutoff in [.7,.75,.8,.85,.9,.95]:
        for engine,group in frame.groupby("engine"):
            snapshots.append({"cutoff":cutoff,"engine":engine,
                              "retained_true_edges":int(((group.retention>=cutoff)&group.population_truth).sum()),
                              "retained_false_edges":int(((group.retention>=cutoff)&~group.population_truth).sum()),
                              "scope":"fixed already-scored X2 comparisons; posthoc, not a rerun or validation"})
    result={"producer_run":RUN,"producer_source":SOURCE,"manifest_sha256":MANIFEST,"case_count":len(completed),
            "target_comparison_rows":len(frame),"scientific_fits":0,"python":platform.python_version(),
            "diagnostic_source_sha256":sha_file(__file__),"input_bundles":fingerprints,"summaries":summaries,
            "rf_fold_comparison_contexts":sum(r["saved_fold_contexts"] for r in sampling),
            "retention_snapshots":snapshots,"saved_X2_warning_counts":dict(warning_counts),
            "limitations":["posthoc same-data descriptions","folds/subsamples are dependent, not independent replications",
                           "max_features contexts were selected by tuning, not randomized comparisons",
                           "no change to historical results or scientific gates"]}
    (args.out/"diagnosis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({"cases":len(completed),"rows":len(frame),"fits":0}),flush=True)


if __name__ == "__main__":
    main()
