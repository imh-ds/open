"""Installed-wheel regression checks, not scientific simulation calibration.
Run with the clean environment's python -I, outside src, using --root/--out.
Historical fixtures/configs retain their existing samples and assessment budgets.
"""
import argparse, pathlib, json, sys, time, hashlib, shutil, warnings, importlib.metadata as metadata
import numpy as np
import pandas as pd
import open
from open.config import load_config, OpenConfig, AnalysisConfig, StabilityConfig, PerformanceCVConfig, TuningCVConfig
from open.layer_stability import fit_layer_consensus
from open.layer_io import write_layer_outputs, load_layer_bundle
from open.layer_visualization import plot_layer_network
from open.layer_artifacts import portable
from open.metadata import validate_metadata
from open.reporting import analyze_researcher, write_researcher_outputs
import matplotlib
matplotlib.use("Agg")

def digest(path):
 with pathlib.Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("case",choices=("chain","chain_missing","shared_predictors","direct_report"));parser.add_argument("--root",type=pathlib.Path,required=True);parser.add_argument("--out",type=pathlib.Path,required=True);args=parser.parse_args()
 module=pathlib.Path(open.__file__).resolve();prefix=pathlib.Path(sys.prefix).resolve()
 assert module.is_relative_to(prefix) and "site-packages" in module.parts and sys.flags.isolated, (module,prefix)
 out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 proof=dict(case=args.case,source_scope="existing fixed regression fixture, not new independent simulation",installed_module=str(module),environment_prefix=str(prefix),isolated_mode=bool(sys.flags.isolated),python=sys.version,distribution_version=metadata.version("open-predictive-expansion"))
 def meta(frame,root):return validate_metadata(tuple(frame),[{"name":n,"type":"continuous"} for n in frame],root)
 with warnings.catch_warnings(record=True) as captured:
  warnings.simplefilter("always")
  if args.case=="direct_report":
   rng=np.random.default_rng(27);x=rng.normal(size=24);frame=pd.DataFrame({"y":x+rng.normal(scale=.1,size=24),"x":x});frame.to_csv(out/"sample.csv",index=False)
   config=OpenConfig(AnalysisConfig("y",20,max_depth=3),performance_cv=PerformanceCVConfig(2,1),tuning_cv=TuningCVConfig(2),stability=StabilityConfig(n_resamples=1,scope="layer_consensus"))
   result=analyze_researcher(frame,"y",tuple(frame),config,outer_folds=2,effect_bins=2)
   write_researcher_outputs(result,out/"report",metadata=meta(frame,"y"),plots=True)
   saved=json.loads((out/"report/evaluation.json").read_text(encoding="utf-8"));assert len(saved["predictions"])==24 and len(saved["folds"])==2
   assert result.reference.assigned_depth=={"y":0,"x":1};assert result.effects["target"]=="y"
   assert all((out/"report"/n).is_file() for n in ("prediction.png","reference_network.png","root_curves.png","REPORT.md","effects/comparison.json"))
   text=(out/"report/REPORT.md").read_text(encoding="utf-8");assert "not SEM global fit" in text and "not assessed" in text
   before={str(f.relative_to(out)):digest(f) for f in out.rglob("*") if f.is_file()}
   try:write_researcher_outputs(result,out/"report",metadata=meta(frame,"y"),plots=True)
   except FileExistsError:pass
   else:raise AssertionError("existing report overwritten")
   assert before=={str(f.relative_to(out)):digest(f) for f in out.rglob("*") if f.is_file()}
   proof.update(fixture_source="tests/test_reporting.py::data and make_config(n_resamples=1)",complete_predictions=24,outer_folds=2,assigned_depth=result.reference.assigned_depth,effects_present=True,no_overwrite_verified=True,scores=saved["scores"],sample_sha256=digest(out/"sample.csv"))
  else:
   original=args.root/"examples/results/2026-10-08-layer-fixtures"/args.case
   for name in ("sample.csv","config.yaml"):shutil.copyfile(original/name,out/name);assert digest(original/name)==digest(out/name)
   frame=pd.read_csv(out/"sample.csv");config=load_config(out/"config.yaml");root=config.analysis.root;expected=json.loads((original/"summary.json").read_text())
   result=fit_layer_consensus(frame,root,tuple(frame),config)
   assert result.status=="COMPLETE" and result.assigned_depth==expected["assigned_depth"]
   assert {k:list(v) for k,v in result.parents.items()}==expected["parents"]
   evidence=[t for layer in result.layers for t in layer.evidence];assert all(t.complete and len(t.assessments)==20 for t in evidence)
   assert sum(len(t.assessments) for t in evidence)==expected["assessment_count"]
   write_layer_outputs(result,out/"bundle",metadata=meta(frame,root));bundle=load_layer_bundle(out/"bundle");plot_layer_network(bundle,output_path=out/"network.png")
   if args.case=="shared_predictors":
    retained=[r for r in evidence[0].relations if r.qualifies];assert len(retained)==1 and retained[0].relation=="SHARED_GROUP" and set(retained[0].members)=={"Scale_A","Scale_B"}
   missing=[]
   if args.case=="chain_missing":
    target=next(t for t in evidence if t.target=="Depression")
    missing=[dict(observed=z.local.n_observed_target_rows,missing=z.local.n_missing_target_rows) for z in target.assessments]
    assert all(z["observed"]+z["missing"]==64 and z["missing"] in (0,1) for z in missing)
   proof.update(fixture_source=str(original.relative_to(args.root)),sample_sha256=digest(out/"sample.csv"),config_sha256=digest(out/"config.yaml"),status=result.status,assigned_depth=result.assigned_depth,parents=result.parents,assessments=sum(len(t.assessments) for t in evidence),planned_per_target=20,missing_target_coverage=missing,bundle_reload_and_plot=True)
 proof.update(status="passed",seconds=time.perf_counter()-started,warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured],files=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(out.rglob("*")) if f.is_file()])
 (out/"verification.json").write_text(json.dumps(portable(proof),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print(json.dumps(dict(case=args.case,status="passed",seconds=proof["seconds"],warnings=len(captured))),flush=True)

if __name__=="__main__":main()
