"""No-fit completion audit of saved installed-wheel checks.
The initial recorder's mappingproxy serialization failed after its workflow
assertions. This verifier reads and checks those complete outputs, without fits.
"""
import pathlib,json,sys,argparse,hashlib,importlib.metadata as metadata
import numpy as np
import pandas as pd
import open
from open.layer_io import load_layer_bundle,write_layer_outputs
from open.layer_artifacts import portable
from open.layer_visualization import plot_layer_network
from open.metadata import validate_metadata
from open.reporting import ResearcherAnalysis,write_researcher_outputs

def digest(p):
 with pathlib.Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("case");p.add_argument("--root",type=pathlib.Path,required=True);p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args();out=a.out
 module=pathlib.Path(open.__file__).resolve();assert module.is_relative_to(pathlib.Path(sys.prefix).resolve()) and "site-packages" in module.parts and sys.flags.isolated
 if a.case=="direct_report":
  d=out/"report";v=json.loads((d/"evaluation.json").read_text(encoding="utf-8"));assert v["status"]=="COMPLETE" and not v["supplement_failures"]
  assert len(v["predictions"])==24 and {x["row_position"] for x in v["predictions"]}==set(range(24)) and len(v["folds"])==2
  frame=pd.read_csv(out/"sample.csv");rng=np.random.default_rng(27);x=rng.normal(size=24);assert np.allclose(frame["x"],x,rtol=0,atol=1e-15) and np.allclose(frame["y"],x+rng.normal(scale=.1,size=24),rtol=0,atol=1e-15)
  for row in v["predictions"]:
   assert np.isfinite([row[k] for k in ("observed","baseline","reduced","full")]).all()
   fold=v["folds"][row["fold"]];assert row["row_position"] not in fold["train_positions"]
  dirs=[d/"reference",d/"fold_0",d/"fold_1"]
  for b in dirs:load_layer_bundle(b)
  meta=json.loads((d/"reference/analysis_metadata.json").read_text());assert meta["assigned_depth"]=={"y":0,"x":1}
  assert all((d/n).is_file() for n in ("prediction.png","reference_network.png","root_curves.png","REPORT.md","effects/comparison.json"))
  text=(d/"REPORT.md").read_text(encoding="utf-8");assert "not SEM global fit" in text and "not assessed" in text
  extra=dict(predictions=24,assessment_folds=2,effects_and_three_graphs=True,each_person_held_out=True)
 else:
  d=out/"bundle";bundle=load_layer_bundle(d);meta=json.loads((d/"analysis_metadata.json").read_text());expected=a.root/"examples/results/2026-10-08-layer-fixtures"/a.case
  reference=json.loads((expected/"summary.json").read_text());assert meta["assigned_depth"]==reference["assigned_depth"] and meta["parents"]==reference["parents"]
  assert all(digest(expected/n)==digest(out/n) for n in ("sample.csv","config.yaml"))
  ledger=json.loads((d/"assessment_ledger.json").read_text())["assessments"];assert len(ledger)==reference["assessment_count"] and all(z["status"]=="completed" for z in ledger)
  counts={};warnings=0
  for z in ledger:counts[z["target"]]=counts.get(z["target"],0)+1;warnings+=len(z["warnings"])
  assert set(counts.values())=={20};assert (out/"network.png").is_file()
  edges=pd.read_csv(d/"edges.csv");groups=pd.read_csv(d/"shared_groups.csv")
  if a.case=="shared_predictors":assert len(edges)==0 and len(groups)==1 and groups.iloc[0]["selected_count"]==20
  if a.case=="chain_missing":
   rows=[z for z in ledger if z["target"]=="Depression"];assert all(z["n_observed_target_rows"]+z["n_missing_target_rows"]==64 and z["n_missing_target_rows"] in (0,1) for z in rows)
  extra=dict(assessments=len(ledger),planned_per_target=20,completed_per_target=counts,assessment_warning_count=warnings,reference_structure_matched=True)
 v=dict(status="passed",case=a.case,installed_module=str(module),isolated_mode=True,distribution_version=metadata.version("open-predictive-expansion"),new_fits_for_this_audit=0,new_refits_for_this_audit=0,assigned_depth=meta["assigned_depth"],provenance=meta["provenance"],scope="fixed regression evidence; no independent scientific replication",recorder_recovery="initial workflow completed but final mappingproxy serialization failed; saved exports independently checked with no fits",**extra)
 (out/"verification.json").write_text(json.dumps(portable(v),indent=2)+"\n",encoding="utf-8");print(json.dumps(v),flush=True)

if __name__=="__main__":main()
