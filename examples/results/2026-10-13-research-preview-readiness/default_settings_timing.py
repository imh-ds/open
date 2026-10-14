"""One-assessment timing probe of existing defaults on the historical chain.
No operating-characteristic claim; the configured 100 assessments are not run.
"""
import argparse,pathlib,json,time,hashlib
import pandas as pd
import open
from open.config import load_config
from open.layer_stability import fit_layer_consensus
from open.layer_artifacts import portable
from open.layer_io import write_layer_outputs
from open.metadata import validate_metadata
p=argparse.ArgumentParser();p.add_argument("--root",type=pathlib.Path,required=True);p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
assert "site-packages" in pathlib.Path(open.__file__).parts
c=load_config(a.root/"configs/default.yaml");d=pd.read_csv(a.root/"examples/results/2026-10-08-layer-fixtures/chain/sample.csv")
start=time.perf_counter();r=fit_layer_consensus(d,"Anxiety",tuple(d),c,n_resamples=1);seconds=time.perf_counter()-start
m=validate_metadata(tuple(d),[{"name":n,"type":"continuous"} for n in d],"Anxiety");write_layer_outputs(r,a.out/"bundle",metadata=m)
v=dict(status=r.status,scope="native timing only, one effective assessment per evaluated target; not default-budget validation",configured_resamples=c.stability.n_resamples,effective_resamples=1,performance_folds=c.performance_cv.folds,performance_repeats=c.performance_cv.repeats,tuning_folds=c.tuning_cv.folds,seconds=seconds,rough_100_assessment_seconds=seconds*100,projection_limit="linear extrapolation assumes identical candidates/expansion and constant per-assessment cost; not a measured 100-assessment run or worst-case bound",assigned_depth=portable(r.assigned_depth),provenance=portable(r.provenance),installed_module=open.__file__)
(a.out/"timing.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(v))
