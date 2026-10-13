"""Audit frozen confirmation records and independently recompute prediction decisions.

No fitting, tuning, or new study identities. Run from the frozen source/runtime.
"""
import argparse,json,sys,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples'))
import pair_confidence_study as study

def verify_task(item):
 c,folder=item;folder=Path(folder);v=study.check_result(c,folder)
 assert study.io.read(folder/'verified.json')==study.verification(c,folder)
 assert not (folder/'error.json').exists()
 values=np.zeros((2,2));den=0.;positive=[0,0];error=0.;checks=0
 for i,f in enumerate(v['comparisons']):
  y=np.asarray(f['observed']);base=np.mean((y-np.asarray(f['baseline_predicted']))**2);ref=f['reference_mse']
  for j,z in enumerate(f['pairs']):
   joint=np.mean((y-np.asarray(z['predicted']))**2);matched=np.mean((y-np.asarray(z['matched_predicted']))**2)
   actual=np.array([(base-joint)/ref,(matched-joint)/ref,(base-matched)/ref]);saved=np.array([z['overall_gain'],z['matched_gain'],z['marginal_gain']]);np.testing.assert_allclose(actual,saved,rtol=1e-10,atol=1e-12);error=max(error,float(abs(actual-saved).max()));checks+=1
   if i<3:values[j]+=(np.array([base,matched])-joint)*len(y);positive[j]+=int(matched>joint)
  if i<3:den+=ref*len(y)
 rows=study.result_rows(v)
 for j,r in enumerate(rows):
  overall,matched=values[j]/den;score=max(0.,matched) if overall>0 and positive[j]>=2 else 0.
  np.testing.assert_allclose([r['cv_overall'],r['cv_matched'],r['score']],[overall,matched,score],rtol=1e-10,atol=1e-12)
  assert r['positive_folds']==positive[j]
  r['score']=float(score)
 return rows,checks,error

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);study.freeze('confirmation');rule=study.validate_rule();planned=study.rules.tasks('confirmation');expected={c['id'] for c in planned}
 assert {p.name for p in a.input.glob('confidence--*') if p.is_dir()}==expected
 stamps=list(a.input.glob('shard-*.json'));assert len(stamps)==130
 for p in stamps:
  v=study.io.read(p);i=v['id'];assert p.name==f'shard-{i}.json' and v==dict(stage='confirmation',id=i,tasks=study.shard_tasks('confirmation',i),manifest=study.identity('confirmation'))
 rows=[];checks=0;error=0.;start=time.monotonic()
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,(r,n,e) in enumerate(pool.map(verify_task,[(c,str(a.input/c['id'])) for c in planned],chunksize=16),1):
   rows+=r;checks+=n;error=max(error,e)
   if i%500==0:print('Verified frozen tasks',i,flush=True)
 for r in rows:r['qualifies']=bool(r['score']>rule['thresholds'][r['cell']])
 frame=pd.DataFrame(rows);old=pd.read_csv(a.input/'case_metrics.csv');keys=['id','pair'];assert set(frame.columns)==set(old.columns)
 pd.testing.assert_frame_equal(frame[old.columns].sort_values(keys).reset_index(drop=True),old.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
 saved=study.io.read(a.input/'audit.json');assert saved['complete']==5200 and saved['missing']==0 and not saved['audit_errors'] and not saved['failed_tasks'];assert saved['manifest']==study.identity('confirmation') and saved['thresholds_sha256']==study.threshold_digest()
 summary=study.confirmation_summary(rows,[],expected)
 for key in ('null_qualified','null_cells','signal_cells'):assert summary[key]==saved[key]
 for cell in summary['null_cells']:
  n,k=cell['assessed'],cell['above'];upper=float(beta.ppf(1-.05/20,k+1,n-k)) if k<n else 1.;np.testing.assert_allclose(upper,cell['upper_bound'],atol=1e-12,rtol=1e-10)
 for cell in summary['signal_cells']:
  n,k=cell['assessed'],cell['detected'];lo=float(beta.ppf(.05,k,n-k+1)) if k else 0.;sim=float(beta.ppf(.05/40,k,n-k+1)) if k else 0.;np.testing.assert_allclose([lo,sim],[cell['lower_bound'],cell['simultaneous_lower_bound']],atol=1e-12,rtol=1e-10)
 frame.to_csv(a.output/'independent-metrics.csv',index=False);(a.output/'independent-summary.json').write_text(json.dumps(summary,indent=2)+'\n');check=dict(datasets=5200,pair_comparisons=checks,pooled_pair_rows=len(frame),max_score_difference=error,shards=130,null_gate=summary['null_qualified'],false_alarms=sum(c['above'] for c in summary['null_cells']),null_datasets=4000,refits_performed=0,audit_seconds=time.monotonic()-start);(a.output/'independent-check.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check),flush=True)
if __name__=='__main__':main()
