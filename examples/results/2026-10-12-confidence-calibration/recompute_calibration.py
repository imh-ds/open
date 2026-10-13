"""Recompute every calibration score from retained prediction vectors without fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rule=json.loads((a.input/'learned-thresholds.json').read_text());expected={r['id']:r for r in rule['rows']};rows=[];checks=0;maxerror=0.;thresholds={};classes={}
for path in sorted(a.input.glob('confidence--*/result.json')):
 v=json.loads(path.read_text());c=v['task'];assert c['id'] in expected;assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[c['id']]['result_sha256'];num=np.zeros((2,2));den=0.;positive=[0,0]
 for i,f in enumerate(v['comparisons']):
  y=np.asarray(f['observed']);base=np.mean((y-np.asarray(f['baseline_predicted']))**2);ref=f['reference_mse']
  for j,z in enumerate(f['pairs']):
   joint=np.mean((y-np.asarray(z['predicted']))**2);matched=np.mean((y-np.asarray(z['matched_predicted']))**2)
   values=np.array([(base-joint)/ref,(matched-joint)/ref,(base-matched)/ref]);saved=np.array([z['overall_gain'],z['matched_gain'],z['marginal_gain']]);np.testing.assert_allclose(values,saved,rtol=1e-10,atol=1e-12);maxerror=max(maxerror,float(abs(values-saved).max()));checks+=1
   if i<3:num[j]+=(np.array([base,matched])-joint)*len(y);positive[j]+=int(matched>joint)
  if i<3:den+=ref*len(y)
 scores=[]
 for j in range(2):
  overall,matched=num[j]/den;score=max(0.,matched) if overall>0 and positive[j]>=2 else 0.;scores.append(score)
  rows.append(dict(id=c['id'],cell=c['cell'],scenario=c['scenario'],pair=f'x1:x{j+2}',score=score,cv_overall=overall,cv_matched=matched,positive_folds=positive[j]))
 score=max(scores);np.testing.assert_allclose(score,expected[c['id']]['score'],rtol=1e-10,atol=1e-12);thresholds[c['cell']]=max(thresholds.get(c['cell'],0.),score);classes.setdefault((c['cell'],c['scenario']),[]).append(score)
assert len(classes)==20 and all(len(x)==100 for x in classes.values()) and len(rows)==4000
for cell,value in thresholds.items():np.testing.assert_allclose(value,rule['thresholds'][cell],rtol=1e-10,atol=1e-12)
frame=pd.DataFrame(rows);old=pd.read_csv(a.input/'case_metrics.csv');keys=['id','pair'];cols=keys+['score','cv_overall','cv_matched','positive_folds'];pd.testing.assert_frame_equal(frame[cols].sort_values(keys).reset_index(drop=True),old[cols].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
frame.to_csv(a.output/'independent-metrics.csv',index=False)
summary=[dict(cell=cell,threshold=thresholds[cell],noise_max=max(classes[cell,'noise']),additive_max=max(classes[cell,'additive']),eligible_noise=sum(x>0 for x in classes[cell,'noise']),eligible_additive=sum(x>0 for x in classes[cell,'additive'])) for cell in thresholds]
pd.DataFrame(summary).to_csv(a.output/'threshold-summary.csv',index=False)
check=dict(datasets=2000,pair_comparisons=checks,metric_rows=len(rows),max_score_difference=maxerror,thresholds_reproduced=len(thresholds),result_hashes_checked=2000);(a.output/'independent-check.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check));print(pd.DataFrame(summary).to_string(index=False))
