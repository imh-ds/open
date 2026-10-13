"""Explicit prediction and population-risk arithmetic for extracted preflight; no fits."""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
threshold_path=a.input/'fixed-thresholds.json'
if not threshold_path.exists():threshold_path=a.input.parents[1]/'examples/study_manifests/pair-confidence-thresholds.json'
thresholds=json.loads(threshold_path.read_text())['thresholds'];rows=[];checks=0;population_checks=0;error=0.
for path in sorted(a.input.glob('prospective--*/result.json')):
 v=json.loads(path.read_text());num=np.zeros((2,2));den=0.;positive=[0,0]
 for i,f in enumerate(v['comparisons']):
  y=np.asarray(f['observed']);base=np.mean((y-np.asarray(f['baseline_predicted']))**2);ref=f['reference_mse']
  for j,z in enumerate(f['pairs']):
   joint=np.mean((y-np.asarray(z['predicted']))**2);matched=np.mean((y-np.asarray(z['matched_predicted']))**2)
   expected=np.array([(base-joint)/ref,(matched-joint)/ref,(base-matched)/ref]);saved=np.array([z['overall_gain'],z['matched_gain'],z['marginal_gain']]);np.testing.assert_allclose(expected,saved,rtol=1e-10,atol=1e-12);error=max(error,float(abs(expected-saved).max()));checks+=1
   if i<3:num[j]+=(np.array([base,matched])-joint)*len(y);positive[j]+=int(matched>joint)
  o=v['population'][i];am=np.mean((y-np.asarray(o['additive_predicted']))**2);jm=np.mean((y-np.asarray(o['joint_predicted']))**2);np.testing.assert_allclose([am,jm,(am-jm)/ref],[o['additive_mse'],o['joint_mse'],o['gap']],rtol=1e-10,atol=1e-12);population_checks+=1
  if i<3:den+=ref*len(y)
 for j in range(2):
  overall,matched=num[j]/den;score=max(0.,matched) if overall>0 and positive[j]>=2 else 0.
  rows.append(dict(id=v['task']['id'],pair=f'x1:x{j+2}',cv_overall=overall,cv_matched=matched,score=score,positive_folds=positive[j],above_threshold=score>thresholds[v['task']['cell']],ideal_independent_gap=v['population'][3]['gap'] if j==0 else None))
frame=pd.DataFrame(rows);old=pd.read_csv(a.input/'case_metrics.csv');keys=['id','pair'];pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True),old[frame.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12);frame.to_csv(a.output/'independent-metrics.csv',index=False)
check=dict(pair_comparisons=checks,population_comparisons=population_checks,pooled_rows=len(rows),max_score_difference=error);(a.output/'independent-check.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check))
