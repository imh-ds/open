"""Explicit raw-prediction arithmetic for extracted construction evidence; no fits."""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[];checks=0;error=0.
for path in sorted(a.input.rglob('result.json')):
 v=json.loads(path.read_text());num=np.zeros((2,2));den=0.
 for i,f in enumerate(v['comparisons']):
  y=np.array(f['observed']);base=np.mean((y-np.array(f['baseline_predicted']))**2);ref=f['reference_mse']
  for j,z in enumerate(f['pairs']):
   joint=np.mean((y-np.array(z['predicted']))**2);matched=np.mean((y-np.array(z['matched_predicted']))**2)
   expected=np.array([(base-joint)/ref,(matched-joint)/ref,(base-matched)/ref]);saved=np.array([z['overall_gain'],z['matched_gain'],z['marginal_gain']]);np.testing.assert_allclose(expected,saved,rtol=1e-10,atol=1e-12);error=max(error,float(abs(expected-saved).max()));checks+=1
   if i<3:num[j]+=(np.array([base,matched])-joint)*len(y)
  if i<3:den+=ref*len(y)
 for j in range(2):rows.append(dict(id=v['task']['id'],pair=j,cv_overall=num[j,0]/den,cv_matched=num[j,1]/den))
frame=pd.DataFrame(rows);frame.to_csv(a.output/'independent-metrics.csv',index=False)
old=pd.read_csv(a.input/'independent-metrics.csv');pd.testing.assert_frame_equal(frame.sort_values(['id','pair']).reset_index(drop=True),old.sort_values(['id','pair']).reset_index(drop=True))
print(json.dumps(dict(pair_comparisons=checks,max_score_difference=error,metric_rows=len(rows))))
