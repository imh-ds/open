"""Check retained supplemental replay arithmetic without fitting any models."""
from pathlib import Path
import argparse,gzip,json,hashlib
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
r=json.loads((a.evidence/'outer-replay-receipt.json').read_text());b=(a.evidence/'outer-replay-raw.json.gz').read_bytes();assert hashlib.sha256(b).hexdigest()==r['raw_vectors']['sha256'];raw=gzip.decompress(b);assert hashlib.sha256(raw).hexdigest()==r['raw_vectors']['uncompressed_sha256'];cases=json.loads(raw);rows=[];max_drift=0.;max_gain=0.
for case in cases:
 v=json.loads((a.input/case['id']/'result.json').read_text());num=np.zeros(2);den=0.
 for f,saved in zip(case['folds'],v['comparisons'][:3]):
  y=np.asarray(f['observed']);np.testing.assert_array_equal(y,v['comparisons'][3]['observed']);pred=[np.asarray(f[k]) for k in ('baseline_predicted','joint_predicted','matched_predicted')];risks=np.array([np.mean((y-z)**2) for z in pred]);np.testing.assert_allclose(risks,f['risks'],rtol=1e-10,atol=1e-12);num+=(risks[[0,2]]-risks[1])*len(y);den+=f['reference_mse']*len(y)
  old=[saved['baseline_predicted'],saved['pairs'][0]['predicted'],saved['pairs'][0]['matched_predicted']];sd=saved['pairs'][0]['curve']['training_sd'];drifts=[float(np.max(np.abs(np.asarray(q)-z))/sd) for q,z in zip(f['original_assessment_refit_predicted'],old)];np.testing.assert_allclose(drifts,f['max_normalized_prediction_drifts'],rtol=1e-10,atol=1e-12);max_drift=max(max_drift,*drifts)
  oy=np.asarray(saved['observed']);op=[np.asarray(z) for z in f['original_assessment_refit_predicted']];gain=float(abs((np.mean((oy-op[2])**2)-np.mean((oy-op[1])**2))/saved['reference_mse']-saved['pairs'][0]['matched_gain']));np.testing.assert_allclose(gain,f['cv_matched_gain_delta'],atol=1e-12);max_gain=max(max_gain,gain)
 rows.append(dict(id=case['id'],same_training_independent_overall=num[0]/den,same_training_independent_matched=num[1]/den))
frame=pd.DataFrame(rows).sort_values('id').reset_index(drop=True);old=pd.read_csv(a.evidence/'outer-replay.csv')[frame.columns].sort_values('id').reset_index(drop=True);pd.testing.assert_frame_equal(frame,old,check_dtype=False,rtol=1e-10,atol=1e-12);np.testing.assert_allclose([max_drift,max_gain],[r['maximum_normalized_prediction_drift'],r['maximum_cv_gain_delta']],rtol=1e-10,atol=1e-12);print(json.dumps(dict(cases=len(cases),folds=len(cases)*3,max_normalized_prediction_drift=max_drift,max_gain_delta=max_gain,refits=0)))
