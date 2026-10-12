"""Recompute diagnostic gains from retained predictions; no model refits."""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
p=args.input;r=pd.read_csv(p/'metrics.csv');maxerr=0.
for identity,g in r.groupby('id'):
    t=pd.read_csv(p/(identity+'.csv'));y=t.observed.to_numpy();old=np.mean((y-t.original_additive.to_numpy())**2)
    for row in g.to_dict('records'):
        mode=row['mode'];matched=np.mean((y-t[mode+'_matched_additive'].to_numpy())**2);joint=np.mean((y-t[mode+'_archived_joint'].to_numpy())**2);ref=row['reference_mse']
        derived=np.array([(old-joint)/ref,(old-matched)/ref,(matched-joint)/ref]);saved=np.array([row['original_gain'],row['marginal_change_gain'],row['matched_joint_gain']])
        maxerr=max(maxerr,float(np.max(np.abs(derived-saved))))
        np.testing.assert_allclose(derived,saved,rtol=1e-10,atol=1e-12)
assert len(r)==270 and r.id.nunique()==90
print(json.dumps(dict(prediction_files=90,metric_rows=270,max_gain_difference=maxerr)))
