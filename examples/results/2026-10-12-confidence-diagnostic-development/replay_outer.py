"""Post-hoc fixed-setting replay: same outer models, independent assessment rows.

No tuning, new decision rule, or new independent simulation replications.
"""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'examples'))
import confidence_diagnostic_study as study
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);study.freeze();rows=[];fits=0;raw=[];start=time.monotonic();max_drift=0.;max_gain_delta=0.
for path in sorted(a.input.glob('diagnostic--*/result.json')):
 v=study.io.read(path);c=v['task']
 if c['scenario']!='mixed' or c['target_gap']!=.04:continue
 X,y,_=study.old.models.generate(c,c['n'],c['seed']);T,ty,_=study.old.models.generate(c,1000,c['test_seed']);num=np.zeros(2);den=0.;folds=[]
 for f in v['comparisons'][:3]:
  train=f['train_indices'];test=f['assessment_indices'];z=f['pairs'][0]
  settings=[(study.old.models._estimator(),f['baseline_parameters'],f['baseline_predicted']),
            (study.old.models.estimator((0,1)),z['parameters'],z['predicted']),
            (study.old.models._estimator(),z['matched_parameters'],z['matched_predicted'])]
  preds=[];original=[];drifts=[]
  for model,params,saved in settings:
   model.set_params(**params).fit(X.iloc[train],y[train]);op=model.predict(X.iloc[test]);drift=float(np.max(np.abs(op-np.asarray(saved)))/np.std(y[train]));max_drift=max(max_drift,drift);drifts.append(drift);original.append(op);preds.append(model.predict(T));fits+=1
  assert all(np.isfinite(q).all() for q in preds+original), 'nonfinite replay predictions'
  cv_risks=np.array([np.square(y[test]-q).mean() for q in original]);gain_delta=float(abs((cv_risks[2]-cv_risks[1])/f['reference_mse']-z['matched_gain']));max_gain_delta=max(max_gain_delta,gain_delta)
  risks=np.array([np.square(ty-q).mean() for q in preds]);reference=float(np.square(ty-y[train].mean()).mean());num+=(risks[[0,2]]-risks[1])*len(ty);den+=reference*len(ty)
  folds.append(dict(original_assessment_refit_predicted=[q.tolist() for q in original],max_normalized_prediction_drifts=drifts,cv_matched_gain_delta=gain_delta,training_n=len(train),reference_mse=reference,observed=ty.tolist(),baseline_predicted=preds[0].tolist(),joint_predicted=preds[1].tolist(),matched_predicted=preds[2].tolist(),risks=risks.tolist()))
 old=study.result_rows(v)[0];rows.append(dict(id=c['id'],n=c['n'],rho=c['rho'],target_gap=c['target_gap'],cv_matched=old['cv_matched'],same_training_independent_matched=float(num[1]/den),same_training_independent_overall=float(num[0]/den),full_training_independent_matched=old['independent_matched'],fixed_threshold=old['fixed_threshold'],cv_training_n=len(v['comparisons'][0]['train_indices']),full_training_n=c['n'],scope='same 1000 assessment rows across outer models; dependent diagnostics, no decision replacement'))
 raw.append(dict(id=c['id'],folds=folds))
assert len(rows)==24 and fits==216
# Supplemental numerical precision screen, separate from untouched frozen CI model guard.
assert max_drift<=1e-6 and max_gain_delta<=1e-7, 'replay insufficiently precise for diagnostic use'
pd.DataFrame(rows).to_csv(a.output/'outer-replay.csv',index=False);study.io.write(a.output/'outer-replay-raw.json',raw);study.io.write(a.output/'outer-replay-receipt.json',dict(cases=24,saved_setting_refits=216,maximum_normalized_prediction_drift=max_drift,maximum_cv_gain_delta=max_gain_delta,diagnostic_precision_limits=dict(normalized_prediction_drift=1e-6,cv_gain_delta=1e-7),historical_CI_guard_changed=False,tuning_fits=0,seconds=time.monotonic()-start,manifest=study.identity(),source='post-hoc diagnostic; post-hoc cross-runtime numerical replay; precision screen separate from historical frozen guard; original study decisions unchanged'))
print(pd.DataFrame(rows).groupby(['n','rho'])[['cv_matched','same_training_independent_matched','full_training_independent_matched']].median().to_string());print('Saved-setting refits',fits)
