from pathlib import Path
import argparse,json,sys,numpy as np,pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();base=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
table=pd.read_csv(base/'case_metrics.csv');checked=0;maxcv=0.;maxind=0.;foldrows=[];selections=[]
for folder in sorted(base.glob('one-se--*')):
 v=json.loads((folder/'result.json').read_text());c=v['task']
 for mode in ('v2','research','conservative'):
  for pi,pair in enumerate(('x1:x2','x1:x3')):
   numer=denom=0.
   for fi,f in enumerate(v['folds']+[v['independent']]):
    y=np.asarray(f['v2']['observed']);b=np.asarray(f['v2']['pairs'][pi]['baseline_predicted']);r=f['v2']['pairs'][pi] if mode=='v2' else f[mode][pi];pred=np.asarray(r['predicted'])
    mse=float(np.mean((y-pred)**2));gain=(np.mean((y-b)**2)-mse)/f['v2']['reference_mse'];assert abs(mse-r['joint_mse'])<1e-8
    if fi<3:
     numer+=(np.mean((y-b)**2)-mse)*len(y);denom+=f['v2']['reference_mse']*len(y)
     foldrows.append(dict(id=c['id'],rho=c['rho'],scenario=c['scenario'],n=c['n'],p=c['p'],fold=fi,mode=mode,pair=pair,gain=gain,mse=mse))
    else:ind=gain
   row=table[(table.id==c['id'])&(table['mode']==mode)&(table.pair==pair)].iloc[0]
   maxcv=max(maxcv,abs(row.cv_gain-numer/denom));maxind=max(maxind,abs(row.independent_gain-ind));checked+=1
 for fi,f in enumerate(v['folds']+[v['independent']]):
  for pi,s in enumerate(f['selection']):
   losses=np.asarray(s['fold_losses']);means=losses.mean(1);best=int(np.argmin(means));threshold=means[best]+np.std(losses[best],ddof=1)/np.sqrt(3);eligible=np.flatnonzero(means<=threshold).tolist();pars=s['candidate_parameters']
   ix=min(eligible,key=lambda i:(-pars[i]['ridge__alpha']/pars[i]['curves__joint_weight']**2,pars[i]['curves__n_knots'],-pars[i]['ridge__alpha'],pars[i]['curves__joint_weight']))
   assert best==s['best_index'] and eligible==s['eligible_indices'] and ix==s['selected_index'] and abs(threshold-s['threshold'])<1e-12
   selections.append(dict(id=c['id'],fold=fi,pair=pi,changed=ix!=best,minimum_joint_penalty=pars[best]['ridge__alpha']/pars[best]['curves__joint_weight']**2,selected_joint_penalty=pars[ix]['ridge__alpha']/pars[ix]['curves__joint_weight']**2))
folds=pd.DataFrame(foldrows);folds.to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(selections).to_csv(out/'selection_metrics.csv',index=False)
summary=[]
for subgroup,mask in (('N80',table.n==80),('N>=300',table.n>=300)):
 for (rho,scenario,mode),g in table[mask&(table.pair=='x1:x2')].groupby(['rho','scenario','mode']):
  summary.append(dict(subgroup=subgroup,rho=rho,scenario=scenario,mode=mode,count=len(g),cv_above_02=int((g.cv_gain>.02).sum()),test_above_02=int((g.independent_gain>.02).sum()),cv_min=float(g.cv_gain.min()),test_min=float(g.independent_gain.min()),test_median=float(g.independent_gain.median())))
s=pd.DataFrame(summary);s.to_csv(out/'findings_summary.csv',index=False)
check=dict(case_rows=checked,max_cv_gain_difference=maxcv,max_independent_gain_difference=maxind,selection_records=len(selections),selection_records_changed=sum(r['changed'] for r in selections),arithmetic='independent explicit raw-prediction risk pooling and candidate preference recomputation; no model refits')
(out/'independent-check.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check));print(s.to_string(index=False));print('WORST N80 OUTER FOLDS');print(folds[(folds.n==80)&(folds.pair=='x1:x2')].groupby('mode').gain.agg(['min','median',lambda g:(g<-.1).sum(),lambda g:(g<-1).sum()]).to_string())
