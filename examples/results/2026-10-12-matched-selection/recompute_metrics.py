from pathlib import Path
import argparse,json,numpy as np,pandas as pd
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();base=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
table=pd.read_csv(base/'case_metrics.csv');checked=0;maxerr=0.;foldrows=[]
for folder in sorted(base.glob('matched--*')):
 v=json.loads((folder/'result.json').read_text());c=v['task']
 for mode in ('v2','research','conservative'):
  for pi,pair in enumerate(('x1:x2','x1:x3')):
   numer=matchednumer=denom=0.
   for fi,f in enumerate(v['folds']+[v['independent']]):
    y=np.asarray(f['v2']['observed']);b=np.asarray(f['v2']['pairs'][pi]['baseline_predicted']);r=f['v2']['pairs'][pi] if mode=='v2' else f[mode][pi];pred=np.asarray(r['predicted']);a=np.asarray(r['matched_additive_predicted']);ref=f['v2']['reference_mse']
    old=float(np.mean((y-b)**2));jmse=float(np.mean((y-pred)**2));amse=float(np.mean((y-a)**2));gain=(old-jmse)/ref;matched=(amse-jmse)/ref;marginal=(old-amse)/ref
    np.testing.assert_allclose([r['joint_mse'],r['matched_additive_mse'],r['gain'],r['matched_gain'],r['marginal_gain']],[jmse,amse,gain,matched,marginal],rtol=1e-10,atol=1e-12)
    assert r['matched_parameters']=={k:r['parameters'][k] for k in ('curves__n_knots','ridge__alpha')}
    if fi<3:
     numer+=(old-jmse)*len(y);matchednumer+=(amse-jmse)*len(y);denom+=ref*len(y)
     foldrows.append(dict(id=c['id'],rho=c['rho'],scenario=c['scenario'],n=c['n'],p=c['p'],fold=fi,mode=mode,pair=pair,gain=gain,matched_gain=matched,marginal_gain=marginal,joint_mse=jmse))
    else:ind,indmatched,indmarginal=gain,matched,marginal
   row=table[(table.id==c['id'])&(table['mode']==mode)&(table.pair==pair)].iloc[0]
   derived=np.array([numer/denom,matchednumer/denom,(numer-matchednumer)/denom,ind,indmatched,indmarginal]);saved=row[['cv_gain','cv_matched_gain','cv_marginal_gain','independent_gain','independent_matched_gain','independent_marginal_gain']].to_numpy(float);maxerr=max(maxerr,float(abs(derived-saved).max()));np.testing.assert_allclose(derived,saved,rtol=1e-10,atol=1e-12);checked+=1
folds=pd.DataFrame(foldrows);folds.to_csv(out/'fold_metrics.csv',index=False)
summary=[]
for subgroup,mask in (('N80',table.n==80),('N>=300',table.n>=300),('all',table.n>0)):
 for (pair,rho,scenario,mode),g in table[mask].groupby(['pair','rho','scenario','mode']):
  for score,cvcol,testcol in (('overall','cv_gain','independent_gain'),('matched','cv_matched_gain','independent_matched_gain')):
   summary.append(dict(subgroup=subgroup,pair=pair,rho=rho,scenario=scenario,mode=mode,score=score,count=len(g),cv_above_02=int((g[cvcol]>.02).sum()),test_above_02=int((g[testcol]>.02).sum()),cv_min=float(g[cvcol].min()),test_min=float(g[testcol].min()),test_median=float(g[testcol].median())))
s=pd.DataFrame(summary);s.to_csv(out/'findings_summary.csv',index=False)
check=dict(case_rows=checked,pair_comparisons=5760,max_all_six_gain_difference=maxerr,arithmetic='explicit raw-prediction risk and decomposition checks; no runner aggregation or model refits')
(out/'independent-check.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps(check));print('TARGET N>=300');print(s[(s.subgroup=='N>=300')&(s.pair=='x1:x2')].to_string(index=False));print('NULL INDEPENDENT >.02 ALL60 EACH');print(table[table.scenario.isin(['noise','additive'])].groupby(['pair','scenario','mode'])[['independent_gain','independent_matched_gain']].agg([lambda x:(x>.02).sum(),'max']).to_string());print('N80 OUTER TAILS TARGET');print(folds[(folds.n==80)&(folds.pair=='x1:x2')].groupby('mode')[['gain','matched_gain']].agg(['min',lambda x:(x<-.1).sum(),lambda x:(x<-1).sum()]).to_string())
