"""Construction checks for comparator v2; no screening/calibration acceptance."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import pair_screen_study as generator
import numpy as np
import pandas as pd
from open.pair_validation import PairValidationConfig,validate_pairs,_estimator


def source_hashes():
    paths=list((ROOT/'src/open').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'examples/pair_screen_study.py']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(paths)}


def verify_comparison(v,X,y):
    train=v['train_indices'];test=v['assessment_indices']
    base=_estimator().set_params(**v['baseline_parameters']).fit(X.iloc[train],y[train])
    basepred=base.predict(X.iloc[test]);observed=y[test]
    reference=float(np.square(observed-y[train].mean()).mean())
    np.testing.assert_allclose(v['observed'],observed)
    for p in v['pairs']:
        assert p['status']=='available'
        pair=tuple(X.columns.get_loc(n) for n in p['members'])
        model=_estimator(pair).set_params(**p['parameters']).fit(X.iloc[train],y[train])
        pred=model.predict(X.iloc[test])
        np.testing.assert_allclose(pred,p['predicted'],rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(basepred,p['baseline_predicted'],rtol=1e-9,atol=1e-9)
        risks=[float(np.square(observed-z).mean()) for z in (basepred,pred)]
        np.testing.assert_allclose([p['baseline_mse'],p['joint_mse'],p['reference_mse'],p['gain']],
            [*risks,reference,(risks[0]-risks[1])/reference])
    return 1+len(v['pairs'])


def run(output):
    generator.validate_environment();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    context=dict(sources=source_hashes(),environment=generator.environment(),
        purpose='24 fresh comparator-only construction checks, one dataset per condition; no calibrated detection claim',
        pairs=[['x1','x2'],['x1','x3']],candidate_selection='fixed to isolate comparator; no RF shortlisting',
        validation=asdict(PairValidationConfig()))
    generator.write(output/'context.json',context);start=time.monotonic();rows=[];refits=0;fits=0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for scenario in generator.SCENARIOS:
            for n,p in generator.POINTS:
                i=len(rows);c=dict(id=f'{scenario}-n{n}-p{p}',scenario=scenario,n=n,p=p,
                    seed=98000000+i,test_seed=108000000+i,split_seed=118000000+i)
                X,y,_,truth=generator.generate(c,n,c['seed'])
                T,test_y,_,_=generator.generate(c,1000,c['test_seed'])
                pairs=tuple(tuple(v) for v in context['pairs'])
                cfg=PairValidationConfig(seed=c['split_seed'])
                folds=[validate_pairs(X,y,tuple(a.tolist()),tuple(b.tolist()),pairs,config=cfg)
                       for a,b in generator.partitions(c)]
                combined=pd.concat([X,T],ignore_index=True);combined_y=np.concatenate([y,test_y])
                independent=validate_pairs(combined,combined_y,tuple(range(n)),tuple(range(n,n+1000)),pairs,config=cfg)
                for v in folds:refits+=verify_comparison(v,X,y)
                refits+=verify_comparison(independent,combined,combined_y)
                fits+=sum(v['additional_fits'] for v in [*folds,independent])
                folder=output/c['id']
                generator.write(folder/'result.json',dict(case=c,truth=truth,folds=folds,independent=independent))
                X.assign(outcome=y).to_csv(folder/'discovery.csv',index=False)
                T.assign(outcome=test_y).to_csv(folder/'independent.csv',index=False)
                selected=[v['pairs'][0] for v in folds]
                gain=sum((v['baseline_mse']-v['joint_mse'])*len(v['predicted']) for v in selected)/sum(
                    v['reference_mse']*len(v['predicted']) for v in selected)
                rows.append(dict(scenario=scenario,n=n,p=p,cv_pair_gain=gain,
                    independent_pair_gain=independent['pairs'][0]['gain'],
                    max_absolute_assessment_prediction=max(abs(z) for v in folds for p in v['pairs'] for z in p['predicted'])))
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    summary=dict(complete=len(rows),planned=24,additional_fitting_tuning_fits=fits,
        saved_setting_refits=refits,elapsed_seconds=time.monotonic()-start,
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured],
        acceptance='none: construction smoke only',source_hashes=source_hashes())
    assert summary['source_hashes']==context['sources'] and len(rows)==24
    generator.write(output/'audit.json',summary)
    print({k:summary[k] for k in ('complete','planned','additional_fitting_tuning_fits','saved_setting_refits','elapsed_seconds')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'.superpowers/spline-stability-smoke')
    run(parser.parse_args().output)
