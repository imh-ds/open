"""Post-hoc fixed-setting marginal matching; does not select or adopt models.

python replay.py --input EXTRACTED/.superpowers/pair-conservative --output FRESH
Uses the frozen D-037 runtime/source guard and saved identities.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples'))
import pair_conservative_study as frozen
from open.pair_validation import _estimator
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('choose a fresh output directory')
    args.output.mkdir(parents=True)
    manifest=frozen.freeze()
    flagged=json.loads((ROOT/'examples/results/2026-10-12-conservative-selection/noise_positive_settings.json').read_text())
    flagged_ids={r['id'] for r in flagged}
    tasks=[c for c in frozen.tasks() if c['scenario']=='noise' or
           (c['rho']==.7 and c['n']>=300 and c['scenario'] in ('pure','mixed'))]
    assert len(tasks)==90 and len(flagged_ids)==3
    rows=[];refits=[];fits=0;start=time.monotonic()
    for c in tasks:
        folder=args.input/c['id']
        value=frozen.io.read(folder/'result.json')
        if (value['task']!=c or value['context']!=frozen.context(c) or
            frozen.io.read(folder/'verified.json')!=frozen.verification(c,folder)):
            raise ValueError('saved evidence context or digests differ')
        v=value['independent']
        X,y,*_=frozen.generate(c,c['n'],c['seed'])
        T,ty,*_=frozen.generate(c,1000,c['test_seed'])
        np.testing.assert_array_equal(ty,v['v2']['observed'])
        reference=float(np.mean((ty-y.mean())**2))
        np.testing.assert_allclose(reference,v['v2']['reference_mse'],rtol=1e-12)
        original=np.asarray(v['v2']['pairs'][0]['baseline_predicted'])
        old_mse=float(np.mean((ty-original)**2))
        predictions=dict(observed=ty,original_additive=original)
        cache={}
        for mode in ('v2','research','conservative'):
            record=v['v2']['pairs'][0] if mode=='v2' else v[mode][0]
            settings=record['parameters']
            key=(settings['curves__n_knots'],settings['ridge__alpha'])
            if key not in cache:
                model=_estimator().set_params(curves__n_knots=key[0],ridge__alpha=key[1]).fit(X,y)
                cache[key]=model.predict(T);fits+=1
            matched=cache[key];joint=np.asarray(record['predicted'])
            matched_mse=float(np.mean((ty-matched)**2));joint_mse=float(np.mean((ty-joint)**2))
            original_gain=(old_mse-joint_mse)/reference
            marginal_gain=(old_mse-matched_mse)/reference
            joint_gain=(matched_mse-joint_mse)/reference
            np.testing.assert_allclose(original_gain,record['gain'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(original_gain,marginal_gain+joint_gain,rtol=1e-12,atol=1e-12)
            predictions[mode+'_matched_additive']=matched
            predictions[mode+'_archived_joint']=joint
            rows.append(dict(id=c['id'],scenario=c['scenario'],rho=c['rho'],n=c['n'],p=c['p'],mode=mode,
                knots=key[0],alpha=key[1],joint_weight=settings.get('curves__joint_weight',1.),
                baseline_knots=v['v2']['baseline_parameters']['curves__n_knots'],
                baseline_alpha=v['v2']['baseline_parameters']['ridge__alpha'],
                reference_mse=reference,original_additive_mse=old_mse,matched_additive_mse=matched_mse,
                joint_mse=joint_mse,original_gain=original_gain,marginal_change_gain=marginal_gain,
                matched_joint_gain=joint_gain))
            if c['id'] in flagged_ids and mode=='conservative':
                replay=frozen.estimator((0,1)).set_params(**settings).fit(X,y).predict(T);fits+=1
                diff=float(np.max(np.abs(replay-joint)))
                refits.append(dict(id=c['id'],max_prediction_difference=diff,
                    mse_difference=float(abs(np.mean((ty-replay)**2)-joint_mse)),
                    strict_replay_pass=bool(np.allclose(replay,joint,rtol=1e-9,atol=1e-9))))
                predictions['conservative_local_joint_replay']=replay
        pd.DataFrame(predictions).to_csv(args.output/(c['id']+'.csv'),index=False)
    frame=pd.DataFrame(rows);frame.to_csv(args.output/'metrics.csv',index=False)
    frozen.freeze()
    receipt=dict(scope='post-hoc saved-case diagnosis; not fresh validation or an interaction decision',
        tasks=90,noise_tasks=60,signal_tasks=30,metric_rows=len(frame),new_fixed_setting_fits=fits,
        elapsed_seconds=time.monotonic()-start,source_commit='87420b857ef9cbc65f6713a9c8ed9a652727843d',
        source_hashes=manifest['sources'],environment=manifest['environment'],
        diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        local_joint_replays=refits,definition='original_gain = marginal_change_gain + matched_joint_gain',
        caveat='matched settings fix basis and alpha, but marginal coefficients are refitted jointly; not an isolated causal effect')
    frozen.io.write(args.output/'receipt.json',receipt)
    print(json.dumps(receipt))
    print(frame[(frame.id.isin(flagged_ids))&(frame['mode']=='conservative')][[
        'id','original_gain','marginal_change_gain','matched_joint_gain']].to_string(index=False))
    print(frame[frame.scenario!='noise'].groupby(['scenario','mode'])[['original_gain','matched_joint_gain']].agg(['median',lambda x:(x>.02).sum()]).to_string())


if __name__=='__main__':
    main()
