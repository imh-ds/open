"""Bounded fresh development study; previous frozen studies remain unchanged."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import pair_screen_study as prior
import numpy as np
import pandas as pd
from open.interactions import PairScreenConfig,screen_pair_interactions,_adaptive_detail,_contrast
from open.pair_validation import PairValidationConfig,validate_pairs,_estimator
from open.learners.trees import GRIDS
from sklearn.model_selection import ParameterGrid

MANIFEST=ROOT/'examples/study_manifests/pair-validation.json'
SCREEN=PairScreenConfig(max_details=2,adaptive_details=True)
VALIDATION=PairValidationConfig()
CUTOFFS=(0.,.01,.02,.05)


def tasks():
    result=[]
    for scenario in prior.SCENARIOS:
        for n,p in prior.POINTS:
            for rep in range(10):
                i=len(result)
                c=dict(id=f'{scenario}-n{n}-p{p}-r{rep:02d}',scenario=scenario,n=n,p=p,
                    replicate=rep,seed=68000000+i,test_seed=78000000+i,split_seed=88000000+i)
                result.append(dict(id=c['id']+'--random_forest',case=c,engine='random_forest'))
    return result


def shards():return [f'{i:03d}' for i in range(120)]


def shard_tasks(identity):
    if identity not in shards():raise ValueError('foreign shard')
    i=int(identity);return tasks()[2*i:2*i+2]


@lru_cache(maxsize=1)
def manifest():
    paths=list((ROOT/'src/open').rglob('*.py'))+[ROOT/n for n in (
        'examples/pair_validation_study.py','examples/pair_validation_protocol.md',
        'examples/pair_screen_study.py','.github/workflows/pair-validation.yml',
        '.github/actions/pair-validation-setup/action.yml')]
    return dict(schema='open.pair_validation_study.v1',tasks=tasks(),shards=shards(),
        screen=asdict(SCREEN),validation=json.loads(json.dumps(asdict(VALIDATION))),cutoffs=list(CUTOFFS),
        environment=prior.EXPECTED_ENV,budget_runner_minutes=3675,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                 for p in sorted(paths)})


def freeze(check=True):
    prior.validate_environment();manifest.cache_clear();value=manifest()
    if check:
        if prior.read(MANIFEST)!=value:raise ValueError('frozen validation context differs')
    else:prior.write(MANIFEST,value)
    return value


def context(task):return prior.fingerprint(dict(task=task,manifest=manifest()))


def verification(task,folder):
    return dict(context=context(task),saved_setting_refits=True,
        result=prior.fingerprint(prior.read(folder/'result.json')),
        receipt=prior.fingerprint(prior.read(folder/'receipt.json')),
        timing=prior.fingerprint(prior.read(folder/'timing.json')),
        predictions=hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest())


def selected(screen):
    ranked=[p for p in screen['pairs'] if p['screen']['status']=='available'
            and p['screen']['standardized_rms']>1e-12]
    ranked.sort(key=lambda p:(-p['screen']['standardized_rms'],p['members']))
    return tuple(tuple(p['members']) for p in ranked[:2])


def assessment_details(model,X,pairs,sd):
    # Fixed and adaptive modes each have a shared cap across the locked pairs.
    fixed_remaining=adaptive_remaining=SCREEN.prediction_rows
    details=[]
    for pair in pairs:
        fixed=_contrast(model,X,pair,SCREEN.detail_rows,SCREEN.detail_bins,SCREEN,sd,fixed_remaining)
        adaptive=_adaptive_detail(model,X,pair,SCREEN,sd,adaptive_remaining)
        fixed_remaining-=fixed['prediction_rows'];adaptive_remaining-=adaptive['prediction_rows']
        details.append(dict(members=list(pair),fixed=fixed,adaptive=adaptive))
    return details


def fold_fit(task,X,y,train,test):
    train=list(train);test=list(test);c=task['case']
    model,params=prior.fit('random_forest',c,X.iloc[train],y[train])
    pred=model.predict(X.iloc[test]);sd=float(np.std(y[train]))
    screen=screen_pair_interactions(model,X.iloc[train],tuple(X),outcome_sd=sd,config=SCREEN)
    pairs=selected(screen)
    validation=validate_pairs(X,y,tuple(train),tuple(test),pairs,
        config=PairValidationConfig(seed=c['split_seed']))
    details=assessment_details(model,X.iloc[test],pairs,sd)
    return dict(train=train,test=test,params=params,predicted=pred.tolist(),observed=y[test].tolist(),
        baseline=np.full(len(test),y[train].mean()).tolist(),screen=screen,
        validation=validation,assessment_details=details)


def pair_summary(folds,pair):
    records=[p for f in folds for p in f['validation']['pairs']
             if p['members']==pair and p['status']=='available']
    if records:
        risks=sum((p['baseline_mse']-p['joint_mse'])*len(p['predicted']) for p in records)
        reference=sum(p['reference_mse']*len(p['predicted']) for p in records)
        gain=risks/reference if reference>0 else None
    else:gain=None
    return dict(members=pair,planned_folds=3,assessed_folds=len(records),
        complete_gain=gain if len(records)==3 else None,conditional_gain=gain)


def metrics(value):
    folds=value['folds'];truth=value['truth']
    pairs=sorted({tuple(p['members']) for f in folds for p in f['validation']['pairs']})
    summaries=[pair_summary(folds,list(p)) for p in pairs]
    true=pair_summary(folds,truth) if truth else None
    details=[d for f in folds for d in f['assessment_details']]
    return dict(true_pair=true,pairs=summaries,
        true_fixed_details=sum(d['members']==truth and d['fixed']['status']=='available' for d in details),
        true_adaptive_details=sum(d['members']==truth and d['adaptive']['status']=='available' for d in details),
        fixed_details=sum(d['fixed']['status']=='available' for d in details),
        adaptive_details=sum(d['adaptive']['status']=='available' for d in details),
        validation_fits=sum(f['validation']['additional_fits'] for f in folds))


def check_validation(v,X,y,train,test,refit,seed):
    expected=asdict(PairValidationConfig(seed=seed))
    expected=json.loads(json.dumps(expected))
    if v['config']!=expected:raise ValueError('validation config differs')
    if v['train_indices']!=list(train) or v['assessment_indices']!=list(test):raise ValueError('validation split differs')
    cost=len(VALIDATION.knots)*len(VALIDATION.alphas)*VALIDATION.inner_folds+1
    available=[p for p in v['pairs'] if p['status']=='available']
    expected_fits=cost*(1+len(available)) if v['baseline_parameters'] is not None else 0
    if v['additional_fits']!=expected_fits:
        raise ValueError('fit count differs')
    if v['additional_fits']>VALIDATION.max_fits:raise ValueError('fit budget differs')
    if not v['pairs']:
        if v['baseline_parameters'] is not None:raise ValueError('unexpected baseline fit')
        return
    if v['baseline_parameters'] is None or len(available)!=len(v['pairs']):
        raise ValueError('expected planned comparisons unavailable')
    grid=list(ParameterGrid({'curves__n_knots':list(VALIDATION.knots),'ridge__alpha':list(VALIDATION.alphas)}))
    if any(p not in grid for p in [v['baseline_parameters']]+[p['parameters'] for p in available]):
        raise ValueError('validation settings outside grid')
    sizes=[len(a) for a,b in prior.partitions(dict(n=len(train),split_seed=seed))]
    if v['inner_training_n']!=sizes:raise ValueError('validation inner sample sizes differ')
    observed=y[list(test)];reference=float(np.square(observed-y[list(train)].mean()).mean())
    np.testing.assert_allclose(v['observed'],observed)
    np.testing.assert_allclose(v['reference_mse'],reference)
    if refit:
        base=_estimator().set_params(**v['baseline_parameters']).fit(X.iloc[list(train)],y[list(train)])
        basepred=base.predict(X.iloc[list(test)])
    for p in v['pairs']:
        if p['status']!='available':continue
        base_mse=float(np.square(observed-p['baseline_predicted']).mean())
        joint_mse=float(np.square(observed-p['predicted']).mean())
        np.testing.assert_allclose([p['baseline_mse'],p['joint_mse'],p['reference_mse'],p['gain']],
            [base_mse,joint_mse,reference,(base_mse-joint_mse)/reference])
        np.testing.assert_allclose(v['baseline_mse'],base_mse)
        if refit:
            np.testing.assert_allclose(basepred,p['baseline_predicted'],rtol=1e-9,atol=1e-9)
            pair=tuple(X.columns.get_loc(n) for n in p['members'])
            joint=_estimator(pair).set_params(**p['parameters']).fit(X.iloc[list(train)],y[list(train)])
            np.testing.assert_allclose(joint.predict(X.iloc[list(test)]),p['predicted'],rtol=1e-9,atol=1e-9)


def check_result(task,folder,refit=False):
    v=prior.read(folder/'result.json');receipt=prior.read(folder/'receipt.json')
    if (v['task']!=task or v['context']!=context(task) or v['status']!='complete'
        or v['selection_scope']!='outer training only'):raise ValueError('result context differs')
    if receipt!=dict(task=task,context=context(task),environment=prior.EXPECTED_ENV):raise ValueError('runtime receipt differs')
    if refit:prior.validate_environment()
    c=task['case'];X,y,_,truth=prior.generate(c,c['n'],c['seed'])
    if len(v['folds'])!=3 or v['truth']!=truth:raise ValueError('planned fold/truth differs')
    for f,(train,test) in zip(v['folds'],prior.partitions(c)):
        if f['train']!=train.tolist() or f['test']!=test.tolist():raise ValueError('RF split differs')
        if f['params'] not in list(ParameterGrid(GRIDS['random_forest'])):raise ValueError('RF grid differs')
        np.testing.assert_allclose(f['observed'],y[test])
        np.testing.assert_allclose(f['baseline'],y[train].mean())
        if tuple(tuple(p['members']) for p in f['validation']['pairs'])!=selected(f['screen']):
            raise ValueError('training shortlist differs')
        check_validation(f['validation'],X,y,train,test,refit,c['split_seed'])
        if refit:
            model,_=prior.fit('random_forest',c,X.iloc[train],y[train],f['params'])
            np.testing.assert_allclose(model.predict(X.iloc[test]),f['predicted'],rtol=1e-9,atol=1e-9)
            screen=screen_pair_interactions(model,X.iloc[train],tuple(X),outcome_sd=float(np.std(y[train])),config=SCREEN)
            if prior.fingerprint(screen)!=prior.fingerprint(f['screen']):raise ValueError('refit training screen differs')
            details=assessment_details(model,X.iloc[test],selected(screen),float(np.std(y[train])))
            if prior.fingerprint(details)!=prior.fingerprint(f['assessment_details']):raise ValueError('refit plot support differs')
    if metrics(v)!=v['metrics']:raise ValueError('summary differs')
    T,test_y,oracle,_=prior.generate(c,1000,c['test_seed'])
    if v['final_params'] not in list(ParameterGrid(GRIDS['random_forest'])):raise ValueError('final RF grid differs')
    pred=pd.read_csv(folder/'predictions.csv')
    if (list(pred)!=['observed','predicted','baseline','oracle'] or len(pred)!=1000
        or not np.isfinite(pred.to_numpy()).all()):raise ValueError('test schema differs')
    np.testing.assert_allclose(pred.observed,test_y);np.testing.assert_allclose(pred.oracle,oracle)
    np.testing.assert_allclose(pred.baseline,y.mean())
    expected=prior.skill(test_y,pred.predicted,pred.baseline)
    np.testing.assert_allclose([expected[k] for k in expected],[v['test'][k] for k in expected])
    combined=pd.concat([X,T],ignore_index=True);combined_y=np.concatenate([y,test_y])
    check_validation(v['independent_validation'],combined,combined_y,range(len(X)),range(len(X),len(combined)),refit,c['split_seed'])
    if refit:
        model,_=prior.fit('random_forest',c,X,y,v['final_params'])
        np.testing.assert_allclose(model.predict(T),pred.predicted,rtol=1e-9,atol=1e-9)
        screen=screen_pair_interactions(model,X,tuple(X),outcome_sd=float(np.std(y)),config=SCREEN)
        if prior.fingerprint(screen)!=prior.fingerprint(v['final_screen']):raise ValueError('final shortlist screen differs')
    if tuple(tuple(p['members']) for p in v['independent_validation']['pairs'])!=selected(v['final_screen']):
        raise ValueError('final locked shortlist differs')
    return v


def run_task(task,output):
    if task not in tasks():raise ValueError('unplanned task')
    prior.validate_environment();folder=Path(output)/task['id'];identity=context(task)
    if (folder/'result.json').exists():
        (folder/'verified.json').unlink(missing_ok=True)
        try:v=check_result(task,folder,refit=True)
        except Exception:
            (folder/'result.json').replace(folder/'invalid-result.json');raise
        prior.write(folder/'verified.json',verification(task,folder));return v
    prior.write(folder/'receipt.json',dict(task=task,context=identity,environment=prior.environment()))
    c=task['case'];X,y,_,truth=prior.generate(c,c['n'],c['seed'])
    start=time.monotonic();folds=[]
    resumed_warnings=[]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for i,(train,test) in enumerate(prior.partitions(c)):
            path=folder/f'fold-{i}.json'
            if path.exists():
                saved=prior.read(path)
                if saved['context']!=identity:raise ValueError('checkpoint context differs')
                f=saved['result']
                resumed_warnings.extend(saved['warnings'])
            else:
                warning_start=len(captured)
                f=fold_fit(task,X,y,train.tolist(),test.tolist())
                prior.write(path,dict(context=identity,result=f,warnings=[
                    dict(category=w.category.__name__,message=str(w.message)) for w in captured[warning_start:]]))
            folds.append(f)
        T,test_y,oracle,_=prior.generate(c,1000,c['test_seed'])
        model,params=prior.fit('random_forest',c,X,y)
        final_screen=screen_pair_interactions(model,X,tuple(X),outcome_sd=float(np.std(y)),config=SCREEN)
        combined=pd.concat([X,T],ignore_index=True);combined_y=np.concatenate([y,test_y])
        independent=validate_pairs(combined,combined_y,tuple(range(len(X))),tuple(range(len(X),len(combined))),
            selected(final_screen),config=PairValidationConfig(seed=c['split_seed']))
        pred=pd.DataFrame(dict(observed=test_y,predicted=model.predict(T),baseline=np.full(len(T),y.mean()),oracle=oracle))
    temp=folder/'predictions.tmp.csv';pred.to_csv(temp,index=False);temp.replace(folder/'predictions.csv')
    v=dict(task=task,context=identity,status='complete',truth=truth,folds=folds,
        selection_scope='outer training only',final_params=params,final_screen=final_screen,
        independent_validation=independent,test=prior.skill(test_y,pred.predicted,pred.baseline),
        fitting_elapsed_seconds=time.monotonic()-start,
        warnings=resumed_warnings+[dict(category=w.category.__name__,message=str(w.message)) for w in captured])
    v['metrics']=metrics(v);prior.write(folder/'result.json',v)
    verification_start=time.monotonic()
    try:check_result(task,folder,refit=True)
    except Exception:
        (folder/'result.json').replace(folder/'invalid-result.json');raise
    comparator_refits=sum(int(f['validation']['baseline_parameters'] is not None)+
        sum(p['status']=='available' for p in f['validation']['pairs']) for f in folds)
    comparator_refits+=int(independent['baseline_parameters'] is not None)+sum(p['status']=='available' for p in independent['pairs'])
    prior.write(folder/'timing.json',dict(total_seconds=time.monotonic()-start,
        verification_seconds=time.monotonic()-verification_start,
        verification_comparator_refits=comparator_refits,verification_RF_refits=4,
        fitting_RF_fits=4*(len(list(ParameterGrid(GRIDS['random_forest'])))*3+1)))
    prior.write(folder/'verified.json',verification(task,folder))
    return v


def run_shard(identity,output):
    freeze();prior.write(Path(output)/f'shard-{identity}.json',dict(id=identity,task_ids=[t['id'] for t in shard_tasks(identity)]))
    errors=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(run_task,t,output):t for t in shard_tasks(identity)}
        for f in as_completed(fs):
            t=fs[f]
            try:f.result()
            except Exception as e:
                errors.append(t['id']);prior.write(Path(output)/t['id']/'error.json',dict(task=t,context=context(t),error=repr(e)))
    if errors:raise RuntimeError(f'failed tasks retained: {errors}')


def plot_report(output,cells):
    import matplotlib.pyplot as plt
    available=[c for c in cells if c['complete'] and c['true_complete'] is not None]
    if not available:return False
    fig,axes=plt.subplots(1,2,figsize=(13,max(4,len(available)*.45)),layout='constrained')
    for i,c in enumerate(available):
        axes[0].scatter(c['true_conditional_gains'],[i+.1]*len(c['true_conditional_gains']),
            s=24,color='#bfc8d1',label='Selected folds only' if i==0 else None)
        axes[0].scatter(c['true_complete_gains'],[i-.1]*len(c['true_complete_gains']),
            s=28,color='#245e9c',label='All three folds assessed' if i==0 else None)
        axes[1].barh(i-.15,c['true_fixed_details'],height=.3,color='#bfc8d1',label='Fixed grid' if i==0 else None)
        axes[1].barh(i+.15,c['true_adaptive_details'],height=.3,color='#245e9c',label='Adaptive grid' if i==0 else None)
    labels=[f"{c['scenario']} N={c['n']}, p={c['p']} ({c['complete']}/10 cases)" for c in available]
    for ax in axes:
        ax.set_yticks(range(len(available)),labels);ax.legend();ax.invert_yaxis()
    axes[0].axvline(0,color='black',linewidth=.8)
    axes[0].set_xlabel('True-pair held-out skill gain; each dot is one dataset')
    axes[1].set_xlim(0,30);axes[1].set_xlabel('Supported true-pair detail folds / 30 planned per cell')
    fig.suptitle('Development diagnostics: coverage and prediction gains\nMissing or unselected comparisons are unknown; no cutoff accepted')
    fig.savefig(output/'validation-summary.png',dpi=150);plt.close(fig)
    return True


def aggregate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);planned=tasks()
    if {p.name for p in output.iterdir() if p.is_dir()}-{t['id'] for t in planned}:raise ValueError('foreign evidence')
    results={};errors=[];failed=[]
    for t in planned:
        folder=output/t['id']
        if (folder/'error.json').exists():
            error=prior.read(folder/'error.json')
            if error['task']!=t or error['context']!=context(t):raise ValueError('failure context differs')
            failed.append(dict(id=t['id'],error=error['error']))
        if (folder/'result.json').exists():
            try:
                v=check_result(t,folder)
                if prior.read(folder/'verified.json')!=verification(t,folder):raise ValueError('verification digest differs')
                results[t['id']]=v
            except Exception as e:errors.append(dict(id=t['id'],error=repr(e)))
    cells=[];cutoffs=[]
    for scenario in prior.SCENARIOS:
        for n,p in prior.POINTS:
            ts=[t for t in planned if (t['case']['scenario'],t['case']['n'],t['case']['p'])==(scenario,n,p)]
            rs=[results[t['id']] for t in ts if t['id'] in results]
            truth=scenario in prior.INTERACTION_SCENARIOS
            true=[r['metrics']['true_pair'] for r in rs if truth]
            universe=p*(p-1)//2
            complete_pairs=sum(q['complete_gain'] is not None for r in rs for q in r['metrics']['pairs'])
            cells.append(dict(scenario=scenario,n=n,p=p,planned=10,complete=len(rs),missing=10-len(rs),
                true_complete=sum(t['complete_gain'] is not None for t in true) if truth else None,
                true_assessed_fold_counts=[t['assessed_folds'] for t in true] if truth else None,
                true_complete_gains=[t['complete_gain'] for t in true if t['complete_gain'] is not None],
                true_conditional_gains=[t['conditional_gain'] for t in true if t['conditional_gain'] is not None],
                true_fixed_details=sum(r['metrics']['true_fixed_details'] for r in rs) if truth else None,
                true_adaptive_details=sum(r['metrics']['true_adaptive_details'] for r in rs) if truth else None,
                all_fixed_details=sum(r['metrics']['fixed_details'] for r in rs),
                all_adaptive_details=sum(r['metrics']['adaptive_details'] for r in rs),
                planned_pair_fold_slots=10*3*universe,
                assessed_pair_fold_slots=sum(q['assessed_folds'] for r in rs for q in r['metrics']['pairs']),
                complete_pair_count=complete_pairs,planned_pair_count=10*universe,
                mean_RF_test_skill=float(np.mean([r['test']['skill'] for r in rs])) if rs else None))
            for cutoff in CUTOFFS:
                complete_true=sum(t['complete_gain'] is not None and t['complete_gain']>cutoff for t in true)
                conditional_true=sum(t['conditional_gain'] is not None and t['conditional_gain']>cutoff for t in true)
                incidental_complete=sum(any(q['members']!=r['truth'] and q['complete_gain'] is not None
                    and q['complete_gain']>cutoff for q in r['metrics']['pairs']) for r in rs)
                incidental_conditional=sum(any(q['members']!=r['truth'] and q['conditional_gain'] is not None
                    and q['conditional_gain']>cutoff for q in r['metrics']['pairs']) for r in rs)
                unknown=sum(t['complete_gain'] is None for t in true)
                cutoffs.append(dict(scenario=scenario,n=n,p=p,cutoff=cutoff,planned=10,complete=len(rs),
                    true_complete_count=complete_true if truth else None,
                    true_complete_bounds=[complete_true,complete_true+unknown+10-len(rs)] if truth else None,
                    true_conditional_count=conditional_true if truth else None,
                    any_incidental_complete_count=incidental_complete,
                    any_incidental_conditional_count=incidental_conditional,
                    complete_incidental_pairs=sum(q['members']!=r['truth'] and q['complete_gain'] is not None
                        for r in rs for q in r['metrics']['pairs']),
                    conditional_incidental_pairs=sum(q['members']!=r['truth'] and q['conditional_gain'] is not None
                        for r in rs for q in r['metrics']['pairs']),
                    unknown_incidental_pair_count=10*(universe-int(truth))-sum(
                        q['members']!=r['truth'] and q['complete_gain'] is not None for r in rs for q in r['metrics']['pairs']),
                    unassessed_incidental_pair_count=10*(universe-int(truth))-sum(
                        q['members']!=r['truth'] and q['conditional_gain'] is not None for r in rs for q in r['metrics']['pairs']),
                    incidental_scope='shortlisted pairs only; unselected pairs unknown, not a false-positive rate',
                    missing=10-len(rs)))
    report=dict(planned=240,complete=len(results),missing=240-len(results),audit_errors=errors,
        acceptance='none: development only',cells=cells,cutoffs=cutoffs,failed_tasks=failed,
        warning_count=sum(len(r['warnings']) for r in results.values()),
        additional_validation_fits=sum(r['metrics']['validation_fits']+r['independent_validation']['additional_fits'] for r in results.values()))
    timings=[prior.read(output/t['id']/'timing.json') for t in planned if t['id'] in results]
    report['cost']={key:sum(t[key] for t in timings) for key in (
        'total_seconds','verification_seconds','verification_comparator_refits','verification_RF_refits','fitting_RF_fits')}
    prior.write(output/'audit.json',report)
    pd.DataFrame(cells).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(output/'gain_sensitivity.csv',index=False)
    lines=['# Pair validation development results','',f"Complete {len(results)}/240; audit errors {len(errors)}.",
        'No cutoff accepted. Unselected folds are unknown; complete and conditional gains differ.',
        'Incidental counts cover shortlisted pairs only; they are not calibrated false-positive rates. See pair-fold coverage in summary.csv.',
        f"Failed tasks: {len(failed)}. Cost includes {report['additional_validation_fits']} comparator fitting/tuning fits and {report['cost']['verification_comparator_refits']} saved-setting comparator verification refits.",'',
        '|Scenario|N|p|Complete /10|True complete pairs|True fixed / adaptive detail folds|',
        '|---|---:|---:|---:|---:|---|']
    for c in cells:
        lines.append(f"|{c['scenario']}|{c['n']}|{c['p']}|{c['complete']}|{c['true_complete']}|{c['true_fixed_details']} / {c['true_adaptive_details']}|")
    if plot_report(output,cells):lines+=['','![Development gains and support](validation-summary.png)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','matrix','shard','aggregate','smoke'])
    p.add_argument('--write',action='store_true');p.add_argument('--id');p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'.superpowers/pair-validation');a=p.parse_args()
    if a.command=='freeze':freeze(not a.write);print('frozen validation context verified');return
    freeze()
    if a.command=='matrix':print(json.dumps(shards()))
    elif a.command=='shard':run_shard(a.id,a.output)
    elif a.command=='smoke':run_task(tasks()[120],a.output)
    else:
        if a.input:prior.merge_shards([d for d in a.input.iterdir() if d.is_dir()],a.output)
        r=aggregate(a.output);print(json.dumps({k:r[k] for k in ('planned','complete','missing')}))
        if r['missing'] or r['audit_errors']:sys.exit(1)


if __name__=='__main__':main()
