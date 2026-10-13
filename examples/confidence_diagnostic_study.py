"""Frozen, resumable fresh-development diagnostics; no confidence-rule adoption."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,time,warnings,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import confidence_diagnostic as d
old=d.previous;io=old.io
import numpy as np
import pandas as pd
MANIFEST=ROOT/'examples/study_manifests/confidence-diagnostic.json'


def sources():
    v=old.sources()
    for name in ('examples/confidence_diagnostic.py','examples/confidence_diagnostic_study.py','examples/confidence_diagnostic_protocol.md',
                 '.github/workflows/confidence-diagnostic.yml','.github/actions/confidence-diagnostic-setup/action.yml'):
        p=ROOT/name
        if p.exists():v[name]=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return v


def manifest():
    old.validate_rule()
    return dict(schema='open.confidence_diagnostic.v1',tasks=d.tasks(),shards=d.shards(),environment=io.EXPECTED_ENV,
        sources=sources(),thresholds_sha256=old.threshold_digest(),prior_manifest=io.fingerprint(old.manifest('confirmation')),
        tasks_per_shard=10,workers=2,budget_minutes=d.BUDGET_MINUTES,fits_per_task=692,saved_setting_refits=20,
        independent_n=1000,inner_folds=3,outer_folds=3,production_accepted=False)


def freeze(check=True):
    old.freeze('confirmation');v=manifest()
    if check:
        if io.read(MANIFEST)!=v:raise ValueError('frozen diagnostic context differs')
    else:io.write(MANIFEST,v)
    identity.cache_clear();return v


@lru_cache(None)
def identity():return io.fingerprint(manifest())


def context(c):return io.fingerprint(dict(task=c,manifest=identity()))


def shard_tasks(i):
    if i not in d.shards():raise ValueError('foreign diagnostic shard')
    return d.tasks()[10*int(i):10*(int(i)+1)]


def population_record(c,data,v):
    X,y,train,test=data;a,j=d.population_predictions(X,c);a=a[test];j=j[test];observed=np.asarray(y)[test]
    am=float(np.square(observed-a).mean());jm=float(np.square(observed-j).mean())
    return dict(additive_predicted=a.tolist(),joint_predicted=j.tolist(),additive_mse=am,joint_mse=jm,
        gap=(am-jm)/v['reference_mse'],scope='ideal population predictions; diagnostic only, never model selection or eligibility')


def audit_population(c,data,v,o):
    if o!=population_record(c,data,v):raise ValueError('population diagnostic vectors or risks differ')


def check_checkpoint(c,s,data,refit=False):
    if s['context']!=context(c) or not np.isfinite(s['fitting_seconds']) or s['fitting_seconds']<0 or type(s['warnings']) is not list:
        raise ValueError('checkpoint context or costs differ')
    old.models.audit_comparison(s['result'],*data,c['split_seed'],refit=refit);audit_population(c,data,s['result'],s['population'])


def verification(c,folder):
    names=['receipt.json','result.json','timing.json']+[f'comparison-{i}.json' for i in range(4)]
    return dict(context=context(c),saved_setting_refits=20,digests={n:old.digest(folder/n) for n in names})


def audit_task(c,folder,refit=False):
    v=io.read(folder/'result.json');receipt=io.read(folder/'receipt.json')
    if (v['task']!=c or v['context']!=context(c) or v['status']!='complete' or
        receipt!=dict(task=c,context=context(c),environment=io.environment()) or len(v['comparisons'])!=4 or len(v['population'])!=4):
        raise ValueError('task evidence identity differs')
    records=[]
    for i,data in enumerate(old.comparisons(c)):
        s=io.read(folder/f'comparison-{i}.json');check_checkpoint(c,s,data,refit);records.append(s)
        if s['result']!=v['comparisons'][i] or s['population']!=v['population'][i]:raise ValueError('result differs from checkpoint')
    if v['warnings']!=[w for s in records for w in s['warnings']]:raise ValueError('warning evidence differs')
    t=io.read(folder/'timing.json')
    if (t['fits']!=692 or t['saved_setting_refits']!=20 or type(t['resumed']) is not int or not 0<=t['resumed']<=4 or
        t['fits_this_attempt']!=173*(4-t['resumed']) or not all(np.isfinite(t[k]) and t[k]>=0 for k in
        ('total_seconds','verification_seconds','retained_fitting_seconds'))):raise ValueError('timing evidence differs')
    np.testing.assert_allclose(t['retained_fitting_seconds'],sum(s['fitting_seconds'] for s in records),rtol=1e-12)
    return v


def run_task(c,output):
    if c not in d.tasks():raise ValueError('foreign task')
    io.validate_environment();folder=Path(output)/c['id'];folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    receipt=dict(task=c,context=context(c),environment=io.environment())
    if (folder/'receipt.json').exists() and io.read(folder/'receipt.json')!=receipt:raise ValueError('existing receipt differs')
    io.write(folder/'receipt.json',receipt)
    if (folder/'verified.json').exists():
        v=audit_task(c,folder)
        if io.read(folder/'verified.json')!=verification(c,folder):raise ValueError('verification stamp differs')
        return v
    records=[];resumed=0
    for i,data in enumerate(old.comparisons(c)):
        path=folder/f'comparison-{i}.json'
        if path.exists():s=io.read(path);check_checkpoint(c,s,data);resumed+=1
        else:
            tick=time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');v=old.models.compare(*data,c['split_seed'])
            s=dict(context=context(c),result=v,population=population_record(c,data,v),fitting_seconds=time.monotonic()-tick,
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]);io.write(path,s)
        records.append(s)
    v=dict(task=c,context=context(c),status='complete',comparisons=[s['result'] for s in records],population=[s['population'] for s in records],
        warnings=[w for s in records for w in s['warnings']]);io.write(folder/'result.json',v)
    costs=dict(fits=692,saved_setting_refits=20,resumed=resumed,fits_this_attempt=173*(4-resumed),retained_fitting_seconds=sum(s['fitting_seconds'] for s in records))
    io.write(folder/'timing.json',dict(costs,total_seconds=time.monotonic()-start,verification_seconds=0.))
    tick=time.monotonic();audit_task(c,folder,True)
    io.write(folder/'timing.json',dict(costs,total_seconds=time.monotonic()-start,verification_seconds=time.monotonic()-tick));io.write(folder/'verified.json',verification(c,folder));return v


def result_rows(v):
    rows=old.result_rows(v);thresholds=old.validate_rule()['thresholds'];c=v['task']
    for pos,r in enumerate(rows):
        info=d.decision(old.models.fold_records(v['comparisons'][:3],pos),thresholds[c['cell']])
        r.update(eligible=info['eligible'],above_threshold=info['above_threshold'],decision_reasons=';'.join(info['reasons']),
            fixed_threshold=info['threshold'],population_target_gap=c['target_gap'],
            ideal_independent_gap=v['population'][3]['gap'] if pos==0 else None,
            ideal_independent_joint_mse=v['population'][3]['joint_mse'] if pos==0 else None,
            model_independent_joint_mse=v['comparisons'][3]['pairs'][pos]['joint_mse'])
        np.testing.assert_allclose([r['cv_overall'],r['cv_matched'],r['score']],[info['overall'],info['matched'],info['score']],rtol=1e-12)
    return rows


def aggregate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);planned=d.tasks();complete={};errors=[];failures=[];history=[];rows=[];unknown=[];cost=0
    for c in planned:
        folder=output/c['id']
        if not folder.exists():continue
        if (folder/'error.json').exists():
            try:
                f=io.read(folder/'error.json')
                if f['task']!=c or f['context']!=context(c):raise ValueError('failure context differs')
                failures.append(dict(id=c['id'],failure=f))
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if (folder/'result.json').exists():
            try:
                v=audit_task(c,folder)
                if io.read(folder/'verified.json')!=verification(c,folder):raise ValueError('verification stamp differs')
                complete[c['id']]=v;rows+=result_rows(v)
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        data=old.comparisons(c)
        for i in range(4):
            path=folder/f'comparison-{i}.json'
            if path.exists():
                try:check_checkpoint(c,io.read(path),data[i]);cost+=173
                except Exception as e:errors.append(dict(id=c['id'],checkpoint=path.name,error=repr(e)))
        if c['id'] not in complete and ((folder/'receipt.json').exists() or any(folder.glob('comparison-*.json'))):unknown.append(c['id'])
    ids={c['id'] for c in planned}
    for p in output.glob('diagnostic--*'):
        if p.is_dir() and p.name not in ids:errors.append(dict(id=p.name,error='foreign task'))
    for p in output.glob('shard-*.json'):
        try:
            v=io.read(p);i=v['id']
            if p.name!=f'shard-{i}.json' or v!=dict(id=i,tasks=shard_tasks(i),manifest=identity()):raise ValueError('shard stamp differs')
        except Exception as e:errors.append(dict(id=p.name,error=repr(e)))
    audit=dict(planned=60,complete=len(complete),missing=60-len(complete),missing_ids=[c['id'] for c in planned if c['id'] not in complete],
        audit_errors=errors,failed_tasks=[f for f in failures if f['id'] not in complete],historical_failed_tasks=[f for f in failures if f['id'] in complete],
        checkpoint_fits=cost,fits=692*len(complete),saved_setting_refits=20*len(complete),unknown_attempt_cost_ids=unknown,
        manifest=identity(),warning_count=sum(len(v['warnings']) for v in complete.values()),production_accepted=False,
        scope='fresh diagnostic development; no new error guarantee, threshold learning or D-040 rescue; checkpoint costs overlap complete totals')
    io.write(output/'audit.json',audit);io.write(output/'assurance.json',d.analytical_report());pd.DataFrame(rows).to_csv(output/'case_metrics.csv',index=False)
    report(output,rows,audit);return audit


def report(output,rows,audit):
    text=['# Fresh confidence diagnostic development', '',f"{audit['complete']}/60 datasets complete; {audit['missing']} missing; {len(audit['audit_errors'])} audit errors.",
        'Small development probe; observed counts and gain summaries are diagnostics, not accepted reliability labels. D-040 failed outcome and thresholds remain unchanged.', '',
        'Hypothetical fixed 1% false-alarm rate: historical per-setting assurance pass probability 67.7%; 20 independent equal-rate settings jointly about 0.0405%. Rates differ in practice; no posterior interpretation.', '']
    if rows:
        frame=pd.DataFrame(rows);true=frame[frame.pair=='x1:x2'];summary=true.groupby(['cell','scenario','target_gap']).agg(assessed=('id','count'),
            eligible=('eligible','sum'),above=('above_threshold','sum'),cv_matched_median=('cv_matched','median'),independent_matched_median=('independent_matched','median'),
            ideal_gap_median=('ideal_independent_gap','median'),pattern_consistent=('pattern_consistent','sum')).reset_index();summary.to_csv(output/'diagnostic_summary.csv',index=False)
        text+=['Independent gains and ideal population predictions are diagnostic only. Nonpositive raw values and all decision exclusions remain in case_metrics.csv.', '',summary.to_string(index=False),'']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def record_failure(c,folder,error):
    path=Path(folder)/'error.json';attempts=[]
    if path.exists():
        v=io.read(path)
        if v['task']!=c or v['context']!=context(c):raise ValueError('existing failure context differs')
        attempts=v.get('attempts',[v['error']])
    if any(type(x) is not str for x in attempts):raise ValueError('invalid failure history')
    io.write(path,dict(task=c,context=context(c),error=error,attempts=attempts+[error]))


def run_shard(i,output):
    freeze();io.write(Path(output)/f'shard-{i}.json',dict(id=i,tasks=shard_tasks(i),manifest=identity()));failed=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_task,c,output):c for c in shard_tasks(i)}
        for f in as_completed(futures):
            c=futures[f]
            try:f.result()
            except Exception as e:
                record_failure(c,Path(output)/c['id'],repr(e));failed.append(c['id'])
    if failed:raise RuntimeError('failed diagnostic tasks: '+','.join(failed))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','matrix','shard','aggregate'));p.add_argument('--write',action='store_true');p.add_argument('--id');p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,default=ROOT/'.superpowers/confidence-diagnostic');a=p.parse_args()
    if a.command=='freeze':freeze(not a.write);print('frozen diagnostic context verified');return
    freeze()
    if a.command=='matrix':print(json.dumps(d.shards()))
    elif a.command=='shard':run_shard(a.id,a.output)
    else:
        if a.input:io.merge_shards([p for p in a.input.iterdir() if p.is_dir()],a.output)
        v=aggregate(a.output);print(json.dumps({k:v[k] for k in ('planned','complete','missing')}))
        if v['missing'] or v['audit_errors']:sys.exit(1)
if __name__=='__main__':main()
