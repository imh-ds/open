"""Approved D-043 checkpointed research runner. No production API changes."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,time,warnings,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import realistic_interaction_design as d
import realistic_interaction_truth as truth
import prospective_interaction_study as previous
old=previous.old;io=old.io
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

MANIFEST=ROOT/'examples/study_manifests/prospective-interaction-realistic.json'
THRESHOLDS=ROOT/'examples/study_manifests/prospective-interaction-realistic-thresholds.json'
CALIBRATION_AUDIT=ROOT/'examples/study_manifests/prospective-interaction-realistic-calibration-audit.json'
NEW_SOURCES=('examples/realistic_interaction_design.py','examples/realistic_interaction_truth.py',
    'examples/realistic_interaction_study.py','examples/realistic_interaction_report.py',
    'examples/realistic_interaction_verify.py','examples/prospective_interaction_realistic_protocol.md',
    '.github/workflows/realistic-interaction.yml','.github/actions/realistic-interaction-setup/action.yml')


def sources():
    value=previous.sources()
    for name in NEW_SOURCES:
        value[name]=hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return dict(sorted(value.items()))


def historical_seeds():
    return {p.name:io.read(p) for p in sorted((ROOT/'examples/study_manifests').glob('*.json'))
            if not p.name.startswith('prospective-interaction-realistic')}


def manifest():
    return dict(schema='open.realistic_interaction.v1',tasks=d.tasks(),sources=sources(),
        environment=io.EXPECTED_ENV,seed_audit=d.audit_seed_collisions(d.tasks(),historical_seeds()),
        tasks_per_shard=80,workers=2,budget_minutes=3750,fits_per_task=692,saved_setting_refits=20,
        independent_n=1000,inner_folds=3,outer_folds=3,base_grid=old.models.BASE_GRID,joint_grid=old.models.JOINT_GRID,
        planned_fits=9964800,planned_verification_refits=288000,production_accepted=False)


def freeze(check=True):
    io.validate_environment();previous.freeze();v=manifest()
    if check:
        if io.read(MANIFEST)!=v:raise ValueError('frozen realistic context differs')
    else:io.write(MANIFEST,v)
    identity.cache_clear();task_map.cache_clear();thresholds.cache_clear();return v


@lru_cache(None)
def identity():return io.fingerprint(manifest())


@lru_cache(None)
def task_map():return {c['id']:c for c in d.tasks()}


@lru_cache(None)
def thresholds():
    if not THRESHOLDS.exists() or not CALIBRATION_AUDIT.exists():raise ValueError('development requires committed calibration audit and fresh thresholds')
    import realistic_interaction_verify as verify
    a=io.read(CALIBRATION_AUDIT);v=io.read(THRESHOLDS)
    if v!=verify.verify_calibration(a,identity()):raise ValueError('calibration threshold provenance differs')
    return v


def validate_task(c):
    if task_map().get(c.get('id'))!=c:raise ValueError('foreign realistic task')
    if c['stage']=='development':thresholds()


def context(c):
    validate_task(c)
    return io.fingerprint(dict(task=c,manifest=identity(),thresholds=io.fingerprint(thresholds()) if c['stage']=='development' else None))


def shard_tasks(stage,i):
    if stage=='development':thresholds()
    return d.shard_tasks(stage,i)


def comparisons(c):
    x,y=truth.generate(c,c['seed'],c['n']);a,b=truth.generate(c,c['test_seed'],1000)
    values=[(x,y,list(train),list(test)) for train,test in KFold(3,shuffle=True,random_state=c['split_seed']).split(x)]
    values.append((pd.concat([x,a],ignore_index=True),np.concatenate([y,b]),list(range(c['n'])),list(range(c['n'],c['n']+1000))))
    return values


def precompute_truth(c,data):
    # All numerical truth checks finish before any comparison starts fitting.
    for x,y,train,test in data:
        truth.teacher(c,x.iloc[test])
        for pair in old.rules.PAIRS:truth.teacher(c,old.models.surface_inputs(x,pair))


def population_record(c,data,v):
    x,y,train,test=data;p=truth.teacher(c,x.iloc[test]);observed=np.asarray(y)[test]
    return dict(predicted=p.tolist(),mse=float(np.square(observed-p).mean()),
        surfaces=[truth.teacher(c,old.models.surface_inputs(x,pair)).reshape(5,5).tolist() for pair in old.rules.PAIRS],
        ideal_additive='unassessed',scope='measured-input conditional teacher; diagnostic only')


def audit_population(c,data,v,o):
    if o!=population_record(c,data,v):raise ValueError('conditional teacher evidence differs')


def check_checkpoint(c,s,data,refit=False):
    if s['context']!=context(c) or not np.isfinite(s['fitting_seconds']) or s['fitting_seconds']<0 or type(s['warnings']) is not list:
        raise ValueError('checkpoint context or costs differ')
    old.models.audit_comparison(s['result'],*data,c['split_seed'],refit=refit);audit_population(c,data,s['result'],s['population'])


def verification(c,folder):
    names=['receipt.json','result.json','timing.json','attempts.json']+[f'comparison-{i}.json' for i in range(4)]
    return dict(context=context(c),saved_setting_refits=20,digests={n:old.digest(folder/n) for n in names})


def attempt_cost(c,folder):
    path=Path(folder)/'attempts.json'
    if not path.exists():
        return dict(fits=0,verification_refits=0,seconds=0.,unknown_fit_cost=True,unknown_verification_cost=True)
    ledger=io.read(path)
    if ledger['context']!=context(c) or not ledger['attempts']:raise ValueError('attempt ledger context differs')
    for i,a in enumerate(ledger['attempts']):
        if (a['id']!=i or a['status'] not in ('running','failed','complete') or type(a['fits']) is not int or
            a['fits'] not in (0,173,346,519,692) or a['verification_refits'] not in (0,20) or
            type(a['unknown_fit_cost']) is not bool or type(a['unknown_verification_cost']) is not bool or
            not np.isfinite([a['started_at'],a['total_seconds']]).all() or a['total_seconds']<0 or
            (a['status']=='complete' and (a['unknown_fit_cost'] or a['unknown_verification_cost']))):
            raise ValueError('attempt cost record differs')
    return dict(fits=sum(a['fits'] for a in ledger['attempts']),
        verification_refits=sum(a['verification_refits'] for a in ledger['attempts']),
        seconds=sum(a['total_seconds'] for a in ledger['attempts']),
        unknown_fit_cost=any(a['unknown_fit_cost'] for a in ledger['attempts']),
        unknown_verification_cost=any(a['unknown_verification_cost'] for a in ledger['attempts']))


def update_attempt(c,folder,**values):
    path=Path(folder)/'attempts.json';ledger=io.read(path)
    if ledger['context']!=context(c):raise ValueError('attempt update context differs')
    a=ledger['attempts'][-1];a.update(values);a['total_seconds']=time.monotonic()-a['started_monotonic']
    io.write(path,ledger)


def audit_task(c,folder,refit=False):
    if not (folder/'attempts.json').exists():raise ValueError('complete task lacks attempt evidence')
    attempt_cost(c,folder)
    v=io.read(folder/'result.json');receipt=io.read(folder/'receipt.json')
    if (v['task']!=c or v['context']!=context(c) or v['status']!='complete' or
        receipt!=dict(task=c,context=context(c),environment=io.environment()) or len(v['comparisons'])!=4 or len(v['population'])!=4):
        raise ValueError('task evidence identity differs')
    records=[]
    for i,data in enumerate(comparisons(c)):
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
    validate_task(c);folder=Path(output)/c['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'verified.json').exists():return _run_task(c,output)
    path=folder/'attempts.json';ledger=io.read(path) if path.exists() else dict(context=context(c),attempts=[])
    if ledger['context']!=context(c):raise ValueError('existing attempt context differs')
    if path.exists():attempt_cost(c,folder)
    ledger['attempts'].append(dict(id=len(ledger['attempts']),status='running',started_at=time.time(),
        started_monotonic=time.monotonic(),total_seconds=0.,fits=0,verification_refits=0,
        unknown_fit_cost=False,unknown_verification_cost=False))
    io.write(path,ledger)
    try:
        v=_run_task(c,output);update_attempt(c,folder,status='complete')
        io.write(folder/'verified.json',verification(c,folder));return v
    except Exception as e:
        update_attempt(c,folder,status='failed',error=repr(e));raise


def _run_task(c,output):
    validate_task(c)
    io.validate_environment();folder=Path(output)/c['id'];folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    receipt=dict(task=c,context=context(c),environment=io.environment())
    if (folder/'receipt.json').exists() and io.read(folder/'receipt.json')!=receipt:raise ValueError('existing receipt differs')
    io.write(folder/'receipt.json',receipt)
    if (folder/'verified.json').exists():
        v=audit_task(c,folder)
        if io.read(folder/'verified.json')!=verification(c,folder):raise ValueError('verification stamp differs')
        return v
    data_values=comparisons(c);precompute_truth(c,data_values)
    records=[];resumed=0
    for i,data in enumerate(data_values):
        path=folder/f'comparison-{i}.json'
        if path.exists():s=io.read(path);check_checkpoint(c,s,data);resumed+=1
        else:
            tick=time.monotonic()
            update_attempt(c,folder,unknown_fit_cost=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');v=old.models.compare(*data,c['split_seed'])
            s=dict(context=context(c),result=v,population=population_record(c,data,v),fitting_seconds=time.monotonic()-tick,
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]);io.write(path,s)
            a=io.read(folder/'attempts.json')['attempts'][-1]
            update_attempt(c,folder,fits=a['fits']+173,unknown_fit_cost=False)
        records.append(s)
    v=dict(task=c,context=context(c),status='complete',comparisons=[s['result'] for s in records],population=[s['population'] for s in records],
        warnings=[w for s in records for w in s['warnings']]);io.write(folder/'result.json',v)
    costs=dict(fits=692,saved_setting_refits=20,resumed=resumed,fits_this_attempt=173*(4-resumed),retained_fitting_seconds=sum(s['fitting_seconds'] for s in records))
    io.write(folder/'timing.json',dict(costs,total_seconds=time.monotonic()-start,verification_seconds=0.))
    tick=time.monotonic();update_attempt(c,folder,unknown_verification_cost=True);audit_task(c,folder,True)
    update_attempt(c,folder,verification_refits=20,unknown_verification_cost=False)
    io.write(folder/'timing.json',dict(costs,total_seconds=time.monotonic()-start,verification_seconds=time.monotonic()-tick));return v




def record_failure(c,folder,error):
    path=Path(folder)/'error.json';attempts=[]
    if path.exists():
        v=io.read(path)
        if v['task']!=c or v['context']!=context(c):raise ValueError('existing failure context differs')
        attempts=v.get('attempts',[v['error']])
    if any(type(x) is not str for x in attempts):raise ValueError('invalid failure history')
    io.write(path,dict(task=c,context=context(c),error=error,attempts=attempts+[error]))



def run_shard(stage,i,output):
    freeze();planned=shard_tasks(stage,i)
    io.write(Path(output)/f'shard-{stage}-{i}.json',dict(id=i,stage=stage,tasks=planned,manifest=identity(),
        thresholds=io.fingerprint(thresholds()) if stage=='development' else None))
    failed=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_task,c,output):c for c in planned}
        for future in as_completed(futures):
            c=futures[future]
            try:future.result()
            except Exception as e:
                record_failure(c,Path(output)/c['id'],repr(e));failed.append(c['id'])
    if failed:raise RuntimeError('failed prospective tasks: '+','.join(failed))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','matrix','shard','pack','aggregate'))
    p.add_argument('--write',action='store_true');p.add_argument('--stage',choices=('calibration','development'),default='calibration')
    p.add_argument('--id');p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,default=ROOT/'.superpowers/realistic-interaction')
    a=p.parse_args()
    if a.command=='freeze':freeze(not a.write);print('frozen realistic context verified');return
    freeze()
    if a.command=='matrix':
        if a.stage=='development':thresholds()
        print(json.dumps(d.shards(a.stage)))
    elif a.command=='shard':run_shard(a.stage,a.id,a.output)
    elif a.command=='pack':
        import prospective_interaction_archive as archive
        print(json.dumps({k:v for k,v in archive.pack(a.input,a.output).items() if k!='inventory'}))
    else:
        import realistic_interaction_report as report
        v=report.aggregate(a.stage,[a.input],a.output)
        print(json.dumps({k:v[k] for k in ('planned','complete','missing','aggregation_seconds')}))
        if v['missing'] or v['audit_errors']:sys.exit(1)


if __name__=='__main__':main()


