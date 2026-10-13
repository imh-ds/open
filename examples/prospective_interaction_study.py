"""Frozen prospective validation runner; production and historical studies unchanged."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,time,warnings,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import prospective_interaction_design as d
import confidence_diagnostic as diagnostic
old=diagnostic.previous;io=old.io
import numpy as np
import pandas as pd
MANIFEST=ROOT/'examples/study_manifests/prospective-interaction.json'
SELECTION=ROOT/'examples/study_manifests/prospective-interaction-selection.json'
STAGE_A=ROOT/'examples/study_manifests/prospective-interaction-null-audit.json'
NEW_SOURCES=('examples/prospective_interaction_rules.py','examples/prospective_interaction_design.py',
 'examples/prospective_interaction_study.py','examples/prospective_interaction_report.py',
 'examples/prospective_interaction_protocol.md','examples/prospective_interaction_assurance.py',
 'examples/prospective_interaction_assurance.json','examples/confidence_diagnostic.py',
 '.github/workflows/prospective-interaction.yml','.github/actions/prospective-interaction-setup/action.yml')


def sources():
    value=old.sources()
    for name in list(value)+list(NEW_SOURCES):
        p=ROOT/name
        if not p.is_file():raise ValueError('required frozen source missing: '+name)
        value[name]=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return dict(sorted(value.items()))


@lru_cache(None)
def historical_seeds():
    paths=sorted((ROOT/'examples/study_manifests').glob('*.json'))
    return {p.name:io.read(p) for p in paths if not p.name.startswith('prospective-interaction')}


def manifest():
    old.validate_rule()
    return dict(schema='open.prospective_interaction.v1',tasks=d.tasks(),environment=io.EXPECTED_ENV,
        sources=sources(),thresholds_sha256=old.threshold_digest(),prior_manifest=io.fingerprint(old.manifest('confirmation')),
        seed_audit=d.audit_seed_collisions(d.tasks(),historical_seeds()),tasks_per_shard=80,workers=2,
        budget_minutes=6450,fits_per_task=692,saved_setting_refits=20,independent_n=1000,inner_folds=3,outer_folds=3,
        endpoints=dict(false=dict(alpha=.02/36,target=.05),detection=dict(alpha=.02/24,target=.8),shape=dict(alpha=.01/24,target=.9)),
        production_accepted=False)


def freeze(check=True):
    io.validate_environment();old.freeze('confirmation');v=manifest()
    if check:
        if io.read(MANIFEST)!=v:raise ValueError('frozen prospective context differs')
    else:io.write(MANIFEST,v)
    identity.cache_clear();selection.cache_clear();task_map.cache_clear();thresholds_by_cell.cache_clear()
    return v


@lru_cache(None)
def identity():return io.fingerprint(manifest())


@lru_cache(None)
def selection():
    if not SELECTION.exists() or not STAGE_A.exists():raise ValueError('signal stage requires committed null audit and selection')
    s=io.read(SELECTION);d.validate_selection(s,dict(identity=identity()),io.read(STAGE_A));return s


@lru_cache(None)
def task_map():return {c['id']:c for c in d.tasks()}


@lru_cache(None)
def thresholds_by_cell():return old.validate_rule()['thresholds']


def validate_task(c):
    if task_map().get(c.get('id'))!=c:raise ValueError('foreign prospective task')
    if c['stage']=='signal' and c['cell'] not in selection()['cells']:raise ValueError('unassessed signal condition')


def context(c):
    validate_task(c)
    return io.fingerprint(dict(task=c,manifest=identity(),selection=io.fingerprint(selection()) if c['stage']=='signal' else None))


def shard_tasks(stage,i):return d.shard_tasks(stage,i,selection() if stage=='signal' else None)


def population_record(c,data,v):
    X,y,train,test=data;a,j=diagnostic.population_predictions(X,c);a=a[test];j=j[test];observed=np.asarray(y)[test]
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
    validate_task(c)
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
    rows=old.result_rows(v);thresholds=thresholds_by_cell();c=v['task']
    for pos,r in enumerate(rows):
        info=diagnostic.decision(old.models.fold_records(v['comparisons'][:3],pos),thresholds[c['cell']])
        r.update(eligible=info['eligible'],above_threshold=info['above_threshold'],decision_reasons=';'.join(info['reasons']),
            fixed_threshold=info['threshold'],population_target_gap=c['target_gap'],
            ideal_independent_gap=v['population'][3]['gap'] if pos==0 else None,
            ideal_independent_joint_mse=v['population'][3]['joint_mse'] if pos==0 else None,
            model_independent_joint_mse=v['comparisons'][3]['pairs'][pos]['joint_mse'])
        np.testing.assert_allclose([r['cv_overall'],r['cv_matched'],r['score']],[info['overall'],info['matched'],info['score']],rtol=1e-12)
    return rows


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
        selection=io.fingerprint(selection()) if stage=='signal' else None))
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
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','matrix','shard','aggregate','selection'))
    p.add_argument('--write',action='store_true');p.add_argument('--stage',choices=('null','signal'),default='null')
    p.add_argument('--id');p.add_argument('--input',type=Path);p.add_argument('--audit',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'.superpowers/prospective-interaction');a=p.parse_args()
    if a.command=='freeze':freeze(not a.write);print('frozen prospective context verified');return
    freeze()
    if a.command=='matrix':print(json.dumps(d.shards(a.stage,selection() if a.stage=='signal' else None)))
    elif a.command=='shard':run_shard(a.stage,a.id,a.output)
    elif a.command=='selection':
        audit=io.read(a.audit);s=d.make_selection(dict(identity=identity()),audit);io.write(a.output,s)
        print(json.dumps(dict(cells=s['cells'],selected=len(s['selected_ids']),skipped=len(s['skipped_ids']))))
    else:
        import prospective_interaction_report as report
        inputs=[x for x in a.input.iterdir() if x.is_dir()] if a.input else [a.output]
        v=report.aggregate(a.stage,inputs,a.output,io.read(MANIFEST),selection() if a.stage=='signal' else None)
        print(json.dumps({k:v[k] for k in ('planned','complete','missing')}))
        if v['missing'] or v['audit_errors']:sys.exit(1)
if __name__=='__main__':main()
