"""Two-stage fixed-pair confidence research; confirmation requires a frozen rule."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from pathlib import Path
import argparse,hashlib,json,sys,time,warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import pair_screen_study as io
import pair_confidence_rules as rules
import pair_confidence_models as models
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

MANIFESTS={s:ROOT/f'examples/study_manifests/pair-confidence-{s}.json' for s in ('calibration','confirmation')}
THRESHOLDS=ROOT/'examples/study_manifests/pair-confidence-thresholds.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_digest():
    # Git checkouts may change only line endings on Windows.
    return hashlib.sha256(THRESHOLDS.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def sources():
    paths=list((ROOT/'src/open').rglob('*.py'))+[ROOT/f'examples/{n}' for n in (
        'pair_confidence_study.py','pair_confidence_models.py','pair_confidence_rules.py','pair_confidence_protocol.md',
        'pair_correlation_study.py','pair_screen_study.py','spline_stability_smoke.py')]
    paths += [ROOT/p for p in ('.github/workflows/pair-confidence.yml',
        '.github/actions/pair-confidence-setup/action.yml') if (ROOT/p).exists()]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(paths)}


def make_rule(rows):
    return dict(schema='open.pair_confidence_thresholds.v1',status='complete',rule=rules.RULE,
        calibration_manifest=io.fingerprint(manifest('calibration')),rows=rows,
        thresholds=rules.learn_thresholds(rows,rules.tasks('calibration')))


def validate_rule():
    if not THRESHOLDS.exists():raise ValueError('confirmation requires committed calibration thresholds')
    v=io.read(THRESHOLDS)
    if (v.get('schema')!='open.pair_confidence_thresholds.v1' or v.get('status')!='complete' or
        v.get('rule')!=rules.RULE or v.get('calibration_manifest')!=io.fingerprint(manifest('calibration'))):
        raise ValueError('calibration rule provenance differs')
    if rules.learn_thresholds(v['rows'],rules.tasks('calibration'))!=v['thresholds']:
        raise ValueError('thresholds disagree with frozen calibration evidence')
    if any(not isinstance(r.get('result_sha256'),str) or len(r['result_sha256'])!=64 or
           any(c not in '0123456789abcdef' for c in r['result_sha256']) for r in v['rows']):
        raise ValueError('calibration result digests missing')
    return v


def manifest(stage):
    value=dict(schema='open.pair_confidence_study.v1',stage=stage,rule=rules.RULE,
        tasks=rules.tasks(stage),shards=rules.shards(stage),environment=io.EXPECTED_ENV,
        sources=sources(),pairs=[list(p) for p in rules.PAIRS],fits_per_comparison=173,
        fits_per_task=692,saved_setting_refits=20,inner_folds=3,outer_folds=3,
        independent_n=1000,grid=rules.GRID.tolist(),tasks_per_shard=40,total_budget_runner_minutes=5550,
        base_grid=models.BASE_GRID,joint_grid=models.JOINT_GRID)
    if stage=='calibration':value['preregistered_confirmation_tasks']=rules.tasks('confirmation')
    else:
        validate_rule();value['thresholds_sha256']=threshold_digest()
    return value


def freeze(stage,check=True):
    io.validate_environment();value=manifest(stage)
    if check:
        if io.read(MANIFESTS[stage])!=value:raise ValueError('frozen confidence context differs')
    else:io.write(MANIFESTS[stage],value)
    identity.cache_clear()
    return value


@lru_cache(None)
def identity(stage):return io.fingerprint(manifest(stage))


def context(c):return io.fingerprint(dict(task=c,manifest=identity(c['stage'])))


def shard_tasks(stage,shard):
    if shard not in rules.shards(stage):raise ValueError('foreign shard')
    i=int(shard);return rules.tasks(stage)[40*i:40*i+40]


def splits(c):return list(KFold(3,shuffle=True,random_state=c['split_seed']).split(range(c['n'])))


def comparisons(c):
    X,y,_=models.generate(c,c['n'],c['seed']);T,ty,_=models.generate(c,1000,c['test_seed'])
    Z=pd.concat([X,T],ignore_index=True);z=np.concatenate([y,ty])
    return [(X,y,a.tolist(),b.tolist()) for a,b in splits(c)]+[(Z,z,list(range(c['n'])),list(range(c['n'],c['n']+1000)))]


def verification(c,folder):
    names=['receipt.json','result.json','timing.json']+[f'comparison-{i}.json' for i in range(4)]
    return dict(context=context(c),saved_setting_refits=20,digests={n:digest(folder/n) for n in names})


def check_checkpoint(c,saved,data):
    if saved['context']!=context(c):raise ValueError('checkpoint context differs')
    if not np.isfinite(saved['fitting_seconds']) or saved['fitting_seconds']<0 or type(saved['warnings']) is not list:
        raise ValueError('checkpoint timing/warnings differ')
    models.audit_comparison(saved['result'],*data,c['split_seed'])


def check_result(c,folder,refit=False):
    v=io.read(folder/'result.json');receipt=io.read(folder/'receipt.json')
    if (v['task']!=c or v['context']!=context(c) or v['status']!='complete' or
        receipt!=dict(task=c,context=context(c),environment=io.environment()) or len(v['comparisons'])!=4):
        raise ValueError('result or receipt context differs')
    checkpoints=[]
    for i,data in enumerate(comparisons(c)):
        s=io.read(folder/f'comparison-{i}.json');check_checkpoint(c,s,data);checkpoints.append(s)
        if v['comparisons'][i]!=s['result']:raise ValueError('result differs from checkpoint')
        if refit:models.audit_comparison(s['result'],*data,c['split_seed'],refit=True)
    if v['warnings']!=[w for s in checkpoints for w in s['warnings']]:raise ValueError('warning evidence differs')
    t=io.read(folder/'timing.json')
    if (t['fits']!=692 or t['saved_setting_refits']!=20 or type(t['resumed']) is not int or not 0<=t['resumed']<=4 or
        t['fits_this_attempt']!=173*(4-t['resumed']) or not all(np.isfinite(t[k]) and t[k]>=0 for k in (
            'total_seconds','verification_seconds','retained_fitting_seconds'))):raise ValueError('cost record differs')
    np.testing.assert_allclose(t['retained_fitting_seconds'],sum(s['fitting_seconds'] for s in checkpoints),rtol=1e-12)
    return v


def run_task(c,output):
    io.validate_environment()
    if c not in rules.tasks(c['stage']):raise ValueError('foreign task')
    folder=Path(output)/c['id'];folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    receipt=dict(task=c,context=context(c),environment=io.environment())
    if (folder/'receipt.json').exists() and io.read(folder/'receipt.json')!=receipt:raise ValueError('existing receipt differs')
    io.write(folder/'receipt.json',receipt)
    if (folder/'verified.json').exists():
        v=check_result(c,folder)
        if io.read(folder/'verified.json')!=verification(c,folder):raise ValueError('verification stamp differs')
        return v
    values=[];resumed=0
    for i,data in enumerate(comparisons(c)):
        p=folder/f'comparison-{i}.json'
        if p.exists():saved=io.read(p);check_checkpoint(c,saved,data);resumed+=1
        else:
            tick=time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');v=models.compare(*data,c['split_seed'])
            saved=dict(context=context(c),result=v,fitting_seconds=time.monotonic()-tick,
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]);io.write(p,saved)
        values.append(saved)
    v=dict(task=c,context=context(c),status='complete',comparisons=[s['result'] for s in values],
           warnings=[w for s in values for w in s['warnings']]);io.write(folder/'result.json',v)
    costs=dict(fits=692,saved_setting_refits=20,resumed=resumed,fits_this_attempt=173*(4-resumed),
               retained_fitting_seconds=sum(s['fitting_seconds'] for s in values))
    io.write(folder/'timing.json',dict(costs,total_seconds=time.monotonic()-start,verification_seconds=0.))
    tick=time.monotonic();check_result(c,folder,True)
    io.write(folder/'timing.json',dict(costs,total_seconds=time.monotonic()-start,verification_seconds=time.monotonic()-tick))
    io.write(folder/'verified.json',verification(c,folder));return v


def result_rows(v):
    c=v['task'];rows=[];folds=v['comparisons'][:3];ind=v['comparisons'][3]
    for pos,pair in enumerate(rules.PAIRS):
        r=models.fold_records(folds,pos);pattern=rules.pattern_summary([f['pairs'][pos]['curve'] for f in folds])
        rows.append(dict(**c,pair=f'x{pair[0]+1}:x{pair[1]+1}',score=rules.pair_score(r),
            cv_overall=rules.pooled(r,'overall'),cv_matched=rules.pooled(r,'matched'),
            positive_folds=sum(f['matched_additive_mse']>f['joint_mse'] for f in r),
            independent_overall=ind['pairs'][pos]['overall_gain'],independent_matched=ind['pairs'][pos]['matched_gain'],
            oracle_gap=models.oracle_gap(c),pattern_status=pattern['status'],pattern_consistent=pattern['consistent'],
            pattern_cosine=pattern['min_cosine'],common_supported=pattern['common_supported'],
            truth_cosine=truth_cosine(folds,pos,c)))
    return rows


def truth_cosine(folds,pos,c):
    if pos or c['scenario'] not in ('pure','mixed'):return None
    curves=[f['pairs'][pos]['curve'] for f in folds];mask=np.array([z['support'] for z in curves]).all(axis=0)
    mask[2,:]=False;mask[:,2]=False
    if mask.sum()<4:return None
    truth=c['b']*rules.GRID[:,None]*rules.GRID[None,:];t=truth[mask];tn=np.linalg.norm(t)
    cos=[]
    for z in curves:
        a=np.array(z['contrast'])[mask];norm=np.linalg.norm(a)
        if norm<1e-8 or tn<1e-8:return None
        cos.append(float(np.clip(a@t/(norm*tn),-1,1)))
    return min(cos)


def confirmation_summary(rows,errors,completed):
    planned=rules.tasks('confirmation')
    null=[];signals=[]
    for n,p,rho in rules.CELLS:
        cell=rules.cell_id(n,p,rho)
        for scenario in ('noise','additive'):
            cases=[c for c in planned if c['cell']==cell and c['scenario']==scenario]
            assessed=[c for c in cases if c['id'] in completed]
            k=sum(any(r['qualifies'] for r in rows if r['id']==c['id']) for c in assessed);missing=len(cases)-len(assessed)
            upper=rules.false_alarm_bound(k+missing,len(cases))
            null.append(dict(cell=cell,scenario=scenario,planned=len(cases),assessed=len(assessed),above=k,
                unknown=missing,lower=k,upper=k+missing,upper_bound=upper))
        for scenario in ('pure','mixed'):
            for b in (.2,.8):
                group=[r for r in rows if r['cell']==cell and r['scenario']==scenario and r['b']==b and r['pair']=='x1:x2']
                controls=[r for r in rows if r['cell']==cell and r['scenario']==scenario and r['b']==b and r['pair']=='x1:x3']
                k=sum(r['qualifies'] for r in group);missing=30-len(group)
                signals.append(dict(cell=cell,scenario=scenario,b=b,planned=30,assessed=len(group),detected=k,
                    unknown=missing,lower=k,upper=k+missing,lower_bound=rules.detection_bound(k,30),
                    control_detected=sum(r['qualifies'] for r in controls),control_unknown=30-len(controls),
                    control_lower=sum(r['qualifies'] for r in controls),
                    control_upper=sum(r['qualifies'] for r in controls)+30-len(controls),
                    simultaneous_lower_bound=rules.detection_bound(k,30,40),
                    detected_consistent=sum(r['qualifies'] and r['pattern_consistent'] for r in group),
                    stable_wrong=sum(r['pattern_consistent'] and r['truth_cosine'] is not None and r['truth_cosine']<.8 for r in group)))
    null_ids={c['id'] for c in planned if c['scenario'] in ('noise','additive')}
    all_ids={c['id'] for c in planned}
    null_errors=any(e['id'] in null_ids or e['id'] not in all_ids for e in errors)
    qualified=not null_errors and null_ids<=set(completed) and all(c['upper_bound']<=.05 for c in null)
    invalid_ids={e['id'] for e in errors}
    for c in signals:
        ids={t['id'] for t in planned if (t['cell'],t['scenario'],t['b'])==(c['cell'],c['scenario'],c['b'])}
        c['audit_invalid']=bool(ids&invalid_ids)
        c['provisional_good_detection']=bool(qualified and not c['audit_invalid'] and c['unknown']==0 and c['lower_bound']>=.8)
    return dict(null_qualified=bool(qualified),null_cells=null,signal_cells=signals)


def aggregate(stage,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);planned=rules.tasks(stage)
    completed={};errors=[];failures=[];rows=[];unknown=[]
    retained=dict(comparisons=0,fits=0,seconds=0.,warnings=0);partial=dict(retained)
    for c in planned:
        folder=output/c['id']
        if not folder.exists():continue
        if (folder/'error.json').exists():
            try:
                failure=io.read(folder/'error.json');failures.append(dict(id=c['id'],error=failure))
                if failure['task']!=c or failure['context']!=context(c):raise ValueError('failure context differs')
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if (folder/'result.json').exists():
            try:
                v=check_result(c,folder)
                if io.read(folder/'verified.json')!=verification(c,folder):raise ValueError('verified stamp differs')
                completed[c['id']]=v;rows.extend(result_rows(v))
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if c['id'] not in completed and ((folder/'receipt.json').exists() or any(folder.glob('comparison-*.json'))):unknown.append(c['id'])
        data=comparisons(c)
        for i,p in enumerate(folder/f'comparison-{j}.json' for j in range(4)):
            if not p.exists():continue
            try:
                s=io.read(p);check_checkpoint(c,s,data[i])
                for bucket in ([retained,partial] if c['id'] not in completed else [retained]):
                    bucket['comparisons']+=1;bucket['fits']+=173;bucket['seconds']+=s['fitting_seconds'];bucket['warnings']+=len(s['warnings'])
            except Exception as e:errors.append(dict(id=c['id'],checkpoint=p.name,error=repr(e)))
    foreign=[p.name for p in output.glob('confidence--*') if p.is_dir() and p.name not in {c['id'] for c in planned}]
    errors.extend(dict(id=i,error='foreign task identity') for i in foreign)
    for path in output.glob('shard-*.json'):
        try:
            stamp=io.read(path);i=stamp['id']
            if (path.name!=f'shard-{i}.json' or stamp!=dict(stage=stage,id=i,tasks=shard_tasks(stage,i),manifest=identity(stage))):
                raise ValueError('shard stamp differs')
        except Exception as e:errors.append(dict(id=path.name,error=repr(e)))
    audit=dict(stage=stage,planned=len(planned),complete=len(completed),missing=len(planned)-len(completed),
        missing_ids=[c['id'] for c in planned if c['id'] not in completed],audit_errors=errors,
        failed_tasks=[f for f in failures if f['id'] not in completed],historical_failed_tasks=[f for f in failures if f['id'] in completed],
        unfinished_attempt_cost_unknown_ids=unknown,retained_checkpoint_cost=retained,partial_checkpoint_cost=partial,
        fits=692*len(completed),saved_setting_refits=20*len(completed),warning_count=sum(len(v['warnings']) for v in completed.values()),
        cost_scope='Complete task totals overlap checkpoint costs; interrupted work is additional unknown cost.',
        manifest=identity(stage),acceptance='research only; no production confidence rule adopted')
    if stage=='calibration':
        calrows=[dict(id=c['id'],cell=c['cell'],scenario=c['scenario'],
            score=max(r['score'] for r in rows if r['id']==c['id']),result_sha256=digest(output/c['id']/'result.json'))
            for c in planned if c['id'] in completed]
        if audit['missing']==0 and not errors:io.write(output/'learned-thresholds.json',make_rule(calrows))
        else:io.write(output/'learned-thresholds.json',dict(status='unavailable',reason='incomplete or invalid calibration',complete=len(completed)))
        audit['thresholds_ready']=audit['missing']==0 and not errors
    else:
        rule=validate_rule();audit['thresholds_sha256']=threshold_digest()
        for r in rows:r['qualifies']=rules.qualifies(r['score'],rule['thresholds'][r['cell']])
        audit.update(confirmation_summary(rows,errors,set(completed)))
        pd.DataFrame(audit['null_cells']).to_csv(output/'null_summary.csv',index=False)
        pd.DataFrame(audit['signal_cells']).to_csv(output/'signal_summary.csv',index=False)
    io.write(output/'audit.json',audit);pd.DataFrame(rows).to_csv(output/'case_metrics.csv',index=False)
    report=['# Interaction confidence study', '',f"Stage {stage}: {audit['complete']}/{audit['planned']} datasets; missing {audit['missing']}; audit errors {len(errors)}.",
        'Fixed-pair Gaussian spline research only; thresholds are not a general p-value or a posterior probability.',
        'Prediction decisions and descriptive supported-grid pattern flags are separate. Missing results remain unknown.',
        'Complete-task and checkpoint costs overlap; interrupted costs can be unknown.', '']
    if stage=='calibration':report += [f"Threshold artifact ready: {audit['thresholds_ready']}. Confirmation cannot start until this artifact is audited, committed and separately frozen."]
    else:
        report += [f"Simultaneous null gate passed: {audit['null_qualified']}.",
            'Good-detection flags require the null gate and pointwise sensitivity lower bound >=.8; consult simultaneous bounds separately.',
            'Weak detection is uncertain evidence, not proof of no interaction. Stable-looking curves can still be wrong.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    if rows:
        plot(stage,output,rows,audit)
        with (output/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n![Research study summary](SUMMARY.png)\n')
    return audit


def plot(stage,output,rows,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=[rules.cell_id(*c) for c in rules.CELLS]
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    if stage=='calibration':
        scores={i:max(r['score'] for r in rows if r['id']==i) for i in {r['id'] for r in rows}}
        for scenario,color,offset in (('noise','#708090',-.12),('additive','#2166ac',.12)):
            for i,cell in enumerate(cells):
                data=[r for r in rows if r['pair']=='x1:x2' and r['cell']==cell and r['scenario']==scenario]
                axes[0].scatter([i+offset]*len(data),[scores[r['id']] for r in data],s=12,alpha=.5,color=color,
                    label=scenario if i==0 else None)
        if audit['thresholds_ready']:
            thresholds=io.read(output/'learned-thresholds.json')['thresholds']
            axes[0].plot(range(10),[thresholds[c] for c in cells],marker='x',color='#c78b36',label='Frozen rule threshold')
        axes[0].set(ylabel='Two-pair maximum eligible CV score',title='Null calibration; no confirmation results')
        assessed=[sum(r['cell']==c and r['pair']=='x1:x2' for r in rows) for c in cells]
        axes[1].bar(range(10),assessed,color='#2166ac');axes[1].axhline(200,color='black',ls='--',label='Planned per cell')
        axes[1].set(ylabel='Independent calibration datasets',title='Coverage; incomplete cells cannot learn a rule');axes[1].legend()
    else:
        for scenario,color,offset in (('noise','#708090',-.12),('additive','#2166ac',.12)):
            c=[z for z in audit['null_cells'] if z['scenario']==scenario]
            axes[0].plot([cells.index(z['cell'])+offset for z in c],[z['upper_bound'] for z in c],marker='o',ls='',color=color,label=scenario)
        axes[0].axhline(.05,color='black',ls='--',label='Prespecified upper-bound gate')
        axes[0].set(ylabel='Simultaneous 95% false-alarm upper bound',title='Null confirmation: two fixed pairs')
        for scenario,color in (('pure','#2166ac'),('mixed','#c78b36')):
            for b,offset,marker in ((.2,-.13,'o'),(.8,.13,'x')):
                c=[z for z in audit['signal_cells'] if z['scenario']==scenario and z['b']==b]
                axes[1].plot([cells.index(z['cell'])+offset for z in c],[z['lower_bound'] for z in c],ls='',marker=marker,color=color,label=f'{scenario}, b={b}')
        axes[1].axhline(.8,color='black',ls='--');axes[1].set(ylabel='Pointwise 95% detection lower bound',title='Sensitivity; pure b levels are scale checks');axes[1].legend(fontsize=8)
        for ax in axes:ax.set_ylim(-.02,1.02)
    for ax in axes:ax.set_xticks(range(10),cells,rotation=70,fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Research {stage}: {audit['complete']}/{audit['planned']} datasets\nConditional Gaussian spline evidence; not a general significance test or production rule")
    fig.savefig(output/'SUMMARY.png',dpi=150);plt.close(fig)


def run_shard(stage,shard,output):
    freeze(stage);io.write(Path(output)/f'shard-{shard}.json',dict(stage=stage,id=shard,tasks=shard_tasks(stage,shard),manifest=identity(stage)))
    failed=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_task,c,output):c for c in shard_tasks(stage,shard)}
        for f in as_completed(futures):
            c=futures[f]
            try:f.result()
            except Exception as e:
                io.write(Path(output)/c['id']/'error.json',dict(task=c,context=context(c),error=repr(e)));failed.append(c['id'])
    if failed:raise RuntimeError(f'failed tasks retained: {failed}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','matrix','shard','aggregate'))
    parser.add_argument('--stage',choices=('calibration','confirmation'),default='calibration');parser.add_argument('--write',action='store_true')
    parser.add_argument('--id');parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'.superpowers/pair-confidence')
    a=parser.parse_args()
    if a.command=='freeze':freeze(a.stage,not a.write);print('frozen confidence context verified');return
    freeze(a.stage)
    if a.command=='matrix':print(json.dumps(rules.shards(a.stage)))
    elif a.command=='shard':run_shard(a.stage,a.id,a.output)
    else:
        if a.input:io.merge_shards([p for p in a.input.iterdir() if p.is_dir()],a.output)
        v=aggregate(a.stage,a.output);print(json.dumps({k:v[k] for k in ('planned','complete','missing')}))
        if v['missing'] or v['audit_errors']:sys.exit(1)


if __name__=='__main__':main()
