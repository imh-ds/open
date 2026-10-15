"""Fresh optimized RF recursive validation and matched historical comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import json
import os
import zipfile

import qualify_chain_validation as old
import optimized_rf_recursive_audit as audit
import multiengine_preflight as previous
from open.layer_artifacts import portable,decode
from open.layer_types import LocalAssessment
from open.layer_stability import make_layer_sample_plans
from open.learners.trees import GRIDS
from recursive_preview_actions import runtime_check,github_api,load_json
from recursive_preview_runner import run_case,write_json,case_context
from recursive_preview_design import digest,text_hash
from time_scenario_validation import pack
from qualification_artifacts import download_artifacts

ROOT=old.ROOT
WORKFLOW='.github/workflows/optimized-rf-validation.yml'
MANIFEST=ROOT/'examples/study_manifests/optimized-rf-validation.json'
REFERENCE=ROOT/'examples/results/2026-10-15-multiengine-preflight-setup/completed-38788630751/audit.json'
PROTOCOL='examples/optimized_rf_validation_protocol.md'


def specs():
    return [dict(s,id='optimized-'+s['id']) for s in previous.specs() if s['engine']=='random_forest']


def settings(spec):
    c=old.q.design.settings(spec)
    return replace(c,learner=replace(c.learner,library=(spec['engine'],)))


def manifest():
    parent=load_json(ROOT/'examples/study_manifests/multiengine-preflight.json')
    rows=load_json(REFERENCE)['cases']
    references={}
    for spec in specs():
        row=next(r for r in rows if r['spec']['id']==spec['id'].removeprefix('optimized-'))
        path=REFERENCE.parent/'raw'/row['spec']['id']/'case.zip'
        references[spec['id']]=dict(path=path.relative_to(ROOT).as_posix(),
            sha256=row['archive_sha256'],bytes=row['archive_bytes'],row=row)
    paths=set(parent['sources'])|{'examples/study_manifests/multiengine-preflight.json',
        'examples/validate_optimized_rf.py','examples/optimized_rf_recursive_audit.py',
        REFERENCE.relative_to(ROOT).as_posix(),PROTOCOL,WORKFLOW}
    return dict(schema='open.optimized_rf_validation.v1',cases=specs(),
        runtime=parent['runtime'],sources={p:text_hash(ROOT/p) for p in sorted(paths)},
        configurations={s['id']:portable(settings(s)) for s in specs()},
        data_sha256={s['id']:hashlib.sha256(old.q.design.generate(s).to_csv(index=False,lineterminator='\n').encode()).hexdigest() for s in specs()},
        grid_policy=audit.POLICY,nominal_grid=audit.NOMINAL_GRID,singleton_grid=audit.SINGLETON_GRID,
        references=references,checkpoint_reuse=False,workers=4,max_parallel=2,
        case_job_minutes=120,compute_minutes=108,soft_seconds=105*60,
        prepare_minutes=15,audit_minutes=30,registration_minutes=1,
        reserved_minutes=286,ceiling=300,unallocated=14,
        scientific_qualification=False,fixtures_excluded_from_qualification=True)


def checked_manifest():
    value=load_json(MANIFEST)
    if value!=manifest():raise ValueError('optimized frozen source/runtime/identity/budget differs')
    runtime_check(value)
    from open.learners.trees import effective_grid
    if GRIDS['random_forest']!=audit.NOMINAL_GRID or any(
        effective_grid('random_forest',n)!=(audit.SINGLETON_GRID if n==1 else audit.NOMINAL_GRID) for n in (1,2,3,4)):
        raise ValueError('production effective grid differs from frozen audit policy')
    for ref in value['references'].values():
        p=ROOT/ref['path']
        if p.stat().st_size!=ref['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('historical reference archive fingerprint differs')
    return value


def first_attempt():
    old.q.first_attempt()


def once(run,api=github_api):
    page=1
    while True:
        batch=api(f'actions/workflows/optimized-rf-validation.yml/runs?event=workflow_dispatch&per_page=100&page={page}')['workflow_runs']
        if any(str(v['id'])!=str(run) for v in batch):
            raise ValueError('previous pilot dispatch exists; no automatic fit retry')
        if len(batch)<100:break
        page+=1


def prepare(output):
    first_attempt();value=checked_manifest();once(os.environ['GITHUB_RUN_ID'])
    a=dict(manifest=digest(value),github_run=os.environ['GITHUB_RUN_ID'],
        github_sha=os.environ['GITHUB_SHA'],github_attempt='1')
    a['digest']=digest(a)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    matrix=dict(include=[dict(id=s['id']) for s in specs()])
    write_json(output/'authorization.json',a);write_json(output/'matrix.json',matrix)
    return matrix


def authorization(path,value):
    a=load_json(path)
    if (a.get('digest')!=digest({k:v for k,v in a.items() if k!='digest'})
        or a.get('manifest')!=digest(value) or a.get('github_run')!=os.environ['GITHUB_RUN_ID']
        or a.get('github_sha')!=os.environ['GITHUB_SHA'] or a.get('github_attempt')!='1'):
        raise ValueError('pilot authorization source/run differs')
    return a


def run_one(case_id,output,authorization_path):
    value=checked_manifest()
    chosen=[s for s in specs() if s['id']==case_id]
    if len(chosen)!=1:raise ValueError('foreign pilot identity')
    first_attempt();authorization(authorization_path,value)
    spec=chosen[0];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    control=dict(spec=spec,manifest=digest(value),github_run=os.environ['GITHUB_RUN_ID'],
        github_sha=os.environ['GITHUB_SHA'],github_attempt='1',status='STARTED')
    write_json(output/'control.json',control)
    try:
        receipt=run_case(spec,output/'cases'/spec['id'],value,
            data=old.q.design.generate(spec),config=settings(spec),seconds=value['soft_seconds'])
        control.update(status='complete',seconds=receipt['seconds'])
    except BaseException as exc:
        control.update(status='FAILED',failure=type(exc).__name__+': '+str(exc)[:2000]);raise
    finally:
        write_json(output/'control.json',control);pack(output)


def audit_one(args):
    spec,path,value,control=args
    store=None
    try:
        if control.get('spec')!=spec or control.get('manifest')!=digest(value):
            raise ValueError('pilot case control identity differs')
        if any(control.get(k)!=os.environ[e] for k,e in (
            ('github_run','GITHUB_RUN_ID'),('github_sha','GITHUB_SHA'),('github_attempt','GITHUB_RUN_ATTEMPT'))):
            raise ValueError('pilot control source/run/attempt differs')
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or any(not n.startswith('cases/'+spec['id']+'/') for n in z.namelist()):
                raise ValueError('case archive CRC/path differs')
        store=audit.Store(path,spec['id']);receipt=store.json('receipt.json')
        if any(receipt.get(k)!=control.get(k) for k in ('github_run','github_sha','github_attempt')):
            raise ValueError('pilot receipt/control provenance differs')
        data=old.q.design.generate(spec);config=settings(spec)
        row=audit.audit_case(spec,store,value,data=data,config=config)
        if row['status']=='complete' and control['status']!='complete':
            raise ValueError('complete receipt with failed control')
        if row['status']!='complete':
            # An incomplete pilot stays unknown, but completed checkpoint arithmetic
            # is still independently checked. Failed/missing attempts are not retried.
            plans=make_layer_sample_plans(data,config);checks=[];complete_attempts=0
            context=case_context(spec,data,config,value)
            for name in store.names:
                if not name.startswith('checkpoints/') or not name.endswith('.json.gz'):continue
                payload=store.compressed(name);a=decode(LocalAssessment,payload['assessment'])
                if a.sample!=plans[a.sample.run_index]:raise ValueError('partial split identity differs')
                expected=f'checkpoints/d{a.depth}-{a.target}/{a.sample.run_index:03}.json.gz'
                if name!=expected:raise ValueError('partial checkpoint path differs')
                local=digest(dict(case=context,depth=a.depth,target=a.target,candidates=a.candidates,config=config))
                completed=a.status=='completed'
                if completed:
                    checks.append(audit.audit_checkpoint(payload,data.iloc[list(a.sample.sampled_positions)],a,local,config))
                elif payload.get('context')!=local or payload.get('digest')!=digest({k:v for k,v in payload.items() if k!='digest'}):
                    raise ValueError('failed checkpoint context/digest differs')
                for suffix,status in (('.started.json','STARTED'),('.finished.json','completed')):
                    attempt=store.json(name[:-3]+suffix)
                    if ((attempt['context'],attempt['depth'],attempt['target'],attempt['run_index'],attempt['seed'])
                        !=(local,a.depth,a.target,a.sample.run_index,a.sample.seed)
                        or suffix=='.started.json' and attempt['status']!='STARTED'
                        or suffix=='.finished.json' and completed and (attempt['status']!='completed' or attempt['fit_cost_unknown'])
                        or suffix=='.finished.json' and not completed and (attempt['status'] not in ('FAILED',a.status) or not attempt['fit_cost_unknown'])):
                        raise ValueError('partial attempt identity/status differs')
            for name in store.names:
                if name.endswith('.finished.json'):
                    a=store.json(name)
                    complete_attempts+=a['status']=='completed' and not a['fit_cost_unknown']
            if complete_attempts!=len(checks):raise ValueError('partial completed-checkpoint coverage differs')
            row.update(assessments=len(checks),observation_checks=sum(v['observation_checks'] for v in checks),
                outer_fits=sum(v['outer_fits'] for v in checks),tuning_fits=sum(v['tuning_fits'] for v in checks),
                unknown_cost_attempts=row['started_assessments']-complete_attempts)
        row.update(archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            archive_bytes=Path(path).stat().st_size)
        row['grid_policy']=audit.POLICY
        if row['status']=='complete':
            ref=value['references'][spec['id']]
            reference=audit.Store(ROOT/ref['path'],ref['row']['spec']['id'])
            try:row['comparison']=audit.compare_reference(store,reference)
            finally:reference.close()
            row['reference_seconds']=ref['row']['seconds']
            row['reference_tuning_fits']=ref['row']['tuning_fits']
            row['reference_outer_fits']=ref['row']['outer_fits']
            row['reference_archive_bytes']=ref['bytes']
        return row
    except Exception as exc:
        return dict(spec=spec,status='audit_failed',structure=None,fit_cost_unknown=True,
            failure=type(exc).__name__+': '+str(exc)[:2000])
    finally:
        if store:store.close()


def summarize(rows,value):
    found={}
    for row in rows:
        if row['spec'] not in specs():raise ValueError('foreign optimized result')
        key=row['spec']['id']
        if key in found:raise ValueError('duplicate optimized result')
        found[key]=row
    cases=[found.get(s['id'],dict(spec=s,status='missing',structure=None,fit_cost_unknown=True)) for s in specs()]
    complete=[r for r in cases if r['status']=='complete']
    return dict(stage='optimized_timing_and_equivalence',grid_policy=audit.POLICY,
        manifest=digest(value),planned=2,complete=len(complete),unknown=2-len(complete),
        exact=sum(bool(r['structure']['exact']) for r in complete),cases=cases,
        execution_verified=len(complete)==2,
        equivalence_verified=len(complete)==2 and all(r.get('comparison',{}).get('equivalent') is True for r in complete),
        scientific_qualification=False,audit_model_fits=0,fixtures_excluded_from_qualification=True)


def transport(api=github_api):
    run=os.environ['GITHUB_RUN_ID'];sha=os.environ['GITHUB_SHA']
    state=api(f'actions/runs/{run}')
    if (state['head_sha']!=sha or state['run_attempt']!=1 or state['event']!='workflow_dispatch'
        or state['path']!=WORKFLOW):raise ValueError('pilot run source/attempt differs')
    jobs=api(f'actions/runs/{run}/jobs?per_page=100&page=1')['jobs']
    required=[j for j in jobs if j['name'].startswith('case (') or j['name']=='prepare']
    expected={f"case ({s['id']})" for s in specs()}|{'prepare'}
    if len(required)!=3 or {j['name'] for j in required}!=expected:
        raise ValueError('pilot case transport coverage differs')
    for j in required:
        uploads=[s for s in j['steps'] if 'actions/upload-artifact@' in s['name']]
        if j['conclusion']!='success' or len(uploads)!=1 or uploads[0]['conclusion']!='success':
            raise ValueError('pilot case/prepare upload failed')
    return dict(verified=True,run=run,source=sha,case_jobs=2)


def aggregate(inputs,output):
    first_attempt();value=checked_manifest();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];errors=[];proof=dict(verified=False)
    def save():
        report=summarize(rows,value)
        report.update(integrity_errors=list(errors),transport=proof,github_run=os.environ['GITHUB_RUN_ID'],
            github_sha=os.environ['GITHUB_SHA'],github_attempt='1')
        report['execution_verified']=report['execution_verified'] and not errors and proof['verified']
        report['equivalence_verified']=report['equivalence_verified'] and not errors and proof['verified']
        write_json(output/'audit.json',report)
        return report
    save();tasks=[];controls=[]
    for path in Path(inputs).rglob('control.json'):
        try:
            control=load_json(path)
            if control.get('spec') not in specs():raise ValueError('foreign pilot control')
            controls.append((path,control))
        except Exception as exc:errors.append(str(path)+': '+str(exc)[:1000])
    for spec in specs():
        matched=[p for p,c in controls if c.get('spec',{}).get('id')==spec['id']]
        if len(matched)!=1:
            rows.append(dict(spec=spec,status='missing' if not matched else 'audit_failed',structure=None,fit_cost_unknown=True))
            if len(matched)>1:errors.append('duplicate control '+spec['id'])
            continue
        path=matched[0];archive=path.parent/'raw/case.zip'
        if not archive.exists():
            rows.append(dict(spec=spec,status='missing_archive',structure=None,fit_cost_unknown=True));continue
        tasks.append((spec,archive,value,load_json(path)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(audit_one,t):t[0] for t in tasks}
        for future in as_completed(futures):
            spec=futures[future]
            try:row=future.result()
            except Exception as exc:row=dict(spec=spec,status='audit_failed',structure=None,fit_cost_unknown=True,failure=str(exc)[:2000])
            rows.append(row);write_json(output/'case-audits'/(spec['id']+'.json'),row);save()
    try:proof=transport()
    except Exception as exc:errors.append(str(exc)[:2000])
    report=save();write_report(report,output)
    return report


def write_report(report,output):
    lines=['# Optimized RF recursive validation','',
        f"{report['complete']}/2 cases completed and passed independent arithmetic/count checks; {report['exact']}/2 recovered the exact chain.",
        f"Historical prediction/decision equivalence verified: {report['equivalence_verified']}",'',
        '|N|Status|Compute minutes|Original minutes|Tuning fits|Original tuning fits|Archive bytes|Original archive bytes|Exact chain|Equivalent|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in report['cases']:
        lines.append('|'+ '|'.join(map(str,[r['spec']['n'],r['status'],
            round(r['seconds']/60,2) if r.get('seconds') is not None else 'unknown',
            round(r['reference_seconds']/60,2) if r.get('reference_seconds') is not None else 'unknown',
            r.get('tuning_fits','unknown'),r.get('reference_tuning_fits','unknown'),
            r.get('archive_bytes','unknown'),r.get('reference_archive_bytes','unknown'),
            (r.get('structure') or {}).get('exact','unknown'),(r.get('comparison') or {}).get('equivalent','unknown')]))+'|')
    lines+=['','If both comparisons pass, the optimized engine retained the original predictions and recursive decisions on these two fixtures while reducing redundant fitting.',
        'This validates implementation/equivalence and measures runtime. It does not establish false-positive rates, recovery probabilities or general sample-size requirements.',
        'The fixed two-case denominator includes incomplete or failed outcomes. No automatic retry or replacement seeds. No larger study follows automatically.']
    (output/'interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','check','prepare','case','pack','aggregate','download'))
    p.add_argument('--out',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--id');p.add_argument('--authorization',type=Path);a=p.parse_args()
    if a.command=='freeze':write_json(MANIFEST,manifest())
    elif a.command=='check':print('PASS',digest(checked_manifest()),'frozen optimized RF pilot/runtime')
    elif a.command=='prepare':print(json.dumps(prepare(a.out),separators=(',',':')))
    elif a.command=='case':run_one(a.id,a.out,a.authorization)
    elif a.command=='pack':pack(a.out)
    elif a.command=='download':
        first_attempt();checked_manifest()
        download_artifacts(os.environ['GITHUB_RUN_ID'],['optimized-rf-'+s['id'] for s in specs()],a.out,allow_missing=True)
    else:
        r=aggregate(a.inputs,a.out)
        if not r['execution_verified'] or not r['equivalence_verified']:raise SystemExit('Pilot incomplete; retain original evidence and stop without retry.')
