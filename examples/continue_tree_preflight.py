"""Bounded RF-only checkpoint continuation; unchanged scientific identity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time
import zipfile

import multiengine_preflight as m
from recursive_preview_actions import github_api,load_json,runtime_check
from recursive_preview_runner import fit_recorded,file_hashes,write_json,write_gzip
from open.layer_artifacts import build_payloads
from open.metadata import validate_metadata

download_artifacts=m.download_artifacts
ROOT=m.ROOT
WORKFLOW='.github/workflows/tree-preflight-continuation.yml'
MANIFEST=ROOT/'examples/study_manifests/tree-preflight-continuation.json'
PARENT_RUN='38775613523'
PARENT_SHA='ffe85ca57112530871a8ab9e9db7ec5ca8fdf9b2'
REPORT=ROOT/'examples/results/2026-10-15-multiengine-preflight-setup/completed-preflight-audit.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def report():return load_json(REPORT)
def pending():return [s for s in m.specs() if s['engine']=='random_forest']
def producer_row(spec):return next(r for r in report()['cases'] if r['spec']==spec)

def manifest():
    paths=set(load_json(m.MANIFEST)['sources'])|{
        m.MANIFEST.relative_to(ROOT).as_posix(),REPORT.relative_to(ROOT).as_posix(),
        WORKFLOW,'examples/continue_tree_preflight.py','examples/tree_preflight_continuation_protocol.md'}
    return dict(schema='open.tree_preflight_continuation.v1',sources={p:m.text_hash(ROOT/p) for p in sorted(paths)},
        runtime=load_json(m.MANIFEST)['runtime'],cases=pending(),producer_run=PARENT_RUN,producer_sha=PARENT_SHA,
        producer_audit_sha256=sha(REPORT),case_minutes=150,compute_minutes=135,soft_seconds=132*60,
        prepare_minutes=15,audit_minutes=30,registration_minutes=1,reserved_minutes=346,
        previous_reserved_minutes=586,prior_ceiling=600,workers=4,max_parallel=2,
        scientific_qualification=False)

def checked_manifest():
    m.checked_manifest();value=load_json(MANIFEST)
    if value!=manifest():raise ValueError('continuation frozen source/runtime/budget differs')
    runtime_check(value)
    r=report()
    if (r['github_run']!=PARENT_RUN or r['github_sha']!=PARENT_SHA or r['github_attempt']!='1'
        or r['manifest']!=m.digest(load_json(m.MANIFEST)) or r['complete']!=4 or r['exact']!=4):
        raise ValueError('producer audit identity differs')
    for s in pending():
        row=producer_row(s)
        if (row['status']!='incomplete' or row['unknown_cost_attempts']!=0
            or row['assessments']!=row['started_assessments'] or row['assessments']!=row['finished_assessments']):
            raise ValueError('producer has unmatched/failed attempts; cannot refit')
    return value

@contextmanager
def producer_environment():
    values=dict(GITHUB_RUN_ID=PARENT_RUN,GITHUB_SHA=PARENT_SHA,GITHUB_RUN_ATTEMPT='1')
    old={k:os.environ.get(k) for k in values};os.environ.update(values)
    try:yield
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

def validate_original(spec,archive,control):
    expected=producer_row(spec)
    if sha(archive)!=expected['archive_sha256']:raise ValueError('producer archive fingerprint differs')
    with producer_environment():
        row=m.audit_one((spec,archive,load_json(m.MANIFEST),control))
    if row!=expected:raise ValueError('producer independent prefix audit differs')
    return row

def restore(spec,archive,control,directory):
    checked=validate_original(spec,archive,control)
    store=m.audit.Store(archive,spec['id']);directory=Path(directory)
    try:
        directory.mkdir(parents=True,exist_ok=False)
        for name in store.names:
            target=(directory/name).resolve()
            if not target.is_relative_to(directory.resolve()):raise ValueError('unsafe producer path')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(store.read(name))
        return store.json('receipt.json'),checked
    finally:store.close()

def resume(spec,directory,prior,reused,value):
    data=m.old.q.design.generate(spec);config=m.settings(spec)
    if prior['context']!=m.case_context(spec,data,config,load_json(m.MANIFEST)):
        raise ValueError('producer scientific context differs')
    if prior['files']!=file_hashes(directory):raise ValueError('restored producer bytes differ')
    write_json(directory/'producer-receipt.json',prior,exclusive=True)
    receipt=dict(prior,status='STARTED',fit_cost_unknown=True,
        github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],github_attempt='1',
        producer_run=PARENT_RUN,producer_sha=PARENT_SHA,continuation_manifest=m.digest(value),
        reused_assessments=reused['assessments'],prior_seconds=prior['seconds'])
    receipt.pop('failure',None);write_json(directory/'receipt.json',receipt)
    start=time.monotonic()
    try:
        network=fit_recorded(data,config,directory/'checkpoints',prior['context'],start+value['soft_seconds'],workers=4)
        metadata=validate_metadata(tuple(data),[dict(name=n,type='continuous') for n in data],config.analysis.root)
        write_gzip(directory/'network.json.gz',build_payloads(network,metadata))
        if network.status!='COMPLETE':raise RuntimeError(network.status)
        if time.monotonic()-start>value['soft_seconds']:raise TimeoutError('continuation computation deadline')
        receipt.update(status='complete',fit_cost_unknown=False)
    except BaseException as exc:
        receipt.update(status='incomplete',failure=type(exc).__name__+': '+str(exc)[:2000]);raise
    finally:
        unchanged=all(sha(directory/n)==h for n,h in prior['files'].items())
        receipt.update(continuation_seconds=time.monotonic()-start,prior_files_unchanged=unchanged)
        receipt['seconds']=prior['seconds']+receipt['continuation_seconds'];receipt['files']=file_hashes(directory)
        if not unchanged:receipt.update(status='incomplete',fit_cost_unknown=True,failure='producer bytes changed')
        write_json(directory/'receipt.json',receipt)
        if not unchanged:raise ValueError('producer bytes changed')
    return receipt

def producer_transport(api=github_api):
    run=api(f'actions/runs/{PARENT_RUN}')
    if (run['head_sha']!=PARENT_SHA or run['run_attempt']!=1 or run['event']!='workflow_dispatch'
        or run['path']!=m.WORKFLOW or run['status']!='completed' or run['conclusion']!='failure'):
        raise ValueError('producer run identity/status differs')
    jobs=api(f'actions/runs/{PARENT_RUN}/jobs?per_page=100')['jobs']
    expected={f"case ({s['id']})" for s in m.specs()}
    cases=[j for j in jobs if j['name'].startswith('case (')]
    if len(cases)!=6 or {j['name'] for j in cases}!=expected:raise ValueError('producer case coverage differs')
    for j in cases:
        uploads=[s for s in j['steps'] if 'actions/upload-artifact@' in s['name']]
        wanted='failure' if 'random_forest' in j['name'] else 'success'
        if j['conclusion']!=wanted or len(uploads)!=1 or uploads[0]['conclusion']!='success':
            raise ValueError('producer computation/upload status differs')
    return dict(verified=True,run=PARENT_RUN,source=PARENT_SHA)

def prepare(output):
    m.first_attempt();value=checked_manifest();page=1
    while True:
        runs=github_api(f'actions/workflows/tree-preflight-continuation.yml/runs?event=workflow_dispatch&per_page=100&page={page}')['workflow_runs']
        if any(str(r['id'])!=os.environ['GITHUB_RUN_ID'] for r in runs):raise ValueError('prior continuation dispatch; no automatic retry')
        if len(runs)<100:break
        page+=1
    proof=producer_transport();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    auth=dict(manifest=m.digest(value),github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],github_attempt='1')
    auth['digest']=m.digest(auth)
    matrix=dict(include=[dict(id=s['id']) for s in pending()])
    write_json(output/'authorization.json',auth);write_json(output/'matrix.json',matrix);write_json(output/'producer-transport.json',proof)
    return matrix

def run_one(case_id,output,auth):
    value=checked_manifest();selected=[s for s in pending() if s['id']==case_id]
    if len(selected)!=1:raise ValueError('foreign continuation identity')
    m.first_attempt();m.authorization(auth,value)
    spec=selected[0];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    control=dict(spec=spec,manifest=m.digest(value),github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],github_attempt='1',status='STARTED')
    write_json(output/'control.json',control)
    try:
        download_artifacts(PARENT_RUN,['tree-case-'+case_id],output/'producer')
        controls=list((output/'producer').rglob('control.json'))
        if len(controls)!=1:raise ValueError('producer control coverage differs')
        parent=controls[0];directory=output/'cases'/case_id
        prior,reused=restore(spec,parent.parent/'raw/case.zip',load_json(parent),directory)
        receipt=resume(spec,directory,prior,reused,value)
        control.update(status='complete',seconds=receipt['seconds'],continuation_seconds=receipt['continuation_seconds'])
    except BaseException as exc:
        control.update(status='FAILED',failure=type(exc).__name__+': '+str(exc)[:2000]);raise
    finally:write_json(output/'control.json',control);m.pack(output)

def audit_one(args):
    spec,path,control,value=args;store=None
    try:
        if control.get('spec')!=spec or control.get('manifest')!=m.digest(value):raise ValueError('continuation control differs')
        for k,e in (('github_run','GITHUB_RUN_ID'),('github_sha','GITHUB_SHA'),('github_attempt','GITHUB_RUN_ATTEMPT')):
            if control.get(k)!=os.environ[e]:raise ValueError('continuation source/run/attempt differs')
        with zipfile.ZipFile(path) as z:
            if z.testzip() or any(not n.startswith('cases/'+spec['id']+'/') for n in z.namelist()):raise ValueError('continuation archive CRC/path differs')
        store=m.audit.Store(path,spec['id']);receipt=store.json('receipt.json');prior=store.json('producer-receipt.json')
        if hashlib.sha256(store.read('producer-receipt.json')).hexdigest()!=producer_row(spec)['receipt_sha256']:
            raise ValueError('producer receipt fingerprint differs')
        if (receipt.get('continuation_manifest')!=m.digest(value) or receipt.get('producer_run')!=PARENT_RUN
            or receipt.get('producer_sha')!=PARENT_SHA or not receipt.get('prior_files_unchanged')):
            raise ValueError('continuation provenance differs')
        if any(receipt.get(k)!=control.get(k) for k in ('github_run','github_sha','github_attempt')):raise ValueError('receipt provenance differs')
        if any(hashlib.sha256(store.read(n)).hexdigest()!=h for n,h in prior['files'].items()):raise ValueError('producer checkpoint bytes changed')
        row=m.audit.audit_case(spec,store,load_json(m.MANIFEST),data=m.old.q.design.generate(spec),config=m.settings(spec))
        if row['status']=='complete' and control['status']!='complete':raise ValueError('failed control with complete receipt')
        row.update(archive_sha256=sha(path),archive_bytes=Path(path).stat().st_size,
            continuation_seconds=receipt.get('continuation_seconds'),reused_assessments=producer_row(spec)['assessments'],
            producer_archive_sha256=producer_row(spec)['archive_sha256'])
        if row['status']=='complete':
            row['new_assessments']=row['assessments']-row['reused_assessments']
            row['new_outer_fits']=row['outer_fits']-producer_row(spec)['outer_fits']
            row['new_tuning_fits']=row['tuning_fits']-producer_row(spec)['tuning_fits']
        return row
    except Exception as exc:return dict(spec=spec,status='audit_failed',structure=None,fit_cost_unknown=True,failure=type(exc).__name__+': '+str(exc)[:2000])
    finally:
        if store:store.close()

def transport():
    state=github_api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}")
    if (state['head_sha']!=os.environ['GITHUB_SHA'] or state['run_attempt']!=1 or state['event']!='workflow_dispatch'
        or state['path']!=WORKFLOW):raise ValueError('continuation transport identity differs')
    jobs=github_api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}/jobs?per_page=100")['jobs']
    found=[j for j in jobs if j['name'].startswith('case (') or j['name']=='prepare']
    expected={f"case ({s['id']})" for s in pending()}|{'prepare'}
    if len(found)!=3 or {j['name'] for j in found}!=expected:raise ValueError('continuation job coverage differs')
    for j in found:
        uploads=[s for s in j['steps'] if 'actions/upload-artifact@' in s['name']]
        if j['conclusion']!='success' or len(uploads)!=1 or uploads[0]['conclusion']!='success':raise ValueError('continuation case/upload failed')
    return dict(verified=True,run=os.environ['GITHUB_RUN_ID'],source=os.environ['GITHUB_SHA'])

def aggregate(inputs,output):
    m.first_attempt();value=checked_manifest();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[r for r in report()['cases'] if r['spec'] not in pending()];errors=[];proof=dict(verified=False)
    def save():
        result=m.summarize(rows,value)
        result.update(continuation_run=os.environ['GITHUB_RUN_ID'],continuation_sha=os.environ['GITHUB_SHA'],
            producer_run=PARENT_RUN,producer_manifest=m.digest(load_json(m.MANIFEST)),transport=proof,integrity_errors=list(errors))
        result['execution_verified']=result['execution_verified'] and proof['verified'] and not errors
        write_json(output/'audit.json',result);return result
    save();controls=list(Path(inputs).rglob('control.json'));tasks=[]
    for p in controls:
        if load_json(p).get('spec') not in pending():errors.append('foreign continuation control')
    for spec in pending():
        matches=[p for p in controls if load_json(p).get('spec')==spec]
        if len(matches)!=1:
            rows.append(dict(spec=spec,status='missing' if not matches else 'audit_failed',structure=None,fit_cost_unknown=True));continue
        p=matches[0];archive=p.parent/'raw/case.zip'
        if not archive.exists():rows.append(dict(spec=spec,status='missing_archive',structure=None,fit_cost_unknown=True));continue
        tasks.append((spec,archive,load_json(p),value))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(audit_one,t):t[0] for t in tasks}
        for f in as_completed(futures):
            spec=futures[f]
            try:row=f.result()
            except Exception as exc:row=dict(spec=spec,status='audit_failed',structure=None,fit_cost_unknown=True,failure=str(exc)[:2000])
            rows.append(row);write_json(output/'case-audits'/(spec['id']+'.json'),row);save()
    try:proof=transport()
    except Exception as exc:errors.append(str(exc)[:2000])
    result=save();m.write_report(result,output);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','check','prepare','case','pack','aggregate','download'))
    p.add_argument('--out',type=Path);p.add_argument('--inputs',type=Path);p.add_argument('--id');p.add_argument('--authorization',type=Path);a=p.parse_args()
    if a.command=='freeze':write_json(MANIFEST,manifest())
    elif a.command=='check':print('PASS',m.digest(checked_manifest()),'frozen RF checkpoint continuation')
    elif a.command=='prepare':print(json.dumps(prepare(a.out),separators=(',',':')))
    elif a.command=='case':run_one(a.id,a.out,a.authorization)
    elif a.command=='pack':m.pack(a.out)
    elif a.command=='download':
        m.first_attempt();checked_manifest();download_artifacts(os.environ['GITHUB_RUN_ID'],['tree-continued-'+s['id'] for s in pending()],a.out,allow_missing=True)
    else:
        r=aggregate(a.inputs,a.out)
        if not r['execution_verified']:raise SystemExit('Continuation incomplete; retain evidence without retry.')
