"""Resume independent checks of existing chain outputs; never fit a model."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import subprocess

import qualify_chain_validation as chain
import recursive_preview_runner as runner

RUN='38690088093'
SOURCE='34729fd23f820711f31ea081c8e4b46769d3ed4f'
DIAGNOSIS=chain.ROOT/'examples/results/2026-10-15-chain-aggregate-timeout-38690088093/raw'


def forbidden(*args,**kwargs):
    raise RuntimeError('model fitting is prohibited during evidence verification')


# This module is imported afresh by every spawned audit worker.
runner.NATIVE_ASSESS=forbidden


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def check_one(args):
    spec,archive,control_path,value,cache=args
    control=chain.load_json(control_path)
    if (control.get('spec')!=spec or control.get('index')!=spec['index'] or control.get('stage')!='chain'
        or control.get('manifest')!=chain.q.design.digest(value) or control.get('github_run')!=RUN
        or control.get('github_sha')!=SOURCE or control.get('github_attempt')!='1'
        or control.get('status')!='complete'):
        raise ValueError('case control provenance differs')
    context=dict(spec=spec,archive_sha256=sha(archive),control_sha256=sha(control_path),
        manifest=chain.q.design.digest(value),replay_source_sha256=sha(__file__),run=RUN,source=SOURCE)
    path=Path(cache)/(spec['id']+'.json')
    if path.exists():
        saved=chain.load_json(path)
        if (saved.get('context')!=context or saved.get('row_sha256')!=chain.q.design.digest(saved['row'])
            or saved['row'].get('spec')!=spec):
            raise ValueError('cached case verification fingerprint differs')
        return saved['row']
    row=chain.q.audit_fresh_case((spec,archive,value,control))
    if row.get('status')=='complete' and row.get('archive_sha256')!=context['archive_sha256']:
        raise ValueError('audited archive changed during verification')
    chain.write_json(path,dict(context=context,row=row,row_sha256=chain.q.design.digest(row)))
    return row


def saved_api(path):
    prefix='actions/runs/'+RUN
    if path==prefix:
        return chain.load_json(DIAGNOSIS/'run.json')
    if path.startswith(prefix+'/jobs?'):
        page=int(path.rsplit('page=',1)[1])
        return chain.load_json(DIAGNOSIS/f'jobs-page-{page}.json') if page<=2 else dict(jobs=[])
    raise ValueError('unexpected offline evidence request')


def report(rows,value,transport):
    r=chain.summarize(rows,value)
    r.update(github_run=RUN,github_sha=SOURCE,github_attempt='1',
        audit_execution='resumable local independent replay of original recorded observation losses',
        audit_model_fits=0,model_refits=0,original_workflow_conclusion='failure',
        transport=transport,integrity_errors=[],correlated_cases_pending=110)
    r['qualified']=r['qualified'] and transport.get('verified') is True
    return r


def replay(inputs,output,workers=8):
    value=chain.checked_manifest()
    original=saved_api('actions/runs/'+RUN)
    if (original['status']!='completed' or original['conclusion']!='failure'
        or original['head_sha']!=SOURCE or original['run_attempt']!=1):
        raise ValueError('original failed run provenance differs')
    proof=chain.transport(RUN,SOURCE,api=saved_api)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    tasks=[]
    found={}
    for path in Path(inputs).rglob('control.json'):
        control=chain.load_json(path)
        index=control['index']
        if index in found:
            raise ValueError('duplicate case control')
        found[index]=path
    if set(found)!={s['index'] for s in chain.specs()}:
        raise ValueError('requires all110 original case controls')
    for spec in chain.specs():
        path=found[spec['index']]
        tasks.append((spec,path.parent/'raw/case.zip',path,value,output/'case-audits'))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(check_one,t):t[0] for t in tasks}
        for future in as_completed(futures):
            spec=futures[future]
            try:
                row=future.result()
            except Exception as exc:
                row=dict(spec=spec,status='audit_failed',structure=None,fit_cost_unknown=True,
                    failure=type(exc).__name__+': '+str(exc)[:2000])
            rows.append(row)
            current=report(rows,value,proof)
            chain.write_json(output/'audit.json',current)
            print(f"Verified {len(rows)}/110; complete={current['complete']}; {spec['id']}: {row['status']}",flush=True)
    final=report(rows,value,proof)
    final['integrity_errors']=[dict(case=r['spec']['id'],failure=r.get('failure',r['status']))
        for r in rows if r['status']!='complete']
    final['qualified']=final['qualified'] and not final['integrity_errors']
    chain.write_json(output/'audit.json',final)
    chain.q.write_report(final,output)
    with (output/'interpretation.md').open('a',encoding='utf-8') as stream:
        stream.write('\nCompleted independent verification of the existing outputs without refitting. '
            'The original Actions run remains failed; correlated-predictor validation remains pending.\n')
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.workers<=16:
        p.error('workers must be1..16')
    r=replay(a.inputs,a.out,a.workers)
    print(json.dumps({k:r[k] for k in ('planned','complete','qualified','integrity_errors','cells')},indent=2))
    if r['complete']!=110 or r['integrity_errors']:
        raise SystemExit('Verification incomplete; all unknowns retained.')
