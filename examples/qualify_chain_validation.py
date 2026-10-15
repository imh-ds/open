"""Bounded original chain identities, gated by the preserved mixed-run null audit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor

import continue_qualification as prior
import qualify_recursive_validation as q
from recursive_preview_runner import run_case, write_json
from qualification_artifacts import download_artifacts

ROOT = q.ROOT
WORKFLOW = '.github/workflows/chain-qualification.yml'
MANIFEST = ROOT/'examples/study_manifests/chain-qualification.json'
GATE = ROOT/'examples/study_manifests/chain-qualification-noise-gate.json'
EVIDENCE = 'a546ec4a93fab6cb12dd7d5b476485d0b668ea83'
BASE = 'examples/results/2026-10-14-recursive-qualification/combined-38676310704/'
INPUTS = [BASE+'audit/audit.json', BASE+'archive-inventory.json', BASE+'publication.json']
CONTINUED_RUN = '38676310704'
CONTINUED_SHA = 'bab7b69b367ba577b7305ea633d1fe12b412bd78'
load_json = q.load_json


def specs():
    return [s for s in q.stage_specs('signal') if s['scenario']=='chain']


def git_bytes(path, revision=EVIDENCE):
    return subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT)


def blob(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def build_gate():
    """One local no-fit byte check of all final archives before freezing."""
    prior.checked_manifest()
    report, inventory, publication = [json.loads(git_bytes(p)) for p in INPUTS]
    q.validate_noise_report(report,prior.original())
    archives = {}
    for row in report['cases']:
        name = row['spec']['id']
        entry = dict(inventory[name])
        raw = (ROOT/entry['path']).read_bytes()
        if (hashlib.sha256(raw).hexdigest()!=entry['sha256'] or entry['sha256']!=row['archive_sha256']
            or blob(raw)!=publication['all_archive_git_blobs'].get(entry['path'])):
            raise ValueError('preserved archive bytes differ')
        with zipfile.ZipFile(ROOT/entry['path']) as z:
            if z.testzip() is not None:
                raise ValueError('archive CRC differs')
            receipt_raw=z.read('cases/'+name+'/receipt.json')
        receipt=json.loads(receipt_raw)
        continued=row['spec'] in prior.pending()
        if (hashlib.sha256(receipt_raw).hexdigest()!=entry['receipt_sha256']
            or entry['receipt_sha256']!=row['receipt_sha256']
            or receipt['github_run']!=(CONTINUED_RUN if continued else prior.PRODUCER_RUN)
            or receipt['github_sha']!=(CONTINUED_SHA if continued else prior.PRODUCER_SHA)
            or receipt['github_attempt']!='1'):
            raise ValueError('preserved receipt provenance differs')
        entry['git_blob']=blob(raw)
        archives[name]=entry
    gate=dict(schema='open.chain_mixed_null_gate.v1', evidence_commit=EVIDENCE,
        inputs={p:blob(git_bytes(p)) for p in INPUTS}, archives=archives,
        audit_sha256=hashlib.sha256(git_bytes(INPUTS[0])).hexdigest(),
        original_manifest=q.design.digest(prior.original()),
        continuation_manifest=q.design.digest(load_json(prior.MANIFEST)),
        producer_run=prior.PRODUCER_RUN,continuation_run=CONTINUED_RUN,
        original_workflow_conclusion='failure',continuation_workflow_conclusion='failure')
    validate_gate(gate)
    return gate


def validate_gate(gate):
    # Read small local inputs; normalized hashes are frozen by the new manifest.
    report, inventory, publication=[load_json(ROOT/p) for p in INPUTS]
    q.validate_noise_report(report,prior.original())
    if (gate['evidence_commit']!=EVIDENCE or gate['producer_run']!=prior.PRODUCER_RUN
        or gate['continuation_run']!=CONTINUED_RUN
        or gate['original_manifest']!=q.design.digest(prior.original())
        or gate['continuation_manifest']!=q.design.digest(load_json(prior.MANIFEST))
        or gate['original_workflow_conclusion']!='failure'
        or gate['continuation_workflow_conclusion']!='failure'
        or report.get('integrity_errors') or report.get('model_refits')!=0
        or report.get('audit_model_fits')!=0
        or report.get('original_run_still_failed') is not True
        or report.get('continuation_run_still_failed') is not True
        or report.get('producer_run')!=prior.PRODUCER_RUN
        or report.get('continuation_manifest')!=gate['continuation_manifest']):
        raise ValueError('mixed null gate provenance differs')
    # Canonical report bytes in Git are LF; checkout conversion must not alter meaning.
    for p in INPUTS:
        raw=(ROOT/p).read_bytes().replace(b'\r\n',b'\n')
        if gate['inputs'].get(p)!=blob(raw):
            raise ValueError('mixed null input blob differs')
        if p==INPUTS[0] and hashlib.sha256(raw).hexdigest()!=gate['audit_sha256']:
            raise ValueError('mixed null audit SHA256 differs')
    if set(gate['archives'])!={s['id'] for s in q.stage_specs('noise')}:
        raise ValueError('mixed null archive inventory differs')
    for row in report['cases']:
        entry=gate['archives'][row['spec']['id']]
        original=inventory[row['spec']['id']]
        if (any(entry.get(k)!=v for k,v in original.items())
            or entry['sha256']!=row['archive_sha256']
            or entry['receipt_sha256']!=row['receipt_sha256']
            or publication['all_archive_git_blobs'].get(entry['path'])!=entry['git_blob']):
            raise ValueError('mixed null archive fingerprint differs')
    return report


def verify_gate(*, remote=False, api=None):
    gate=load_json(GATE)
    validate_gate(gate)
    proof=dict(gate=q.design.text_hash(GATE),audit_sha256=gate['audit_sha256'],verified=True)
    if remote:
        api=q.github_api if api is None else api
        for run,sha,selected,original in ((prior.PRODUCER_RUN,prior.PRODUCER_SHA,prior.completed(),True),
                                        (CONTINUED_RUN,CONTINUED_SHA,prior.pending(),False)):
            state=api(f'actions/runs/{run}')
            if state['status']!='completed' or state['conclusion']!='failure':
                raise ValueError('historical failed workflow state differs')
            prior.transport(run,sha,selected,original_run=original,api=api)
        tree=api(f'git/trees/{EVIDENCE}?recursive=1')
        if tree.get('truncated'):
            raise ValueError('remote preservation inventory truncated')
        found={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'}
        required=dict(gate['inputs'])
        required.update({v['path']:v['git_blob'] for v in gate['archives'].values()})
        if any(found.get(p)!=h for p,h in required.items()):
            raise ValueError('remote preserved evidence blob differs')
        proof['remote_verified']=True
    return proof


def manifest():
    paths=set(load_json(prior.MANIFEST)['sources']) | set(INPUTS) | {
        'examples/study_manifests/qualification-continuation.json',
        'examples/qualify_chain_validation.py','examples/chain_validation_protocol.md',
        'examples/qualification_artifacts.py',GATE.relative_to(ROOT).as_posix(),WORKFLOW}
    return dict(schema='open.chain_qualification.v1',cases=specs(),
        default_config=prior.original()['default_config'],runtime=prior.original()['runtime'],
        sources={p:q.design.text_hash(ROOT/p) for p in sorted(paths)},
        original_manifest=q.design.digest(prior.original()),
        continuation_manifest=q.design.digest(load_json(prior.MANIFEST)),
        job_minutes=45,compute_minutes=39,soft_seconds=2310,workers=4,max_parallel=20,
        prior_reserved=4863,new_reservation=5026,reserved_total=9889,remaining_ceiling=111,
        ceiling=10000,per_cell=55,wrong_upper=.10,recovery_lower=.80,
        correlated_cases_pending=110)


def checked_manifest():
    prior.checked_manifest()
    value=load_json(MANIFEST)
    if value!=manifest():
        raise ValueError('chain frozen source/identity/budget differs')
    verify_gate()
    return value


def verify_once(current,*,api=None):
    api=q.github_api if api is None else api
    for workflow,allowed in (('chain-qualification.yml',{str(current)}),
        ('recursive-qualification.yml',{prior.PRODUCER_RUN}),
        ('qualification-continuation.yml',{CONTINUED_RUN})):
        page=1
        while True:
            batch=api(f'actions/workflows/{workflow}/runs?event=workflow_dispatch&per_page=100&page={page}')['workflow_runs']
            if any(str(r['id']) not in allowed for r in batch):
                raise ValueError('unexpected prior dispatch; no automatic retry')
            if len(batch)<100:
                break
            page+=1


def case_matrix():
    return dict(include=[dict(index=s['index'],id=s['id']) for s in specs()])


def prepare(output):
    q.first_attempt()
    value=checked_manifest()
    verify_once(os.environ['GITHUB_RUN_ID'])
    proof=verify_gate(remote=True)
    authorization=dict(manifest=q.design.digest(value),github_run=os.environ['GITHUB_RUN_ID'],
        github_sha=os.environ['GITHUB_SHA'],github_attempt='1',noise_gate=proof)
    authorization['digest']=q.design.digest(authorization)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'authorization.json',authorization)
    write_json(output/'matrix.json',case_matrix())
    write_json(output/'artifact-names.json',[f"chain-case-{s['index']}" for s in specs()])
    return case_matrix()


def verify_authorization(path,value):
    a=load_json(path)
    if (a.get('digest')!=q.design.digest({k:v for k,v in a.items() if k!='digest'})
        or a['manifest']!=q.design.digest(value) or a['github_run']!=os.environ['GITHUB_RUN_ID']
        or a['github_sha']!=os.environ['GITHUB_SHA'] or a['github_attempt']!='1'
        or a['noise_gate']!=dict(verify_gate(),remote_verified=True)):
        raise ValueError('chain authorization source/run/gate differs')
    return a


def run_one(index,output,authorization):
    value=checked_manifest()
    selected=[s for s in specs() if type(index) is int and s['index']==index]
    if len(selected)!=1:
        raise ValueError('case identity outside fresh chain subset')
    q.first_attempt()
    verify_authorization(authorization,value)
    spec=selected[0]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    control=dict(stage='chain',index=index,spec=spec,manifest=q.design.digest(value),status='STARTED',
        github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],github_attempt='1')
    write_json(output/'control.json',control)
    try:
        receipt=run_case(spec,output/'cases'/spec['id'],value,seconds=value['soft_seconds'])
        control.update(status='complete',seconds=receipt['seconds'])
    except BaseException as exc:
        control.update(status='FAILED',failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        write_json(output/'control.json',control)
        q.pack(output)


def summarize(rows,value):
    if any(r['spec'] not in specs() for r in rows):
        raise ValueError('foreign chain case')
    report=q.summarize(rows,'signal',value)
    report['cases']=[r for r in report['cases'] if r['spec']['scenario']=='chain']
    report['cells']={k:v for k,v in report['cells'].items() if k.startswith('chain-')}
    report.update(stage='chain',planned=110,complete=sum(r['status']=='complete' for r in report['cases']),
        qualified=all(v['qualified'] for v in report['cells'].values()),correlated_cases_pending=110)
    return report


def transport(run_id,sha,*,api=None):
    api=q.github_api if api is None else api
    run=api(f'actions/runs/{run_id}')
    if (str(run['id'])!=str(run_id) or run['head_sha']!=sha or run['run_attempt']!=1
        or run['event']!='workflow_dispatch' or run['path']!=WORKFLOW):
        raise ValueError('chain transport source/attempt differs')
    jobs,page=[],1
    while True:
        batch=api(f'actions/runs/{run_id}/jobs?per_page=100&page={page}')['jobs']
        jobs+=batch
        if len(batch)<100:
            break
        page+=1
    expected={f"case ({s['id']})" for s in specs()}
    found=[j for j in jobs if j['name'].startswith('case (')]
    required=found+[j for j in jobs if j['name']=='prepare']
    if len(found)!=110 or {j['name'] for j in found}!=expected or len(required)!=111:
        raise ValueError('chain transport coverage differs')
    for job in required:
        uploads=[s for s in job['steps'] if 'actions/upload-artifact@' in s['name']]
        if job['conclusion']!='success' or len(uploads)!=1 or uploads[0]['conclusion']!='success':
            raise ValueError('chain case/prepare upload failed')
    return dict(verified=True,run_id=str(run_id),source=sha,attempt=1,case_jobs=110)


def aggregate(inputs,output):
    q.first_attempt()
    value=checked_manifest()
    planned={s['index']:s for s in specs()}
    report=summarize([],value)
    report.update(github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],
        github_attempt='1',integrity_errors=[],transport=dict(verified=False))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'audit.json',report)
    rows,controls=[],{}
    try:
        for path in sorted(Path(inputs).rglob('control.json')):
            try:
                c=load_json(path)
                i=c['index']
                if type(i) is not int or i not in planned or c['spec']!=planned[i] or c['stage']!='chain':
                    raise ValueError('foreign chain control')
                if (c['manifest']!=q.design.digest(value) or c['github_run']!=os.environ['GITHUB_RUN_ID']
                    or c['github_sha']!=os.environ['GITHUB_SHA'] or c['github_attempt']!='1'):
                    raise ValueError('chain control source/run differs')
                controls.setdefault(i,[]).append(path)
            except Exception as exc:
                report['integrity_errors'].append(f'{path}: {type(exc).__name__}: {str(exc)[:1000]}')
        tasks=[]
        for index,paths in controls.items():
            if len(paths)!=1:
                report['integrity_errors'].append(f'duplicate case {index}')
                rows.append(dict(spec=planned[index],status='audit_failed',structure=None,fit_cost_unknown=True))
                continue
            path=paths[0]
            archive=path.parent/'raw/case.zip'
            if not archive.exists():
                rows.append(dict(spec=planned[index],status='missing_archive',structure=None,fit_cost_unknown=True))
            else:
                tasks.append((planned[index],archive,value,load_json(path)))
        with ProcessPoolExecutor(max_workers=4) as pool:
            rows+=list(pool.map(q.audit_fresh_case,tasks))
        report.update(summarize(rows,value))
        try:
            report['transport']=transport(os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA'])
        except Exception as exc:
            report['integrity_errors'].append(type(exc).__name__+': '+str(exc)[:1000])
        report['qualified']=report['qualified'] and not report['integrity_errors'] and report['transport']['verified']
    except BaseException as exc:
        report['qualified']=False
        report['integrity_errors'].append(type(exc).__name__+': '+str(exc)[:1000])
        raise
    finally:
        write_json(output/'audit.json',report)
        q.write_report(report,output)
        with (output/'interpretation.md').open('a',encoding='utf-8') as stream:
            stream.write('\nOnly the chain subset was evaluated. The 110 correlated-predictor cases remain pending.\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('gate','freeze','check','prepare','case','aggregate','pack','download'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--index',type=int)
    parser.add_argument('--authorization',type=Path)
    a=parser.parse_args()
    if a.command=='gate':
        write_json(GATE,build_gate())
    elif a.command=='freeze':
        write_json(MANIFEST,manifest())
    elif a.command=='check':
        print('PASS',q.design.digest(checked_manifest()),'frozen chain source/runtime/gate')
    elif a.command=='prepare':
        print(json.dumps(prepare(a.out),separators=(',',':')))
    elif a.command=='case':
        run_one(a.index,a.out,a.authorization)
    elif a.command=='aggregate':
        if not aggregate(a.inputs,a.out)['qualified']:
            raise SystemExit('Chain did not qualify; preserve evidence, no automatic follow-on.')
    elif a.command=='download':
        q.first_attempt()
        checked_manifest()
        download_artifacts(os.environ['GITHUB_RUN_ID'],[f"chain-case-{s['index']}" for s in specs()],a.out,
                           allow_missing=True)
    else:
        q.pack(a.out)
