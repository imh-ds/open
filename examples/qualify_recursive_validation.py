"""Fresh 55-per-cell qualification using unchanged recorded production recursion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import subprocess
import zipfile

from open.layer_artifacts import portable
import recursive_preview_design as design
from recursive_preview_actions import load_json,runtime_check,github_api,find_one
from recursive_preview_runner import run_case,write_json
from recursive_preview_audit import Store,audit_case
from time_scenario_validation import pack

ROOT=design.ROOT
MANIFEST=ROOT/'examples/study_manifests/recursive-qualification.json'
GATE=ROOT/'examples/study_manifests/recursive-qualification-noise-gate.json'
PROOF=ROOT/'examples/results/2026-10-14-recursive-preview-validation/scenario-timing-completed-38645875109/audit/verification.json'


def reference():
    return load_json(ROOT/'examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38621370295/raw/producer-manifest.json')


def specs():
    return [s for s in design.specs() if 1<=s['repetition']<=55]


def stage_specs(stage):
    if stage not in ('noise','signal'):
        raise ValueError('foreign stage')
    return [s for s in specs() if (s['scenario']=='noise')==(stage=='noise')]


def job_cap(spec):
    return {('noise',80):14,('noise',200):19,('chain',80):40,('chain',200):40,
        ('correlated',80):25,('correlated',200):30}[spec['scenario'],spec['n']]


def case_matrix(stage):
    return [dict(index=s['index'],id=s['id'],job_minutes=job_cap(s),compute_minutes=job_cap(s)-5)
            for s in stage_specs(stage)]


def manifest():
    old=reference()
    proof=load_json(PROOF)
    known=sorted(int(k) for k in proof['cases'])
    if proof['status']!='PASS' or known!=[0,80,160,240,320,400]:
        raise ValueError('timing history/inventory differs')
    if set(known)&{s['index'] for s in specs()}:
        raise ValueError('qualification identity was already fitted')
    paths=set(old['sources']) | {'examples/qualify_recursive_validation.py',
        '.github/workflows/recursive-qualification.yml','examples/qualification_validation_protocol.md',
        'examples/time_scenario_validation.py',PROOF.relative_to(ROOT).as_posix(),
        'examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38621370295/raw/producer-manifest.json'}
    return dict(schema='open.recursive_qualification.v1',cases=specs(),historical_fit_indices=known,
        original_manifest=design.digest(old),default_config=old['default_config'],runtime=old['runtime'],
        sources={p:design.text_hash(ROOT/p) for p in sorted(paths)},workers=4,per_cell=55,
        wrong_upper=.10,recovery_lower=.80,allocated_runner_minutes=9897,ceiling=10000,
        historical_allocations=506,registration_minutes=1,stage_control_minutes=150,
        new_case_job_minutes=sum(job_cap(s) for s in specs()),max_parallel=20,
        excluded_pilots_are_descriptive=True)


def checked_manifest():
    value=load_json(MANIFEST)
    if value!=manifest():
        raise ValueError('qualification manifest differs from frozen source/settings/identities')
    if any(design.text_hash(ROOT/p)!=h for p,h in reference()['sources'].items()):
        raise ValueError('original frozen sources changed')
    runtime_check(value)
    return value


def first_attempt():
    if (os.environ.get('GITHUB_RUN_ATTEMPT')!='1' or not os.environ.get('GITHUB_RUN_ID')
        or not os.environ.get('GITHUB_SHA')):
        raise ValueError('requires first Actions attempt; no automatic fit retry')


def verify_once(stage,current_run,noise_run=None,*,api=None):
    api=github_api if api is None else api
    allowed={str(current_run)} | ({str(noise_run)} if stage=='signal' and noise_run else set())
    page=1
    while True:
        runs=api(f'actions/workflows/recursive-qualification.yml/runs?event=workflow_dispatch&per_page=100&page={page}')['workflow_runs']
        if any(str(r['id']) not in allowed for r in runs):
            raise ValueError('previous qualification dispatch exists; no automatic retry/new run')
        if len(runs)<100:
            return
        page+=1


def summarize(rows,stage,value):
    planned=stage_specs(stage)
    by_id={}
    for row in rows:
        if row['spec'] not in planned:
            raise ValueError('foreign case row')
        name=row['spec']['id']
        if name in by_id:
            raise ValueError('duplicate case row')
        by_id[name]=row
    cases=[by_id.get(s['id'],dict(spec=s,status='missing',structure=None,fit_cost_unknown=True)) for s in planned]
    cells={}
    for scenario in (('noise',) if stage=='noise' else ('chain','correlated')):
        for n in (80,200):
            cell=[r for r in cases if (r['spec']['scenario'],r['spec']['n'])==(scenario,n)]
            complete=[r for r in cell if r['status']=='complete']
            unknown=55-len(complete)
            wrong=sum(r['structure']['incorrect'] for r in complete)
            exact=sum(r['structure']['exact'] for r in complete)
            def bound(k):
                if not unknown:
                    return design.interval(k,n=55)
                return dict(count=k,lower=None,upper=None,possible_counts=[k,k+unknown],
                    inclusive_lower=design.interval(k,n=55)['lower'],
                    inclusive_upper=design.interval(k+unknown,n=55)['upper'])
            incorrect,recovery=bound(wrong),bound(exact)
            cells[f'{scenario}-n{n}']=dict(planned=55,complete=len(complete),unknown=unknown,
                incorrect=incorrect,recovery=recovery,
                qualified=not unknown and incorrect['upper']<=.10 and
                    (stage=='noise' or recovery['lower']>=.80))
    return dict(stage=stage,manifest=design.digest(value),planned=len(cases),
        complete=sum(r['status']=='complete' for r in cases),cases=cases,cells=cells,
        qualified=all(c['qualified'] for c in cells.values()),audit_model_fits=0,
        interval_scope='marginal per cell; not general error control',pilots_excluded=True)


def validate_noise_report(report,value):
    expected=summarize(report['cases'],'noise',value)
    if not expected['qualified'] or any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError('noise gate incomplete, edited or not qualified')
    return True


def transport(stage,run_id,sha,*,completed=False,api=github_api):
    run=api(f'actions/runs/{run_id}')
    if (str(run['id'])!=str(run_id) or run['head_sha']!=sha or run['run_attempt']!=1
        or run['event']!='workflow_dispatch' or run['path']!='.github/workflows/recursive-qualification.yml'
        or (completed and (run['status']!='completed' or run['conclusion']!='success'))):
        raise ValueError('qualification run source/attempt/completion differs')
    jobs=[]
    page=1
    while True:
        batch=api(f'actions/runs/{run_id}/jobs?per_page=100&page={page}')['jobs']
        jobs+=batch
        if len(batch)<100:
            break
        page+=1
    expected={f"case ({s['id']})" for s in stage_specs(stage)}
    found=[j for j in jobs if j['name'].startswith('case (')]
    if len(found)!=len(expected) or {j['name'] for j in found}!=expected:
        raise ValueError('case job transport coverage differs')
    required=found+[j for j in jobs if j['name']=='prepare' or completed and j['name']=='aggregate']
    if len([j for j in required if j['name']=='prepare'])!=1 or completed and len([j for j in required if j['name']=='aggregate'])!=1:
        raise ValueError('stage preparation/aggregation transport missing')
    for job in required:
        uploads=[s for s in job['steps'] if 'actions/upload-artifact@' in s['name']]
        if job['conclusion']!='success' or len(uploads)!=1 or uploads[0]['conclusion']!='success':
            raise ValueError('stage case/upload transport failed')
    return dict(run_id=str(run_id),source=sha,attempt=1,case_jobs=len(found),verified=True,
        workflow_completed=completed)


def committed(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('gate path outside repository')
    rel=path.relative_to(ROOT).as_posix()
    if subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)!=path.read_bytes():
        raise ValueError('gate evidence must match committed bytes')


def verify_preserved_archives(archives,*,git=subprocess.check_output):
    paths=[v['path'] for v in archives.values()]
    inventory=git(['git','ls-tree','-r','HEAD','--',*paths],cwd=ROOT)
    found={line.split('\t',1)[1]:line.split()[2] for line in inventory.decode().splitlines()}
    for entry in archives.values():
        blob=entry['git_blob']
        if len(blob)!=40 or any(c not in '0123456789abcdef' for c in blob) or found.get(entry['path'])!=blob:
            raise ValueError('noise raw archives not preserved at committed Git blobs')
        raw=git(['git','cat-file','blob',blob],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('preserved archive SHA256 differs from audited evidence')


def verify_noise_gate(noise_run,*,remote=True):
    value=checked_manifest()
    if not noise_run:
        raise ValueError('signal requires preserved committed noise gate')
    committed(GATE)
    gate=load_json(GATE)
    if gate['manifest']!=design.digest(value) or str(gate['noise_run'])!=str(noise_run):
        raise ValueError('noise gate manifest/run differs')
    path=(ROOT/gate['audit_path']).resolve()
    committed(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=gate['audit_sha256']:
        raise ValueError('noise audit digest differs')
    report=load_json(path)
    validate_noise_report(report,value)
    if (report['github_run']!=str(noise_run) or report['github_sha']!=gate['noise_source']
        or report.get('transport',{}).get('verified') is not True or report.get('integrity_errors')):
        raise ValueError('noise report provenance or transport differs')
    archives=gate.get('raw_archives',{})
    if set(archives)!={s['id'] for s in stage_specs('noise')}:
        raise ValueError('committed noise archive inventory incomplete')
    for row in report['cases']:
        entry=archives[row['spec']['id']]
        if entry['sha256']!=row['archive_sha256']:
            raise ValueError('noise archive fingerprint differs from audited evidence')
        if not (ROOT/entry['path']).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('noise archive path outside repository')
    if remote:
        verify_preserved_archives(archives)
    proof=transport('noise',noise_run,gate['noise_source'],completed=True) if remote else None
    return dict(gate_sha256=hashlib.sha256(GATE.read_bytes()).hexdigest(),audit_sha256=gate['audit_sha256'],
        noise_run=str(noise_run),transport=proof)


def prepare(stage,output,noise_run=None):
    first_attempt()
    value=checked_manifest()
    verify_once(stage,os.environ['GITHUB_RUN_ID'],noise_run)
    authorization=dict(stage=stage,manifest=design.digest(value),github_run=os.environ['GITHUB_RUN_ID'],
        github_sha=os.environ['GITHUB_SHA'],github_attempt='1',noise_gate=None)
    if stage=='signal':
        authorization['noise_gate']=verify_noise_gate(noise_run)
    authorization['digest']=design.digest(authorization)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'authorization.json',authorization)
    write_json(output/'matrix.json',dict(include=case_matrix(stage)))
    return dict(include=case_matrix(stage))


def verify_authorization(stage,path,value):
    authorization=load_json(path)
    signed={k:v for k,v in authorization.items() if k!='digest'}
    if (authorization.get('digest')!=design.digest(signed) or authorization['stage']!=stage
        or authorization['manifest']!=design.digest(value) or authorization['github_run']!=os.environ['GITHUB_RUN_ID']
        or authorization['github_sha']!=os.environ['GITHUB_SHA'] or authorization['github_attempt']!='1'):
        raise ValueError('stage authorization source/run differs')
    if stage=='signal':
        proof=verify_noise_gate(authorization['noise_gate']['noise_run'],remote=False)
        saved=authorization['noise_gate']
        if (proof['gate_sha256']!=saved['gate_sha256'] or proof['audit_sha256']!=saved['audit_sha256']
            or saved.get('transport',{}).get('verified') is not True):
            raise ValueError('signal authorization gate differs')
    return authorization


def run_one(stage,index,output,authorization=None):
    value=checked_manifest()
    selected=[s for s in stage_specs(stage) if type(index) is int and s['index']==index]
    if len(selected)!=1:
        raise ValueError('case identity outside fresh stage')
    first_attempt()
    verify_authorization(stage,authorization,value)
    spec=selected[0]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    control=dict(stage=stage,index=index,spec=spec,manifest=design.digest(value),status='STARTED',
        github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],github_attempt='1')
    write_json(output/'control.json',control)
    try:
        receipt=run_case(spec,output/'cases'/spec['id'],value,seconds=(job_cap(spec)-5.5)*60)
        control.update(status='complete',seconds=receipt['seconds'])
    except BaseException as exc:
        control.update(status='FAILED',failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        write_json(output/'control.json',control)
        pack(output)


def audit_fresh_case(args):
    spec,archive,value,control=args
    store=None
    try:
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or {n.split('/')[1] for n in z.namelist()}!={spec['id']}:
                raise ValueError('case archive CRC/identity differs')
        store=Store(archive,spec['id'])
        receipt=store.json('receipt.json')
        if any(receipt.get(k)!=control[k] for k in ('github_run','github_sha','github_attempt')):
            raise ValueError('receipt provenance differs from case control')
        row=audit_case(spec,store,value)
        if row['status']=='complete' and control['status']!='complete':
            raise ValueError('completed receipt has failed case control')
        row['archive_sha256']=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
        return row
    except Exception as exc:
        return dict(spec=spec,status='audit_failed',structure=None,fit_cost_unknown=True,
            started_assessments=sum(n.endswith('.started.json') for n in store.names) if store else None,
            finished_assessments=sum(n.endswith('.finished.json') for n in store.names) if store else None,
            failure=type(exc).__name__+': '+str(exc)[:2000])
    finally:
        if store:
            store.close()


def aggregate_stage(stage,inputs,output):
    first_attempt()
    value=checked_manifest()
    planned={s['index']:s for s in stage_specs(stage)}
    report=summarize([],stage,value)
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
                if type(i) is not int or i not in planned or c['spec']!=planned[i] or c['stage']!=stage:
                    raise ValueError('foreign case control')
                if (c['manifest']!=design.digest(value) or c['github_run']!=os.environ['GITHUB_RUN_ID']
                    or c['github_sha']!=os.environ['GITHUB_SHA'] or c['github_attempt']!='1'):
                    raise ValueError('case control source/run differs')
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
            rows+=list(pool.map(audit_fresh_case,tasks))
        report.update(summarize(rows,stage,value))
        try:
            report['transport']=transport(stage,os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA'])
        except Exception as exc:
            report['integrity_errors'].append(type(exc).__name__+': '+str(exc)[:1000])
        report['qualified']=report['qualified'] and not report['integrity_errors'] and report['transport']['verified']
    except BaseException as exc:
        report['qualified']=False
        report['integrity_errors'].append(type(exc).__name__+': '+str(exc)[:1000])
        raise
    finally:
        write_json(output/'audit.json',report)
        write_report(report,output)
    return report


def write_report(report,output):
    lines=['# Fresh recursive qualification','',
        f"{report['complete']}/{report['planned']} planned independent datasets completed and passed integrity checks.",
        f"Stage {report['stage']}: {'met' if report['qualified'] else 'did not meet'} the prespecified qualification criteria.",'',
        '|Condition|Complete|Unknown|Any incorrect finding|Exact networks|Qualified|','|---|---:|---:|---:|---:|---|']
    for label,c in report['cells'].items():
        lines.append(f"|{label}|{c['complete']}/55|{c['unknown']}|{c['incorrect']['count']}|{c['recovery']['count']}|{c['qualified']}|")
    lines+=['','Unknown cases prevent positive qualification; failure is not an empty network.',
        'Timing pilots are excluded. Results apply only to these continuous Gaussian default-elastic-net reference conditions.',
        'Incorrect findings mean extra/wrong retained relations; missing links and wrong depths remain in saved case rows.',
        'No probability of truth for empirical arrows, causal validation or general sample-size rule is implied. No model fits in audit.']
    (output/'interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=list(report['cells'])
    for ax,metric,title,cutoff in zip(axes,('incorrect','recovery'),
            ('Any incorrect finding','Exact network recovery'),(.10,.80)):
        for i,label in enumerate(labels):
            c=report['cells'][label]
            b=c[metric]
            if c['unknown']:
                ax.text(i,.5,f"{c['complete']}/55 complete\nunknown",ha='center',fontsize=8)
            else:
                rate=b['count']/55
                ax.errorbar(i,rate,yerr=[[rate-b['lower']],[b['upper']-rate]],fmt='o',capsize=4)
        ax.axhline(cutoff,color='grey',linestyle='--',linewidth=1)
        ax.set(title=title,ylim=(-.03,1.03),xticks=range(len(labels)),xticklabels=labels,
            ylabel='Proportion of independent datasets')
        ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Reference conditions; intervals are marginal per cell')
    fig.tight_layout()
    fig.savefig(output/'qualification-rates.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','check','prepare','case','aggregate','pack'))
    p.add_argument('--stage',choices=('noise','signal'))
    p.add_argument('--out',type=Path)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--index',type=int)
    p.add_argument('--authorization',type=Path)
    p.add_argument('--noise-run')
    a=p.parse_args()
    if a.command=='freeze':
        write_json(MANIFEST,manifest())
    elif a.command=='check':
        print('PASS',design.digest(checked_manifest()),'frozen qualification source/runtime')
    elif a.command=='prepare':
        print(design.json.dumps(prepare(a.stage,a.out,a.noise_run),separators=(',',':')))
    elif a.command=='case':
        run_one(a.stage,a.index,a.out,a.authorization)
    elif a.command=='aggregate':
        result=aggregate_stage(a.stage,a.inputs,a.out)
        if not result['qualified']:
            raise SystemExit('Stage did not qualify; evidence retained, no automatic continuation.')
    else:
        pack(a.out)
