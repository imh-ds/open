"""Replay completed scenario timing evidence and planning arithmetic; never fit models."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path.cwd()/'examples'))
import time_scenario_validation as timing
import recursive_preview_runner as runner
from recursive_preview_audit import Store,audit_case
from recursive_preview_design import digest

BASE=Path(__file__).resolve().parents[1]
RAW=BASE/'raw'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def seconds(start,end):
    return (datetime.fromisoformat(end.replace('Z','+00:00'))-
            datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds()


def no_fits(*args,**kwargs):
    raise AssertionError('No model fitting is permitted during evidence replay')


def main():
    runner.NATIVE_ASSESS=no_fits
    timing.run_case=no_fits
    manifest=load(RAW/'timing-manifest.json')
    assert manifest==timing.checked_manifest()
    old=timing.reference()
    run,jobs,aggregate=load(RAW/'run.json'),load(RAW/'jobs.json')['jobs'],load(RAW/'aggregate.json')
    assert run['id']==38645875109 and run['run_attempt']==1 and run['event']=='workflow_dispatch'
    assert run['head_sha']=='f1c1e3851bc19504a3ff6131c2f6bc3524ce5450' and run['conclusion']=='success'
    assert aggregate['timing_manifest']==digest(manifest) and aggregate['status']=='COMPLETE_TIMING_ONLY'
    assert not aggregate['scientific_gate_accepted'] and not aggregate['original_preflight_reclassified']
    assert aggregate['audit_model_fits']==0 and aggregate['timing_only']
    assert [row['index'] for row in aggregate['cases']]==[0,80,320,400]
    active=[j for j in jobs if j['name']!='register']
    assert len(active)==5 and all(j['conclusion']=='success' for j in active)
    assert next(j for j in jobs if j['name']=='register')['conclusion']=='skipped'
    artifacts=load(RAW/'artifacts.json')['artifacts']
    assert {a['name'] for a in artifacts}=={'scenario-timing-aggregate',*(f'scenario-timing-{i}' for i in (0,80,320,400))}
    assert all(a['size_in_bytes']>0 and a['workflow_run']['id']==run['id']
               and a['workflow_run']['head_sha']==run['head_sha'] for a in artifacts)
    rows,audits=[],{}
    for item in aggregate['cases']:
        index=item['index']
        spec=old['cases'][index]
        control=load(RAW/f'control-{index}.json')
        assert control==item['control'] and control['index']==index
        assert control['github_run']==str(run['id']) and control['github_sha']==run['head_sha']
        assert control['github_attempt']=='1' and control['timing_manifest']==digest(manifest)
        assert control['status']=='COMPLETE_TIMING_ONLY' and not control['scientific_gate_accepted']
        archive=RAW/f'case-{index}.zip'
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert {n.split('/')[1] for n in z.namelist()}=={spec['id']}
        store=Store(archive,spec['id'])
        try:
            audit=audit_case(spec,store,old)
            receipt=store.json('receipt.json')
            for key in ('github_run','github_sha','github_attempt'):
                assert receipt[key]==control[key]
            assert receipt['seconds']==control['case_seconds']
        finally:
            store.close()
        assert audit==control['audit']==item['audit'] and audit['status']=='complete'
        assert audit['warnings']==0 and audit['structure']['exact'] and not audit['structure']['incorrect']
        assert not audit['fit_cost_unknown'] and item['fit_cost_unknown'] is False
        job=next(j for j in jobs if j['name']==f'timing ({index})')
        assert all(s['conclusion']=='success' for s in job['steps'])
        audits[str(index)]=audit
        rows.append(dict(index=index,scenario=spec['scenario'],n=spec['n'],case_seconds=audit['seconds'],
            timing_scope='uninterrupted',compute_audit_seconds=control['compute_audit_seconds'],
            job_seconds=seconds(job['started_at'],job['completed_at']),assessments=audit['assessments'],
            warnings=audit['warnings'],exact_structure=True,archive_bytes=archive.stat().st_size))
    chains=timing.audit_saved_chains(manifest)
    assert chains==aggregate['reused_chains']
    prior=load(BASE.parent/'timing-completed-38629900589/raw/timing.json')
    for index in (160,240):
        audit=chains[str(index)]
        assert audit==prior['cases'][str(index)]['audit'] and audit['structure']['exact']
        assert audit['warnings']==0
        audits[str(index)]=audit
        rows.append(dict(index=index,scenario='chain',n=old['cases'][index]['n'],case_seconds=audit['seconds'],
            timing_scope='cumulative across two runs' if index==160 else 'uninterrupted',
            compute_audit_seconds=None,job_seconds=None,assessments=audit['assessments'],warnings=0,
            exact_structure=True,archive_bytes=None))
    rows.sort(key=lambda r:('noise','chain','correlated').index(r['scenario'])*2+(r['n']==200))
    compute_remaining=79*sum(r['case_seconds']/60 for r in rows)
    illustrative_caps=[15,20,40,40,30,35]
    caps_remaining=79*sum(illustrative_caps)
    # Keep original reserved caps rather than refunding earlier unused time.
    overhead_reserved=250+256+150
    result=dict(status='PASS',run_id=run['id'],source=run['head_sha'],timing_manifest=digest(manifest),
        audit_model_fits=0,new_case_assessments=sum(audits[str(i)]['assessments'] for i in (0,80,320,400)),
        reused_chain_assessments=sum(audits[str(i)]['assessments'] for i in (160,240)),
        total_observation_checks=sum(a['observation_checks'] for a in audits.values()),
        total_outer_fit_records=sum(a['outer_fits'] for a in audits.values()),
        total_tuning_fit_records=sum(a['tuning_fits'] for a in audits.values()),
        warnings=sum(a['warnings'] for a in audits.values()),cases=audits,
        current_run_job_minutes=sum(seconds(j['started_at'],j['completed_at']) for j in active)/60,
        current_run_rounded_job_minutes=sum(math.ceil(seconds(j['started_at'],j['completed_at'])/60) for j in active),
        current_run_wall_minutes=seconds(run['run_started_at'],run['updated_at'])/60,
        planning=dict(original_planned_datasets=480,timed_independent_datasets=6,
            remaining_datasets=474,remaining_job_reservation=9344,
            remaining_compute_projection_minutes=compute_remaining,
            remaining_setup_audit_upload_headroom_minutes=9344-compute_remaining,
            mean_overhead_allowance_per_remaining_case=(9344-compute_remaining)/474,
            illustrative_single_case_caps=illustrative_caps,
            illustrative_full_study_reserved_minutes=overhead_reserved+1+caps_remaining,
            example_50_per_cell_reserved_minutes=overhead_reserved+1+49*sum(illustrative_caps),
            caveat='One timing dataset per cell; projections/caps are planning illustrations, not approved allocations or worst-case bounds.'),
        scientific_gate_accepted=False)
    write_json=runner.write_json
    write_json(BASE/'audit/verification.json',result)
    write_json(BASE/'audit/raw-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir()) if p.is_file()})
    with (BASE/'audit/timing-summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    plot(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.5))
    values=[r['case_seconds']/60 for r in rows]
    labels=[f"{r['scenario'].title()}\nN={r['n']}" for r in rows]
    bars=ax.bar(labels,values,color=['#5B8DB8']*2+['#466A94']*2+['#4E9877']*2)
    ax.bar_label(bars,labels=[f'{v:.1f} min' for v in values],padding=4)
    ax.set(ylabel='Minutes for one full-default dataset',ylim=(0,33),
        title='Measured runtime varies by network structure')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.12,.03,'One dataset per condition; timings are not worst-case bounds.\nChain N=80 combines its original run and continuation. Setup/audit/upload excluded.',fontsize=9)
    fig.subplots_adjust(bottom=.21,top=.88)
    fig.savefig(BASE/'timing-by-condition.png',dpi=160)
    fig.savefig(BASE/'timing-by-condition.svg')
    p=BASE/'timing-by-condition.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    main()
