"""Independently re-audit the two completed timing networks; zero model fits."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path.cwd()/'examples'))
from recursive_preview_audit import Store,audit_case
from recursive_preview_design import digest
from recursive_preview_actions import runtime_check
import time_recursive_preview as probe

BASE = Path(__file__).resolve().parents[1]
RAW = BASE/'raw'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    run,report = load(RAW/'run.json'),load(RAW/'timing.json')
    manifest,original = load(RAW/'timing-manifest.json'),load(RAW/'reference-manifest.json')
    assert manifest == probe.checked_manifest()
    assert original == probe.original_manifest()
    runtime_check(original)
    assert run['conclusion'] == 'success' and run['run_attempt'] == 1
    assert run['head_sha'] == report['github_sha'] == 'e4798c8febfb7a92b81f574c8c93ce61cf7b9183'
    assert report['github_run'] == str(run['id']) == '38629900589'
    assert report['timing_manifest'] == digest(manifest)
    assert report['status'] == 'COMPLETE_TIMING_ONLY'
    assert report['timing_only'] and not report['scientific_gate_accepted'] and not report['original_preflight_reclassified']
    assert set(report['cases']) == {'160','240'}
    jobs = load(RAW/'jobs.json')['jobs']
    job = next(j for j in jobs if j['name'] == 'timing')
    uploads = [s for s in job['steps'] if 'actions/upload-artifact@' in s['name']]
    assert job['conclusion'] == 'success' and len(uploads) == 1 and uploads[0]['conclusion'] == 'success'
    assert all(j['conclusion'] == 'skipped' for j in jobs if j['name'] != 'timing')
    artifacts = load(RAW/'artifacts.json')['artifacts']
    assert len(artifacts) == 1 and artifacts[0]['name'] == 'recursive-preview-timing'
    assert artifacts[0]['size_in_bytes'] > 0 and not artifacts[0]['expired']
    archive = RAW/'timing-cases.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert {n.split('/')[1] for n in z.namelist()} == {original['cases'][i]['id'] for i in (160,240)}
        # The download contains an exact loose duplicate. Verify before archiving it locally.
        loose = RAW/'recursive-preview-timing/.superpowers/recursive-timing/cases'
        if loose.exists():
            files = sorted(p for p in loose.rglob('*') if p.is_file())
            assert len(files) == len(z.namelist())
            for p in files:
                assert p.read_bytes() == z.read('cases/'+p.relative_to(loose).as_posix())
    prior = load(RAW/'original-case160-receipt.json')
    assert hashlib.sha256((RAW/'original-case160-receipt.json').read_bytes()).hexdigest() == manifest['producer']['receipt_sha256']
    rows,audits = [],{}
    for index in (160,240):
        spec = original['cases'][index]
        store = Store(archive,spec['id'])
        try:
            checked = audit_case(spec,store,original)
            assert checked == report['cases'][str(index)]['audit']
            receipt = store.json('receipt.json')
            assert receipt['github_run'] == str(run['id']) and receipt['github_sha'] == run['head_sha']
            if index == 160:
                for name,sha in prior['files'].items():
                    assert hashlib.sha256(store.read(name)).hexdigest() == sha
                assert receipt['reused_assessments'] == 144 and receipt['new_assessments'] == 156
                assert receipt['seconds'] == prior['seconds']+receipt['continuation_seconds']
            reused = report['reuse_audit'] if index == 160 else {}
            assert report['cases'][str(index)]['fit_counts'] == probe.split_fit_counts(checked,reused)
            assert checked['status'] == 'complete' and checked['warnings'] == 0
            assert checked['structure']['exact'] and not checked['structure']['incorrect']
            audits[str(index)] = checked
            rows.append(dict(n=spec['n'],index=index,case_seconds=receipt['seconds'],
                prior_seconds=report['cases'][str(index)]['prior_seconds'],
                continuation_seconds=report['cases'][str(index)]['continuation_seconds'],
                assessments=checked['assessments'],reused_assessments=report['cases'][str(index)]['reused_assessments'],
                new_assessments=report['cases'][str(index)]['new_assessments'],
                exact_structure=checked['structure']['exact'],warnings=checked['warnings']))
        finally:
            store.close()
    result = dict(run_id=run['id'],source=run['head_sha'],timing_manifest=digest(manifest),
        audit_status='PASS',audit_model_fits=0,cases=audits,
        assessments=sum(a['assessments'] for a in audits.values()),
        observation_checks=sum(a['observation_checks'] for a in audits.values()),
        reused_assessments=144,new_assessments=456,reused_original_files_preserved=len(prior['files']),
        outer_fits=sum(a['outer_fits'] for a in audits.values()),
        tuning_fits=sum(a['tuning_fits'] for a in audits.values()),warnings=0,
        both_expected_chains_recovered=True,original_failed_preflights_reclassified=False,
        scientific_qualification=False,scope='Two chain datasets: timing and observed structural correctness, not estimated error rates.')
    (BASE/'audit/verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (BASE/'audit/timing-summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.iterdir()) if p.is_file()}
    (BASE/'audit/raw-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    plot(rows)
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'},indent=2))

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,graph) = plt.subplots(1,2,figsize=(10,4.1),gridspec_kw={'width_ratios':[1.1,1]})
    prior = [r['prior_seconds']/60 for r in rows]
    new = [r['continuation_seconds']/60 for r in rows]
    ax.barh(['N = 80','N = 200'],prior,color='#a9b7c6',label='Prior run (reused)')
    ax.barh(['N = 80','N = 200'],new,left=prior,color='#3478ad',label='Timing run')
    ax.axvline(15,color='#a14545',linestyle='--',label='Old case allowance')
    for i,r in enumerate(rows):
        ax.text(r['case_seconds']/60+.35,i,f"{r['case_seconds']/60:.1f} min",va='center')
    ax.set_xlim(0,34);ax.set_xlabel('Case computation minutes');ax.invert_yaxis()
    ax.set_title('Measured time with unchanged defaults')
    handles,labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False,fontsize=9)
    graph.set_xlim(0,1);graph.set_ylim(0,1)
    graph.axis('off');graph.set_title('Recovered in both cases')
    for x,label in ((.12,'X1'),(.5,'X2'),(.88,'Y')):
        graph.text(x,.56,label,ha='center',va='center',fontsize=14,bbox=dict(boxstyle='round,pad=.5',fc='#e6eff6',ec='#3478ad'))
    for a,b in ((.21,.41),(.59,.79)):
        graph.annotate('',xy=(b,.56),xytext=(a,.56),arrowprops=dict(arrowstyle='->',color='#3478ad',lw=2))
    graph.text(.5,.23,'Correct indirect and direct links\nNo nuisance predictors retained',ha='center',fontsize=10)
    fig.suptitle('Recursive timing probe: two completed chain datasets',fontsize=13)
    fig.text(.5,.02,'N80 time combines two runs; N200 is uninterrupted. Two cases do not estimate error rates.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,top=.78,bottom=.3,wspace=.25)
    fig.savefig(BASE/'timing-and-chain.png',dpi=160)
    fig.savefig(BASE/'timing-and-chain.svg')
    svg = BASE/'timing-and-chain.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

if __name__ == '__main__':
    main()
