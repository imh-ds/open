"""Qualification contracts, with real fitting prohibited in every test."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    import qualify_recursive_validation as q
    import recursive_preview_runner as runner
    def forbidden(*args,**kwargs):
        pytest.fail('test reached a real model-fitting boundary')
    monkeypatch.setattr(q,'run_case',forbidden)
    monkeypatch.setattr(runner,'NATIVE_ASSESS',forbidden)
    monkeypatch.setenv('GITHUB_RUN_ID','12345')
    monkeypatch.setenv('GITHUB_SHA','test-source')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')
    monkeypatch.setattr(q,'github_api',lambda p:dict(workflow_runs=[]))


def complete_rows(stage,wrong=0,exact=55):
    import qualify_recursive_validation as q
    rows=[]
    for s in q.stage_specs(stage):
        r=s['repetition']
        rows.append(dict(spec=s,status='complete',structure=dict(incorrect=r<=wrong,
            exact=wrong<r<=exact+wrong,missing=[],extra=[],depth_matches={})))
    return rows


def test_fixed_fresh_subset_never_includes_any_pilot():
    import qualify_recursive_validation as q
    cases=q.specs()
    assert len(cases)==330 and len({s['index'] for s in cases})==330
    assert {s['repetition'] for s in cases}==set(range(1,56))
    assert not {0,80,160,240,320,400}&{s['index'] for s in cases}
    assert len(q.stage_specs('noise'))==110 and len(q.stage_specs('signal'))==220
    assert q.specs()[0]['generator_seed']==1600001000


def test_55_case_confidence_guards_and_original_denominators():
    import qualify_recursive_validation as q
    report=q.summarize(complete_rows('signal',wrong=1,exact=50),'signal',q.manifest())
    assert report['qualified']
    assert all(c['planned']==55 and c['qualified'] for c in report['cells'].values())
    assert not q.summarize(complete_rows('signal',wrong=2,exact=50),'signal',q.manifest())['qualified']
    assert not q.summarize(complete_rows('signal',wrong=1,exact=49),'signal',q.manifest())['qualified']
    assert len(q.design.specs())==480


def test_missing_or_failed_cases_remain_unknown_not_empty_networks():
    import qualify_recursive_validation as q
    rows=complete_rows('noise')
    rows[0]=dict(spec=rows[0]['spec'],status='audit_failed',structure=None)
    report=q.summarize(rows[:-1],'noise',q.manifest())
    assert not report['qualified'] and report['planned']==110
    assert len(report['cases'])==110 and sum(c['unknown'] for c in report['cells'].values())==2
    assert all(c['incorrect']['upper'] is None for c in report['cells'].values())


def test_duplicates_and_foreign_rows_fail_closed():
    import qualify_recursive_validation as q
    rows=complete_rows('noise')
    with pytest.raises(ValueError,match='duplicate'):
        q.summarize(rows+[rows[0]],'noise',q.manifest())
    rows[0]['spec']=q.design.specs()[0]
    with pytest.raises(ValueError,match='foreign'):
        q.summarize(rows,'noise',q.manifest())


def test_budget_and_timed_settings_are_unchanged():
    import qualify_recursive_validation as q
    m=q.manifest()
    assert m['allocated_runner_minutes']==9897
    assert sum(q.job_cap(s) for s in q.specs())==9240
    assert m['default_config']==q.reference()['default_config']
    assert m['historical_fit_indices']==[0,80,160,240,320,400]
    assert len(q.case_matrix('noise'))==110 and len(q.case_matrix('signal'))==220


def test_signal_cannot_prepare_without_committed_exact_noise_gate(tmp_path,monkeypatch):
    import qualify_recursive_validation as q
    monkeypatch.setattr(q,'checked_manifest',lambda:q.manifest())
    with pytest.raises((ValueError,FileNotFoundError)):
        q.prepare('signal',tmp_path/'out',None)
    assert not (tmp_path/'out/matrix.json').exists()


def test_rerun_and_pilot_identity_rejected_before_output(tmp_path,monkeypatch):
    import qualify_recursive_validation as q
    monkeypatch.setattr(q,'checked_manifest',lambda:q.manifest())
    with pytest.raises(ValueError,match='identity'):
        q.run_one('noise',0,tmp_path/'out')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    with pytest.raises(ValueError,match='attempt'):
        q.run_one('noise',1,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_interrupted_case_keeps_control_and_atomic_archive(tmp_path,monkeypatch):
    import qualify_recursive_validation as q
    monkeypatch.setattr(q,'checked_manifest',lambda:q.manifest())
    def interrupted(spec,directory,manifest,*,seconds):
        assert spec['index']==1 and seconds==510
        q.write_json(directory/'receipt.json',dict(status='STARTED',fit_cost_unknown=True))
        raise TimeoutError('no new batches')
    monkeypatch.setattr(q,'run_case',interrupted)
    q.prepare('noise',tmp_path/'prepare')
    with pytest.raises(TimeoutError):
        q.run_one('noise',1,tmp_path/'out',tmp_path/'prepare/authorization.json')
    assert q.load_json(tmp_path/'out/control.json')['status']=='FAILED'
    assert (tmp_path/'out/raw/case.zip').is_file()


def test_gate_rejects_edited_or_incomplete_noise_results():
    import qualify_recursive_validation as q
    m=q.manifest()
    report=q.summarize(complete_rows('noise'),'noise',m)
    q.validate_noise_report(report,m)
    report['cases'][0]['status']='missing'
    with pytest.raises(ValueError,match='gate'):
        q.validate_noise_report(report,m)


def test_manifest_tampering_or_old_source_changes_stop_before_execution(monkeypatch):
    import qualify_recursive_validation as q
    value=q.manifest()
    original=q.load_json
    value['cases'][0]['analysis_seed']+=1
    monkeypatch.setattr(q,'load_json',lambda p:copy.deepcopy(value) if p==q.MANIFEST else original(p))
    with pytest.raises(ValueError,match='manifest'):
        q.checked_manifest()


def test_transport_checks_all_pages_and_successful_uploads():
    import qualify_recursive_validation as q
    jobs=[]
    for spec in q.stage_specs('noise'):
        jobs.append(dict(name=f"case ({spec['id']})",conclusion='success',steps=[
            dict(name='actions/upload-artifact@pinned',conclusion='success')]))
    jobs.append(dict(name='prepare',conclusion='success',steps=[
        dict(name='actions/upload-artifact@pinned',conclusion='success')]))
    def api(path):
        if '/jobs?' in path:
            return dict(jobs=jobs[:100] if path.endswith('page=1') else jobs[100:])
        return dict(id=12345,head_sha='source',run_attempt=1,event='workflow_dispatch',
            path='.github/workflows/recursive-qualification.yml')
    assert q.transport('noise',12345,'source',api=api)['case_jobs']==110
    jobs[100]['steps'][0]['conclusion']='failure'
    with pytest.raises(ValueError,match='transport'):
        q.transport('noise',12345,'source',api=api)


def test_new_dispatch_cannot_automatically_retry_previous_failed_stage():
    import qualify_recursive_validation as q
    def api(path):
        return dict(workflow_runs=[dict(id=12345),dict(id=12222)])
    with pytest.raises(ValueError,match='dispatch'):
        q.verify_once('noise','12345',None,api=api)


def test_wrong_run_authorization_rejected_before_fit(tmp_path,monkeypatch):
    import qualify_recursive_validation as q
    monkeypatch.setattr(q,'checked_manifest',lambda:q.manifest())
    q.prepare('noise',tmp_path/'prepare')
    monkeypatch.setenv('GITHUB_RUN_ID','other')
    with pytest.raises(ValueError,match='authorization'):
        q.run_one('noise',1,tmp_path/'out',tmp_path/'prepare/authorization.json')
    assert not (tmp_path/'out').exists()


def test_broken_archive_returns_unknown_without_aborting_other_audits(tmp_path):
    import qualify_recursive_validation as q
    p=tmp_path/'bad.zip'
    p.write_bytes(b'broken')
    row=q.audit_fresh_case((q.specs()[0],p,q.manifest(),{}))
    assert row['status']=='audit_failed' and row['fit_cost_unknown']
    assert row['started_assessments'] is None


def test_workflow_sparse_inputs_budget_and_failure_preservation():
    import fnmatch
    import yaml
    import qualify_recursive_validation as q
    wf=yaml.safe_load((q.ROOT/'.github/workflows/recursive-qualification.yml').read_text())
    assert wf['jobs']['case']['strategy']['max-parallel']==20
    assert wf['jobs']['case']['strategy']['fail-fast'] is False
    assert wf['jobs']['prepare']['timeout-minutes']*2+wf['jobs']['aggregate']['timeout-minutes']*2==150
    assert 'always()' in wf['jobs']['aggregate']['if']
    for name in ['prepare','case','aggregate']:
        paths=wf['jobs'][name]['steps'][0]['with']['sparse-checkout'].splitlines()
        assert all(any(fnmatch.fnmatch(p,pattern) for pattern in paths) for p in q.manifest()['sources'])


def test_preserved_git_archive_must_match_both_blob_and_audit_sha():
    import hashlib
    import qualify_recursive_validation as q
    data=b'original preserved archive'
    archives={'case':dict(path='examples/results/frozen/case.zip',git_blob='a'*40,
        sha256=hashlib.sha256(data).hexdigest())}
    def git(args,**kwargs):
        return b'100644 blob '+b'a'*40+b'\texamples/results/frozen/case.zip\n' if args[1]=='ls-tree' else data
    q.verify_preserved_archives(archives,git=git)
    archives['case']['sha256']='b'*64
    with pytest.raises(ValueError,match='SHA256'):
        q.verify_preserved_archives(archives,git=git)


def test_aggregate_missing_cases_persists_all_rows_and_graph(tmp_path,monkeypatch):
    import qualify_recursive_validation as q
    monkeypatch.setattr(q,'checked_manifest',lambda:q.manifest())
    monkeypatch.setattr(q,'transport',lambda *a,**k:dict(verified=True))
    report=q.aggregate_stage('noise',tmp_path/'inputs',tmp_path/'out')
    assert report['planned']==110 and report['complete']==0 and not report['qualified']
    assert len(q.load_json(tmp_path/'out/audit.json')['cases'])==110
    assert (tmp_path/'out/qualification-rates.png').stat().st_size>0
