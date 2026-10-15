"""Execution contracts only: no real local assessment may run."""
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))


@pytest.fixture(autouse=True)
def no_fits(monkeypatch):
    import recursive_preview_runner as runner
    monkeypatch.setattr(runner, 'NATIVE_ASSESS', lambda *a, **k: pytest.fail('local fit'))
    monkeypatch.setenv('GITHUB_RUN_ID', '999')
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')


def test_original_chain_identities_and_budget():
    import qualify_chain_validation as c
    assert [s['index'] for s in c.specs()] == list(range(161,216))+list(range(241,296))
    assert c.specs() == [s for s in c.q.stage_specs('signal') if s['scenario']=='chain']
    m = c.manifest()
    assert m['reserved_total'] == 9889 and m['remaining_ceiling'] == 111
    assert m['new_reservation'] == 110*45+15+60+1
    assert m['soft_seconds'] == 2310


def rows(c, exact=55, wrong=0):
    result=[]
    for s in c.specs():
        r=s['repetition']
        result.append(dict(spec=s,status='complete',structure=dict(exact=r<=exact,incorrect=r<=wrong)))
    return result


def test_fixed_per_cell_endpoints_and_missing_are_unknown():
    import qualify_chain_validation as c
    m={}
    assert c.summarize(rows(c,50,1),m)['qualified']
    assert not c.summarize(rows(c,49,1),m)['qualified']
    assert not c.summarize(rows(c,55,2),m)['qualified']
    report=c.summarize(rows(c)[1:],m)
    assert report['planned']==110 and report['complete']==109 and not report['qualified']
    assert report['cells']['chain-n80']['unknown']==1
    assert set(report['cells'])=={'chain-n80','chain-n200'}
    with pytest.raises(ValueError,match='foreign'):
        c.summarize([dict(spec=c.q.stage_specs('noise')[0])],m)
    with pytest.raises(ValueError,match='duplicate'):
        c.summarize(rows(c)+rows(c)[:1],m)


def test_gate_rejects_changed_report_or_archive_inventory(monkeypatch):
    import qualify_chain_validation as c
    gate=c.load_json(c.GATE)
    assert c.validate_gate(gate)['complete']==110
    edited=copy.deepcopy(gate)
    edited['archives'].pop(next(iter(edited['archives'])))
    with pytest.raises(ValueError):
        c.validate_gate(edited)
    edited=copy.deepcopy(gate)
    edited['audit_sha256']='0'*64
    with pytest.raises(ValueError):
        c.validate_gate(edited)


def test_foreign_case_and_retry_never_start(tmp_path,monkeypatch):
    import qualify_chain_validation as c
    monkeypatch.setattr(c,'checked_manifest',lambda: {})
    for index in (0,160,320,True):
        with pytest.raises(ValueError,match='identity'):
            c.run_one(index,tmp_path/'out',None)
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    with pytest.raises(ValueError,match='first'):
        c.run_one(161,tmp_path/'out',None)
    assert not (tmp_path/'out').exists()


def test_workflow_caps_and_paginated_download():
    import qualify_chain_validation as c
    w=yaml.safe_load((c.ROOT/c.WORKFLOW).read_text())
    assert w['jobs']['case']['timeout-minutes']==45
    assert w['jobs']['case']['strategy']['max-parallel']==20
    assert w['jobs']['aggregate']['timeout-minutes']==60
    text=(c.ROOT/c.WORKFLOW).read_text()
    assert 'qualification_artifacts.py' in text
    assert 'pattern: chain-case-*' not in text
    assert 'if: always()' in text


def test_authorization_is_bound_to_current_source_run_and_remote_gate(tmp_path,monkeypatch):
    import qualify_chain_validation as c
    proof=dict(gate='abc',audit_sha256='def',verified=True)
    monkeypatch.setattr(c,'verify_gate',lambda **k: proof)
    value={}
    def write(**updates):
        a=dict(manifest=c.q.design.digest(value),github_run='999',github_sha='source',
            github_attempt='1',noise_gate=dict(proof,remote_verified=True))
        a.update(updates)
        a['digest']=c.q.design.digest(a)
        c.write_json(tmp_path/'auth.json',a)
    write()
    c.verify_authorization(tmp_path/'auth.json',value)
    for update in (dict(github_sha='other'),dict(github_run='998'),dict(github_attempt='2'),
                   dict(noise_gate=proof),dict(manifest='other')):
        write(**update)
        with pytest.raises(ValueError,match='authorization'):
            c.verify_authorization(tmp_path/'auth.json',value)


def test_gate_live_checks_failed_historical_runs_and_remote_preservation(monkeypatch):
    import qualify_chain_validation as c
    gate=c.load_json(c.GATE)
    monkeypatch.setattr(c.prior,'transport',lambda *a,**k: None)
    tree=dict(gate['inputs'])
    tree.update({v['path']:v['git_blob'] for v in gate['archives'].values()})
    def api(path):
        if path.startswith('git/trees/'):
            return dict(truncated=False,tree=[dict(path=p,sha=h,type='blob') for p,h in tree.items()])
        return dict(status='completed',conclusion='failure')
    assert c.verify_gate(remote=True,api=api)['remote_verified']
    tree.pop(next(iter(tree)))
    with pytest.raises(ValueError,match='preserved'):
        c.verify_gate(remote=True,api=api)
    with pytest.raises(ValueError,match='historical'):
        c.verify_gate(remote=True,api=lambda p: dict(status='completed',conclusion='success'))


def test_previous_dispatch_and_paginated_jobs_are_checked():
    import qualify_chain_validation as c
    c.verify_once('999',api=lambda p: dict(workflow_runs=[]))
    with pytest.raises(ValueError,match='prior dispatch'):
        c.verify_once('999',api=lambda p: dict(workflow_runs=[dict(id=998)]))
    jobs=[dict(name=f"case ({s['id']})",conclusion='success',
        steps=[dict(name='Run actions/upload-artifact@pin',conclusion='success')]) for s in c.specs()]
    jobs.append(dict(name='prepare',conclusion='success',steps=jobs[0]['steps']))
    def api(path):
        if '/jobs?' in path:
            return dict(jobs=jobs[:100] if path.endswith('page=1') else jobs[100:])
        return dict(id=999,head_sha='source',run_attempt=1,event='workflow_dispatch',path=c.WORKFLOW)
    assert c.transport('999','source',api=api)['case_jobs']==110
    jobs[-2]['conclusion']='failure'
    with pytest.raises(ValueError,match='upload failed'):
        c.transport('999','source',api=api)


def test_case_uses_original_runner_and_frozen_soft_limit(tmp_path,monkeypatch):
    import qualify_chain_validation as c
    value=dict(soft_seconds=2310)
    monkeypatch.setattr(c,'checked_manifest',lambda: value)
    monkeypatch.setattr(c,'verify_authorization',lambda *a: None)
    calls=[]
    def run(spec,directory,manifest,seconds):
        calls.append((spec,manifest,seconds))
        return dict(seconds=12)
    monkeypatch.setattr(c,'run_case',run)
    c.run_one(161,tmp_path/'case',None)
    assert calls==[(c.specs()[0],value,2310)]
    assert c.load_json(tmp_path/'case/control.json')['status']=='complete'


def test_empty_aggregate_keeps_110_unknowns_and_cannot_qualify(tmp_path,monkeypatch):
    import qualify_chain_validation as c
    monkeypatch.setattr(c,'checked_manifest',lambda: {})
    monkeypatch.setattr(c,'transport',lambda *a: dict(verified=True))
    monkeypatch.setattr(c.q,'write_report',lambda *a: None)
    result=c.aggregate(tmp_path/'missing',tmp_path/'out')
    assert result['complete']==0 and result['planned']==110 and not result['qualified']
    assert all(v['unknown']==55 for v in result['cells'].values())


def test_partial_download_preserves_available_cases_and_failed_transfers(tmp_path):
    import subprocess
    import qualification_artifacts as a
    records=[dict(id=1,name='case-a',expired=False),dict(id=2,name='case-b',expired=False)]
    calls=[]
    def transfer(command,check):
        calls.append(command[command.index('--name')+1])
        if calls[-1]=='case-a':
            raise subprocess.CalledProcessError(1,command)
    with pytest.raises(ValueError,match='failed'):
        a.download_artifacts('999',['case-a','case-b','case-c'],tmp_path/'out',
            api=lambda p:dict(total_count=2,artifacts=records),run=transfer,allow_missing=True)
    assert calls==['case-a','case-b']
    import json
    inventory=json.loads((tmp_path/'out/download-inventory.json').read_text())
    assert inventory['missing']==['case-c'] and inventory['failed']==['case-a']
