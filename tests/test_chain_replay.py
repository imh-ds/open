import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def task(tmp_path,c):
    archive=tmp_path/'case.zip';archive.write_bytes(b'archive')
    control=tmp_path/'control.json'
    spec=c.chain.specs()[0]
    value={}
    c.chain.write_json(control,dict(stage='chain',index=spec['index'],spec=spec,
        manifest=c.chain.q.design.digest(value),github_run=c.RUN,github_sha=c.SOURCE,
        github_attempt='1',status='complete'))
    return spec,archive,control,value,tmp_path/'cache'


def test_replay_caches_completed_checks_and_resumes_without_reaudit(tmp_path,monkeypatch):
    import replay_chain_verification as c
    args=task(tmp_path,c)
    calls=[]
    def audit(t):
        calls.append(t)
        return dict(spec=t[0],status='complete',archive_sha256=c.sha(t[1]),structure={})
    monkeypatch.setattr(c.chain.q,'audit_fresh_case',audit)
    first=c.check_one(args)
    assert first['status']=='complete' and len(calls)==1
    assert c.check_one(args)==first and len(calls)==1
    args[1].write_bytes(b'changed')
    with pytest.raises(ValueError,match='cached'):
        c.check_one(args)


def test_edited_cached_result_or_source_control_is_rejected(tmp_path,monkeypatch):
    import replay_chain_verification as c
    args=task(tmp_path,c)
    monkeypatch.setattr(c.chain.q,'audit_fresh_case',lambda t: dict(spec=t[0],status='complete',
        archive_sha256=c.sha(t[1]),structure={}))
    c.check_one(args)
    p=args[-1]/(args[0]['id']+'.json')
    value=json.loads(p.read_text());value['row']['status']='wrong'
    c.chain.write_json(p,value)
    with pytest.raises(ValueError,match='cached'):
        c.check_one(args)
    p.unlink()
    control=json.loads(args[2].read_text());control['github_sha']='wrong'
    c.chain.write_json(args[2],control)
    with pytest.raises(ValueError,match='control'):
        c.check_one(args)


def test_no_fit_guard_fails_closed():
    import replay_chain_verification as c
    import recursive_preview_runner as runner
    with pytest.raises(RuntimeError,match='prohibited'):
        runner.NATIVE_ASSESS(None)


def test_empty_summary_cannot_pass():
    import replay_chain_verification as c
    r=c.report([],{},dict(verified=True))
    assert r['planned']==110 and r['complete']==0 and not r['qualified']
    assert r['original_workflow_conclusion']=='failure'
