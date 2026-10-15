import importlib.util,sys
from pathlib import Path
import pytest
from study_fixture_helpers import isolate_prospective


def module(tmp,monkeypatch):
    p=Path(__file__).parents[1]/'examples/prospective_interaction_study.py'
    assert p.exists(),'prospective runner missing'
    sys.path.insert(0,str(p.parent));s=importlib.util.spec_from_file_location('prospective_study',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.MANIFEST=tmp/'manifest.json';monkeypatch.setattr(m,'sources',lambda: m.old.sources())
    isolate_prospective(m,tmp,monkeypatch);return m


def test_freeze_context_and_signal_guard(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);m.freeze();c=m.d.tasks()[0]
    assert m.manifest()['fits_per_task']==692 and len(m.manifest()['tasks'])==25200
    signal=next(c for c in m.d.tasks() if c['stage']=='signal')
    with pytest.raises(ValueError):m.run_task(signal,tmp_path/'out')
    a=m.context(c);monkeypatch.setattr(m,'sources',lambda:{'wrong':'source'});m.identity.cache_clear()
    assert m.context(c)!=a
    with pytest.raises(ValueError):m.freeze()


def test_verified_resume_and_prediction_tamper(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];out=tmp_path/'out'
    v=m.run_task(c,out);assert len(v['comparisons'])==4
    assert m.io.read(out/c['id']/'timing.json')['saved_setting_refits']==20
    monkeypatch.setattr(m.old.models,'compare',lambda *a:pytest.fail('resume refitted'))
    assert m.run_task(c,out)==v
    v['comparisons'][0]['pairs'][0]['predicted'][0]+=1;m.io.write(out/c['id']/'result.json',v)
    with pytest.raises((ValueError,AssertionError)):m.run_task(c,out)


def test_partial_cost_and_repeat_history(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];folder=tmp_path/'out'/c['id']
    m.record_failure(c,folder,'first');m.record_failure(c,folder,'second')
    assert m.io.read(folder/'error.json')['attempts']==['first','second']
    data=m.old.comparisons(c)[0];v=m.old.models.compare(*data,c['split_seed'])
    s=dict(context=m.context(c),result=v,population=m.population_record(c,data,v),fitting_seconds=1.,warnings=[])
    m.check_checkpoint(c,s,data);s['population']['gap']+=1
    with pytest.raises(ValueError):m.check_checkpoint(c,s,data)


def test_failed_verification_attempt_survives_completed_resume(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];out=tmp_path/'out';original=m.audit_task
    def interrupted(c,folder,refit=False):
        if refit:raise RuntimeError('interrupted verification with unknown refit cost')
        return original(c,folder,refit)
    monkeypatch.setattr(m,'audit_task',interrupted)
    with pytest.raises(RuntimeError):m.run_task(c,out)
    m.record_failure(c,out/c['id'],'verification failed')
    monkeypatch.setattr(m,'audit_task',original)
    monkeypatch.setattr(m.old.models,'compare',lambda *a:pytest.fail('resume refitted tuning'))
    m.run_task(c,out)
    ledger=m.io.read(out/c['id']/'attempts.json')
    assert len(ledger['attempts'])==2
    assert ledger['attempts'][0]['unknown_verification_cost']
    assert ledger['attempts'][1]['verification_refits']==20 and ledger['attempts'][1]['fits']==0
    assert m.attempt_cost(c,out/c['id'])['unknown_verification_cost']
    import prospective_interaction_report as report
    monkeypatch.setattr(report,'study',m)
    a=report.aggregate('null',[out],out,m.manifest())
    assert a['complete']==1 and c['id'] in a['unknown_attempt_cost_ids']
    assert a['known_attempt_cost']['verification_refits']==20


def test_interrupted_fit_remains_unknown_after_success(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];out=tmp_path/'out';original=m.old.models.compare
    def interrupted(*args):raise RuntimeError('interrupted partial tuning')
    monkeypatch.setattr(m.old.models,'compare',interrupted)
    with pytest.raises(RuntimeError):m.run_task(c,out)
    monkeypatch.setattr(m.old.models,'compare',original);m.run_task(c,out)
    a=m.attempt_cost(c,out/c['id']);assert a['fits']==692 and a['unknown_fit_cost']
