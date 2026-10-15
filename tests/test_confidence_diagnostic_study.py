import importlib.util,sys,json
from study_fixture_helpers import isolate_pair_calibration
from pathlib import Path
import numpy as np
import pytest


def module(tmp,monkeypatch):
    root=Path(__file__).parents[1];p=root/'examples/confidence_diagnostic_study.py'
    assert p.exists(), 'diagnostic runner missing'
    sys.path.insert(0,str(root/'examples'))
    spec=importlib.util.spec_from_file_location('diagnostic_study',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Exercise real provenance checks with isolated unit-test evidence. The
    # committed calibration is bound to historical source and must stay frozen.
    isolate_pair_calibration(m.old,tmp/'pair',monkeypatch)
    m.MANIFEST=tmp/'manifest.json';m.freeze(False);return m


def test_frozen_context_and_shards(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);m.freeze();assert len(m.shard_tasks('005'))==10
    with pytest.raises(ValueError):m.shard_tasks('006')
    a=m.manifest();assert a['thresholds_sha256']==m.old.threshold_digest() and a['tasks']==m.d.tasks()
    monkeypatch.setattr(m,'sources',lambda:dict(changed='context'))
    with pytest.raises(ValueError):m.freeze()


def test_oracle_vectors_are_diagnostic_and_audited(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[18];data=m.old.comparisons(c)[0]
    X,y,train,test=data;v=m.old.models.compare(*data,c['split_seed']);o=m.population_record(c,data,v)
    np.testing.assert_allclose(o['joint_predicted'],m.d.population_predictions(X,c)[1][test])
    m.audit_population(c,data,v,o)
    o['joint_predicted'][0]+=1
    with pytest.raises((ValueError,AssertionError)):m.audit_population(c,data,v,o)


def test_partial_evidence_retains_missing_and_cost(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];data=m.old.comparisons(c)[0];v=m.old.models.compare(*data,c['split_seed']);out=tmp_path/'out'
    m.io.write(out/c['id']/'comparison-0.json',dict(context=m.context(c),result=v,population=m.population_record(c,data,v),fitting_seconds=1.,warnings=[]))
    a=m.aggregate(out);assert a['complete']==0 and a['missing']==60 and not a['audit_errors']
    assert a['checkpoint_fits']==173 and a['unknown_attempt_cost_ids']==[c['id']]
    assert not (out/'learned-thresholds.json').exists() and a['production_accepted'] is False


def test_verified_resume_and_prediction_tampering(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];out=tmp_path/'out';v=m.run_task(c,out);assert len(v['comparisons'])==4
    assert m.io.read(out/c['id']/'timing.json')['saved_setting_refits']==20
    monkeypatch.setattr(m.old.models,'compare',lambda *a,**k:pytest.fail('verified resume refitted'))
    assert m.run_task(c,out)==v
    p=out/c['id']/'result.json';v['comparisons'][0]['pairs'][0]['predicted'][0]+=1;m.io.write(p,v)
    with pytest.raises((ValueError,AssertionError)):m.run_task(c,out)


def test_foreign_stamps_and_error_context_fail_audit(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);out=tmp_path/'out';m.io.write(out/'shard-999.json',dict(id='999'))
    c=m.d.tasks()[0];m.io.write(out/c['id']/'error.json',dict(task=c,context='wrong'))
    a=m.aggregate(out);assert len(a['audit_errors'])==2 and a['missing']==60


def test_repeated_failures_preserve_every_attempt(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];folder=tmp_path/'out'/c['id']
    m.record_failure(c,folder,'first failure');m.record_failure(c,folder,'second failure')
    a=m.io.read(folder/'error.json');assert a['attempts']==['first failure','second failure'] and a['error']=='second failure'
    b=m.aggregate(tmp_path/'out');assert len(b['failed_tasks'])==1 and b['failed_tasks'][0]['failure']['attempts']==a['attempts']
