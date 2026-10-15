import sys
from pathlib import Path
import numpy as np
import pytest
from study_fixture_helpers import isolate_caches, isolate_prospective
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def module(tmp,monkeypatch):
    assert (Path(__file__).parents[1]/'examples/realistic_interaction_study.py').exists(),'realistic runner missing'
    import realistic_interaction_study as m
    isolate_prospective(m.previous,tmp/'previous',monkeypatch)
    isolate_caches(m,monkeypatch)
    monkeypatch.setattr(m,'MANIFEST',tmp/'manifest.json')
    monkeypatch.setattr(m,'THRESHOLDS',tmp/'thresholds.json')
    monkeypatch.setattr(m,'CALIBRATION_AUDIT',tmp/'calibration-audit.json')
    monkeypatch.setattr(m,'sources',lambda: m.old.sources())
    m.freeze(False);return m


def test_guard_and_splits(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0]
    assert m.manifest()['budget_minutes']==3750
    with pytest.raises(ValueError,match='calibration'):m.context(m.d.stage_tasks('development')[0])
    data=m.comparisons(c);assert len(data)==4
    assert all(len(z[2])==100 and len(z[3])==50 for z in data[:3])
    assert len(data[3][2])==150 and len(data[3][3])==1000
    np.testing.assert_array_equal(data[0][0],data[3][0].iloc[:150])
    with pytest.raises(ValueError):m.context(dict(c,seed=1))


def test_real_resume_refits_and_tamper(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];out=tmp_path/'out'
    v=m.run_task(c,out);assert len(v['comparisons'])==4
    assert m.attempt_cost(c,out/c['id'])['verification_refits']==20
    monkeypatch.setattr(m.old.models,'compare',lambda *a:pytest.fail('resume refitted'))
    assert m.run_task(c,out)==v
    v['population'][0]['predicted'][0]+=1;m.io.write(out/c['id']/'result.json',v)
    with pytest.raises((ValueError,AssertionError)):m.run_task(c,out)


def test_precomputed_truth_failure_prevents_any_fit(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0]
    def fail(*args):raise ValueError('quadrature disagreement')
    monkeypatch.setattr(m.truth,'teacher',fail)
    monkeypatch.setattr(m.old.models,'compare',lambda *a:pytest.fail('fit before truth validation'))
    with pytest.raises(ValueError,match='quadrature'):m.run_task(c,tmp_path/'out')
    cost=m.attempt_cost(c,tmp_path/'out'/c['id'])
    assert cost['fits']==0 and not cost['unknown_fit_cost']


def test_failed_fit_cost_retained(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);c=m.d.tasks()[0];out=tmp_path/'out'
    def fail(*args):raise RuntimeError('interrupted fit')
    monkeypatch.setattr(m.old.models,'compare',fail)
    with pytest.raises(RuntimeError):m.run_task(c,out)
    cost=m.attempt_cost(c,out/c['id']);assert cost['unknown_fit_cost'] and cost['fits']==0
