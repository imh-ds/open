import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def study():
    path=Path(__file__).parents[1]/'examples/pair_conservative_study.py'
    assert path.exists(), 'conservative runner missing'
    spec=importlib.util.spec_from_file_location('pair_conservative_study',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def params(weight=1.,knots=3,alpha=10.):
    return dict(curves__joint_weight=weight,curves__n_knots=knots,ridge__alpha=alpha)


def test_one_se_prefers_joint_penalty_not_weight_alone():
    m=study()
    p=[params(100,5,100),params(1,3,.1),params(10,3,100)]
    s=m.choose_one_se(p,[[1.,2.,3.],[2.1,2.1,2.1],[2.2,2.2,2.2]])
    assert s['best_index']==0 and s['selected_index']==2
    assert s['best_se']==pytest.approx(1/np.sqrt(3))
    assert s['eligible_indices']==[0,1,2]


def test_zero_band_and_deterministic_ties():
    m=study();p=[params(10,5,100),params(1,3,1)]
    s=m.choose_one_se(p,[[2.,2.,2.],[2.,2.,2.]])
    assert p[s['selected_index']]==params(1,3,1)
    reverse=m.choose_one_se(p[::-1],[[2.,2.,2.],[2.,2.,2.]])
    assert p[::-1][reverse['selected_index']]==p[s['selected_index']]
    assert m.choose_one_se(p,[[1.,1.,1.],[2.,2.,2.]])['selected_index']==0


def test_invalid_inner_evidence_rejected():
    m=study()
    for p,loss in (([],[]),([params()],[[1.,np.nan,2.]]),([params()],[[1.]]),
                   ([params(),params()],[[1.,2.],[1.,2.]]),([params(1000)],[[1.,2.]])):
        with pytest.raises(ValueError):m.choose_one_se(p,loss)


def test_saved_development_case_selects_restrained_alternative():
    m=study();data=json.loads((Path(__file__).parents[1]/'examples/results/2026-10-12-small-n-diagnosis/candidate_scores.json').read_text())
    p=[{k:r[k] for k in params()} for r in data]
    selection=m.choose_one_se(p,[r['split_losses'] for r in data])
    assert p[selection['selected_index']]==params(1.,3,10.)
    assert data[selection['selected_index']]['outer_mse']<3


def test_fresh_tasks_and_frozen_roundtrip(tmp_path,monkeypatch):
    m=study();tasks=m.tasks();assert len(tasks)==240 and min(t['seed'] for t in tasks)==178000000
    assert all(t['id'].startswith('one-se--') for t in tasks)
    assert [t for s in m.shards() for t in m.shard_tasks(s)]==tasks
    monkeypatch.setattr(m,'MANIFEST',tmp_path/'manifest.json');m.freeze(False);m.freeze()


def test_selection_does_not_use_assessment_outcomes():
    m=study();c=m.tasks()[0];X,y,*_=m.generate(c,c['n'],c['seed']);train,test=m.splits(c)[0]
    a=m.compare(X,y,train,test,c['split_seed']);changed=y.copy();changed[test]+=100
    b=m.compare(X,changed,train,test,c['split_seed'])
    assert a['additional_fits']==223<=224
    assert a['selection']==b['selection']
    assert a['conservative'][0]['parameters']==b['conservative'][0]['parameters']
    assert a['conservative'][0]['predicted']==b['conservative'][0]['predicted']
    assert a['conservative'][0]['gain']!=b['conservative'][0]['gain']


def test_native_audit_and_tampered_selection(tmp_path):
    m=study();c=m.tasks()[0];m.run_task(c,tmp_path);r=m.aggregate(tmp_path)
    assert r['complete']==1 and r['missing']==239 and r['saved_setting_refits']==28
    p=tmp_path/c['id']/'result.json';v=m.io.read(p)
    v['folds'][0]['selection'][0]['threshold']+=1;m.io.write(p,v)
    m.io.write(tmp_path/c['id']/'verified.json',m.verification(c,tmp_path/c['id']))
    assert m.aggregate(tmp_path)['complete']==0


def test_known_failure_replay_preserves_minimum_and_restraints_risk():
    m=study()
    import pair_correlation_study as prior
    c=next(c for c in prior.tasks() if c['id']=='pure-rho0.9-n80-p5-r00')
    X,y,*_=m.generate(c,c['n'],c['seed']);train,test=m.splits(c)[1]
    v=m.compare(X,y,train,test,c['split_seed'])
    assert v['research'][0]['parameters']==params(100.,5,100.)
    assert v['research'][0]['joint_mse']>10.
    assert v['conservative'][0]['parameters']==params(1.,3,10.)
    assert v['conservative'][0]['joint_mse']<3.
    m.check_comparison(json.loads(json.dumps(v)),X,y,train.tolist(),test.tolist(),c['split_seed'],refit=True)


def test_unknown_failure_retains_planned_denominator(tmp_path):
    m=study();c=m.tasks()[0]
    m.io.write(tmp_path/c['id']/'error.json',dict(task=c,context=m.context(c),error='controlled failure'))
    r=m.aggregate(tmp_path)
    assert r['missing']==240 and r['failed_tasks']==[dict(id=c['id'],error='controlled failure')]
    assert all(cell['planned']==5 for cell in r['cells'])


def test_completed_checkpoint_cost_is_retained_when_task_fails(tmp_path):
    m=study();c=m.tasks()[0];X,y,*_=m.generate(c,c['n'],c['seed']);train,test=m.splits(c)[0]
    checkpoint=dict(context=m.context(c),result=m.compare(X,y,train,test,c['split_seed']),
        fitting_seconds=12.3,warnings=[dict(category='UserWarning',message='retained test warning')])
    m.io.write(tmp_path/c['id']/'comparison-0.json',checkpoint)
    m.io.write(tmp_path/c['id']/'error.json',dict(task=c,context=m.context(c),error='interrupted'))
    r=m.aggregate(tmp_path)
    assert r['complete']==0 and r['fitting_tuning_fits']==0
    assert r['retained_checkpoint_cost']['fitting_tuning_fits']==223
    assert r['partial_checkpoint_cost']['fitting_seconds']==12.3
    assert r['partial_checkpoint_cost']['warning_count']==1
    assert c['id'] in r['unfinished_attempt_cost_unknown_ids']


def test_successful_task_keeps_old_failure_as_history(tmp_path):
    m=study();c=m.tasks()[0];m.run_task(c,tmp_path)
    m.io.write(tmp_path/c['id']/'error.json',dict(task=c,context=m.context(c),error='prior attempt failed'))
    r=m.aggregate(tmp_path)
    assert r['complete']==1 and not r['failed_tasks']
    assert r['historical_failed_tasks']==[dict(id=c['id'],error='prior attempt failed')]
