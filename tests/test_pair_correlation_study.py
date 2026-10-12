import importlib.util
from pathlib import Path
import numpy as np
import pytest


def study():
    path=Path(__file__).parents[1]/'examples/pair_correlation_study.py'
    assert path.exists(),'correlation research runner missing'
    spec=importlib.util.spec_from_file_location('pair_correlation_study',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_disjoint_design_and_frozen_roundtrip(tmp_path,monkeypatch):
    m=study();tasks=m.tasks()
    assert len(tasks)==240 and len({t['id'] for t in tasks})==240
    assert [t for s in m.shards() for t in m.shard_tasks(s)]==tasks
    assert min(t['seed'] for t in tasks)>=148000000
    monkeypatch.setattr(m,'MANIFEST',tmp_path/'manifest.json')
    m.freeze(False);assert m.freeze()==m.io.read(m.MANIFEST)


def test_oracle_matches_gaussian_risk_and_orthogonality():
    m=study();c=next(t for t in m.tasks() if t['scenario']=='mixed' and t['rho']==.7)
    X,y,full,add=m.generate(c,200000,c['test_seed'])
    residual=full-add
    expected=.64*(1-.7**2)**2/(1+.7**2)
    assert np.mean(residual**2)==pytest.approx(expected,rel=.025)
    for feature in (X.x1,X.x2,X.x1**2-1,X.x2**2-1):
        assert abs(np.mean(residual*feature))<.015
    assert m.oracle_gap(c)==pytest.approx(.0212558609910821,rel=1e-12)


def test_weight_tuning_uses_only_training_and_keeps_tiny_features_small():
    m=study();c=m.tasks()[0];X,y,_,_=m.generate(c,80,c['seed'])
    train,test=m.splits(c)[0]
    a=m.compare(X,y,train,test,c['split_seed']);changed=y.copy();changed[test]+=100
    b=m.compare(X,changed,train,test,c['split_seed'])
    assert a['additional_fits']==221 and a['additional_fits']<=224
    assert a['research'][0]['parameters']==b['research'][0]['parameters']
    assert a['research'][0]['predicted']==b['research'][0]['predicted']
    assert a['research'][0]['gain']!=b['research'][0]['gain']
    assert np.max(np.abs(a['research'][0]['predicted']))<10


def test_native_audit_and_changed_evidence_rejected(tmp_path):
    m=study();c=m.tasks()[145];m.run_task(c,tmp_path)
    r=m.aggregate(tmp_path)
    assert r['complete']==1 and r['missing']==239 and r['acceptance']=='none: development only'
    import pandas as pd
    rows=pd.read_csv(tmp_path/'case_metrics.csv')
    assert (rows[rows.pair=='x1:x3'].oracle_gap==0).all()
    assert (rows[rows.pair=='x1:x2'].oracle_gap>0).all()
    file=tmp_path/c['id']/'result.json';v=m.io.read(file)
    v['folds'][0]['research'][0]['gain']+=1;m.io.write(file,v)
    assert m.aggregate(tmp_path)['complete']==0


def test_missing_and_failed_identities_are_preserved(tmp_path):
    m=study();c=m.tasks()[0]
    m.io.write(tmp_path/c['id']/'error.json',dict(task=c,context=m.context(c),error='controlled failure'))
    r=m.aggregate(tmp_path)
    assert r['complete']==0 and r['missing']==240
    assert r['failed_tasks']==[dict(id=c['id'],error='controlled failure')]


def test_weight_grid_does_not_restore_original_numerical_explosion():
    from test_pair_validation import unstable_correlated_case
    m=study();X,y,train,test=unstable_correlated_case()
    for weight in m.WEIGHTS:
        model=m.estimator((0,1)).set_params(curves__n_knots=5,ridge__alpha=100.,curves__joint_weight=weight)
        pred=model.fit(X.iloc[train],y[train]).predict(X.iloc[test])
        assert np.max(np.abs(pred))<15
        assert np.square(y[test]-pred).mean()<8


def test_guard_is_sensitive_to_source_changes(tmp_path,monkeypatch):
    m=study();monkeypatch.setattr(m,'MANIFEST',tmp_path/'manifest.json')
    original=m.manifest;m.freeze(False)
    def changed():
        v=original();v['sources']['examples/pair_correlation_study.py']='tampered';return v
    monkeypatch.setattr(m,'manifest',changed)
    with pytest.raises(ValueError,match='frozen'):m.freeze()


def test_rejects_changed_solver_even_if_verification_stamp_recomputed(tmp_path):
    m=study();c=m.tasks()[0];m.run_task(c,tmp_path);folder=tmp_path/c['id']
    v=m.io.read(folder/'result.json');v['folds'][0]['v2']['solver']['tol']=1e-3
    m.io.write(folder/'result.json',v);m.io.write(folder/'verified.json',m.verification(c,folder))
    assert m.aggregate(tmp_path)['complete']==0


def test_pooled_gain_uses_weighted_risks_not_mean_of_ratios():
    m=study();folds=[]
    for base,joint,ref,n in ((3.,2.,2.,10),(10.,5.,100.,20)):
        p=dict(joint_mse=joint,predicted=[0.]*n)
        folds.append(dict(v2=dict(baseline_mse=base,reference_mse=ref,pairs=[p]),research=[p]))
    assert m.pooled_gain(dict(folds=folds),'v2',0)==pytest.approx(110/2020)


def test_checkpoint_resume_does_not_retune_and_reports_reused_work(tmp_path,monkeypatch):
    m=study();c=m.tasks()[0];m.run_task(c,tmp_path);folder=tmp_path/c['id']
    for name in ('result.json','verified.json','timing.json','predictions.csv'):(folder/name).unlink()
    def no_retuning(*args,**kwargs):raise AssertionError('resumed checkpoint retuned')
    monkeypatch.setattr(m,'compare',no_retuning)
    m.run_task(c,tmp_path)
    cost=m.io.read(folder/'timing.json')
    assert cost['fitting_tuning_fits_this_attempt']==0 and cost['resumed_comparisons']==4
    assert cost['fitting_tuning_fits']==884 and cost['saved_setting_refits']==20
    assert m.aggregate(tmp_path)['complete']==1


def test_offline_checker_rejects_contradictory_baseline_records():
    import copy,json
    m=study();c=m.tasks()[0];X,y,_,_=m.generate(c,c['n'],c['seed']);train,test=m.splits(c)[0]
    v=json.loads(json.dumps(m.compare(X,y,train,test,c['split_seed'])))
    m.check_comparison(v,X,y,train.tolist(),test.tolist(),c['split_seed'])
    for field in ('parameters','risk','prediction','sizes'):
        changed=copy.deepcopy(v)
        if field=='parameters':changed['v2']['baseline_parameters']['ridge__alpha']=999.
        if field=='risk':changed['v2']['pairs'][0]['baseline_mse']=-100.
        if field=='prediction':changed['v2']['pairs'][1]['baseline_predicted'][0]+=1.
        if field=='sizes':changed['v2']['inner_training_n'][0]+=1
        with pytest.raises((ValueError,AssertionError)):
            m.check_comparison(changed,X,y,train.tolist(),test.tolist(),c['split_seed'])
