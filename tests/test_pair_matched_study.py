import importlib.util,json
from pathlib import Path
import numpy as np
import pytest


def study():
    path=Path(__file__).parents[1]/'examples/pair_matched_study.py'
    assert path.exists(), 'matched runner missing'
    spec=importlib.util.spec_from_file_location('pair_matched_study',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_matched_settings_scores_and_cost():
    m=study();c=m.tasks()[0];X,y,*_=m.generate(c,c['n'],c['seed']);a,b=m.splits(c)[0]
    v=m.compare(X,y,a,b,c['split_seed'])
    assert v['additional_fits']==229<=232
    for mode in ('v2','research','conservative'):
        records=v['v2']['pairs'] if mode=='v2' else v[mode]
        for r in records:
            assert r['matched_parameters']=={k:r['parameters'][k] for k in ('curves__n_knots','ridge__alpha')}
            assert r['gain']==pytest.approx(r['marginal_gain']+r['matched_gain'])
            pred=np.asarray(r['matched_additive_predicted']);mse=np.mean((y[b]-pred)**2)
            assert r['matched_additive_mse']==pytest.approx(mse)
            assert r['matched_gain']==pytest.approx((mse-r['joint_mse'])/v['v2']['reference_mse'])
    m.check_comparison(json.loads(json.dumps(v)),X,y,a.tolist(),b.tolist(),c['split_seed'],refit=True)


def test_assessment_outcomes_cannot_select_matched_settings_or_predictions():
    m=study();c=m.tasks()[0];X,y,*_=m.generate(c,c['n'],c['seed']);a,b=m.splits(c)[0]
    v=m.compare(X,y,a,b,c['split_seed']);changed=y.copy();changed[b]+=100
    w=m.compare(X,changed,a,b,c['split_seed'])
    assert v['selection']==w['selection']
    for r,s in zip(v['v2']['pairs']+v['research']+v['conservative'],w['v2']['pairs']+w['research']+w['conservative']):
        assert r['matched_parameters']==s['matched_parameters']
        assert r['matched_additive_predicted']==s['matched_additive_predicted']
        assert r['predicted']==s['predicted']
        assert r['matched_gain']!=s['matched_gain']


def test_known_noise_gain_is_not_interaction_gain():
    m=study()
    import pair_conservative_study as old
    c=next(c for c in old.tasks() if c['id']=='one-se--noise-rho0.7-n80-p5-r03')
    X,y,*_=m.generate(c,c['n'],c['seed']);T,ty,*_=m.generate(c,1000,c['test_seed'])
    import pandas as pd
    v=m.compare(pd.concat([X,T],ignore_index=True),np.concatenate([y,ty]),range(80),range(80,1080),c['split_seed'])
    r=v['conservative'][0]
    assert r['gain']>.04 and abs(r['matched_gain'])<.0001
    assert r['marginal_gain']>.04


def test_native_audit_rejects_matched_parameters_and_gain_tampering(tmp_path):
    m=study();c=m.tasks()[0];m.run_task(c,tmp_path);r=m.aggregate(tmp_path)
    assert r['complete']==1 and r['missing']==239 and r['saved_setting_refits']==52
    assert r['fitting_tuning_fits']==916
    table=m.pd.read_csv(tmp_path/'case_metrics.csv')
    assert {'cv_matched_gain','independent_matched_gain','cv_marginal_gain','independent_marginal_gain'}<=set(table)
    sensitivity=m.pd.read_csv(tmp_path/'gain_sensitivity.csv');assert set(sensitivity.score)=={'overall','matched'}
    p=tmp_path/c['id']/'result.json';v=m.io.read(p)
    v['folds'][0]['research'][0]['matched_parameters']['curves__n_knots']=99
    m.io.write(p,v);m.io.write(tmp_path/c['id']/'verified.json',m.verification(c,tmp_path/c['id']))
    assert m.aggregate(tmp_path)['complete']==0


def test_score_tampering_is_rejected():
    m=study();c=m.tasks()[0];X,y,*_=m.generate(c,c['n'],c['seed']);a,b=m.splits(c)[0]
    v=json.loads(json.dumps(m.compare(X,y,a,b,c['split_seed'])))
    v['conservative'][0]['matched_gain']+=.1
    with pytest.raises(AssertionError):m.check_comparison(v,X,y,a.tolist(),b.tolist(),c['split_seed'])


def test_fresh_design_and_partial_cost_remain_auditable(tmp_path,monkeypatch):
    m=study();tasks=m.tasks();assert len(tasks)==240 and tasks[0]['seed']==208000000
    assert all(t['id'].startswith('matched--') for t in tasks)
    assert [t for s in m.shards() for t in m.shard_tasks(s)]==tasks
    monkeypatch.setattr(m,'MANIFEST',tmp_path/'manifest.json');m.freeze(False);m.freeze()
    c=tasks[0];X,y,*_=m.generate(c,c['n'],c['seed']);a,b=m.splits(c)[0]
    m.io.write(tmp_path/c['id']/'comparison-0.json',dict(context=m.context(c),result=m.compare(X,y,a,b,c['split_seed']),fitting_seconds=1.,warnings=[]))
    r=m.aggregate(tmp_path);assert r['missing']==240 and r['partial_checkpoint_cost']['fitting_tuning_fits']==229
    assert c['id'] in r['unfinished_attempt_cost_unknown_ids']


def test_overall_and_matched_cv_pool_risks_before_ratios():
    m=study()
    def fold(base,reference,joint,matched,n):
        return dict(v2=dict(baseline_mse=base,reference_mse=reference),
            research=[dict(joint_mse=joint,matched_additive_mse=matched,predicted=[0.]*n)])
    v=dict(folds=[fold(10.,3.,6.,8.,2),fold(4.,10.,3.,2.,1)])
    assert m.pooled_gain(v,'research',0)==pytest.approx(9/16)
    assert m.pooled_gain(v,'research',0,'matched')==pytest.approx(3/16)
    with pytest.raises(ValueError):m.pooled_gain(v,'research',0,'wrong score')
