import importlib.util
from pathlib import Path

import pytest


def study():
    path=Path(__file__).parents[1]/'examples/pair_validation_study.py'
    assert path.exists(),'validation study runner missing'
    spec=importlib.util.spec_from_file_location('pair_validation_study',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_new_disjoint_design_and_shards():
    m=study()
    tasks=m.tasks()
    assert len(tasks)==240 and len({t['id'] for t in tasks})==240
    assert len(m.shards())==120
    assert [t for s in m.shards() for t in m.shard_tasks(s)]==tasks
    assert min(t['case']['seed'] for t in tasks)>=68000000
    with pytest.raises(ValueError):m.shard_tasks('120')


def test_frozen_manifest_roundtrip(tmp_path,monkeypatch):
    m=study();monkeypatch.setattr(m,'MANIFEST',tmp_path/'manifest.json')
    m.freeze(check=False)
    assert m.freeze()==m.prior.read(m.MANIFEST)


def test_partial_audit_cannot_nominate_rule(tmp_path):
    m=study();r=m.aggregate(tmp_path)
    assert r['planned']==240 and r['complete']==0 and r['missing']==240
    assert r['acceptance']=='none: development only'
    assert len(r['cells'])==24
    assert all(c['planned']==10 and c['complete']==0 for c in r['cells'])
    assert r['cutoffs'][0]['unknown_incidental_pair_count']==100
    assert r['cutoffs'][0]['unassessed_incidental_pair_count']==100


def test_error_only_identity_and_reason_remain_visible(tmp_path):
    m=study();t=m.tasks()[0]
    m.prior.write(tmp_path/t['id']/'error.json',dict(task=t,context=m.context(t),error='controlled failure'))
    r=m.aggregate(tmp_path)
    assert r['missing']==240 and r['failed_tasks']==[dict(id=t['id'],error='controlled failure')]


def test_native_validation_task_refits_and_bound_receipt(tmp_path):
    m=study();task=m.tasks()[0]
    r=m.run_task(task,tmp_path)
    assert len(r['folds'])==3 and r['selection_scope']=='outer training only'
    assert all(f['validation']['additional_fits']<=100 for f in r['folds'])
    m.check_result(task,tmp_path/task['id'],refit=True)
    p=tmp_path/task['id']/'result.json';value=m.prior.read(p)
    value['folds'][0]['validation']['baseline_mse']+=1
    m.prior.write(p,value)
    assert m.aggregate(tmp_path)['complete']==0


def test_missing_candidate_is_unknown():
    m=study()
    folds=[dict(validation=dict(pairs=[])) for _ in range(3)]
    v=m.pair_summary(folds,['x1','x2'])
    assert v['assessed_folds']==0 and v['complete_gain'] is None


def test_validator_rejects_changed_seed_grid_and_fit_count():
    import copy
    from open.pair_validation import PairValidationConfig,validate_pairs
    m=study();task=m.tasks()[0];c=task['case']
    X,y,_,_=m.prior.generate(c,c['n'],c['seed'])
    train,test=m.prior.partitions(c)[0]
    v=validate_pairs(X,y,tuple(train.tolist()),tuple(test.tolist()),(('x1','x2'),),
        config=PairValidationConfig(seed=c['split_seed']))
    for key,value in [('config',{**v['config'],'seed':0}),('additional_fits',0),
                       ('baseline_parameters',{'curves__n_knots':7,'ridge__alpha':1.})]:
        broken=copy.deepcopy(v);broken[key]=value
        with pytest.raises(ValueError):m.check_validation(broken,X,y,train,test,False,c['split_seed'])
