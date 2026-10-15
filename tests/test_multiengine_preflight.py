"""No fitting in the preflight's local execution/audit tests."""
import copy
import gzip
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


@pytest.fixture(autouse=True)
def no_fits(monkeypatch):
    import recursive_preview_runner as r
    monkeypatch.setattr(r,'NATIVE_ASSESS',lambda *a,**k:pytest.fail('local model fit'))
    monkeypatch.setenv('GITHUB_RUN_ID','999')
    monkeypatch.setenv('GITHUB_SHA','source')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')


def test_six_fixture_identities_defaults_and_approved_budget():
    import multiengine_preflight as m
    cases=m.specs();assert len(cases)==6 and len({s['id'] for s in cases})==6
    assert {s['engine'] for s in cases}=={'random_forest','gbm','lightgbm'}
    assert {s['index'] for s in cases}=={160,240}
    for s in cases:
        config=m.settings(s)
        assert config.learner.library==(s['engine'],)
        assert config.stability.n_resamples==100 and config.thresholds.stability==.70
        assert config.performance_cv.folds==5 and config.performance_cv.repeats==2
        assert config.tuning_cv.folds==3
    v=m.manifest()
    assert v['reserved_minutes']==586 and v['ceiling']==600 and v['unallocated']==14
    assert v['runtime']['lightgbm']=='4.6.0'


def risk_fixture(engine):
    import multiengine_recursive_audit as a
    from open.layer_artifacts import decode
    from open.layer_types import LocalAssessment
    from open.learners.trees import GRIDS
    import recursive_preview_audit as old
    import recursive_preview_design as d
    from dataclasses import replace
    spec=next(s for s in d.specs() if s['index']==161)
    archive=d.ROOT/'examples/results/2026-10-15-chain-verification-38690088093/raw/cases/chain-case-161/raw/case.zip'
    with old.Store(archive,spec['id']).zip as z:
        payload=json.loads(gzip.decompress(z.read('cases/'+spec['id']+'/checkpoints/d0-Y/000.json.gz')))
    assessment=decode(LocalAssessment,payload['assessment'])
    risk=copy.deepcopy(next(r['risk'] for r in payload['risks'] if r['origin']=='analysis' and len(r['risk']['predictors'])==len(assessment.candidates)))
    risk['learner']=engine
    for fold in risk['folds']:fold['fitted_parameters']={k:v[0] for k,v in GRIDS[engine].items()}
    risk['tuning_model_fit_count']=len(risk['folds'])*4*3
    config=d.settings(spec);config=replace(config,learner=replace(config.learner,library=(engine,)))
    return a,risk,d.generate(spec).iloc[list(assessment.sample.sampled_positions)],assessment,config


@pytest.mark.parametrize('engine',['random_forest','gbm','lightgbm'])
def test_engine_grid_and_count_checks(engine):
    a,risk,data,assessment,config=risk_fixture(engine)
    args=(data,assessment.sample,assessment.target,assessment.candidates,tuple(risk['predictors']))
    assert a.audit_risk(risk,*args,engine=engine)>0
    assert a.expected_tuning(risk,config,True)==len(risk['folds'])*4*3
    bad=copy.deepcopy(risk);bad['tuning_model_fit_count']+=1
    with pytest.raises(ValueError,match='tuning count'):
        a.verify_tuning(bad,config,True)
    bad=copy.deepcopy(risk);bad['learner']='elastic_net'
    with pytest.raises(ValueError,match='engine'):
        a.audit_risk(bad,*args,engine=engine)
    bad=copy.deepcopy(risk);bad['folds'][0]['fitted_parameters']={'alpha':1,'l1_ratio':.5}
    with pytest.raises(ValueError,match='parameters'):
        a.audit_risk(bad,*args,engine=engine)
    bad=copy.deepcopy(risk);bad['folds'][0]['fitted_parameters'][next(iter(bad['folds'][0]['fitted_parameters']))]=999
    with pytest.raises(ValueError,match='parameters'):
        a.audit_risk(bad,*args,engine=engine)


def test_unknown_and_wrong_structure_have_different_statuses():
    import multiengine_preflight as m
    spec=m.specs()[0]
    r=m.summarize([dict(spec=spec,status='complete',structure=dict(exact=False,incorrect=True))],{})
    assert r['planned']==6 and r['complete']==1 and r['unknown']==5
    assert r['exact']==0 and not r['execution_verified'] and r['scientific_qualification'] is False
    with pytest.raises(ValueError,match='duplicate'):
        m.summarize([dict(spec=spec,status='complete')]*2,{})


def test_invalid_case_and_attempt_do_not_fit(tmp_path,monkeypatch):
    import multiengine_preflight as m
    monkeypatch.setattr(m,'checked_manifest',lambda:{})
    with pytest.raises(ValueError,match='identity'):
        m.run_one('foreign',tmp_path/'out',None)
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    with pytest.raises(ValueError,match='first'):
        m.run_one(m.specs()[0]['id'],tmp_path/'out',None)
    assert not (tmp_path/'out').exists()


def test_authorization_rejects_changed_source_and_manifest(tmp_path):
    import multiengine_preflight as m
    a=dict(manifest=m.digest({}),github_run='999',github_sha='source',github_attempt='1')
    a['digest']=m.digest(a);path=tmp_path/'auth.json';m.write_json(path,a)
    m.authorization(path,{})
    a['github_sha']='changed';a['digest']=m.digest({k:v for k,v in a.items() if k!='digest'})
    m.write_json(path,a)
    with pytest.raises(ValueError,match='authorization'):m.authorization(path,{})


def test_wrong_chain_is_reported_without_calling_it_execution_failure():
    import multiengine_preflight as m
    rows=[dict(spec=s,status='complete',structure=dict(exact=False,incorrect=True)) for s in m.specs()]
    r=m.summarize(rows,{})
    assert r['complete']==6 and r['execution_verified'] and r['exact']==0
    assert r['scientific_qualification'] is False


def test_empty_aggregate_saves_all_unknowns(tmp_path,monkeypatch):
    import multiengine_preflight as m
    monkeypatch.setattr(m,'checked_manifest',lambda:{})
    monkeypatch.setattr(m,'transport',lambda:dict(verified=True))
    r=m.aggregate(tmp_path/'missing',tmp_path/'out')
    assert r['complete']==0 and r['unknown']==6 and not r['execution_verified']
    assert (tmp_path/'out/audit.json').exists()


def test_workflow_bounds_and_same_candidate_rows():
    import multiengine_preflight as m
    import yaml
    w=yaml.safe_load((m.ROOT/m.WORKFLOW).read_text())
    assert w['jobs']['case']['timeout-minutes']==90
    assert w['jobs']['case']['strategy']['max-parallel']==6
    assert w['jobs']['aggregate']['timeout-minutes']==30
    hashes=m.manifest()['data_sha256']
    for n in (80,200):
        assert len({hashes[s['id']] for s in m.specs() if s['n']==n})==1

def test_hard_timeout_preserves_auditable_prefix_without_sealed_receipt(tmp_path):
    import multiengine_preflight as m
    import zipfile
    spec=m.specs()[0];config=m.settings(spec);data=m.old.q.design.generate(spec)
    value={'runtime':{'lightgbm':'4.6.0'}}
    control=dict(spec=spec,manifest=m.digest(value),github_run='999',github_sha='source',github_attempt='1',status='STARTED')
    receipt=dict(control,context=m.case_context(spec,data,config,value),config=m.portable(config),
        python='3.11.9',module='C:/site-packages/open/__init__.py',runtime=value['runtime'],fit_cost_unknown=True)
    archive=tmp_path/'case.zip';prefix='cases/'+spec['id']+'/'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(prefix+'receipt.json',json.dumps(receipt))
        z.writestr(prefix+'config.json',json.dumps(m.portable(config)))
        z.writestr(prefix+'data.csv',data.to_csv(index=False,lineterminator='\n'))
        z.writestr(prefix+'checkpoints/d0-Y/000.json.started.json',json.dumps({'status':'STARTED'}))
    row=m.audit_one((spec,archive,value,control))
    assert row['status']=='STARTED',row
    assert row['assessments']==0 and row['unknown_cost_attempts']==1
    assert row['fit_cost_unknown'] and not row['receipt_inventory_sealed']
    assert m.summarize([row],value)['unknown']==6

def test_failed_assessment_does_not_discard_completed_partial_checkpoint(tmp_path,monkeypatch):
    import multiengine_preflight as m
    import zipfile
    from dataclasses import replace
    from open.layer_artifacts import portable
    _,_,_,original,_=risk_fixture('random_forest')
    spec=m.specs()[0];config=m.settings(spec);data=m.old.q.design.generate(spec)
    value={'runtime':{'lightgbm':'4.6.0'}}
    control=dict(spec=spec,manifest=m.digest(value),github_run='999',github_sha='source',github_attempt='1',status='FAILED')
    context=m.case_context(spec,data,config,value)
    receipt=dict(control,status='incomplete',context=context,config=portable(config),
        python='3.11.9',module='C:/site-packages/open/__init__.py',runtime=value['runtime'],fit_cost_unknown=True)
    files={'config.json':json.dumps(portable(config)).encode(),'data.csv':data.to_csv(index=False,lineterminator='\n').encode()}
    plans=m.make_layer_sample_plans(data,config)
    checked=[]
    def check(payload,sampled,assessment,local,c):
        checked.append(assessment.sample.run_index)
        return dict(observation_checks=7,outer_fits=2,tuning_fits=4)
    monkeypatch.setattr(m.audit,'audit_checkpoint',check)
    for i in (0,1):
        a=replace(original,sample=plans[i],status='completed' if i==0 else 'failed',local=original.local if i==0 else None,
            failure_type=None if i==0 else 'RuntimeError',failure_message=None if i==0 else 'Synthetic software-test failure')
        local=m.digest(dict(case=context,depth=a.depth,target=a.target,candidates=a.candidates,config=config))
        payload=dict(context=local,assessment=portable(a),risks=[]);payload['digest']=m.digest(payload)
        name=f'checkpoints/d{a.depth}-{a.target}/{i:03}.json'
        files[name+'.gz']=gzip.compress(json.dumps(payload).encode())
        attempt=dict(context=local,depth=a.depth,target=a.target,run_index=i,seed=a.sample.seed,status='STARTED',fit_cost_unknown=True)
        files[name+'.started.json']=json.dumps(attempt).encode()
        attempt.update(status='completed' if i==0 else 'FAILED',fit_cost_unknown=i!=0)
        files[name+'.finished.json']=json.dumps(attempt).encode()
    import hashlib
    receipt['files']={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    files['receipt.json']=json.dumps(receipt).encode()
    archive=tmp_path/'case.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for n,v in files.items():z.writestr('cases/'+spec['id']+'/'+n,v)
    row=m.audit_one((spec,archive,value,control))
    assert row['status']=='incomplete',row
    assert checked==[0] and row['assessments']==1
    assert row['unknown_cost_attempts']==1 and row['observation_checks']==7
