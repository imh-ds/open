import importlib.util
from pathlib import Path
import pytest


def module():
    spec = importlib.util.spec_from_file_location('selection_investigation',
        Path(__file__).parents[1]/'examples/selection_investigation.py')
    value = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_diagnostic_identities_and_variants_are_fixed_and_unique():
    study = module()
    tasks = study.diagnostic_tasks()
    assert len(tasks) == len({t['id'] for t in tasks}) == 74
    assert sum(t['kind'] == 'model' for t in tasks) == 64
    assert sum(t['kind'] == 'loco' for t in tasks) == 10
    assert {t['scope'] for t in tasks if t['kind']=='model'} == {'full', 'sample0'}


def test_shard_ranges_are_disjoint_and_skip_existing_work():
    study = module()
    comparison = dict(depth=1,target='x1',candidates=['x2','x3'],missing=list(range(16,100)))
    tasks = study.assessment_tasks([comparison])
    assert len(tasks) == 9
    ids = [i for t in tasks for i in t['indices']]
    assert sorted(ids) == list(range(16,100))
    assert len(ids) == len(set(ids))
    assert max(len(t['indices']) for t in tasks) <= 10


def test_manifest_rejects_changed_diagnostic_and_completion_rules():
    study = module()
    manifest = study.make_manifest()
    study.validate_manifest(manifest)
    manifest['tasks'][0]['variant'] = 'unplanned'
    with pytest.raises(ValueError,match='manifest'):
        study.validate_manifest(manifest)


def test_artifact_merge_rejects_conflicting_duplicates(tmp_path):
    study = module()
    a,b,d = tmp_path/'a',tmp_path/'b',tmp_path/'out'
    a.mkdir();b.mkdir()
    (a/'x.json').write_text('same');(b/'x.json').write_text('same')
    study.merge_files([a,b],d)
    assert (d/'x.json').read_text() == 'same'
    (b/'x.json').write_text('different')
    with pytest.raises(ValueError,match='duplicate'):
        study.merge_files([b],d)


def test_known_product_feature_is_deterministic_and_does_not_use_outcome():
    import pandas as pd
    study = module()
    data = pd.DataFrame({'y':[100.,-100.],'x1':[2.,3.],'x2':[4.,5.]})
    assert study.features(data,'product').iloc[:,0].tolist() == [8.,15.]
    data.y = 0
    assert study.features(data,'product').iloc[:,0].tolist() == [8.,15.]


def test_frontier_requires_all_planned_assessments_before_advancing(tmp_path):
    from dataclasses import replace
    import pandas as pd
    from layer_helpers import make_local,make_assessment,make_config
    study = module()
    config=make_config(n_resamples=2)
    data=pd.DataFrame({n:range(20) for n in ('y','a','b')})
    plans=study.pilot.engine.make_layer_sample_plans(data,config)
    context=study.pilot.checkpoint_context(data,0,'y',('a','b'),config)
    for i in (0,):
        a=make_assessment(make_local('y',('a','b'),unique=('a',)),index=i)
        study.pilot.save_checkpoint(tmp_path/'d0-y'/f'{i:03d}.json.gz',replace(a,sample=plans[i]),context)
    pending=study.pending_comparisons(data,config,tmp_path)
    assert pending == [dict(depth=0,target='y',candidates=['a','b'],missing=[1])]


def test_failed_assessment_is_not_replaced_or_used_to_advance(tmp_path):
    from dataclasses import replace
    import pandas as pd
    from layer_helpers import make_local,make_assessment,make_config
    study=module(); config=make_config(n_resamples=2)
    data=pd.DataFrame({n:range(20) for n in ('y','a','b')})
    plans=study.pilot.engine.make_layer_sample_plans(data,config)
    context=study.pilot.checkpoint_context(data,0,'y',('a','b'),config)
    a=make_assessment(make_local('y',('a','b'),unique=('a',)),index=0)
    a=replace(a,sample=plans[0],status='failed',local=None,failure_type='RuntimeError',failure_message='failure')
    study.pilot.save_checkpoint(tmp_path/'d0-y'/'000.json.gz',a,context)
    with pytest.raises(ValueError,match='failed assessment retained'):
        study.pending_comparisons(data,config,tmp_path)


def test_complete_layer_uses_production_consensus_before_next_comparison(tmp_path):
    from dataclasses import replace
    import pandas as pd
    from layer_helpers import make_local,make_assessment,make_config
    study=module();config=make_config(n_resamples=2)
    data=pd.DataFrame({n:range(20) for n in ('y','a','b')})
    plans=study.pilot.engine.make_layer_sample_plans(data,config)
    context=study.pilot.checkpoint_context(data,0,'y',('a','b'),config)
    for i in range(2):
        a=make_assessment(make_local('y',('a','b'),unique=('a',)),index=i)
        study.pilot.save_checkpoint(tmp_path/'d0-y'/f'{i:03d}.json.gz',replace(a,sample=plans[i]),context)
    assert study.pending_comparisons(data,config,tmp_path) == [dict(depth=1,target='a',candidates=['b'],missing=[0,1])]


def test_new_workflow_uses_account_capacity_and_six_hour_timeout():
    import yaml
    root=Path(__file__).parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/selection-investigation.yml').read_text())
    jobs=workflow['jobs']
    for name in ('diagnostic','layer-1','layer-2'):
        assert jobs[name]['strategy']['max-parallel']==256
        assert jobs[name]['strategy']['fail-fast'] is False
        assert jobs[name]['timeout-minutes']==360
    assert jobs['layer-2']['needs']=='aggregate-1'
    assert jobs['aggregate-2']['needs']==['aggregate-1','layer-2']


def test_only_diagnostic_grid_has_deeper_trees():
    study=module()
    assert study.model_grid('gbm_deeper')['max_depth']==[2,3,5]
    assert study.model_grid('gbm_shipped')['max_depth']==[2,3]


def test_partial_shard_preserves_checkpoint_and_withholds_next_layer(tmp_path):
    study=module()
    output=tmp_path/'output'; downloads=tmp_path/'downloads'
    task=dict(id='d1-x1-010',depth=1,target='x1',candidates=['x2'],indices=[16,17])
    study.pilot.write_json(output/'pending.json',[task])
    folder=downloads/'completion-1-d1-x1-010'
    study.pilot.write_json(folder/'receipt.json',dict(task=task,manifest=study.pilot.fingerprint(study.make_manifest())))
    checkpoint=folder/'checkpoints/d1-x1/016.json.gz'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    with pytest.raises(ValueError,match='partial checkpoints preserved'):
        study.merge_shards(output,downloads)
    assert (output/'continuation'/study.CHAIN/'checkpoints/d1-x1/016.json.gz').read_bytes()==b'checkpoint'


def test_foreign_shard_files_are_not_merged(tmp_path):
    study=module()
    output=tmp_path/'output';downloads=tmp_path/'downloads'
    task=dict(id='d1-x1-010',depth=1,target='x1',candidates=['x2'],indices=[16])
    study.pilot.write_json(output/'pending.json',[task])
    folder=downloads/'completion-1-d1-x1-010'
    study.pilot.write_json(folder/'receipt.json',dict(task=task,manifest=study.pilot.fingerprint(study.make_manifest())))
    checkpoint=folder/'checkpoints/d1-x1/017.json.gz'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'foreign')
    with pytest.raises(ValueError,match='foreign assessment files'):
        study.merge_shards(output,downloads)
    assert not (output/'continuation').exists()


def test_worker_failure_still_leaves_other_checkpoints_in_uploaded_directory(tmp_path,monkeypatch):
    from concurrent.futures import Future
    study=module()
    task=dict(id='d1-x1-010',depth=1,target='x1',candidates=['x2'],indices=[16,17])
    monkeypatch.setattr(study,'stage',lambda output:[task])
    def assessment(data,depth,target,candidates,plan,config,path,context):
        if plan.run_index==16:
            raise RuntimeError('worker error')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'saved')
        return 'computed'
    class Pool:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def submit(self,func,*args):
            future=Future()
            try: future.set_result(func(*args))
            except Exception as exc: future.set_exception(exc)
            return future
    monkeypatch.setattr(study,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(study.pilot,'one_assessment',assessment)
    with pytest.raises(RuntimeError,match='worker error'):
        study.run_assessments(task,tmp_path)
    assert (tmp_path/'shard/receipt.json').exists()
    assert (tmp_path/'shard/checkpoints/d1-x1/017.json.gz').read_bytes()==b'saved'


def test_actual_linear_diagnostic_audit_rejects_fabricated_prediction(tmp_path):
    import pandas as pd
    study=module()
    task=next(t for t in study.diagnostic_tasks() if t['kind']=='model' and
        t['case']=='interaction-n80-p5' and t['scope']=='full' and t['variant']=='linear')
    study.run_task(task,tmp_path)
    assert study.audit_task(task,tmp_path)['status']=='complete'
    path=tmp_path/'diagnostics'/task['id']/'predictions.csv'
    predictions=pd.read_csv(path,float_precision='round_trip')
    predictions.loc[0,'predicted']+=1
    predictions.to_csv(path,index=False)
    with pytest.raises(AssertionError):
        study.audit_task(task,tmp_path)
