"""Continuation contract tests; real fitting is prohibited."""
import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))

@pytest.fixture(autouse=True)
def no_fits(monkeypatch):
    import recursive_preview_runner as r
    monkeypatch.setattr(r,'NATIVE_ASSESS',lambda *a,**k:pytest.fail('local model fitting'))
    monkeypatch.setenv('GITHUB_RUN_ID','999');monkeypatch.setenv('GITHUB_SHA','source');monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')

def test_only_two_incomplete_rf_cases_and_fixed_budget():
    import continue_tree_preflight as c
    assert len(c.pending())==2 and {s['engine'] for s in c.pending()}=={'random_forest'}
    assert {c.producer_row(s)['assessments'] for s in c.pending()}=={192,216}
    v=c.manifest();assert v['reserved_minutes']==346 and v['case_minutes']==150
    assert v['soft_seconds']==132*60 and v['compute_minutes']==135
    assert v['previous_reserved_minutes']==586 and v['prior_ceiling']==600
    assert not v['scientific_qualification']

def test_foreign_or_retry_does_not_download_or_fit(tmp_path,monkeypatch):
    import continue_tree_preflight as c
    monkeypatch.setattr(c,'checked_manifest',lambda:{})
    monkeypatch.setattr(c,'download_artifacts',lambda *a,**k:pytest.fail('unexpected download'))
    with pytest.raises(ValueError,match='identity'):c.run_one('foreign',tmp_path/'out',None)
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    with pytest.raises(ValueError,match='first'):c.run_one(c.pending()[0]['id'],tmp_path/'out',None)

def test_changed_original_archive_is_rejected(tmp_path):
    import continue_tree_preflight as c
    archive=tmp_path/'case.zip';archive.write_bytes(b'changed')
    with pytest.raises(ValueError,match='fingerprint'):c.validate_original(c.pending()[0],archive,{})

def test_resume_preserves_all_original_checkpoint_bytes_on_timeout(tmp_path,monkeypatch):
    import continue_tree_preflight as c
    import hashlib,zipfile
    spec=c.pending()[0];data=c.m.old.q.design.generate(spec);config=c.m.settings(spec)
    contents={'data.csv':data.to_csv(index=False,lineterminator='\n').encode(),
        'config.json':json.dumps(c.m.portable(config)).encode(),'checkpoints/saved-prefix.bin':b'unchanged checkpoint fixture'}
    prior=dict(spec=spec,context=c.m.case_context(spec,data,config,c.load_json(c.m.MANIFEST)),
        manifest=c.m.digest(c.load_json(c.m.MANIFEST)),config=c.m.portable(config),status='incomplete',seconds=4538.906,
        files={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()})
    archive=tmp_path/'producer.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for n,v in dict(contents,**{'receipt.json':json.dumps(prior).encode()}).items():z.writestr('cases/'+spec['id']+'/'+n,v)
    monkeypatch.setattr(c,'validate_original',lambda *a,**k:c.producer_row(spec))
    directory=tmp_path/'cases'/spec['id']
    prior,reused=c.restore(spec,archive,{},directory)
    hashes=dict(prior['files'])
    def stop(data,config,checkpoint_directory,context,deadline,workers):
        assert context==prior['context'] and config==c.m.settings(spec) and workers==4
        raise TimeoutError('software-test interruption')
    monkeypatch.setattr(c,'fit_recorded',stop)
    with pytest.raises(TimeoutError):c.resume(spec,directory,prior,reused,{'soft_seconds':1})
    receipt=c.load_json(directory/'receipt.json')
    assert receipt['status']=='incomplete' and receipt['github_run']=='999'
    assert receipt['prior_files_unchanged'] and receipt['reused_assessments']==216
    assert all(c.sha(directory/n)==h for n,h in hashes.items())
    assert receipt['context']==prior['context'] and receipt['manifest']==prior['manifest']

def test_changed_scientific_context_cannot_start_a_continuation(tmp_path,monkeypatch):
    import continue_tree_preflight as c
    monkeypatch.setattr(c,'fit_recorded',lambda *a,**k:pytest.fail('fit despite changed context'))
    with pytest.raises(ValueError,match='scientific context'):
        c.resume(c.pending()[0],tmp_path,{'context':'changed'},c.producer_row(c.pending()[0]),{'soft_seconds':1})
    assert not (tmp_path/'receipt.json').exists()

def test_workflow_budget_and_two_case_transport():
    import continue_tree_preflight as c
    import yaml
    w=yaml.safe_load((c.ROOT/c.WORKFLOW).read_text())
    assert w['jobs']['case']['timeout-minutes']==150 and w['jobs']['case']['strategy']['max-parallel']==2
    assert sum(w['jobs'][j]['timeout-minutes'] for j in ('register','prepare','aggregate'))+2*150==346
    step=next(s for s in w['jobs']['case']['steps'] if s.get('name')=='Recorded production recursion; no automatic fit retry')
    assert step['timeout-minutes']==135
    assert w['jobs']['case']['steps'][-1]['with']['name']=='tree-continued-${{ matrix.id }}'

def test_report_hash_is_portable_across_windows_line_endings(tmp_path):
    import continue_tree_preflight as c
    a=tmp_path/'lf.json';b=tmp_path/'crlf.json'
    a.write_bytes(b'{"complete":4}\n');b.write_bytes(b'{"complete":4}\r\n')
    assert c.report_hash(a)==c.report_hash(b)
