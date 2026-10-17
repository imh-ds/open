"""Frozen pilot controls and evidence handling, without scientific model fits."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))


def test_approved_cases_and_paired_generator():
    import lightgbm_effect_pilot as pilot
    small, large = pilot.cases(200), pilot.cases(300)
    assert len(small) == len(large) == 30
    assert len({s['id'] for s in small + large}) == 60
    seeds = sorted(s['analysis_seed'] for s in small + large)
    assert min(b-a for a,b in zip(seeds, seeds[1:])) >= 1000
    for a,b in zip(small,large,strict=True):
        assert a['generator_seed'] == b['generator_seed']
        pd.testing.assert_frame_equal(pilot.generate(a), pilot.generate(b).iloc[:200])
        x1,e2,ey,_,_ = np.random.default_rng(a['generator_seed']).normal(size=(5,300))
        effect = a['effect_percent']/100
        expected = effect*x1 + np.sqrt(1-effect**2)*e2
        np.testing.assert_array_equal(pilot.generate(b).X2, expected)
        np.testing.assert_array_equal(pilot.generate(b).Y,
                                     effect*expected+np.sqrt(1-effect**2)*ey)
        config = pilot.settings(a)
        assert config.learner.library == ('lightgbm',)
        assert config.stability.n_resamples == 100
        assert config.thresholds.stability == .70
        assert (config.performance_cv.folds,config.performance_cv.repeats,config.tuning_cv.folds)==(5,2,3)
    with pytest.raises(ValueError):
        pilot.cases(80)


def test_manifest_and_sources_reject_tampering():
    import lightgbm_effect_pilot as pilot
    manifest = pilot.manifest(200)
    assert manifest['reserved_runner_minutes'] == 1981
    assert manifest['runner_minute_ceiling'] == 2000
    assert len(manifest['data_sha256']) == 30
    assert pilot.validate_manifest(manifest) == pilot.digest(manifest)
    manifest['cases'][0]['effect_percent'] = 60
    with pytest.raises(ValueError, match='manifest'):
        pilot.validate_manifest(manifest)


def test_expired_budget_preserves_unknown_without_fitting(tmp_path,monkeypatch):
    import lightgbm_effect_pilot as pilot
    manifest=pilot.manifest(200)
    monkeypatch.setenv('GITHUB_RUN_ID','123')
    monkeypatch.setenv('GITHUB_SHA','a'*40)
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')
    monkeypatch.setenv('GITHUB_EVENT_NAME','workflow_dispatch')
    monkeypatch.setattr(pilot,'runtime_check',lambda m: None)
    monkeypatch.setattr(pilot.runner,'run_case',lambda *a,**kw: pytest.fail('expired case fitted'))
    path=tmp_path/'manifest.json'
    pilot.write_json(path,manifest)
    out=tmp_path/'case'
    result=pilot.execute(path,manifest['cases'][0]['id'],out,job_started_at=0,now=100000)
    assert result['status']=='incomplete' and result['fit_cost_unknown']
    assert (out/'raw/case.zip').is_file()
    with pytest.raises(FileExistsError):
        pilot.execute(path,manifest['cases'][0]['id'],out,job_started_at=0,now=100000)


def test_missing_results_remain_unknown_in_every_denominator():
    import lightgbm_effect_pilot as pilot
    planned=pilot.cases(200)
    spec=planned[0]
    row=dict(case_id=spec['id'],status='complete',structure=dict(exact=True,incorrect=False),
             edges=[['X1','X2'],['X2','Y']])
    summary=pilot.summarize(planned,[row])
    assert summary[0]['planned']==10 and summary[0]['complete']==1 and summary[0]['unknown']==9
    assert summary[0]['direct_links']==summary[0]['indirect_links']==1
    assert summary[0]['exact_networks']==1
    assert summary[1]['unknown']==summary[2]['unknown']==10
    with pytest.raises(ValueError,match='duplicate'):
        pilot.summarize(planned,[row,row])
    with pytest.raises(ValueError,match='foreign'):
        pilot.summarize(planned,[dict(row,case_id='foreign')])


def test_first_attempt_dispatch_only(monkeypatch):
    import lightgbm_effect_pilot as pilot
    monkeypatch.setenv('GITHUB_EVENT_NAME','push')
    with pytest.raises(ValueError):
        pilot.first_attempt()


def test_workflow_reservation_and_defaults():
    import lightgbm_effect_pilot as pilot
    import yaml
    root=Path(__file__).parents[1]
    w=yaml.safe_load((root/'.github/workflows/lightgbm-effect-pilot.yml').read_text())
    assert w['on']['workflow_dispatch']['inputs']['sample_size']['default']=='200'
    assert w['jobs']['case']['strategy']['max-parallel']==20
    assert w['jobs']['case']['timeout-minutes']==60
    assert 30*60+sum(w['jobs'][j]['timeout-minutes'] for j in ['register','prepare','aggregate','publish'])==1981
    assert w['jobs']['aggregate']['if']=='${{ always() && needs.prepare.result == \'success\' }}'
    assert pilot.RUNTIME['lightgbm']=='4.6.0'


def test_candidate_snapshot_uses_planned_not_conditional_denominator():
    import lightgbm_effect_pilot as pilot
    spec=pilot.cases(200)[0]
    target=dict(target='X2',assessments=list(range(100)),candidates_evidence=[
        dict(source='X1',classes={'PROXIMAL_UNIQUE':dict(selected=72,planned=100,completed=100)}),
        dict(source='Z1',classes={})])
    payload={'layer_evidence.json':dict(layers=[dict(depth=1,evidence=[target])])}
    rows=pilot.candidate_rows(spec,payload)
    assert rows[0]['proximal_selected']==72
    assert rows[0]['planned']==rows[0]['completed']==100
    assert rows[0]['truth'] and not rows[1]['truth']
    assert rows[1]['proximal_selected']==0 and rows[1]['planned']==100


def test_aggregate_with_missing_and_incomplete_archives_preserves_them(tmp_path,monkeypatch):
    import lightgbm_effect_pilot as pilot
    manifest=pilot.manifest(200)
    for key,value in dict(GITHUB_RUN_ID='123',GITHUB_SHA='a'*40,
                          GITHUB_RUN_ATTEMPT='1',GITHUB_EVENT_NAME='workflow_dispatch').items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(pilot,'runtime_check',lambda m: None)
    path=tmp_path/'authorization/manifest.json'
    pilot.write_json(path,manifest)
    pilot.write_json(path.parent/'producer.json',dict(run='123',source='a'*40,attempt=1,
                     manifest=pilot.digest(manifest)))
    spec=manifest['cases'][0]
    download=tmp_path/'download'
    pilot.execute(path,spec['id'],download/('lgbm-case-'+spec['id']),job_started_at=0,now=100000)
    report=pilot.aggregate(path,download,tmp_path/'report')
    assert not report['execution_complete']
    assert report['cells'][0]['unknown']==10
    assert report['cases'][0]['status']=='incomplete'
    assert report['cases'][1]['status']=='audit_failed'
    inventory=pilot.load(tmp_path/'report/inventory.json')
    assert len(inventory)==1
    assert len(inventory[0]['members'])==1
    assert (tmp_path/'report/retention-snapshots.csv').is_file()


def test_interrupted_case_gets_separate_packaging_step(tmp_path,monkeypatch):
    import lightgbm_effect_pilot as pilot
    manifest=pilot.manifest(200)
    for key,value in dict(GITHUB_RUN_ID='123',GITHUB_SHA='a'*40,
                          GITHUB_RUN_ATTEMPT='1',GITHUB_EVENT_NAME='workflow_dispatch').items():
        monkeypatch.setenv(key,value)
    path=tmp_path/'authorization/manifest.json'
    pilot.write_json(path,manifest)
    pilot.write_json(path.parent/'producer.json',dict(run='123',source='a'*40,attempt=1,
                     manifest=pilot.digest(manifest)))
    spec=manifest['cases'][0]
    out=tmp_path/'interrupted'
    raw=out/'cases'/spec['id']
    raw.mkdir(parents=True)
    (raw/'partial-checkpoint.json').write_text('{"status":"STARTED"}')
    control=pilot.preserve_interrupted(path,spec['id'],out)
    assert control['status']=='incomplete' and control['fit_cost_unknown']
    assert (out/'raw/case.zip').is_file()
    assert pilot.preserve_interrupted(path,spec['id'],out)==control


def test_incomplete_case_still_checks_raw_source(tmp_path,monkeypatch):
    import lightgbm_effect_pilot as pilot
    import zipfile
    manifest=pilot.manifest(200)
    for key,value in dict(GITHUB_RUN_ID='123',GITHUB_SHA='a'*40,
                          GITHUB_RUN_ATTEMPT='1',GITHUB_EVENT_NAME='workflow_dispatch').items():
        monkeypatch.setenv(key,value)
    spec=manifest['cases'][0]
    folder=tmp_path/'case'; archive=folder/'raw/case.zip'
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('cases/'+spec['id']+'/receipt.json',
                   '{"github_run":"wrong","github_sha":"wrong","github_attempt":"1"}')
    pilot.write_json(folder/'control.json',dict(case_id=spec['id'],spec=spec,
        manifest=pilot.digest(manifest),source='a'*40,run='123',attempt=1,
        status='incomplete',fit_cost_unknown=True,archive_sha256=pilot.evidence.sha256_file(archive)))
    with pytest.raises(ValueError,match='producer'):
        pilot.audit_saved(spec,folder,manifest)


@pytest.mark.parametrize('job',['prepare','case','aggregate','publish'])
def test_sparse_source_freeze_is_import_complete(tmp_path,job):
    import shutil
    import subprocess
    import yaml
    root=Path(__file__).parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/lightgbm-effect-pilot.yml').read_text())
    checkout=next(s for s in workflow['jobs'][job]['steps'] if s.get('uses','').startswith('actions/checkout@'))
    for pattern in checkout['with']['sparse-checkout'].splitlines():
        for source in root.glob(pattern):
            if source.is_file():
                dest=tmp_path/source.relative_to(root)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,dest)
            elif source.is_dir() and pattern=='src/**':
                shutil.copytree(source,tmp_path/source.relative_to(root),dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__'))
    code="import sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path[:0]=[str(p/'src'),str(p/'examples')]; import lightgbm_effect_pilot as m; assert len(m.manifest(200)['cases'])==30; assert len(m.manifest(300)['cases'])==30"
    result=subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)],cwd=tmp_path,
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
