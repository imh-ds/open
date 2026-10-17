"""Matched reference pilot: same data/splits, different learners, fixed budget."""
import copy
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def test_references_match_original_data_and_every_nonlearner_setting():
    import linear_effect_pilot as pilot
    original=pilot.load(pilot.ORIGINAL)
    specs=pilot.cases(200)
    assert len(specs)==60
    assert len({s['id'] for s in specs})==60
    for s in specs:
        source=next(c for c in original['cases'] if c['id']==s['paired_lightgbm_case_id'])
        assert s['analysis_seed']==source['analysis_seed']
        pd.testing.assert_frame_equal(pilot.generate(s),pilot.original_generate(source))
        expected=copy.deepcopy(original['configurations'][source['id']])
        expected['learner']['library']=[s['engine']]
        assert pilot.portable(pilot.settings(s))==expected
    with pytest.raises(ValueError):pilot.cases(300)


def test_ols_audit_policy_is_zero_search_and_restores_legacy_globals():
    import linear_reference_audit as audit
    import multiengine_recursive_audit as legacy
    from open.config import AnalysisConfig,OpenConfig,LearnerConfig
    cfg=OpenConfig(analysis=AnalysisConfig('Y',7),learner=LearnerConfig(library=('ols',)))
    fn=legacy.valid_parameters
    with audit.policy():
        assert legacy.valid_parameters({},'ols')
        assert not legacy.valid_parameters({'alpha':1},'ols')
        assert legacy.expected_tuning({'folds':[1,2]},cfg,True)==0
    assert legacy.valid_parameters is fn


def test_workflow_caps_and_sources_match_approved_allocation():
    import linear_effect_pilot as pilot
    import yaml
    workflow=yaml.safe_load((pilot.ROOT/'.github/workflows/linear-effect-pilot.yml').read_text())
    assert 30*sum(pilot.CASE_CAPS.values())+sum(workflow['jobs'][j]['timeout-minutes'] for j in ['register','prepare','aggregate','publish'])==3871
    m=pilot.manifest(200)
    assert m['runner_minute_ceiling']==4000 and m['reserved_runner_minutes']==3871
    assert len(m['cases'])==60
    assert pilot.validate_manifest(m)==pilot.digest(m)
    assert workflow['jobs']['case']['timeout-minutes']=='${{ matrix.case_cap_minutes }}'
    assert workflow['jobs']['case']['strategy']['max-parallel']==20


def test_summary_keeps_engines_separate_and_missing_cases_unknown():
    import linear_effect_pilot as pilot
    planned=pilot.cases(200)
    spec=next(s for s in planned if s['engine']=='ols' and s['effect_percent']==30)
    row=dict(case_id=spec['id'],status='complete',edges=[['X2','Y']],
             structure=dict(exact=False,incorrect=False))
    result=pilot.summarize(planned,[row])
    assert len(result)==6 and all(c['planned']==10 for c in result)
    assert sum(c['unknown'] for c in result)==59
    assert next(c for c in result if c['engine']=='ols' and c['effect']==.3)['direct_links']==1


@pytest.mark.parametrize('job',['prepare','case','aggregate','publish'])
def test_sparse_reference_checkout_can_freeze_controls(tmp_path,job):
    import shutil
    import subprocess
    import yaml
    import linear_effect_pilot as pilot
    workflow=yaml.safe_load((pilot.ROOT/'.github/workflows/linear-effect-pilot.yml').read_text())
    checkout=next(s for s in workflow['jobs'][job]['steps'] if s.get('uses','').startswith('actions/checkout@'))
    for pattern in checkout['with']['sparse-checkout'].splitlines():
        for source in pilot.ROOT.glob(pattern):
            if source.is_file():
                dest=tmp_path/source.relative_to(pilot.ROOT)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,dest)
            elif source.is_dir() and pattern=='src/**':
                shutil.copytree(source,tmp_path/source.relative_to(pilot.ROOT),dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__'))
    code="import sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path[:0]=[str(p/'src'),str(p/'examples')];import linear_effect_pilot as m;assert len(m.manifest(200)['cases'])==60"
    result=subprocess.run([sys.executable,'-I','-c',code,str(tmp_path)],cwd=tmp_path,
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr


def test_expired_reference_case_is_preserved_without_fitting(tmp_path,monkeypatch):
    import linear_effect_pilot as pilot
    for key,value in dict(GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_RUN_ATTEMPT='1',
                          GITHUB_RUN_ID='123',GITHUB_SHA='a'*40).items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(pilot,'runtime_check',lambda m:None)
    monkeypatch.setattr(pilot.runner,'run_case',lambda *a,**k:pytest.fail('expired budget fitted'))
    manifest=pilot.manifest(200); path=tmp_path/'manifest.json'
    pilot.write_json(path,manifest)
    out=tmp_path/'output'
    result=pilot.execute(path,manifest['cases'][0]['id'],out,job_started_at=0,now=100000)
    assert result['status']=='incomplete' and (out/'raw/case.zip').is_file()
    row=pilot.audit_saved(manifest['cases'][0],out,manifest)
    assert row['status']=='incomplete' and row['scientific_fits']==0
