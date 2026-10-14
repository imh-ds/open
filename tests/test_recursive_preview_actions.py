import importlib
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
ROOT = Path(__file__).parents[1]


def test_workflow_manual_only_and_exact_timeout_budget():
    path = ROOT/'.github/workflows/recursive-preview-validation.yml'
    assert path.exists(), 'missing staged workflow'
    workflow = yaml.safe_load(path.read_text())
    jobs = workflow['jobs']
    assert jobs['register']['timeout-minutes'] == 1
    assert jobs['preflight']['timeout-minutes'] == 120
    assert jobs['shard']['timeout-minutes'] == 40
    assert jobs['shard']['strategy']['max-parallel'] == 20
    assert jobs['prepare']['timeout-minutes'] == 15
    assert jobs['aggregate']['timeout-minutes'] == 60
    assert 'workflow_dispatch' in jobs['preflight']['if']
    assert 'workflow_dispatch' in jobs['prepare']['if']
    assert 1+120+2*15+2*60+(80+160)*40 == 9871
    text = path.read_text()
    assert 'persist-credentials: false' in text
    assert 'recursive_preview_*.py' in text
    import fnmatch
    for job in ('preflight','prepare','shard','aggregate'):
        checkout = jobs[job]['steps'][0]['with']
        assert any(fnmatch.fnmatch('examples/study_manifests/recursive-preview-validation.json',p)
                   for p in checkout['sparse-checkout'].splitlines())
    assert 'github.token' in text and 'actions: read' in text


def test_frozen_source_or_runtime_edit_is_rejected(tmp_path, monkeypatch):
    m = importlib.import_module('recursive_preview_actions')
    import recursive_preview_design as d
    manifest = d.manifest()
    d.validate_manifest(manifest)
    edited = dict(manifest, workers=2)
    with pytest.raises(ValueError, match='manifest'):
        d.validate_manifest(edited)
    assert m.indices('noise', 79) == [158,159]
    assert m.indices('signal', 0) == [160,161]
    with pytest.raises(ValueError): m.indices('signal', 160)


def test_archive_preserves_exact_cases_without_fitting(tmp_path):
    import zipfile
    m = importlib.import_module('recursive_preview_actions')
    case = tmp_path/'case'; case.mkdir()
    (case/'data.csv').write_bytes(b'1,2\n')
    out = tmp_path/'saved.zip'
    m.archive_cases([case], out)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ['cases/case/data.csv']
        assert z.read(z.namelist()[0]) == b'1,2\n'


def test_preflight_gate_cannot_be_accepted_from_status_alone():
    m = importlib.import_module('recursive_preview_actions')
    with pytest.raises(ValueError, match='preflight'):
        m.validate_preflight({'status':'PASS'}, {})


def test_original_preflight_run_must_include_successful_raw_uploads():
    m = importlib.import_module('recursive_preview_actions')
    report = {'github_run':'123', 'github_sha':'source'}
    steps = [dict(name='Run actions/upload-artifact@sha', conclusion='success',
                  started_at='2026-10-14T00:00:00Z', completed_at='2026-10-14T00:00:02Z') for _ in range(2)]
    def api(path):
        if path.endswith('/jobs?per_page=100'):
            return {'jobs':[dict(name='preflight', conclusion='success', steps=steps)]}
        if path.endswith('/artifacts'):
            return {'artifacts':[dict(name=n,expired=False,size_in_bytes=200)
                for n in ('recursive-preview-preflight-control','recursive-preview-preflight-raw')]}
        return dict(id=123,status='completed',conclusion='success',head_sha='source',run_attempt=1)
    assert m.verify_preflight_transport(report, api)['raw_uploaded_bytes'] == 200
    steps[-1]['conclusion'] = 'failure'
    with pytest.raises(ValueError, match='upload'):
        m.verify_preflight_transport(report, api)


def test_reuse_requires_original_archive_and_case_fingerprints(tmp_path, monkeypatch):
    import hashlib
    m = importlib.import_module('recursive_preview_actions')
    monkeypatch.setattr(m, 'validate_preflight', lambda *args:True)
    case = tmp_path/'case'; case.mkdir()
    (case/'receipt.json').write_text('{}')
    archive = tmp_path/'raw.zip'; m.archive_cases([case], archive)
    control = dict(raw_sha256={'preflight-cases.zip':'wrong'},
                   planned_cases={'160':{'receipt_sha256':'wrong'}})
    spec = dict(index=160,id='case')
    with pytest.raises(ValueError, match='archive differs'):
        m.restore_preflight(archive,tmp_path/'restored',spec,{},control)
    control['raw_sha256']['preflight-cases.zip'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='case differs'):
        m.restore_preflight(archive,tmp_path/'restored',spec,{},control)
