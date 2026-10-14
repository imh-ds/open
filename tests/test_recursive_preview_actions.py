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
