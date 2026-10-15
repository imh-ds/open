import sys
from pathlib import Path
import pytest
import yaml
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'examples'))


def test_workflow_budget_and_sparse_sources():
    path=ROOT/'.github/workflows/realistic-interaction.yml'
    assert path.exists(),'realistic workflow missing'
    workflow=yaml.safe_load(path.read_text());jobs=workflow['jobs'];worker=jobs['study']
    assert worker['timeout-minutes']==20 and worker['strategy']['max-parallel']==256
    assert jobs['prepare']['timeout-minutes']==15 and jobs['aggregate']['timeout-minutes']==60
    assert [s['timeout-minutes'] for s in worker['steps'] if 'timeout-minutes' in s]==[4,12,3,1]
    assert workflow['concurrency']['cancel-in-progress'] is False
    assert jobs['prepare']['if']=="github.event_name == 'workflow_dispatch'"
    assert 'prospective_interaction_assurance.json' in worker['steps'][0]['with']['sparse-checkout']
    assert 'examples/*protocol.md' in worker['steps'][0]['with']['sparse-checkout']


def test_realistic_freeze_and_legacy_preserved(tmp_path,monkeypatch):
    from study_fixture_helpers import isolate_prospective
    import realistic_interaction_study as s
    isolate_prospective(s.previous,tmp_path/'previous',monkeypatch)
    assert (ROOT/'.github/workflows/realistic-interaction.yml').exists(),'realistic workflow missing'
    s.previous.freeze();m=s.manifest()
    assert m['planned_fits']==9964800 and m['planned_verification_refits']==288000
    assert m['seed_audit']['collisions']==0
    assert 'examples/realistic_interaction_truth.py' in m['sources']
    assert all(s.sources()[p]==digest for p,digest in s.previous.sources().items())
