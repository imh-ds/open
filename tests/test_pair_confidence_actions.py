from pathlib import Path
import yaml


def test_actions_matrix_stage_and_budget():
    root=Path(__file__).parents[1];p=root/'.github/workflows/pair-confidence.yml'
    assert p.exists(),'confidence workflow missing'
    w=yaml.safe_load(p.read_text());events=w.get('on',w.get(True))
    assert events['workflow_dispatch']['inputs']['stage']['options']==['calibration','confirmation']
    assert w['jobs']['prepare']['if']=="github.event_name == 'workflow_dispatch'"
    worker=w['jobs']['study'];assert worker['strategy']['max-parallel']==256 and not worker['strategy']['fail-fast']
    assert worker['timeout-minutes']==30
    assert w['jobs']['prepare']['timeout-minutes']==15 and w['jobs']['aggregate']['timeout-minutes']==60
    assert 180*worker['timeout-minutes']+2*(15+60)==5550
    upload=next(s for s in worker['steps'] if 'upload-artifact' in s.get('uses',''))
    assert upload['if']=='always()' and upload['with']['include-hidden-files']
    assert 'inputs.stage' in upload['with']['name']
    assert 'pair-confidence-setup' in str(w)


def test_setup_pins_runtime_and_checks_requested_stage():
    p=Path(__file__).parents[1]/'.github/actions/pair-confidence-setup/action.yml'
    assert p.exists(),'confidence setup missing'
    a=yaml.safe_load(p.read_text());assert a['inputs']['stage']['required']
    command=' '.join(s.get('run','') for s in a['runs']['steps'])
    assert 'scikit-learn==1.3.0' in command and 'scipy==1.11.2' in command
    assert 'pair_confidence_study.py freeze --stage' in command and '--write' not in command
