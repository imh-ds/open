import sys,importlib.util
from pathlib import Path
import yaml
import pytest

ROOT=Path(__file__).parents[1]


def workflow():
    p=ROOT/'.github/workflows/prospective-interaction.yml'
    assert p.exists(),'prospective Actions workflow missing'
    return yaml.safe_load(p.read_text())


def test_pinned_staged_caps_and_partial_upload():
    v=workflow();jobs=v['jobs'];assert jobs['prepare']['timeout-minutes']==15 and jobs['aggregate']['timeout-minutes']==60
    worker=jobs['study'];assert worker['timeout-minutes']==20 and worker['strategy']['max-parallel']==256
    assert worker['strategy']['fail-fast'] is False and worker['runs-on']=='windows-2022'
    assert worker['steps'][1]['timeout-minutes']==4 and worker['steps'][2]['timeout-minutes']==12
    assert worker['steps'][3]['if']=='always()'
    assert jobs['prepare']['if']=="github.event_name == 'workflow_dispatch'"
    assert all(v['env'][k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    setup=ROOT/'.github/actions/prospective-interaction-setup/action.yml';assert setup.exists()
    text=setup.read_text();assert '3.11.9' in text and 'scipy==1.11.2' in text
    assert 'freeze' in text and 'prospective_interaction_study.py' in text


def test_sparse_source_guard_and_crlf(tmp_path):
    workflow();sys.path.insert(0,str(ROOT/'examples'));import prospective_interaction_study as m
    sources=m.sources();assert 'examples/prospective_interaction_report.py' in sources
    assert m.old.threshold_digest()=='755a6fe65f908c8b3f61a6ca4fa2d5aa5622a912519b53238b470773dbecbf17'
    p=tmp_path/'converted';p.write_bytes((ROOT/'examples/prospective_interaction_rules.py').read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    import hashlib
    assert hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==sources['examples/prospective_interaction_rules.py']
    for name in sources:assert (ROOT/name).exists()
    text=(ROOT/'.github/workflows/prospective-interaction.yml').read_text()
    assert 'examples/prospective_interaction_assurance.json' in text
    assert 'merge-multiple: false' in text and 'examples/results/' not in text
