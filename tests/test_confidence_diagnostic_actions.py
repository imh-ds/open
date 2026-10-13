from pathlib import Path
import yaml
import importlib.util,sys


def workflow():
    root=Path(__file__).parents[1];p=root/'.github/workflows/confidence-diagnostic.yml'
    assert p.exists(), 'diagnostic workflow missing'
    return root,yaml.safe_load(p.read_text())


def test_six_shards_budget_and_no_automatic_fits():
    root,w=workflow();assert w['jobs']['prepare']['if']=="github.event_name == 'workflow_dispatch'"
    assert w['jobs']['study']['strategy']['max-parallel']==256
    assert not w['jobs']['study']['strategy']['fail-fast']
    assert 6*w['jobs']['study']['timeout-minutes']+w['jobs']['prepare']['timeout-minutes']+w['jobs']['aggregate']['timeout-minutes']==125
    upload=next(s for s in w['jobs']['study']['steps'] if 'upload-artifact' in s.get('uses',''))
    assert upload['if']=='always()' and upload['with']['include-hidden-files']
    command=' '.join(s.get('run','') for job in w['jobs'].values() for s in job['steps'])
    assert '--write' not in command


def test_sparse_checkout_covers_every_frozen_source_and_excludes_archives():
    root,w=workflow();sys.path.insert(0,str(root/'examples'));import confidence_diagnostic_study as m
    import fnmatch
    for job in w['jobs'].values():
        checkout=next(s for s in job['steps'] if 'actions/checkout@' in s.get('uses',''))
        assert checkout['with']['sparse-checkout-cone-mode'] is False
        patterns=checkout['with']['sparse-checkout'].splitlines()
        for p in m.sources():assert any(fnmatch.fnmatchcase(p,pattern) for pattern in patterns),p
        for p in ('examples/study_manifests/confidence-diagnostic.json','examples/study_manifests/pair-confidence-calibration.json','examples/study_manifests/pair-confidence-confirmation.json','examples/study_manifests/pair-confidence-thresholds.json'):
            assert any(fnmatch.fnmatchcase(p,x) for x in patterns),p
        assert not any(fnmatch.fnmatchcase('examples/results/2026-10-13-confidence-confirmation/raw-evidence.zip.part000',x) for x in patterns)


def test_dedicated_setup_checks_both_frozen_contexts():
    root,w=workflow();p=root/'.github/actions/confidence-diagnostic-setup/action.yml'
    assert p.exists(), 'diagnostic setup missing'
    a=yaml.safe_load(p.read_text());command=' '.join(s.get('run','') for s in a['runs']['steps'])
    assert 'scikit-learn==1.3.0' in command and 'confidence_diagnostic_study.py freeze' in command
    assert '--write' not in command
