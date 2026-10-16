import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))


def test_chain_manifest_preserves_settings_pairs_and_fresh_identities():
    import tree_recursive_signal_study as study
    value = study.build_manifest()
    assert len(value["cases"]) == 330
    assert value["stage"] == "signal-chain"
    assert value["runner_minute_ceiling"] == 30000
    assert value["reserved_runner_minutes"] == 28536
    old = json.loads((study.ROOT / "examples/study_manifests/tree-recursive-null.json").read_text())
    for field in ("generator_seed", "analysis_seed"):
        assert not {c[field] for c in old["cases"]} & {c[field] for c in value["cases"]}
    bases = sorted({c['analysis_seed'] for c in value['cases']})
    assert len(bases) == 110
    assert all(b-a >= 1000 for a,b in zip(bases,bases[1:]))
    for n in (80, 200):
        cases = study.cases_for_n(value, n)
        assert len(cases) == 165
        for repetition in range(1, 56):
            paired = [c for c in cases if c["repetition"] == repetition]
            assert len({value["data_sha256"][c["id"]] for c in paired}) == 1
        config = study.settings(cases[0])
        assert config.stability.n_resamples == 100






def test_signal_actions_have_cell_audit_and_combined_publication():
    import tree_recursive_signal_actions as actions
    assert callable(actions.aggregate)
    assert callable(actions.combine)
    assert callable(actions.publish)


@pytest.mark.parametrize("job_name", ["prepare", "case-n80", "case-n200", "audit", "combine", "publish"])
def test_signal_workflow_sparse_checkout_is_import_complete(tmp_path, job_name):
    import shutil
    import subprocess
    import yaml
    root = Path(__file__).parents[1]
    workflow = yaml.safe_load((root / '.github/workflows/tree-recursive-signal-qualification.yml').read_text())
    job = workflow['jobs'][job_name]
    checkout = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/checkout@'))
    for pattern in checkout['with']['sparse-checkout'].splitlines():
        if pattern == 'src/**':
            shutil.copytree(root / 'src', tmp_path / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        else:
            target = tmp_path / pattern
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / pattern, target)
    code = "import sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path[:0]=[str(p/'src'),str(p/'examples')]; import tree_recursive_signal_actions as a; import recursive_preview_actions; import qualification_artifacts; m,d=a.checked_manifest(); a.verify_sources(m)"
    result = subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert workflow['jobs']['publish']['permissions']['contents'] == 'write'
    assert workflow['permissions']['contents'] == 'read'


def test_signal_prepares_both_waves_and_seals_expired_case_without_fitting(tmp_path, monkeypatch):
    import tree_recursive_signal_actions as actions
    for name, value in {"GITHUB_RUN_ATTEMPT":"1", "GITHUB_EVENT_NAME":"workflow_dispatch",
                        "GITHUB_RUN_ID":"123", "GITHUB_SHA":"a"*40}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(actions, "runtime_check", actions.verify_sources)
    monkeypatch.setattr(actions.runner, "run_case", lambda *a, **k: pytest.fail("fit during no-fit preparation/deadline"))
    result = actions.prepare(tmp_path/'authorization')
    assert all(len(json.loads(value)['include']) == 165 for value in result.values())
    matrix = json.loads(result['matrix80'])['include']
    row = actions.run_case(matrix[0]['id'], tmp_path/'authorization/wave-n80', tmp_path/'expired',
                           job_started_at=0, expected_sample_size=80)
    assert row['status'] == 'incomplete'
    assert row['fit_cost_unknown'] is True
    assert (tmp_path/'expired/raw/case.zip').is_file()


def test_workflow_caps_and_required_sources_match_frozen_budget():
    import yaml
    import tree_recursive_signal_study as study
    root = Path(__file__).parents[1]
    w = yaml.safe_load((root/'.github/workflows/tree-recursive-signal-qualification.yml').read_text())
    caps = 55*sum(sum(by_n.values()) for by_n in study.CASE_CAPS.values())
    overhead = 1+30+6*120+15+270
    assert caps+overhead == 28536 < 30000
    assert w['jobs']['audit']['timeout-minutes'] == 120
    assert w['jobs']['publish']['timeout-minutes'] == 270
    assert w['jobs']['combine']['timeout-minutes'] == 15
    for name in ['prepare','case-n80','case-n200','audit','combine','publish']:
        checkout = next(s for s in w['jobs'][name]['steps'] if s.get('uses','').startswith('actions/checkout@'))
        sparse = checkout['with']['sparse-checkout'].splitlines()
        assert all(source in sparse for source in study.SOURCE_FILES)


def test_signal_cell_audits_and_combination_keep_all_330_identities(tmp_path, monkeypatch):
    import shutil
    import qualification_artifacts
    import tree_recursive_signal_actions as a
    manifest, digest = a.checked_manifest()
    for name, value in {"GITHUB_RUN_ATTEMPT":"1", "GITHUB_EVENT_NAME":"workflow_dispatch",
                        "GITHUB_RUN_ID":"123", "GITHUB_SHA":"a"*40}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(a, "checked_manifest", lambda *args, **kwargs: (manifest, digest))
    monkeypatch.setattr(a, "verify_transport", lambda *args: {"verified": True, "errors": []})
    downloads = []
    monkeypatch.setattr(qualification_artifacts, "download_artifacts", lambda run,names,path,**kwargs: downloads.extend(names))
    monkeypatch.setattr(a, "audit_case_artifact", lambda expected,*args: dict(expected,status="complete",
                        fit_cost_unknown=False,archive_sha256="c"*64,false_network=False,exact_network=True))
    directories = {}
    for n in (80, 200):
        for engine in a.study.ENGINES:
            out = tmp_path/f'{engine}-{n}'
            doc = a.aggregate(tmp_path/'downloads',out,engine=engine,sample_size=n)
            assert doc['cell']['complete'] == 55
            directories[f'tree-signal-report-{engine}-n{n}'] = out/'cell-report.json'
    assert len(downloads) == len(set(downloads)) == 330

    def get_reports(run,names,path,**kwargs):
        for name in names:
            target = Path(path)/name/'cell-report.json'
            target.parent.mkdir(parents=True)
            shutil.copyfile(directories[name],target)
    monkeypatch.setattr(qualification_artifacts,'download_artifacts',get_reports)
    result = a.combine(tmp_path/'reports',tmp_path/'combined')
    assert len(result['cells']) == 6 and all(result['engine_eligibility'].values())
    assert sum(c['complete'] for c in result['cells']) == 330
    bad = json.loads(next(iter(directories.values())).read_text())
    bad['source_sha'] = 'b'*40
    next(iter(directories.values())).write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='identity'):
        a.combine(tmp_path/'reports-bad',tmp_path/'combined-bad')
