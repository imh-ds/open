"""Timing transport must not retry fits, lose failed identities, or accept foreign work."""
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


@pytest.fixture(autouse=True)
def actions_identity(monkeypatch):
    import time_scenario_validation as t
    import recursive_preview_runner as runner
    def forbidden(*args,**kwargs):
        pytest.fail('regression test reached a real scientific fitting boundary')
    monkeypatch.setattr(t,'run_case',forbidden)
    monkeypatch.setattr(runner,'NATIVE_ASSESS',forbidden)
    monkeypatch.setenv('GITHUB_RUN_ID','12345')
    monkeypatch.setenv('GITHUB_SHA','test-source')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','1')


def test_only_original_four_identities_can_start(tmp_path, monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t, 'checked_manifest', lambda: t.manifest())
    with pytest.raises(ValueError, match='identity'):
        t.run_case_timing(160, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_case_interruption_preserves_control_and_archive_without_retry(tmp_path, monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t, 'checked_manifest', lambda: t.manifest())
    def interrupted(spec, directory, manifest, *, seconds):
        assert spec['index'] == 0 and seconds == 2880
        t.write_json(directory/'receipt.json', dict(status='incomplete', fit_cost_unknown=True))
        raise TimeoutError('unfinished batch')
    monkeypatch.setattr(t, 'run_case', interrupted)
    out = tmp_path/'out'
    with pytest.raises(TimeoutError):
        t.run_case_timing(0, out)
    report = t.load_json(out/'control.json')
    assert report['status'] == 'FAILED' and report['index'] == 0
    assert report['scientific_gate_accepted'] is False
    assert (out/'raw/case.zip').is_file()
    with pytest.raises(FileExistsError):
        t.run_case_timing(0, out)


def test_manifest_tamper_and_changed_sources_fail_before_fitting(monkeypatch):
    import time_scenario_validation as t
    frozen = t.manifest()
    original = t.load_json
    monkeypatch.setattr(t, 'runtime_check', lambda m: None)
    monkeypatch.setattr(t, 'load_json', lambda p: copy.deepcopy(frozen) if p == t.MANIFEST else original(p))
    # Keep real producer reads while presenting an edited new freeze.
    frozen['case_indices'] = [1,80,320,400]
    with pytest.raises(ValueError, match='manifest'):
        t.checked_manifest()


def test_chain_proof_reuses_actual_archive_without_any_fitting(monkeypatch):
    import time_scenario_validation as t
    import recursive_preview_runner as runner
    def forbidden(*args, **kwargs):
        pytest.fail('saved chain evidence must never be fitted')
    monkeypatch.setattr(runner, 'NATIVE_ASSESS', forbidden)
    rows = t.audit_saved_chains(t.manifest())
    assert set(rows) == {'160','240'}
    assert sum(row['assessments'] for row in rows.values()) == 600
    assert sum(row['observation_checks'] for row in rows.values()) == 1076653
    assert all(row['structure']['exact'] for row in rows.values())


def test_chain_archive_tamper_is_rejected_before_audit(monkeypatch):
    import time_scenario_validation as t
    manifest = t.manifest()
    manifest['chains']['archive_sha256'] = 'edited'
    with pytest.raises(ValueError, match='archive'):
        t.audit_saved_chains(manifest)


def test_missing_cases_keep_all_four_rows(tmp_path, monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t, 'checked_manifest', lambda: t.manifest())
    monkeypatch.setattr(t, 'audit_saved_chains', lambda m: {})
    result = t.aggregate_timing(tmp_path/'inputs', tmp_path/'aggregate')
    assert result['status'] == 'INCOMPLETE_TIMING_ONLY'
    assert [r['index'] for r in result['cases']] == [0,80,320,400]
    assert all(r['status'] == 'missing' for r in result['cases'])
    assert result['scientific_gate_accepted'] is False
    assert result['audit_model_fits'] == 0


def test_foreign_control_or_duplicate_artifacts_rejected(tmp_path, monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t, 'checked_manifest', lambda: t.manifest())
    monkeypatch.setattr(t, 'audit_saved_chains', lambda m: {})
    t.write_json(tmp_path/'inputs/foreign/control.json', {'index':160})
    with pytest.raises(ValueError, match='foreign'):
        t.aggregate_timing(tmp_path/'inputs', tmp_path/'aggregate')
    result = t.load_json(tmp_path/'aggregate/aggregate.json')
    assert result['status'] == 'AUDIT_FAILED' and len(result['cases']) == 4


def test_actions_rerun_cannot_start_duplicate_scientific_fits(tmp_path,monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t,'checked_manifest',lambda:t.manifest())
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    def forbidden(*args,**kwargs):
        pytest.fail('rerun reached the scientific fitting boundary')
    monkeypatch.setattr(t,'run_case',forbidden)
    with pytest.raises(ValueError,match='attempt'):
        t.run_case_timing(0,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_duplicate_controls_cannot_count_as_two_cases(tmp_path,monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t,'checked_manifest',lambda:t.manifest())
    value=t.manifest()
    control=dict(index=0,timing_manifest=t.design.digest(value),github_run='12345',
        github_sha='test-source',github_attempt='1',scientific_gate_accepted=False)
    for name in ['a','b']:
        t.write_json(tmp_path/'inputs'/name/'control.json',control)
    with pytest.raises(ValueError,match='duplicate'):
        t.aggregate_timing(tmp_path/'inputs',tmp_path/'aggregate')


def test_other_run_control_is_not_accepted_as_current_evidence(tmp_path,monkeypatch):
    import time_scenario_validation as t
    monkeypatch.setattr(t,'checked_manifest',lambda:t.manifest())
    control=dict(index=0,timing_manifest=t.design.digest(t.manifest()),github_run='other-run',
        github_sha='test-source',github_attempt='1',scientific_gate_accepted=False)
    t.write_json(tmp_path/'inputs/a/control.json',control)
    with pytest.raises(ValueError,match='source/run'):
        t.aggregate_timing(tmp_path/'inputs',tmp_path/'aggregate')


def test_workflow_covers_frozen_inputs_and_bounded_always_aggregate():
    import fnmatch
    import time_scenario_validation as t
    wf = yaml.safe_load((t.ROOT/'.github/workflows/scenario-validation-timing.yml').read_text())
    jobs = wf['jobs']
    assert jobs['timing']['strategy']['matrix']['index'] == [0,80,320,400]
    assert jobs['timing']['strategy']['fail-fast'] is False
    assert 4*jobs['timing']['timeout-minutes']+jobs['aggregate']['timeout-minutes']+jobs['register']['timeout-minutes'] == 256
    assert 'always()' in jobs['aggregate']['if']
    paths = jobs['timing']['steps'][0]['with']['sparse-checkout'].splitlines()
    assert all(any(fnmatch.fnmatch(n,p) for p in paths) for n in t.manifest()['sources'])


def test_preservation_rebuilds_corrupt_archive_from_original_files(tmp_path):
    import zipfile
    import time_scenario_validation as t
    t.write_json(tmp_path/'cases/original/receipt.json',dict(status='STARTED'))
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/case.zip').write_bytes(b'interrupted ZIP')
    t.pack(tmp_path)
    with zipfile.ZipFile(tmp_path/'raw/case.zip') as z:
        assert z.testzip() is None
        assert z.read('cases/original/receipt.json') == (tmp_path/'cases/original/receipt.json').read_bytes()


def test_packing_interruption_does_not_publish_partial_zip(tmp_path,monkeypatch):
    import time_scenario_validation as t
    t.write_json(tmp_path/'cases/original/receipt.json',dict(status='STARTED'))
    def interrupted(cases,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(b'partial ZIP')
        raise RuntimeError('packing interruption')
    monkeypatch.setattr(t,'archive_cases',interrupted)
    with pytest.raises(RuntimeError,match='packing interruption'):
        t.pack(tmp_path)
    assert not (tmp_path/'raw/case.zip').exists()


def test_bad_case_does_not_prevent_other_case_or_chain_audits(tmp_path,monkeypatch):
    import zipfile
    import time_scenario_validation as t
    monkeypatch.setattr(t,'checked_manifest',lambda:t.manifest())
    seen=[]
    def chains(value):
        seen.append('chains')
        return {}
    monkeypatch.setattr(t,'audit_saved_chains',chains)
    def audit(spec,store,old):
        seen.append(spec['index'])
        return dict(status='incomplete')
    monkeypatch.setattr(t,'audit_case',audit)
    for index in (0,80):
        folder=tmp_path/'inputs'/str(index)
        control=dict(index=index,timing_manifest=t.design.digest(t.manifest()),github_run='12345',
            github_sha='test-source',github_attempt='1',scientific_gate_accepted=False,status='FAILED')
        t.write_json(folder/'control.json',control)
        (folder/'raw').mkdir()
        with zipfile.ZipFile(folder/'raw/case.zip','w') as z:
            prefix='cases/'+t.reference()['cases'][index]['id']+'/'
            z.writestr(prefix+'receipt.json','invalid JSON' if index==0 else t.design.json.dumps(control))
    result=t.aggregate_timing(tmp_path/'inputs',tmp_path/'aggregate')
    assert [r['status'] for r in result['cases']] == ['audit_failed','incomplete','missing','missing']
    assert seen == [80,'chains']
    assert result['status'] == 'AUDIT_FAILED'
