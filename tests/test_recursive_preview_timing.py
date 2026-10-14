from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def test_timing_workflow_removes_case_limit_within_remaining_budget():
    import time_recursive_preview as t
    workflow = yaml.safe_load((t.ROOT/'.github/workflows/recursive-preview-timing.yml').read_text())
    assert workflow['jobs']['register']['timeout-minutes'] == 1
    assert workflow['jobs']['timing']['timeout-minutes'] == 82
    assert 9917+1+82 == 10000
    m = t.manifest()
    assert m['case_seconds'] is None
    assert m['case_indices'] == [160,240]
    assert m['new_fixture_fits'] == 0
    import fnmatch
    patterns = workflow['jobs']['timing']['steps'][0]['with']['sparse-checkout'].splitlines()
    assert all(any(fnmatch.fnmatch(n,p) for p in patterns) for n in m['sources'])


def test_restore_real_completed_work_has_no_refits(tmp_path,monkeypatch):
    import time_recursive_preview as t
    import recursive_preview_runner as runner
    from open.layer_stability import make_layer_sample_plans
    def forbid(*a,**k):
        pytest.fail('saved assessment was refitted')
    monkeypatch.setattr(runner,'NATIVE_ASSESS',forbid)
    directory = tmp_path/'case'
    receipt, audit = t.restore_partial(t.PRODUCER/'raw/recursive-preview-preflight-raw/preflight-all-attempts.zip',directory,t.manifest())
    assert audit['assessments'] == 144
    assert audit['observation_checks'] == 174477
    assert receipt['status'] == 'incomplete'
    spec = t.original_manifest()['cases'][160]
    data,config = t.design.generate(spec),t.design.settings(spec)
    plans = make_layer_sample_plans(data,config)
    for path in sorted(directory.rglob('*.json.gz')):
        payload = runner.read_gzip(path)
        assessment = payload['assessment']
        assert runner.one_assessment(data,assessment['depth'],assessment['target'],
            tuple(assessment['candidates']),plans[assessment['sample']['run_index']],
            config,path,payload['context']) == 'reused'
    assert runner.file_hashes(directory) == receipt['files']


@pytest.mark.parametrize('field',['archive_sha256','receipt_sha256'])
def test_restore_rejects_edited_producer_fingerprints(tmp_path,field):
    import time_recursive_preview as t
    m = t.manifest()
    m['producer'][field] = 'edited'
    with pytest.raises(ValueError,match='fingerprint'):
        t.restore_partial(t.PRODUCER/'raw/recursive-preview-preflight-raw/preflight-all-attempts.zip',tmp_path/'case',m)


def test_resume_removes_deadline_and_keeps_prior_receipt_on_failure(tmp_path,monkeypatch):
    import math
    import time_recursive_preview as t
    directory = tmp_path/'cases/case'
    directory.mkdir(parents=True)
    prior = dict(status='incomplete',seconds=903.781,context='original',files={},failure='old timeout')
    def stop(data,config,checkpoints,context,deadline):
        assert context == 'original'
        assert math.isinf(deadline)
        raise RuntimeError('simulated interruption; no fits')
    monkeypatch.setattr(t,'fit_recorded',stop)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        t.resume_case(directory,prior,t.manifest(),tmp_path)
    assert t.load_json(tmp_path/'provenance/original-case160-receipt.json') == prior
    current = t.load_json(directory/'receipt.json')
    assert current['status'] == 'incomplete'
    assert current['fit_cost_unknown'] is True
    assert current['prior_seconds'] == 903.781
    assert current['seconds'] >= current['prior_seconds']


def test_fit_cost_breakdown_separates_reuse_from_new_work():
    import time_recursive_preview as t
    assert t.split_fit_counts(dict(outer_fits=12200,tuning_fits=508000),
        dict(outer_fits=12191,tuning_fits=507000)) == dict(
            reused=dict(outer_fits=12191,tuning_fits=507000),
            new=dict(outer_fits=9,tuning_fits=1000))
    with pytest.raises(ValueError,match='fit count'):
        t.split_fit_counts(dict(outer_fits=1,tuning_fits=1),dict(outer_fits=2,tuning_fits=0))
