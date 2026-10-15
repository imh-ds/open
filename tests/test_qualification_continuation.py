"""Execution-only continuation tests: real model fitting is forbidden."""
import copy
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))


@pytest.fixture(autouse=True)
def no_fits(monkeypatch):
    import recursive_preview_runner as runner
    def forbidden(*a, **k):
        pytest.fail('real model fit attempted locally')
    monkeypatch.setattr(runner, 'NATIVE_ASSESS', forbidden)
    monkeypatch.setenv('GITHUB_RUN_ID', '999')
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')


def test_exact_same_identity_partition_and_budget():
    import continue_qualification as c
    m = c.manifest()
    assert len(c.pending()) == 38 and len(c.completed()) == 72
    assert {s['n'] for s in c.pending()} == {80}
    assert sorted(s['index'] for s in c.pending()+c.completed()) == sorted(
        s['index'] for s in c.q.stage_specs('noise'))
    assert m['original_manifest'] == c.design.digest(c.original())
    assert m['reserved_total'] == 4863 and m['remaining_ceiling'] == 5137
    assert m['continuation_minutes'] == 38*60+15+60+1
    assert m['soft_seconds'] == 39*60


def test_changed_producer_partition_is_rejected(monkeypatch):
    import continue_qualification as c
    edited = copy.deepcopy(c.producer())
    edited['cases'][0]['status'] = 'missing'
    monkeypatch.setattr(c, 'producer', lambda: edited)
    with pytest.raises(ValueError, match='partition'):
        c.pending()


def test_completed_case_cannot_be_dispatched(tmp_path):
    import continue_qualification as c
    with pytest.raises(ValueError, match='identity'):
        c.run_one(c.completed()[0]['index'], tmp_path/'missing.zip', tmp_path/'out', None)
    assert not (tmp_path/'out').exists()


def test_producer_archive_edit_is_rejected_before_restore(tmp_path):
    import continue_qualification as c
    archive = tmp_path/'bad.zip'
    archive.write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive'):
        c.validate_archive(c.pending()[0], archive, replay=True)


@pytest.mark.parametrize('index', [23, 25, 32])
def test_real_deadline_late_complete_and_interrupted_prefixes_are_reusable(index):
    import continue_qualification as c
    root = c.q.ROOT/'examples/results/2026-10-14-recursive-qualification/noise-38655036215/raw'
    if not (root/f'case-{index}.zip').exists():
        pytest.skip('large producer archives are retained in the evidence checkout')
    spec = next(s for s in c.pending() if s['index'] == index)
    checked = c.validate_archive(spec, root/f'case-{index}.zip', replay=True)
    assert checked['assessments'] >= 80
    assert checked['observation_checks'] > 0
    assert checked['original_status'] == 'incomplete'
    assert checked['unknown_attempts'] == 0


def test_resume_keeps_original_context_and_receipt_on_timeout(tmp_path, monkeypatch):
    import continue_qualification as c
    spec = c.pending()[0]
    directory = tmp_path/'cases'/spec['id']
    directory.mkdir(parents=True)
    prior = dict(context='original-context', status='incomplete', seconds=590,
                 files={}, failure='old timeout')
    def interrupted(data, config, checkpoints, context, deadline, workers=4):
        assert context == 'original-context' and workers == 4
        raise TimeoutError('simulated timeout; no fits')
    monkeypatch.setattr(c, 'fit_recorded', interrupted)
    with pytest.raises(TimeoutError):
        c.resume(spec, directory, prior, dict(assessments=0), c.manifest(), tmp_path)
    assert c.load_json(tmp_path/'provenance/original-receipt.json') == prior
    current = c.load_json(directory/'receipt.json')
    assert current['status'] == 'incomplete' and current['producer_run'] == c.PRODUCER_RUN
    assert current['context'] == 'original-context' and current['seconds'] >= 590


def test_late_completed_network_is_audited_and_preserved_without_reconstruction(tmp_path, monkeypatch):
    import continue_qualification as c
    def forbidden(*a, **k):
        pytest.fail('late-completed network must not be reconstructed or fitted again')
    monkeypatch.setattr(c, 'fit_recorded', forbidden)
    spec = next(s for s in c.pending() if s['index'] == 23)
    archive = c.ROOT/'examples/results/2026-10-14-recursive-qualification/noise-38655036215/raw/case-23.zip'
    if not archive.exists():
        pytest.skip('large producer archives are retained in the evidence checkout')
    directory = tmp_path/'cases'/spec['id']
    prior, reused = c.restore(spec, archive, directory)
    original_network = (directory/'network.json.gz').read_bytes()
    receipt = c.resume(spec, directory, prior, reused, c.manifest(), tmp_path)
    assert receipt['status'] == 'complete' and receipt['prior_files_unchanged']
    assert (directory/'network.json.gz').read_bytes() == original_network
    assert c.file_hashes(directory) == prior['files']


def test_combined_rows_keep_unknowns_and_reject_duplicates():
    import continue_qualification as c
    rows = c.producer()['cases']
    result = c.combine(rows[:1])
    assert result['planned'] == 110 and result['complete'] == 72
    assert not result['qualified']
    assert result['cells']['noise-n80']['unknown'] == 38
    with pytest.raises(ValueError, match='duplicate'):
        c.combine([rows[0], rows[0]])
    audited = dict(next(r for r in rows if r['status'] == 'complete'), original_status='complete')
    assert c.combine([audited])['complete'] == 72


def test_failed_original_case_is_not_required_to_be_successful():
    import continue_qualification as c
    specs = c.completed()
    jobs = [dict(name=f"case ({s['id']})", conclusion='success',
                 steps=[dict(name='Run actions/upload-artifact@pinned', conclusion='success')]) for s in specs]
    jobs.append(dict(name='case (noise-n80-r25)', conclusion='failure', steps=[]))
    def api(path):
        if '/jobs?' in path:
            return dict(jobs=jobs)
        return dict(id=int(c.PRODUCER_RUN), head_sha=c.PRODUCER_SHA, run_attempt=1,
                    event='workflow_dispatch', path='.github/workflows/recursive-qualification.yml',
                    status='completed', conclusion='failure')
    assert c.transport(c.PRODUCER_RUN, c.PRODUCER_SHA, specs, original_run=True, api=api)['case_jobs'] == 72
    jobs[0]['steps'][0]['conclusion'] = 'failure'
    with pytest.raises(ValueError, match='upload'):
        c.transport(c.PRODUCER_RUN, c.PRODUCER_SHA, specs, original_run=True, api=api)


def test_once_only_and_attempt_checks():
    import continue_qualification as c
    c.verify_once('999', api=lambda p: dict(workflow_runs=[dict(id=999)]))
    with pytest.raises(ValueError, match='previous'):
        c.verify_once('999', api=lambda p: dict(workflow_runs=[dict(id=998)]))


def test_workflow_caps_downloads_and_source_coverage():
    import continue_qualification as c
    import fnmatch
    w = yaml.safe_load((c.ROOT/c.WORKFLOW).read_text())
    assert w['jobs']['case']['timeout-minutes'] == 60
    assert w['jobs']['case']['strategy']['max-parallel'] == 20
    assert w['jobs']['case']['strategy']['fail-fast'] is False
    assert w['jobs']['prepare']['timeout-minutes'] == 15
    assert w['jobs']['aggregate']['timeout-minutes'] == 60
    for key in ('prepare', 'case', 'aggregate'):
        patterns = w['jobs'][key]['steps'][0]['with']['sparse-checkout'].splitlines()
        assert all(any(fnmatch.fnmatch(n,p) for p in patterns) for n in c.manifest()['sources'])
    compute = next(s for s in w['jobs']['case']['steps'] if s.get('name') == 'Continue same dataset')
    assert compute['timeout-minutes'] == 48
