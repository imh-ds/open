"""Versioned optimized-grid audits and a bounded fresh RF validation."""
import fnmatch
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


@pytest.fixture(autouse=True)
def no_scientific_fits(monkeypatch):
    import recursive_preview_runner as runner
    monkeypatch.setattr(runner, 'NATIVE_ASSESS',
        lambda *a, **k: pytest.fail('local scientific fit forbidden'))


def test_workflow_budget_sparse_sources_and_reference_archives():
    import validate_optimized_rf as v
    value = v.manifest()
    workflow = yaml.safe_load((v.ROOT/v.WORKFLOW).read_text())
    jobs = workflow['jobs']
    assert (jobs['register']['timeout-minutes'] + jobs['prepare']['timeout-minutes']
        + 2*jobs['case']['timeout-minutes'] + jobs['aggregate']['timeout-minutes']) == 286
    assert jobs['case']['strategy']['max-parallel'] == 2
    required = set(value['sources']) | {v.MANIFEST.relative_to(v.ROOT).as_posix()}
    required |= {r['path'] for r in value['references'].values()}
    for name in ('prepare', 'case', 'aggregate'):
        checkout = next(s for s in jobs[name]['steps'] if s.get('uses', '').startswith('actions/checkout@'))
        patterns = checkout['with']['sparse-checkout'].split()
        assert all(any(fnmatch.fnmatch(p, pattern) for pattern in patterns) for p in required)


def test_saved_checkpoint_under_explicit_new_count_policy_without_fits():
    import validate_optimized_rf as v
    from open.layer_artifacts import decode
    from open.layer_types import LocalAssessment
    spec = v.specs()[0]
    ref = v.manifest()['references'][spec['id']]
    store = v.audit.Store(v.ROOT/ref['path'], ref['row']['spec']['id'])
    try:
        payload = store.compressed('checkpoints/d0-Y/000.json.gz')
    finally:
        store.close()
    # A software fixture only: adjust a COPY of old counters to the explicit new
    # policy. Predictions stay archived; this is not a fresh scientific result.
    config = v.settings(spec)
    for entry in payload['risks']:
        risk = entry['risk']
        risk['tuning_model_fit_count'] = v.audit.expected_tuning(
            risk, config, risk['tuning_model_fit_count'] > 0)
    payload['assessment']['local']['tuning_model_fit_count'] = sum(
        r['risk']['tuning_model_fit_count'] for r in payload['risks'])
    assessment = decode(LocalAssessment, payload['assessment'])
    payload['digest'] = v.digest({k: val for k, val in payload.items() if k != 'digest'})
    data = v.old.q.design.generate(spec).iloc[list(assessment.sample.sampled_positions)]
    result = v.audit.audit_checkpoint(payload, data, assessment, payload['context'], config)
    assert result['observation_checks'] > 0
    assert result['tuning_fits'] == 360


@pytest.mark.parametrize('foreign_control', [False, True])
def test_aggregate_gates_equivalence_on_integrity_and_transport(tmp_path, monkeypatch, foreign_control):
    import validate_optimized_rf as v
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_RUN_ID', 'unit-run')
    monkeypatch.setenv('GITHUB_SHA', 'unit-sha')
    monkeypatch.setattr(v, 'checked_manifest', lambda: {})
    monkeypatch.setattr(v, 'summarize', lambda *a: dict(execution_verified=True,
        equivalence_verified=True, complete=2, exact=2, cases=[]))
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    if foreign_control:
        v.write_json(inputs/'control.json', {'spec': {'id': 'foreign'}})
        monkeypatch.setattr(v, 'transport', lambda: {'verified': True})
    else:
        def fail():
            raise ValueError('controlled transport failure')
        monkeypatch.setattr(v, 'transport', fail)
    result = v.aggregate(inputs, tmp_path/'out')
    assert not result['execution_verified'] and not result['equivalence_verified']
    assert result['integrity_errors']


def config():
    return SimpleNamespace(learner=SimpleNamespace(library=('random_forest',)),
                           tuning_cv=SimpleNamespace(folds=3))


def risk(width=1, count=12, fraction=.7):
    return dict(predictors=['x'+str(i) for i in range(width)],
                folds=[dict(fitted_parameters=dict(max_features=fraction, min_samples_leaf=2))
                       for _ in range(2)], tuning_model_fit_count=count)


def test_singleton_policy_rejects_legacy_counts_without_changing_legacy_auditor():
    import optimized_rf_recursive_audit as new
    import multiengine_recursive_audit as old
    assert new.verify_tuning(risk(), config(), True) == 12
    assert old.verify_tuning(risk(count=24), config(), True) == 24
    with pytest.raises(ValueError, match='tuning count'):
        new.verify_tuning(risk(count=24), config(), True)
    with pytest.raises(ValueError, match='parameters'):
        new.verify_tuning(risk(fraction=1.), config(), True)
    assert new.verify_tuning(risk(width=2, count=24, fraction=1.), config(), True) == 24
    assert new.verify_tuning(risk(count=0, fraction=1.), config(), False) == 0


def test_policy_scope_restores_historical_auditor_after_error(monkeypatch):
    import optimized_rf_recursive_audit as new
    old = new.legacy
    original = old.verify_tuning

    def fail(*args, **kwargs):
        assert old.verify_tuning(risk(), config(), True) == 12
        raise RuntimeError('controlled audit error')

    monkeypatch.setattr(old, 'audit_case', fail)
    with pytest.raises(RuntimeError, match='controlled'):
        new.audit_case({}, None, {'grid_policy': new.POLICY}, config=config())
    assert old.verify_tuning is original
    with pytest.raises(ValueError, match='policy'):
        new.audit_case({}, None, {'grid_policy': 'legacy'}, config=config())


def test_parity_ignores_only_tuning_counts_and_retains_scientific_checks():
    import optimized_rf_recursive_audit as new
    old = dict(tuning_model_fit_count=24, outer_model_fit_count=2,
               predicted=1., observed=1., selection_frequency=.7,
               fitted_parameters={'max_features': .7})
    current = dict(old, tuning_model_fit_count=12)
    assert new.compare_statistics(current, old)['equivalent']
    for key, value in [('outer_model_fit_count', 3), ('predicted', 1.1),
                       ('observed', 1.000000000000001), ('selection_frequency', .71)]:
        changed = dict(current, **{key: value})
        assert not new.compare_statistics(changed, old)['equivalent']


def test_fresh_two_case_protocol_budget_and_no_reuse():
    import validate_optimized_rf as v
    value = v.manifest()
    assert value['reserved_minutes'] == 286 and value['ceiling'] == 300
    assert value['case_job_minutes'] == 120 and value['soft_seconds'] == 105*60
    assert len(v.specs()) == 2 and {s['n'] for s in v.specs()} == {80, 200}
    assert all(s['id'].startswith('optimized-') for s in v.specs())
    assert not value['checkpoint_reuse'] and not value['scientific_qualification']
    for spec in v.specs():
        c = v.settings(spec)
        assert c.stability.n_resamples == 100 and c.performance_cv.folds == 5
        assert c.performance_cv.repeats == 2 and c.tuning_cv.folds == 3


def test_partial_denominator_and_failed_equivalence_remain_visible():
    import validate_optimized_rf as v
    spec = v.specs()[0]
    report = v.summarize([dict(spec=spec, status='complete', structure={'exact': True},
                               comparison={'equivalent': False})], {})
    assert report['planned'] == 2 and report['complete'] == 1 and report['unknown'] == 1
    assert not report['execution_verified'] and not report['equivalence_verified']
    with pytest.raises(ValueError, match='duplicate'):
        v.summarize([dict(spec=spec)]*2, {})


def test_foreign_case_and_retry_cannot_start_fits(tmp_path, monkeypatch):
    import validate_optimized_rf as v
    monkeypatch.setattr(v, 'checked_manifest', lambda: {})
    monkeypatch.setattr(v, 'run_case', lambda *a, **k: pytest.fail('unexpected scientific fit'))
    with pytest.raises(ValueError, match='identity'):
        v.run_one('foreign', tmp_path/'case', None)
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '2')
    with pytest.raises(ValueError, match='first'):
        v.run_one(v.specs()[0]['id'], tmp_path/'case', None)
