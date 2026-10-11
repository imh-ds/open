import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location('sample_size_pilot',
        Path(__file__).parents[1]/'examples/sample_size_pilot.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_manifest_is_twelve_unique_frozen_cases():
    study = module()
    specs = study.specs()
    assert len(specs) == len({s['id'] for s in specs}) == 12
    assert {(s['n'], s['p']) for s in specs} == {(80, 5), (300, 5), (300, 15)}
    assert len({s['seed'] for s in specs}) == 12
    assert all(study.settings(s).stability.n_resamples == 100 for s in specs)
    assert study.settings(specs[-1]).analysis.max_depth == 3


def test_generator_reproducibility_signal_and_oracle():
    study = module()
    for scenario in ('noise', 'additive', 'interaction', 'chain'):
        frame, mean = study.generate(scenario, 50000, 5, 123)
        again, again_mean = study.generate(scenario, 50000, 5, 123)
        np.testing.assert_array_equal(frame, again)
        np.testing.assert_array_equal(mean, again_mean)
        assert list(frame) == ['y', 'x1', 'x2', 'x3', 'x4', 'x5']
        if scenario != 'noise':
            assert np.var(mean)/np.var(frame.y) == pytest.approx(.6, abs=.02)
            assert np.var(frame.y-mean) == pytest.approx(2/3, abs=.02)
    frame, _ = study.generate('chain', 50000, 5, 321)
    assert frame.x1.corr(frame.x2) == pytest.approx(.6, abs=.02)
    assert frame.x2.corr(frame.x3) == pytest.approx(.6, abs=.02)


def test_structure_keeps_upstream_misses_and_extra_edges_visible():
    study = module()
    spec = next(s for s in study.specs() if s['scenario'] == 'chain')
    ideal = {'x1': ('y',), 'x2': ('x1',), 'x3': ('x2',)}
    result = study.structural_scores(spec, ideal, {'y':0, 'x1':1, 'x2':2, 'x3':3})
    assert result['path_recovered'] and result['root_recovered']
    assert result['required_edges_found'] == 3 and result['extra_edges'] == []
    missed = study.structural_scores(spec, {}, {'y':0})
    assert not missed['path_recovered'] and missed['required_edges_found'] == 0
    result = study.structural_scores(spec, {**ideal, 'x4':('y',)},
        {'y':0, 'x1':1, 'x2':2, 'x3':3, 'x4':1})
    assert result['false_root_inputs'] == ['x4']
    assert result['extra_edges'] == [['x4', 'y']]


def test_manifest_rejects_changes_and_normalizes_only_line_endings(tmp_path):
    study = module()
    manifest = study.make_manifest()
    study.validate_manifest(manifest)
    manifest['cases'][0]['seed'] += 1
    with pytest.raises(ValueError, match='manifest'):
        study.validate_manifest(manifest)
    assert study.text_hash(b'a\r\nb\n') == study.text_hash(b'a\nb\n')
    assert study.text_hash(b'a\nb\n') != study.text_hash(b'a\nc\n')


def test_deadline_preserves_checkpoint_and_rejects_stale_context(tmp_path):
    study = module()
    from dataclasses import replace
    from open.layer_stability import make_layer_sample_plans
    spec = study.specs()[0]
    frame, _ = study.generate('noise', 20, 3, 4)
    config = study.settings(spec)
    config = replace(config, stability=replace(config.stability, n_resamples=1))
    plans = make_layer_sample_plans(frame, config)
    adapter = study.CheckpointAssessor(tmp_path, deadline=0, workers=1)
    with pytest.raises(study.BudgetExceeded):
        adapter(frame, depth=0, target='y', candidates=('x1',), plans=plans, config=config)
    assert not list(tmp_path.rglob('*.json.gz'))


def test_prediction_scores_are_signed_and_oracle_is_separate():
    study = module()
    import pandas as pd
    frame = pd.DataFrame({'observed':[0., 2.], 'baseline':[1., 1.],
        'full':[0., 2.], 'retained':[3., 3.], 'oracle':[0., 2.]})
    result = study.prediction_scores(frame)
    assert result['full']['skill'] == 1
    assert result['retained']['skill'] == -4
    assert result['oracle']['skill'] == 1
    assert result['skill_loss'] == 5


def test_row_diagnostics_read_production_gain_and_nested_sizes():
    from types import SimpleNamespace
    from layer_helpers import make_local, make_assessment, make_config
    a = make_assessment(make_local('y', ('a',), unique=('a',)))
    network = SimpleNamespace(layers=[SimpleNamespace(evidence=[
        SimpleNamespace(assessments=[a])])])
    row = module().row_diagnostics(network, make_config())[0]
    assert row['relative_gain'] == .5
    assert row['fit_n'] == [8, 8]
    assert row['tuning_n'] == [4, 4, 4, 4]


def test_worker_returns_pickle_safe_receipt_and_restores_original_identity(tmp_path, monkeypatch):
    from layer_helpers import make_local, make_assessment, make_config
    from dataclasses import replace
    import pandas as pd
    study = module()
    native = make_assessment(make_local('y', ('a',), unique=('a',)))
    plan = replace(native.sample, run_index=3)
    calls = []
    def assess(data, **kwargs):
        calls.append(kwargs)
        return (replace(native, sample=kwargs['plans'][0]),)
    monkeypatch.setattr(study, 'NATIVE_ASSESS', assess)
    path = tmp_path/'assessment.json.gz'
    result = study.one_assessment(pd.DataFrame({'y':range(16), 'a':range(16)}),
        0, 'y', ('a',), plan, make_config(), path, 'context')
    assert result == 'computed'
    assert calls[0]['plans'][0].run_index == 0
    saved = study.load_checkpoint(path, plan, 'context')
    assert saved.sample.run_index == 3
    assert study.one_assessment(None, 0, 'y', ('a',), plan, make_config(), path, 'context') == 'reused'
    with pytest.raises(ValueError, match='context'):
        study.load_checkpoint(path, plan, 'other')


def test_checkpoint_adapter_matches_native_three_depth_recursion(tmp_path, monkeypatch):
    import time
    import pandas as pd
    from layer_helpers import make_config, install_scripted_analyzer
    study = module()
    config = make_config(n_resamples=2)
    frame = pd.DataFrame({n:range(20) for n in ('y', 'x1', 'x2', 'x3')})
    selections = {(t, b):(source,) for t, source in
        (('y','x1'), ('x1','x2'), ('x2','x3')) for b in range(2)}
    calls = install_scripted_analyzer(monkeypatch, selections=selections)
    native = study.engine.fit_layer_consensus(frame, 'y', tuple(frame), config)
    class Future:
        def __init__(self, value): self.value = value
        def result(self): return self.value
    class InlinePool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args): return Future(fn(*args))
    monkeypatch.setattr(study, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setattr(study.engine, 'assess_layer_target',
        study.CheckpointAssessor(tmp_path, time.monotonic()+30, 2))
    checkpointed = study.engine.fit_layer_consensus(frame, 'y', tuple(frame), config)
    assert checkpointed.parents == native.parents
    assert checkpointed.assigned_depth == {'y':0, 'x1':1, 'x2':2, 'x3':3}
    assert len(list(tmp_path.rglob('*.json.gz'))) == 6
    assert study.fingerprint(checkpointed.layers) == study.fingerprint(native.layers)
    count = len(calls)
    resumed = study.engine.fit_layer_consensus(frame, 'y', tuple(frame), config)
    assert len(calls) == count
    assert study.fingerprint(resumed.layers) == study.fingerprint(native.layers)


def test_combined_audit_keeps_all_planned_cases_after_bad_or_missing_artifacts(tmp_path):
    study = module()
    study.write_json(tmp_path/'manifest.json', study.make_manifest())
    study.write_json(tmp_path/study.specs()[0]['id']/'result.json', {'broken':True})
    report = study.audit(tmp_path)
    assert report['planned'] == len(report['cases']) == 12
    assert report['complete'] == 0 and report['incomplete'] == 12
    assert report['audit_errors'] == 1
    assert report['cases'][0]['status'] == 'audit_failed'
    assert all(r['status'] == 'missing' for r in report['cases'][1:])


def test_prediction_verification_rejects_arithmetic_consistent_fabricated_predictions():
    import pandas as pd
    from types import SimpleNamespace
    study = module()
    spec = study.specs()[0]
    config = study.settings(spec)
    train, _ = study.generate('noise', 80, 5, spec['seed'])
    test, mean = study.generate('noise', 10, 5, spec['test_seed'])
    fake = pd.DataFrame(dict(observed=test.y, baseline=np.full(10, train.y.mean()),
        retained=test.y, full=test.y, oracle=mean))
    network = SimpleNamespace(parents={})
    result = {'full_parameters':{'learning_rate':.1, 'max_depth':3}, 'retained_parameters':None}
    with pytest.raises((ValueError, AssertionError)):
        study.verify_refits(train, test, fake, network, result, config)


def test_incomplete_case_cannot_bypass_saved_network_validation(tmp_path):
    study = module()
    spec = study.specs()[0]
    config = study.settings(spec)
    train, _ = study.generate(spec['scenario'], spec['n'], spec['p'], spec['seed'])
    test, _ = study.generate(spec['scenario'], 2000, spec['p'], spec['test_seed'])
    train.to_csv(tmp_path/'train.csv', index=False)
    test.to_csv(tmp_path/'test.csv', index=False)
    context = study.fingerprint(dict(spec=spec, config=config, test_n=2000,
        sources=study.sources(), protocol=study.text_hash(study.PROTOCOL.read_bytes())))
    study.write_json(tmp_path/'result.json', dict(spec=spec, config=study.portable(config),
        context=context, status='incomplete', structural=None, scores=None, diagnostics=[]))
    (tmp_path/'network.json.gz').write_bytes(b'corrupt network')
    with pytest.raises((ValueError, OSError)):
        study.audit_case(spec, tmp_path)


def test_actions_budget_matches_manifest_and_push_cannot_fit_by_accident():
    import yaml
    study = module()
    workflow = yaml.safe_load((Path(__file__).parents[1]/'.github/workflows/sample-size-pilot.yml').read_text())
    jobs = workflow['jobs']
    manifest = study.make_manifest()
    assert jobs['case']['strategy']['max-parallel'] == manifest['max_parallel'] == 3
    assert jobs['case']['timeout-minutes'] == manifest['hard_job_minutes'] == 110
    assert jobs['prepare']['timeout-minutes']+12*110+jobs['audit']['timeout-minutes'] == 1365
    assert '[run-sample-size-pilot]' in jobs['prepare']['if']
    assert 'workflow_dispatch' in jobs['prepare']['if']
