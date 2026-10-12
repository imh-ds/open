import importlib.util
from pathlib import Path

import numpy as np
import pytest


def study():
    path = Path(__file__).parents[1] / 'examples/pair_screen_study.py'
    assert path.exists(), 'pair-screen study runner is not implemented'
    spec = importlib.util.spec_from_file_location('pair_screen_study', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unique_fresh_plan_and_nonoverlapping_shards():
    m = study()
    tasks = m.tasks()
    assert len(tasks) == 960 and len(m.cases()) == 480
    assert len({t['id'] for t in tasks}) == 960
    assert len(m.shards()) == 240
    assert [t for s in m.shards() for t in m.shard_tasks(s)] == tasks
    assert len({c['seed'] for c in m.cases()}) == 480
    assert {c['seed'] for c in m.cases()}.isdisjoint(c['test_seed'] for c in m.cases())
    with pytest.raises(ValueError):
        m.shard_tasks('240')


def test_generator_truth_and_independent_noise():
    m = study()
    case = m.cases()[0]
    X, y, mean, truth = m.generate(case, 20000, 123)
    assert truth is None and np.all(mean == 0)
    case = dict(case, scenario='correlated_mixed')
    X, y, mean, truth = m.generate(case, 20000, 123)
    assert truth == ['x1', 'x2']
    assert abs(X.x1.corr(X.x2) - .7) < .02
    np.testing.assert_allclose(mean, X.x1+.6*X.x2+.8*(X.x1*X.x2-.7))
    assert abs(1-np.var(y-mean)/np.var(y)-.6) < .03


def test_partial_denominators_and_no_automatic_acceptance(tmp_path):
    m = study()
    report = m.aggregate(tmp_path)
    assert report['planned'] == 960 and report['complete'] == 0
    assert report['missing'] == 960 and not report['audit_errors']
    assert len(report['cells']) == 48
    assert all(c['planned'] == 20 and c['complete'] == 0 for c in report['cells'])
    assert all(c['true_top1_bounds'] == [0, 20] for c in report['cells']
               if c['scenario'] in m.INTERACTION_SCENARIOS)
    assert report['acceptance'] == 'none: development only'


def test_foreign_and_conflicting_evidence_rejected(tmp_path):
    m = study()
    (tmp_path / 'foreign').mkdir()
    with pytest.raises(ValueError, match='foreign'):
        m.aggregate(tmp_path)
    a, b, target = (tmp_path / n for n in ('a', 'b', 'target'))
    for root, text in ((a, 'one'), (b, 'two')):
        (root/'case').mkdir(parents=True)
        (root/'case'/'result.json').write_text(text)
    with pytest.raises(ValueError, match='conflict'):
        m.merge_shards([a, b], target)


def test_native_task_saved_setting_refits_and_tampering(tmp_path):
    m = study()
    task = next(t for t in m.tasks() if t['case']['scenario']=='mixed'
                and t['case']['n']==80 and t['engine']=='gbm')
    result = m.run_task(task, tmp_path)
    assert result['status'] == 'complete'
    assert len(result['folds']) == 3
    assert result['screen']['planned_folds'] == 3
    assert result['screen']['additional_model_fits'] == 0
    m.check_result(task, tmp_path/task['id'], refit=True)
    receipt_path = tmp_path/task['id']/'receipt.json'
    receipt = m.read(receipt_path)
    altered = dict(receipt, environment=dict(m.manifest()['environment'],python='0.0.0'))
    m.write(receipt_path, altered)
    with pytest.raises(ValueError, match='runtime'):
        m.check_result(task, tmp_path/task['id'])
    m.write(receipt_path, receipt)
    assert m.run_task(task, tmp_path)['task'] == task
    p = tmp_path/task['id']/'result.json'
    value = m.read(p)
    value['folds'][0]['predicted'][0] += 1
    m.write(p, value)
    with pytest.raises((ValueError, AssertionError)):
        m.check_result(task, p.parent)
    report = m.aggregate(tmp_path)
    assert report['complete'] == 0 and len(report['audit_errors']) == 1


def test_resume_verification_failure_revokes_old_completion(tmp_path):
    m = study()
    task = m.tasks()[1]  # actual GBM noise task
    m.run_task(task, tmp_path)
    folder = tmp_path/task['id']
    value = m.read(folder/'result.json')
    value['folds'][0]['predicted'][0] += 1
    f = value['folds'][0]
    f['mse'] = float(np.square(np.asarray(f['observed'])-f['predicted']).mean())
    value['cv'] = m.skill(np.concatenate([f['observed'] for f in value['folds']]),
        np.concatenate([f['predicted'] for f in value['folds']]),
        np.concatenate([f['baseline'] for f in value['folds']]))
    m.write(folder/'result.json', value)
    m.check_result(task, folder)  # arithmetic is consistent, saved model is not
    assert m.aggregate(tmp_path)['complete'] == 0  # stale refit stamp cannot certify changed bytes
    with pytest.raises(AssertionError):
        m.run_task(task, tmp_path)
    assert not (folder/'verified.json').exists()
    assert not (folder/'result.json').exists()
    assert (folder/'invalid-result.json').exists()
    assert m.aggregate(tmp_path)['complete'] == 0


def test_unassessed_true_pair_is_unknown_not_negative():
    m = study()
    result = dict(truth=['x1','x2'],screen=dict(pairs=[dict(members=['x1','x2'],
        mean_screen_rms=None,shortlisted_folds=0,supported_detail_folds=0)]))
    metrics = m.stats(result)
    assert metrics['true_assessed'] is False
    assert metrics['true_top1'] is None and metrics['true_top5'] is None
    assert all(c['true_exceeds'] is None for c in metrics['sensitivity'])
    assert metrics['any_displayed'] is False  # observed report coverage, not absence


def test_runtime_is_checked_against_frozen_versions(monkeypatch):
    m = study()
    monkeypatch.setattr(m, 'environment',lambda:dict(m.manifest()['environment'],python='0.0.0'))
    with pytest.raises(ValueError,match='runtime'):
        m.validate_environment()


def test_completed_but_unassessed_rank_has_unknown_bounds(tmp_path, monkeypatch):
    m = study()
    planned=[t for t in m.tasks() if t['case']['scenario']=='pure'
             and t['case']['n']==80 and t['engine']=='gbm']
    metrics=m.stats(dict(truth=['x1','x2'],screen=dict(pairs=[dict(members=['x1','x2'],
        mean_screen_rms=None,shortlisted_folds=0,supported_detail_folds=0)])))
    value=dict(metrics=metrics,test=dict(skill=0),screen=dict(prediction_rows=0),
               fitting_screen_elapsed_seconds=1,warnings=[])
    monkeypatch.setattr(m,'check_result',lambda task,folder:value)
    for t in planned:
        folder=tmp_path/t['id']
        m.write(folder/'result.json',{})
        m.write(folder/'receipt.json',{})
        (folder/'predictions.csv').write_text('fixture')
        m.write(folder/'verified.json',m.verification(t,folder))
    report=m.aggregate(tmp_path)
    cell=next(c for c in report['cells'] if c['scenario']=='pure' and c['n']==80 and c['engine']=='gbm')
    assert cell['complete']==20 and cell['true_assessed']==0
    assert cell['true_top1_bounds']==[0,20] and cell['true_top1_wilson'] is None
    assert all(c['true_exceeds_bounds']==[0,20] and c['incidental_exceeds_bounds']==[0,20]
        for c in report['cutoffs'] if c['scenario']=='pure' and c['n']==80 and c['engine']=='gbm')
