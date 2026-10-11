import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location('fresh_learning',
        Path(__file__).parents[1] / 'examples/fresh_learning.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_plan_has_independent_datasets_and_paired_variants():
    study = module()
    cases = study.cases()
    assert len(cases) == len({c['id'] for c in cases}) == 240
    assert len({c['seed'] for c in cases}) == 240
    assert len({c['test_seed'] for c in cases}) == 240
    assert not {c['seed'] for c in cases} & {c['test_seed'] for c in cases}
    assert len(study.tasks()) == 960
    assert len(study.shards()) == 120
    assert all(len(study.shard_tasks(i)) == 8 for i in study.shards())


def test_benchmark_uses_truth_only_and_never_outcome():
    study = module()
    frame, oracle = study.pilot.generate('interaction', 80, 5, 8000000)
    X = study.features(frame, 'benchmark', 'interaction')
    assert list(X) == ['known_product']
    np.testing.assert_allclose(X.known_product, oracle)
    assert 'y' not in study.features(frame, 'gbm_shipped', 'interaction')
    assert list(study.features(frame, 'benchmark', 'noise')) == ['constant']


def test_aggregate_preserves_missing_denominator(tmp_path):
    study = module()
    report = study.aggregate(tmp_path, verify=False)
    assert report['planned'] == 960
    assert report['complete'] == 0
    assert report['missing'] == 960
    assert all(c['planned'] == 20 and c['positive_skill_lower_bound_count'] == 0
               for c in report['cells'])


def test_resume_rejects_changed_context(tmp_path):
    study = module()
    task = study.tasks()[0]
    folder = tmp_path / task['id']
    study.pilot.write_json(folder / 'result.json', {'context': 'foreign', 'status': 'complete'})
    with pytest.raises(ValueError, match='context'):
        study.run_task(task, tmp_path)


def test_completed_benchmark_predictions_and_sizes_are_audited(tmp_path):
    study = module()
    task = study.tasks()[3]
    result = study.run_task(task, tmp_path)
    assert result['status'] == 'complete'
    folder = tmp_path / task['id']
    (folder/'verified.json').unlink()
    study.run_task(task,tmp_path)
    assert (folder/'verified.json').exists()
    result['folds'][0]['inner_training_n'] = [1, 1, 1]
    study.pilot.write_json(folder / 'result.json', result)
    with pytest.raises(ValueError, match='training sizes'):
        study.check_result(task, folder)


def test_parameters_outside_declared_grid_are_rejected(tmp_path):
    study = module()
    task = study.tasks()[0]
    with pytest.raises(ValueError, match='parameters'):
        study.check_parameters(task, {'max_depth':99, 'learning_rate':.1})


def test_nomination_requires_complete_cells_and_prespecified_gains():
    study = module()
    cells = []
    for scenario in ('noise','additive','interaction','chain'):
        for n,p in ((80,5),(300,5),(300,15)):
            cells.append(dict(scenario=scenario,n=n,p=p,variant='gbm_deeper',
                complete=20,paired_n=20,
                mean_paired_difference=.06 if scenario == 'interaction' else 0))
    assert study.nominees(cells) == ['gbm_deeper']
    cells[-1]['complete'] = 19
    assert study.nominees(cells) == []
