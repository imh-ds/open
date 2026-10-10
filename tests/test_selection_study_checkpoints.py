import importlib.util
from pathlib import Path

import pytest

from layer_helpers import make_assessment, make_failed_assessment, make_local, make_sample
from open.layer_artifacts import portable

spec = importlib.util.spec_from_file_location('selection_runner', Path(__file__).parents[1]/'examples/run_selection_reliability.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize('failed', [False, True])
def test_checkpoint_preserves_complete_and_failed_assessment(tmp_path, failed):
    a = make_failed_assessment('y', ('x',), index=1) if failed else make_assessment(make_local('y', ('x',), unique=('x',)), index=1)
    path = tmp_path/'assessment.json.gz'
    runner.save_checkpoint(path, a, 'fixed-data-settings-and-source')
    restored = runner.load_checkpoint(path, a.sample, 'fixed-data-settings-and-source')
    assert portable(restored) == portable(a)


def test_checkpoint_rejects_changed_settings_or_sample(tmp_path):
    a = make_assessment(make_local('y', ('x',), unique=('x',)))
    path = tmp_path/'assessment.json.gz'
    runner.save_checkpoint(path, a, 'fixed-context')
    with pytest.raises(ValueError, match='context'):
        runner.load_checkpoint(path, a.sample, 'changed-context')
    with pytest.raises(ValueError, match='sample'):
        runner.load_checkpoint(path, make_sample(1), 'fixed-context')


def test_unfinished_temp_file_is_not_a_checkpoint(tmp_path):
    path = tmp_path/'assessment.json.gz'
    path.with_name(path.name+'.tmp').write_bytes(b'incomplete')
    assert not path.exists()
    with pytest.raises(FileNotFoundError):
        runner.load_checkpoint(path, make_sample(), 'fixed-context')
