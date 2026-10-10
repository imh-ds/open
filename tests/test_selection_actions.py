import importlib.util
import json
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location('selection_actions', Path(__file__).parents[1]/'examples/selection_actions.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def manifest():
    from dataclasses import asdict
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))
    from run_selection_reliability import config, portable
    return dict(scope='development_root_selection', config=portable(asdict(config())),
                specs=[dict(id='ordinary-n80-r0', scenario='ordinary', n=80, repetition=0, seed=181001)],
                source_hashes={}, execution_sources=[])


def test_transfer_rejects_changed_research_settings():
    m = manifest()
    m['config']['thresholds']['stability'] = .5
    with pytest.raises(ValueError, match='settings'):
        module().validate_manifest(m)


def test_transfer_rejects_unknown_or_repeated_dataset_identity():
    m = manifest()
    m['specs'][0]['seed'] += 1
    with pytest.raises(ValueError, match='identities'):
        module().validate_manifest(m)
    m = manifest()
    m['specs'] *= 2
    with pytest.raises(ValueError, match='identities'):
        module().validate_manifest(m)


def test_transfer_rejects_source_tampering(tmp_path):
    source = tmp_path/'source.py'
    source.write_bytes(b'changed')
    m = manifest()
    m['source_hashes'] = {'source.py': 'wrong'}
    with pytest.raises(ValueError, match='source'):
        module().verify_sources(m, tmp_path)


def test_transfer_checks_resumed_runner_and_original_archive(tmp_path):
    import hashlib
    current = tmp_path/'examples/run_selection_reliability.py'
    current.parent.mkdir()
    current.write_bytes(b'resumed')
    archive = tmp_path/'runner-at-initial-dispatch.py'
    archive.write_bytes(b'initial')
    m = manifest()
    m['source_hashes'] = {'examples/run_selection_reliability.py': hashlib.sha256(b'initial').hexdigest()}
    m['execution_sources'] = [dict(runner_hash=hashlib.sha256(b'resumed').hexdigest())]
    module().verify_sources(m, tmp_path, tmp_path)
    archive.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='archive'):
        module().verify_sources(m, tmp_path, tmp_path)
