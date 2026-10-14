import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


def module():
    return importlib.import_module('recursive_preview_runner')


def test_worker_restores_sample_and_reuses_only_verified_checkpoint(tmp_path, monkeypatch):
    from layer_helpers import make_assessment, make_local, make_config
    m = module()
    assessment = make_assessment(make_local('y', ('a',), unique=('a',)))
    plan = replace(assessment.sample, run_index=3)
    calls = []
    def native(data, **kw):
        calls.append(kw['plans'][0].run_index)
        return (replace(assessment, sample=kw['plans'][0]),)
    monkeypatch.setattr(m, 'NATIVE_ASSESS', native)
    path = tmp_path/'003.json.gz'
    args = (pd.DataFrame({'y':range(16), 'a':range(16)}), 0, 'y', ('a',), plan, make_config(), path, 'context')
    assert m.one_assessment(*args) == 'computed'
    assert m.read_checkpoint(path, plan, 'context')[0].sample == plan
    assert m.one_assessment(*args) == 'reused' and calls == [0]
    with pytest.raises(ValueError, match='context'):
        m.read_checkpoint(path, plan, 'wrong')
    payload = m.read_gzip(path)
    payload['assessment']['target'] = 'edited'
    m.write_gzip(path, payload)
    with pytest.raises(ValueError, match='digest'):
        m.read_checkpoint(path, plan, 'context')


def test_interrupted_started_assessment_cannot_be_refitted(tmp_path):
    from layer_helpers import make_sample, make_config
    m = module()
    path = tmp_path/'000.json.gz'
    m.write_json(path.with_suffix('.started.json'), {'status':'STARTED'})
    with pytest.raises(RuntimeError, match='already started'):
        m.one_assessment(None, 0, 'y', ('a',), make_sample(), make_config(), path, 'context')


def test_parallel_adapter_matches_scripted_recursion_and_restores_native(tmp_path, monkeypatch):
    import time
    from layer_helpers import make_config, install_scripted_analyzer
    m = module()
    c = make_config(n_resamples=2)
    data = pd.DataFrame({n:range(20) for n in ('y','x1','x2','x3')})
    install_scripted_analyzer(monkeypatch, selections={
        (t,b):(s,) for t,s in (('y','x1'),('x1','x2'),('x2','x3')) for b in range(2)})
    native = m.engine.fit_layer_consensus(data, 'y', tuple(data), c)
    class Future:
        def __init__(self, value): self.value = value
        def result(self): return self.value
    class Pool:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args): return Future(fn(*args))
    monkeypatch.setattr(m, 'ProcessPoolExecutor', Pool)
    result = m.fit_recorded(data, c, tmp_path, 'context', time.monotonic()+30)
    assert m.digest(result.layers) == m.digest(native.layers)
    assert m.engine.assess_layer_target is m.NATIVE_ASSESS
    assert len(list(tmp_path.rglob('*.json.gz'))) == 6


def test_completed_case_cannot_reuse_edited_outputs(tmp_path):
    m = module()
    (tmp_path/'data.csv').write_text('original')
    receipt = {'context':'c', 'status':'complete', 'files':m.file_hashes(tmp_path)}
    m.write_json(tmp_path/'receipt.json', receipt)
    assert m.verified_receipt(tmp_path, 'c') == receipt
    (tmp_path/'data.csv').write_text('edited')
    with pytest.raises(ValueError, match='evidence'):
        m.verified_receipt(tmp_path, 'c')
