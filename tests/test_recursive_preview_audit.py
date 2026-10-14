import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


def module():
    return importlib.import_module('recursive_preview_audit')


def baseline():
    from layer_helpers import make_sample
    from open.risk import RiskResult, FoldRisk, ObservationLoss
    from open.layer_artifacts import portable
    import hashlib
    plan = make_sample()
    data = pd.DataFrame({'y':np.arange(16, dtype=float), 'a':np.arange(16, dtype=float)})
    folds = []
    for split in plan.split_plan.splits:
        pred = float(data.y.iloc[list(split.train_indices)].mean())
        rows = tuple(ObservationLoss(i, float(data.y.iloc[i]), pred,
                                   float((data.y.iloc[i]-pred)**2)) for i in split.test_indices)
        folds.append(FoldRisk(split.repeat, split.fold, (), rows,
                              float(np.mean([r.loss for r in rows])), None))
    risk = RiskResult('y', ('a',), (), None, 'mse',
        hashlib.sha256(plan.split_plan.to_json(indent=None).encode()).hexdigest(), plan.seed,
        tuple(folds), float(np.mean([r.loss for f in folds for r in f.observations])), 0, 0)
    return portable(risk), data, plan


def test_risk_audit_rejects_loss_row_coverage_and_baseline_tampering():
    import copy
    m = module()
    risk, data, plan = baseline()
    assert m.audit_risk(risk, data, plan, 'y', ('a',), ()) > 0
    for change in ('loss', 'coverage', 'observed', 'baseline'):
        edited = copy.deepcopy(risk)
        row = edited['folds'][0]['observations'][0]
        if change == 'loss': row['loss'] += 1
        if change == 'coverage': edited['folds'][0]['observations'].pop()
        if change == 'observed': row['observed'] += 1
        if change == 'baseline': row['predicted'] += 1
        with pytest.raises(ValueError):
            m.audit_risk(edited, data, plan, 'y', ('a',), ())


def test_baseline_cannot_claim_an_engine_or_tuning_fits():
    import copy
    m = module()
    risk, data, plan = baseline()
    for field, value in (('learner','elastic_net'), ('tuning_model_fit_count',1)):
        edited = copy.deepcopy(risk)
        edited[field] = value
        with pytest.raises(ValueError):
            m.audit_risk(edited, data, plan, 'y', ('a',), ())


def test_pair_contrasts_anchor_to_selected_full_baseline_rows_and_reused_parameters():
    import copy
    m = module()
    params = {'alpha':1.,'l1_ratio':.5}
    def fold(index, loss):
        return dict(repeat_index=0,fold_index=index,predictors=['z'],
            observations=[dict(row_position=index,loss=loss)],fitted_parameters=params)
    base = dict(aggregate_risk=15.,folds=[fold(0,10.),fold(1,20.)])
    full = dict(aggregate_risk=2.5,folds=[fold(0,1.),fold(1,4.)])
    ablation = dict(predictors=['z'],aggregate_risk=3.,folds=[fold(0,3.)])
    pair = dict(selected_folds=[[0,0]],baseline_risk=10.,full_risk=1.,
                ablated_risk=3.,total_gain=9.,raw=2.,normalized=2/9,ablation=ablation)
    m.audit_pair_group_anchors([pair],[],base,full)
    for field in ('baseline_risk','full_risk','total_gain','raw'):
        edited = copy.deepcopy(pair); edited[field] += 1
        with pytest.raises(ValueError):
            m.audit_pair_group_anchors([edited],[],base,full)
    edited = copy.deepcopy(pair)
    edited['ablation']['folds'][0]['fitted_parameters'] = {'alpha':.1,'l1_ratio':.5}
    with pytest.raises(ValueError, match='parameters'):
        m.audit_pair_group_anchors([edited],[],base,full)


def test_stage_summary_keeps_missing_bad_and_conflicting_cases():
    m = module()
    import recursive_preview_design as d
    specs = d.specs()[:160]
    rows = [dict(spec=s, status='complete', structure=dict(incorrect=False, exact=True)) for s in specs]
    report = m.summarize(rows[:-1], 'noise', 'manifest')
    assert report['planned'] == 160 and report['complete'] == 159 and not report['qualified']
    assert report['cells']['noise-n200']['unknown'] == 1
    report = m.summarize(rows, 'noise', 'manifest')
    assert report['qualified']
    with pytest.raises(ValueError, match='duplicate'):
        m.summarize([*rows, rows[0]], 'noise', 'manifest')


def test_null_selection_is_recomputed_instead_of_trusting_flag():
    m = module()
    import recursive_preview_design as d
    rows = [dict(spec=s, status='complete', structure=dict(incorrect=False, exact=True)) for s in d.specs()[:160]]
    report = m.summarize(rows, 'noise', 'manifest')
    assert m.validate_noise_gate(report, 'manifest')
    for row in report['cases'][:3]: row['structure']['incorrect'] = True
    with pytest.raises(ValueError, match='gate'):
        m.validate_noise_gate(report, 'manifest')


def test_streaming_archive_rejects_duplicate_members(tmp_path):
    import zipfile
    m = module()
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('cases/a/receipt.json', '{}')
        with pytest.warns(UserWarning):
            z.writestr('cases/a/receipt.json', '{"edited":true}')
    with pytest.raises(ValueError, match='duplicate'):
        m.Store(archive, 'a')


def test_full_recorded_production_path_audits_with_scripted_mean_predictor(tmp_path, monkeypatch):
    """Exercise real risk recording/recursion/artifacts; no ML training or scientific fixture."""
    from open.learners.base import Predictor
    from layer_helpers import make_config
    import recursive_preview_runner as runner
    import recursive_preview_design as design
    m = module()
    class Mean(Predictor):
        engine_name = 'elastic_net'
        tuning_grid_size = 25
        def tune(self, *args, **kw): return {'alpha':1., 'l1_ratio':.5}
        def fit(self, X, y, params=None): self.mean = float(np.mean(y)); return self
        def predict(self, X): return np.full(len(X), self.mean)
        def clone(self): return Mean()
    monkeypatch.setattr(runner.analysis, 'make_predictor', lambda c:Mean())
    monkeypatch.setattr(runner.package, '__file__', 'C:/test/site-packages/open/__init__.py')
    native_version = runner.importlib.metadata.version
    monkeypatch.setattr(runner.importlib.metadata, 'version',
        lambda n:design.RUNTIME[n] if n in design.RUNTIME else native_version(n))
    class Future:
        def __init__(self, value): self.value = value
        def result(self): return self.value
    class Pool:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, *args): return Future(fn(*args))
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    spec = {'id':'unit-baseline', 'fixture':'scripted'}
    data = pd.DataFrame({'y':np.arange(20,dtype=float), 'a':np.arange(20,dtype=float)})
    config = make_config(n_resamples=2)
    manifest = {'runtime':design.RUNTIME}
    directory = tmp_path/spec['id']
    runner.run_case(spec, directory, manifest, data=data, config=config)
    store = m.Store(directory)
    row = m.audit_case(spec, store, manifest, data=data, config=config)
    assert row['status'] == 'complete' and row['assessments'] == 2
    assert row['observation_checks'] == 128 and row['structure'] is None
    assert row['outer_fits'] == 8 and row['tuning_fits'] == 400
