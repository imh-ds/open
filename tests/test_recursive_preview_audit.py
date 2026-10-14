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
