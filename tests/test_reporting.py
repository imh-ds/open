from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from open.evaluation import evaluate_procedure
from open.metadata import validate_metadata
import open.reporting as module
from layer_helpers import make_config


def data():
    rng = np.random.default_rng(27)
    x = rng.normal(size=24)
    return pd.DataFrame({'y': x + rng.normal(scale=.1, size=24), 'x': x})


def metadata(d):
    return validate_metadata(tuple(d), [{'name': n, 'type': 'continuous'} for n in d], 'y')


def test_unified_export_preserves_fold_evidence_and_does_not_overwrite(tmp_path, monkeypatch):
    d, c = data(), make_config(n_resamples=1)
    result = module.analyze_researcher(d, 'y', tuple(d), c, outer_folds=2, include_effects=False)
    destination = tmp_path/'report'
    module.write_researcher_outputs(result, destination, metadata=metadata(d))
    payload = json.loads((destination/'evaluation.json').read_text())
    assert payload['schema'] == 'open.procedure_evaluation'
    assert len(payload['folds']) == 2
    assert len(payload['predictions']) == 24
    report = (destination/'REPORT.md').read_text(encoding='utf-8')
    assert 'not SEM global fit' in report
    assert 'not assessed' in report
    assert 'training-mean' in report
    assert 'x → y' in report
    assert (destination/'fold_0'/'assessment_ledger.json').exists()
    assert (destination/'reference'/'edges.csv').exists()
    with pytest.raises(FileExistsError):
        module.write_researcher_outputs(result, destination, metadata=metadata(d))
    monkeypatch.setattr(module, 'write_layer_outputs',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('export failed')))
    with pytest.raises(RuntimeError, match='export failed'):
        module.write_researcher_outputs(result, tmp_path/'failed', metadata=metadata(d))
    assert not (tmp_path/'failed').exists()
    assert not list(tmp_path.glob('.failed.*'))


def test_report_incomplete_and_unavailable_scores_without_fabricated_numbers():
    d = data()
    c = make_config(n_resamples=1)
    # Outer training sets have one row; discovery cannot support inner folds.
    result = module.analyze_researcher(d.iloc[:2], 'y', tuple(d), c, outer_folds=2, include_effects=False)
    report = module.render_researcher_report(result)
    assert 'INCOMPLETE' in report
    assert 'withheld' in report
    assert '0/2 folds completed' in report
    assert all(f.status == 'failed' for f in result.evaluation.folds)


def test_effects_include_contributions_shapes_and_explicit_data_role(tmp_path):
    d = data()
    result = module.analyze_researcher(d, 'y', tuple(d), make_config(n_resamples=1),
                                      outer_folds=2, effect_bins=2)
    report = module.render_researcher_report(result)
    assert 'Q25' in report and 'Curve magnitude' in report and 'ΔCV R²' in report
    assert 'full candidate set' in report
    assert result.effects['target'] == 'y'
    destination = tmp_path/'plots'
    module.write_researcher_outputs(result, destination, metadata=metadata(d), plots=True)
    assert all((destination/n).exists() for n in ('prediction.png', 'reference_network.png', 'root_curves.png'))
    assert (destination/'effects'/'comparison.json').exists()


def test_export_rejects_metadata_mismatch(tmp_path):
    d = data()
    result = module.analyze_researcher(d, 'y', tuple(d), make_config(n_resamples=1),
                                      outer_folds=2, include_effects=False)
    bad = validate_metadata(('y', 'z'), [{'name': n, 'type': 'continuous'} for n in ('y', 'z')], 'y')
    with pytest.raises(ValueError, match='metadata'):
        module.write_researcher_outputs(result, tmp_path/'report', metadata=bad)
    assert not (tmp_path/'report').exists()
