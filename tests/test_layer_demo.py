import json
from pathlib import Path
import pytest

from layer_helpers import make_config
from test_layer_io import result
import open.layer_stability as engine
from examples.run_survey_demo import run_demo
from examples.audit_layer_demo import audit_demo


def test_layer_runner_and_offline_audit(tmp_path, monkeypatch):
    import examples.run_layer_demo as demo
    # Real recursive engine, controlled local assessments; no expensive fitting.
    from layer_helpers import install_scripted_analyzer
    install_scripted_analyzer(monkeypatch, base_seed=613,
        selections={('wellbeing',b):('social_support',) for b in range(5)})
    output = tmp_path/'demo'
    run_demo(output,5,max_depth=3,method='layer_consensus',sensitivity=True)
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['method'] == 'layer_consensus' and summary['status'] == 'COMPLETE'
    assert (output/'network.png').is_file() and (output/'REPORT.md').is_file()
    audit = audit_demo(output)
    assert audit['passed'] and audit['assessment_count'] == 10
    with pytest.raises(FileExistsError):
        run_demo(output,5,method='layer_consensus')
    evidence=output/'bundle/layer_evidence.json'
    original=evidence.read_text(encoding='utf-8')
    payload=json.loads(original)
    payload['layers'][0]['evidence'][0]['candidates']=['social_support']
    evidence.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError): audit_demo(output)
    evidence.write_text(original,encoding='utf-8')
    ledger=output/'bundle/assessment_ledger.json'
    payload=json.loads(ledger.read_text(encoding='utf-8'))
    payload['assessments'][0]['local']['relevance']['unique']['social_support']['raw'] += .1
    ledger.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError): audit_demo(output)


def test_invalid_method_rejected_before_output(tmp_path):
    with pytest.raises(ValueError): run_demo(tmp_path/'bad',method='unknown')
    assert not (tmp_path/'bad').exists()


def test_unrequested_sensitivity_is_explicitly_not_assessed(tmp_path,monkeypatch):
    from layer_helpers import install_scripted_analyzer
    install_scripted_analyzer(monkeypatch,base_seed=613,selections={})
    output=tmp_path/'reference_only'
    run_demo(output,2,method='layer_consensus',sensitivity=False)
    assert 'not assessed' in (output/'REPORT.md').read_text(encoding='utf-8').lower()
