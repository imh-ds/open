import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def test_continuation_manifest_preserves_identities_and_cumulative_budget():
    import recursive_preview_design as d
    value = d.manifest()
    assert value['schema'] == 'open.recursive_reference.v2'
    assert value['allocated_runner_minutes'] == 9917
    assert value['preflight_job_minutes'] == 45
    proof = value['continuation']
    assert proof['producer_run_id'] == '38615814397'
    original = json.loads((d.PRODUCER/'raw/producer-manifest.json').read_text())
    assert value['cases'] == original['cases']
    assert value['default_config'] == original['default_config']
    assert value['runtime'] == original['runtime']
    assert value['preflight_indices'] == original['preflight_indices'] == [160,240]
    assert original['allocated_runner_minutes'] == 9871
    assert proof['producer_manifest_digest'] == d.digest(original)


def test_completed_fixture_is_reused_without_fitting(tmp_path, monkeypatch):
    import recursive_preview_design as d
    import recursive_preview_actions as a
    from open.layer_artifacts import reconstruct_payloads
    def forbid(*args,**kw):
        pytest.fail('completed fixture must never be fitted again')
    monkeypatch.setattr(a,'run_case',forbid)
    base = d.ROOT/'examples/results/2026-10-14-default-preview-pilot/default-pilot-chain/.superpowers/default-pilot/chain/bundle'
    names = ('analysis_metadata.json','layer_evidence.json','assessment_ledger.json')
    baseline,_ = reconstruct_payloads(*(json.loads((base/n).read_text()) for n in names))
    data = pd.read_csv(d.ROOT/'examples/results/2026-10-08-layer-fixtures/chain/sample.csv')
    result = a.reuse_original_fixture(d.PRODUCER/'raw',d.manifest(),baseline,data,tmp_path)
    assert result['new_fixture_fits'] == 0
    assert result['audit']['assessments'] == 200
    assert result['audit']['observation_checks'] == 140761
    assert result['comparison']['equivalent']
    assert (tmp_path/'raw/preflight-fixture.zip').read_bytes() == (d.PRODUCER/'raw/preflight-all-attempts.zip').read_bytes()


def test_reuse_fails_closed_on_original_source_or_archive_edit(tmp_path):
    import recursive_preview_design as d
    import recursive_preview_actions as a
    value = d.manifest()
    value['continuation']['archive_sha256'] = 'edited'
    with pytest.raises(ValueError,match='archive'):
        a.reuse_original_fixture(d.PRODUCER/'raw',value,None,None,tmp_path)
    value = d.manifest()
    key = next(k for k in value['sources'] if k.startswith('src/open/'))
    value['sources'][key] = 'edited'
    with pytest.raises(ValueError,match='package source'):
        a.reuse_original_fixture(d.PRODUCER/'raw',value,None,None,tmp_path)


def test_reuse_rejects_edited_receipt_and_changed_computation_boundary(tmp_path,monkeypatch):
    import recursive_preview_design as d
    import recursive_preview_actions as a
    from types import SimpleNamespace
    from open.config import load_config
    value = d.manifest()
    value['continuation']['fixture_receipt_sha256'] = 'edited'
    baseline = SimpleNamespace(config=load_config(d.ROOT/'configs/default.yaml'))
    with pytest.raises(ValueError,match='receipt fingerprint'):
        a.reuse_original_fixture(d.PRODUCER/'raw',value,baseline,None,tmp_path)
    original_load = a.load_json
    def changed_control(path):
        result = original_load(path)
        if Path(path).name == 'preflight.json':
            result['planned_cases'] = {'160':{'status':'complete'}}
        return result
    monkeypatch.setattr(a,'load_json',changed_control)
    with pytest.raises(ValueError,match='computation boundary'):
        a.reuse_original_fixture(d.PRODUCER/'raw',d.manifest(),baseline,None,tmp_path)


def test_preflight_acceptance_requires_exact_zero_refit_provenance():
    import recursive_preview_design as d
    import recursive_preview_actions as a
    value = d.manifest()
    proof = value['continuation']
    report = dict(status='PASS',manifest=d.digest(value),equivalence=True,
        planned_cases={str(i):dict(index=i,seconds=1,status='complete') for i in (160,240)},
        storage_feasible=True,audit_feasible=True,archive_crc_checked=True,
        fixture_reused=True,fixture_new_fits=0,fixture_producer_run=proof['producer_run_id'],
        fixture_producer_manifest=proof['producer_manifest_digest'],
        raw_sha256={'preflight-fixture.zip':proof['archive_sha256']})
    report['digest'] = d.digest(report)
    assert a.validate_preflight(report,value)
    for key,bad in [('fixture_reused',False),('fixture_new_fits',200),
                    ('fixture_producer_run','other'),('fixture_producer_manifest','other')]:
        edited = copy.deepcopy(report)
        edited[key] = bad
        del edited['digest']
        edited['digest'] = d.digest(edited)
        with pytest.raises(ValueError,match='preflight'):
            a.validate_preflight(edited,value)
