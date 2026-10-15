"""Regression for the aggregate's 100-of-112 artifact download failure."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


def test_inventory_fetches_all_pages_before_selecting_original_cases():
    import qualification_artifacts as a
    records = [dict(id=i+1, name=f'qualification-noise-{i+1}', expired=False) for i in range(112)]
    calls = []
    def api(path):
        calls.append(path)
        page = int(path.rsplit('page=',1)[1])
        return dict(total_count=112, artifacts=records[(page-1)*100:page*100])
    selected = a.select_artifacts('38655036215', [r['name'] for r in records[40:]], api=api)
    assert len(selected) == 72 and selected[-1]['id'] == 112
    assert len(calls) == 2 and calls[-1].endswith('page=2')


@pytest.mark.parametrize('kind', ['missing','duplicate','expired','short-page'])
def test_incomplete_or_ambiguous_inventory_fails_closed(kind):
    import qualification_artifacts as a
    records = [dict(id=1,name='wanted',expired=False)]
    total = 1
    if kind == 'missing':
        records[0]['name'] = 'other'
    elif kind == 'duplicate':
        records.append(dict(id=2,name='wanted',expired=False));total=2
    elif kind == 'expired':
        records[0]['expired'] = True
    else:
        total = 112
    with pytest.raises(ValueError):
        a.select_artifacts('123', ['wanted'], api=lambda p:dict(total_count=total,artifacts=records))


def test_downloader_uses_exact_required_names_and_keeps_artifact_directories(tmp_path):
    import qualification_artifacts as a
    calls = []
    records = [dict(id=1,name='qualification-noise-1',expired=False),
               dict(id=2,name='qualification-noise-135',expired=False)]
    a.download_artifacts('38655036215', [r['name'] for r in records], tmp_path,
                         api=lambda p:dict(total_count=2,artifacts=records),
                         run=lambda args,**kw:calls.append(args))
    assert len(calls) == 2
    assert all('--name' in args and '--pattern' not in args for args in calls)
    assert calls[1][-1] == str(tmp_path/'qualification-noise-135')


def test_offline_replay_rejects_unrecorded_api_requests():
    import recursive_preview_runner as runner
    original = runner.NATIVE_ASSESS
    p = Path(__file__).parents[1]/'examples/results/2026-10-14-recursive-qualification/combined-38676310704/reproduce.py'
    spec = importlib.util.spec_from_file_location('combined_replay',p)
    module = importlib.util.module_from_spec(spec)
    # The archived CLI intentionally installs a process-wide no-fit guard.
    # Scope that import side effect to this test's import, not the whole suite.
    with patch.object(runner,'NATIVE_ASSESS',original):
        spec.loader.exec_module(module)
    assert runner.NATIVE_ASSESS is original, 'archive import leaked its no-fit guard'
    with pytest.raises(ValueError,match='unexpected API'):
        module.saved_api('actions/runs/another-run')


def test_replay_can_stage_preserved_inputs_without_ignored_downloads(tmp_path,monkeypatch):
    import recursive_preview_runner as runner
    original = runner.NATIVE_ASSESS
    p = Path(__file__).parents[1]/'examples/results/2026-10-14-recursive-qualification/combined-38676310704/reproduce.py'
    spec = importlib.util.spec_from_file_location('staging_replay',p)
    module = importlib.util.module_from_spec(spec)
    with patch.object(runner,'NATIVE_ASSESS',original):
        spec.loader.exec_module(module)
    assert runner.NATIVE_ASSESS is original
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'BASE',tmp_path/'combined')
    monkeypatch.setattr(module.c,'completed',lambda:[dict(index=1)])
    monkeypatch.setattr(module.c,'pending',lambda:[dict(index=4)])
    original = tmp_path/'examples/results/2026-10-14-recursive-qualification/noise-38655036215/raw'
    original.mkdir(parents=True)
    (original/'case-1.zip').write_bytes(b'original archive')
    (module.BASE/'raw').mkdir(parents=True)
    for name in ('continued-case-4.zip','continued-control-4.json','original-receipt-4.json','reuse-audit-4.json'):
        (module.BASE/'raw'/name).write_bytes(name.encode())
    inputs,producer_inputs = module.stage_preserved(tmp_path/'staged')
    assert (producer_inputs/'qualification-noise-1/raw/case.zip').read_bytes() == b'original archive'
    assert (inputs/'qualification-continuation-4/raw/case.zip').read_bytes() == b'continued-case-4.zip'
    assert (inputs/'qualification-continuation-4/provenance/original-receipt.json').exists()
    with pytest.raises(FileExistsError):
        module.stage_preserved(tmp_path/'staged')
