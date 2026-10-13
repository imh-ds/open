import sys,importlib.util
from pathlib import Path
import pytest


def load():
    p=Path(__file__).parents[1]/'examples/prospective_interaction_report.py'
    assert p.exists(),'prospective reporting missing'
    sys.path.insert(0,str(p.parent));s=importlib.util.spec_from_file_location('prospective_report',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_null_selection_and_incomplete_gate():
    m=load();tasks=m.study.d.stage_tasks('null');rows=[]
    for c in tasks:rows.append(dict(id=c['id'],cell=c['cell'],benchmark=c['benchmark'],control=False,detected=False,displayed=False,shape_correct=False))
    s,passed=m.strata_summary('null',rows,tasks);assert len(s)==12 and len(passed)==6
    s,passed=m.strata_summary('null',rows[:-1],tasks);assert len(passed)==5 and s[-1]['status']=='incomplete'


def test_zero_curves_and_skipped_signal_denominators():
    m=load();tasks=m.study.d.stage_tasks('signal',{'cells':[m.study.d.cell_ids()[0]]})
    rows=[dict(id=c['id'],cell=c['cell'],benchmark=c['benchmark'],control=False,detected=True,displayed=False,shape_correct=False) for c in tasks]
    s,unused=m.strata_summary('signal',rows,tasks)
    assert len(s)==4 and all(r['detection']['pass'] for r in s)
    assert all(r['shape']['bound'] is None and r['shape']['n']==0 for r in s)


def test_duplicate_conflicts_and_foreign_folders(tmp_path):
    m=load();a=tmp_path/'a';b=tmp_path/'b';out=tmp_path/'out'
    for p in (a,b):(p/'prospective--x').mkdir(parents=True);(p/'prospective--x'/'result.json').write_text('{}')
    receipt=m.merge_inputs([a,b],out);assert receipt['identical_duplicates']==1
    (b/'prospective--x'/'result.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):m.merge_inputs([a,b],out)


def test_plain_report_withholds_amplitude_claim(tmp_path):
    m=load();m.write_report(dict(stage='null',planned=14400,complete=0,missing=14400,audit_errors=[],passed_cells=[]),[],tmp_path)
    s=(tmp_path/'REPORT.md').read_text();assert 'unknown' in s and 'magnitude' in s and 'No' in s


def test_partial_receipt_context_and_unknown_failure_cost(tmp_path,monkeypatch):
    m=load();c=m.study.d.tasks()[0];out=tmp_path/'out'
    m.study.io.write(out/c['id']/'receipt.json',dict(task=c,context='foreign',environment=m.study.io.environment()))
    a=m.aggregate('null',[out],out,m.study.manifest())
    assert a['missing']==14400 and a['audit_errors'] and c['id'] in a['unknown_attempt_cost_ids']
    assert not (out/'proposed-selection.json').exists()
