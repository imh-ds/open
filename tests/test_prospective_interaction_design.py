import sys,json,importlib.util
from pathlib import Path
import pytest


def load():
    p=Path(__file__).parents[1]/'examples/prospective_interaction_design.py'
    assert p.exists(),'prospective manifest design missing'
    sys.path.insert(0,str(p.parent));s=importlib.util.spec_from_file_location('prospective_design',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_universe_shards_and_coefficients():
    m=load();tasks=m.tasks();assert len(tasks)==25200
    assert len([c for c in tasks if c['stage']=='null'])==14400
    assert len(m.shards('null'))==180 and len(m.shards('signal',{'cells':list(m.cell_ids())}))==135
    assert len(m.shard_tasks('null','179'))==80
    for c in tasks:
        if c['target_gap']:assert abs(m.d.previous.models.oracle_gap(c)-c['target_gap'])<1e-13
    assert m.audit_seed_collisions(tasks,[])['unique_seeds']==75600
    with pytest.raises(ValueError):m.audit_seed_collisions(tasks,[{'seed':tasks[0]['seed']}])
    with pytest.raises(ValueError):m.shard_tasks('null','180')
    assert m.shards('signal',{'cells':[]})==[]


def test_selection_context_completion_and_skips():
    m=load();manifest={'identity':'example'};audit=dict(stage='null',manifest='example',complete=14400,missing=0,audit_errors=[],passed_cells=[m.cell_ids()[0]])
    import prospective_interaction_rules as rules
    audit['strata']=[dict(cell=cell,benchmark=scenario,**rules.bounds(0 if cell==m.cell_ids()[0] else 100,1200,'false'))
                     for cell in m.cell_ids() for scenario in ('noise','additive')]
    s=m.make_selection(manifest,audit);assert m.validate_selection(s,manifest,audit)==audit['passed_cells']
    assert len(s['skipped_ids'])==9000 and len(m.shards('signal',s))==23
    for key,value in [('manifest','foreign'),('cells',list(m.cell_ids()))]:
        bad=dict(s);bad[key]=value
        with pytest.raises(ValueError):m.validate_selection(bad,manifest,audit)
    bad=dict(audit,missing=1)
    with pytest.raises(ValueError):m.make_selection(manifest,bad)


def test_selection_recomputes_null_gates_instead_of_trusting_passed_list():
    m=load();manifest={'identity':'example'}
    audit=dict(stage='null',manifest='example',complete=14400,missing=0,audit_errors=[],passed_cells=m.cell_ids(),strata=[])
    with pytest.raises(ValueError):m.make_selection(manifest,audit)


def test_historical_seed_fields_recursively_included():
    m=load();c=m.tasks()[0]
    with pytest.raises(ValueError):m.audit_seed_collisions([c],{'preregistered': [{'test_seed':c['test_seed']}]})
    with pytest.raises(ValueError):m.audit_seed_collisions([c,c],[])
