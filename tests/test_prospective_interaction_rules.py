import importlib.util
import sys
from pathlib import Path
import pytest


def load():
    path=Path(__file__).parents[1]/'examples/prospective_interaction_rules.py'
    assert path.exists(), 'prospective endpoint rules missing'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('prospective_rules',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('endpoint,n,k', [('false',1200,36),('false',450,8),('detection',450,387),('shape',200,194)])
def test_exact_gate_boundaries(endpoint,n,k):
    m=load();assert m.bounds(k,n,endpoint)['pass']
    assert not m.bounds(k+(1 if endpoint=='false' else -1),n,endpoint)['pass']


def test_unknown_and_invalid_counts():
    m=load();assert m.bounds(0,0,'shape')['bound'] is None
    for k,n,e in [(-1,10,'false'),(11,10,'shape'),(True,10,'false'),(1.5,10,'detection'),(0,10,'other')]:
        with pytest.raises(ValueError):m.bounds(k,n,e)


def test_partial_and_separate_shape_support():
    m=load();planned=[dict(id=str(i)) for i in range(450)]
    rows=[dict(id=str(i),detected=True,control=False,displayed=False,shape_correct=False) for i in range(450)]
    r=m.summarize_stratum(rows,planned,'detection');assert r['pass']
    assert not m.summarize_stratum(rows[:-1],planned,'detection')['pass']
    shape=m.summarize_stratum(rows,planned,'shape');assert shape['status']=='unresolved'
    nulls=[m.bounds(0,1200,'false')]*2
    signal=dict(detection=r,control=m.bounds(0,450,'false'),shape=shape)
    q=m.qualify_condition(nulls,signal);assert q['predictive_support'] and not q['curve_pattern_support']
    assert not m.qualify_condition([m.bounds(40,1200,'false'),nulls[0]],signal)['predictive_support']


def test_duplicate_and_foreign_denominators_rejected():
    m=load();planned=[dict(id='a')]
    for rows in ([dict(id='a'),dict(id='a')],[dict(id='foreign')]):
        with pytest.raises(ValueError):m.summarize_stratum(rows,planned,'false')
