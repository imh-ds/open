import importlib.util
from pathlib import Path
import pytest
from sklearn.model_selection import KFold


def module():
    p=Path(__file__).parents[1]/'examples/results/2026-10-13-prospective-interaction-null/recompute_results.py'
    spec=importlib.util.spec_from_file_location('null_evidence',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture():
    c=dict(n=9,split_seed=123)
    splits=[(list(a),list(b)) for a,b in KFold(3,shuffle=True,random_state=123).split(range(9))]+[(list(range(9)),list(range(9,1009)))]
    v=dict(comparisons=[dict(train_indices=a,assessment_indices=b,pairs=[dict(pair=[0,1]),dict(pair=[0,2])]) for a,b in splits])
    names=['receipt.json','result.json','timing.json','attempts.json']+[f'comparison-{i}.json' for i in range(4)]
    return c,v,dict(digests={n:'hash' for n in names})


def test_split_pair_and_digest_shape_tampering():
    m=module();assert hasattr(m,'check_structure'),'independent structural guard missing'
    c,v,verified=fixture();m.check_structure(c,v,verified)
    for kind in ('fold','pair','overlap','digest'):
        c,v,verified=fixture()
        if kind=='fold':v['comparisons'].pop()
        if kind=='pair':v['comparisons'][0]['pairs'].reverse()
        if kind=='overlap':v['comparisons'][0]['train_indices'].append(v['comparisons'][0]['assessment_indices'][0])
        if kind=='digest':verified['digests']={}
        with pytest.raises(AssertionError):m.check_structure(c,v,verified)


def test_threshold_digest_is_pinned():
    m=module();assert hasattr(m,'check_threshold'),'frozen threshold guard missing'
    with pytest.raises(AssertionError):m.check_threshold({'thresholds_sha256':'wrong'},b'{}')


def test_selection_is_bound_to_original_audit():
    m=module();assert hasattr(m,'check_selection'),'exact selection guard missing'
    with pytest.raises(AssertionError):m.check_selection({},dict(passed_cells=[]),dict(tasks=[]))
