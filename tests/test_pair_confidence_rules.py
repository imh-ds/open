import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    p = Path(__file__).parents[1]/'examples/pair_confidence_rules.py'
    assert p.exists(), 'confidence rule implementation missing'
    spec = importlib.util.spec_from_file_location('confidence_rules', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def folds(overall=1., matched=1.):
    return [dict(n=n, reference_mse=r, baseline_mse=3., joint_mse=3.-overall,
                 matched_additive_mse=3.-overall+matched) for n,r in ((2,3.),(1,10.),(1,2.))]


def test_pool_and_eligibility():
    m=module(); f=folds()
    assert m.pooled(f,'matched')==pytest.approx(4/18)
    assert m.pair_score(f)==pytest.approx(4/18)
    assert m.pair_score(folds(overall=-1.))==0.
    f[0]['matched_additive_mse']=f[0]['joint_mse']-.1
    f[1]['matched_additive_mse']=f[1]['joint_mse']-.1
    assert m.pair_score(f)==0.
    f[0]['reference_mse']=float('nan')
    with pytest.raises(ValueError):m.pair_score(f)


def test_threshold_learning_requires_complete_unique_valid_null_rows():
    m=module(); planned=[dict(id=f'{s}{i}',cell='a',scenario=s) for s in ('noise','additive') for i in range(100)]
    rows=[dict(**c,score=i/1000) for i,c in enumerate(planned)]
    assert m.learn_thresholds(rows,planned)=={'a':pytest.approx(.199)}
    for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[dict(rows[-1],score=float('nan'))]):
        with pytest.raises(ValueError):m.learn_thresholds(bad,planned)
    assert not m.qualifies(.199,.199) and m.qualifies(.2,.199)
    with pytest.raises(ValueError):m.qualifies(1.,-1.)


def test_exact_operating_bounds():
    m=module()
    assert m.false_alarm_bound(0,200)==pytest.approx(.02951304960704)
    assert m.false_alarm_bound(2,200)<.05<m.false_alarm_bound(3,200)
    assert m.detection_bound(28,30)>.8>m.detection_bound(27,30)
    assert m.false_alarm_bound(0,0)==1. and m.detection_bound(0,0)==0.


def test_pattern_unknown_and_shape_agreement():
    m=module(); a=np.arange(25.).reshape(5,5); mask=np.ones((5,5),bool)
    curves=[dict(contrast=a.tolist(),support=mask.tolist()) for _ in range(3)]
    assert m.pattern_summary(curves)['consistent']
    curves[1]['contrast']=(-a).tolist()
    assert not m.pattern_summary(curves)['consistent']
    for c in curves:c['support']=np.zeros((5,5),bool).tolist()
    assert m.pattern_summary(curves)['status']=='unresolved_support'
    for c in curves:c.update(support=mask.tolist(),contrast=np.zeros((5,5)).tolist())
    assert m.pattern_summary(curves)['status']=='unresolved_magnitude'


def test_fresh_design_counts_and_seed_separation():
    m=module(); a=m.tasks('calibration');b=m.tasks('confirmation')
    assert len(a)==2000 and len(b)==5200 and len(m.CELLS)==10
    assert len({c['id'] for c in a+b})==7200
    assert len({c['seed'] for c in a+b})==7200
    assert not {c['test_seed'] for c in a}&{c['test_seed'] for c in b}
    assert len(m.shards('calibration'))==50 and len(m.shards('confirmation'))==130
