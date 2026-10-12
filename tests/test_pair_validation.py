from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def api():
    assert (Path(__file__).parents[1]/'src/open/pair_validation.py').exists(), 'validation helper missing'
    from open.pair_validation import PairValidationConfig, validate_pairs
    return PairValidationConfig, validate_pairs


def sample(kind='product',n=600):
    rng=np.random.default_rng(394)
    X=pd.DataFrame(rng.normal(size=(n,3)),columns=['a','b','noise'])
    y=X.a+.6*X.b**2
    if kind=='product':y=y+1.5*X.a*X.b
    return X,y.to_numpy()+rng.normal(scale=.1,size=n)


def test_true_joint_gain_and_shared_baseline_with_bounded_fits():
    Config,validate=api()
    X,y=sample()
    c=Config(knots=(3,),alphas=(.01,1.),max_pairs=2,max_fits=30)
    r=validate(X,y,tuple(range(400)),tuple(range(400,600)),(('a','b'),('a','noise')),config=c)
    assert r['additional_fits']==21  # seven baseline, two seven-fit joint searches
    assert r['pairs'][0]['gain']>.4
    assert r['pairs'][1]['gain']<.01
    assert r['pairs'][0]['baseline_predicted']==r['pairs'][1]['baseline_predicted']


def test_separate_curved_effects_do_not_need_joint_model():
    Config,validate=api()
    X,y=sample('additive')
    r=validate(X,y,tuple(range(400)),tuple(range(400,600)),(('a','b'),),
        config=Config(knots=(3,),alphas=(.01,1.)))
    assert r['pairs'][0]['gain']<.01


def test_assessment_outcomes_do_not_select_settings_or_predictions():
    Config,validate=api()
    X,y=sample(n=120)
    kwargs=dict(config=Config(knots=(3,),alphas=(.1,)))
    a=validate(X,y,tuple(range(80)),tuple(range(80,120)),(('a','b'),),**kwargs)
    y[80:]+=100
    b=validate(X,y,tuple(range(80)),tuple(range(80,120)),(('a','b'),),**kwargs)
    assert a['baseline_parameters']==b['baseline_parameters']
    assert a['pairs'][0]['parameters']==b['pairs'][0]['parameters']
    assert a['pairs'][0]['predicted']==b['pairs'][0]['predicted']
    assert a['pairs'][0]['gain']!=b['pairs'][0]['gain']


def test_invalid_splits_and_insufficient_budget_stay_explicit():
    Config,validate=api()
    X,y=sample(n=120)
    with pytest.raises(ValueError,match='disjoint'):
        validate(X,y,tuple(range(80)),tuple(range(70,120)),(('a','b'),))
    r=validate(X,y,tuple(range(80)),tuple(range(80,120)),(('a','b'),),
        config=Config(knots=(3,),alphas=(1.,),max_fits=4))
    assert r['additional_fits']==4
    assert r['pairs'][0]['status']=='fit_budget' and r['pairs'][0]['gain'] is None
