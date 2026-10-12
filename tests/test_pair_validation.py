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


def test_reversed_duplicate_cannot_consume_two_slots():
    _,validate=api()
    X,y=sample(n=120)
    with pytest.raises(ValueError,match='distinct'):
        validate(X,y,tuple(range(80)),tuple(range(80,120)),
            (('a','b'),('b','a'),('a','noise')))


def unstable_correlated_case():
    # Frozen D-034 case 213: preserve the original data, not a new tuned example.
    from sklearn.model_selection import KFold
    rng=np.random.default_rng(68000213)
    x=rng.normal(size=(300,5));rho=.7
    x[:,1]=rho*x[:,0]+np.sqrt(1-rho*rho)*x[:,1]
    mean=x[:,0]+.6*x[:,1]+.8*(x[:,0]*x[:,1]-rho)
    variance=1+.36+1.2*rho+.64*(1+rho*rho)
    y=mean+rng.normal(scale=np.sqrt(variance*2/3),size=300)
    X=pd.DataFrame(x,columns=['x1','x2','x3','x4','x5'])
    train,test=list(KFold(3,shuffle=True,random_state=88000213).split(X))[1]
    return X,y,train,test


def test_nearly_constant_joint_feature_cannot_explode_native_prediction():
    from open.pair_validation import _estimator
    X,y,train,test=unstable_correlated_case()
    model=_estimator((0,1)).set_params(curves__n_knots=5,ridge__alpha=100.)
    model.fit(X.iloc[train],y[train])
    predicted=model.predict(X.iloc[test])
    # Original prediction was 88.2759, joint MSE 86.052. Bounds check the
    # failure's practical impact without asserting an interaction is detected.
    assert abs(predicted[list(test).index(291)])<15
    assert np.square(y[test]-predicted).mean()<8


def test_nested_comparison_does_not_reintroduce_native_explosion():
    Config,validate=api()
    X,y,train,test=unstable_correlated_case()
    r=validate(X,y,tuple(train.tolist()),tuple(test.tolist()),(('x1','x2'),),
        config=Config(seed=88000213))
    p=r['pairs'][0]
    assert p['status']=='available' and p['joint_mse']<8
    assert np.max(np.abs(p['predicted']))<15


def test_stabilized_result_identifies_scaling_version_and_survives_unit_changes():
    Config,validate=api()
    X,y=sample(n=120)
    options=dict(config=Config(knots=(3,),alphas=(.01,)))
    train=tuple(range(80));assessment=tuple(range(80,120))
    a=validate(X,y,train,assessment,(('a','b'),),**options)
    changed=X.copy();changed.a=1000*changed.a+2000;changed.b=.01*changed.b-50
    b=validate(changed,y,train,assessment,(('a','b'),),**options)
    assert a['schema']=='open.experimental_pair_validation.v2'
    assert a['feature_scaling']=='standardized_marginals_natural_joint'
    np.testing.assert_allclose(a['pairs'][0]['predicted'],b['pairs'][0]['predicted'],rtol=1e-7,atol=1e-7)
