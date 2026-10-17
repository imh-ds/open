"""OLS obeys the learner contract without coefficient shrinkage or tuning fits."""
import numpy as np
import pandas as pd
import pytest

from open.config import OpenConfig, LearnerConfig, AnalysisConfig, PerformanceCVConfig
from open.learners import make_predictor
from open.risk import evaluate_risk
from open.splits import make_split_plan


def test_ols_matches_unpenalized_least_squares_and_clone_is_unfitted():
    config=OpenConfig(analysis=AnalysisConfig(root='Y',random_seed=7),learner=LearnerConfig(library=('ols',)))
    model=make_predictor(config)
    X=pd.DataFrame(dict(x=[0.,1,2,3,4,5],z=[2.,0,3,1,5,4]))
    y=1+2*X.x-.3*X.z
    model.fit(X,y,params={})
    np.testing.assert_allclose(model.predict(X),y,atol=1e-12)
    raw=model._pipeline['regressor'].coef_/model._pipeline['preprocessor']['scaler'].scale_
    np.testing.assert_allclose(raw,[2,-.3],atol=1e-12)
    assert model.tuning_grid_size==0
    with pytest.raises(RuntimeError):
        model.clone().predict(X)
    with pytest.raises(ValueError):
        model.fit(X,y,params={'alpha':.1})


def test_ols_heldout_risk_has_no_tuning_fits_and_reuses_splits():
    config=OpenConfig(analysis=AnalysisConfig(root='Y',random_seed=7),learner=LearnerConfig(library=('ols',)),
                      performance_cv=PerformanceCVConfig(folds=2,repeats=1))
    data=pd.DataFrame(dict(Y=[1.,3,4,7,8,11,12,15,16,19,20,23],
                           X1=list(range(12)),Z=[0.,1,0,1,0,1,0,1,0,1,0,1]))
    plan=make_split_plan(data,config)
    full=evaluate_risk(data,target='Y',candidate_set=('X1','Z'),predictors=('X1','Z'),
                       split_plan=plan,predictor=make_predictor(config),tuning_folds=2)
    params={(f.repeat_index,f.fold_index):f.fitted_parameters for f in full.folds}
    reduced=evaluate_risk(data,target='Y',candidate_set=('X1','Z'),predictors=('X1',),
                          split_plan=plan,predictor=make_predictor(config),fitted_parameters=params)
    assert full.learner==reduced.learner=='ols'
    assert full.tuning_model_fit_count==reduced.tuning_model_fit_count==0
    assert full.outer_model_fit_count==reduced.outer_model_fit_count==2
    assert all(f.fitted_parameters=={} for f in full.folds+reduced.folds)
    assert full.split_plan_id==reduced.split_plan_id
    assert full.aggregate_risk<reduced.aggregate_risk


def test_ols_effect_supplement_reports_real_coefficients_and_zero_search():
    from open.effects import compare_target_engines
    from test_effects import comparison_fixture
    frame,config,plan=comparison_fixture()
    result=compare_target_engines(frame,'y',('x','z'),config,plan,engines=('ols',),bins=3)
    row=result['engines'][0]
    assert row['engine']=='ols' and row['tuning_grid_size']==0
    assert row['tuning_model_fit_count']==0
    assert all(r['b'] is not None and r['beta'] is not None for r in row['effects'])


def test_recursive_selection_prunes_ols_nuisance_without_a_coefficient_penalty():
    from open.config import StabilityConfig,TuningCVConfig
    from open.layer_stability import fit_layer_consensus
    data=pd.DataFrame(dict(Y=1+2*np.arange(24,dtype=float),
                           X=np.arange(24,dtype=float),Z=[0.,1.]*12))
    config=OpenConfig(analysis=AnalysisConfig('Y',7,max_depth=2),
        learner=LearnerConfig(library=('ols',)),performance_cv=PerformanceCVConfig(2,1),
        tuning_cv=TuningCVConfig(2),stability=StabilityConfig(n_resamples=2,scope='layer_consensus'))
    result=fit_layer_consensus(data,'Y',tuple(data),config)
    assert result.status=='COMPLETE'
    assert result.assigned_depth=={'Y':0,'X':1}
    selected=[(r.source,r.target) for layer in result.layers for target in layer.evidence
              for r in target.relations if r.qualifies]
    assert selected==[('X','Y')]
