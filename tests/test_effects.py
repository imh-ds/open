"""Model-agnostic effect definitions and paired evaluation safeguards."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from open.config import AnalysisConfig, OpenConfig, PerformanceCVConfig, TuningCVConfig
from open.splits import make_split_plan


class FunctionModel:
    def __init__(self, function): self.function = function
    def predict(self, X): return self.function(X)


def test_linear_ale_known_contrast_and_unit_invariance():
    from open.effects import accumulated_local_effects
    frame = pd.DataFrame({'x':np.linspace(-2,2,101), 'z':np.linspace(-2,2,101)})
    model = FunctionModel(lambda X: 2*X.x + 9*X.z)
    result = accumulated_local_effects(model,frame,'x',edges=[-2,-1,0,1,2],outcome_sd=4)
    assert result['standardized_contrast'] == pytest.approx(1)
    assert result['status'] == 'available'
    scaled = accumulated_local_effects(FunctionModel(lambda X:100*model.predict(X)),
        frame,'x',edges=[-2,-1,0,1,2],outcome_sd=400)
    assert scaled['standardized_contrast'] == result['standardized_contrast']
    np.testing.assert_allclose(result['standardized_curve'],scaled['standardized_curve'])
    # Perfectly correlated z is held at its local observed values; no credit
    # from z's coefficient leaks into x's ALE differences.
    assert result['raw_contrast'] == pytest.approx(4)
    assert result['standardized_curve_sd'] == pytest.approx(.5*np.std(frame.x,ddof=0))
    assert scaled['standardized_curve_sd'] == pytest.approx(result['standardized_curve_sd'])


def test_u_shape_requires_curve_even_when_endpoint_contrast_is_zero():
    from open.effects import accumulated_local_effects
    frame = pd.DataFrame({'x':np.linspace(-2,2,101)})
    result = accumulated_local_effects(FunctionModel(lambda X:X.x**2),frame,'x',
        edges=[-2,-1,0,1,2],outcome_sd=1)
    assert result['standardized_contrast'] == pytest.approx(0)
    assert np.ptp(result['standardized_curve']) == pytest.approx(4)
    assert result['standardized_curve_sd'] > 1
    assert result['magnitude_status'] == 'available'


def test_empty_bins_missing_values_and_degenerate_scales_are_explicit():
    from open.effects import accumulated_local_effects
    frame=pd.DataFrame({'x':[-2,-1.5,1.5,2,np.nan]})
    model=FunctionModel(lambda X:X.x)
    result=accumulated_local_effects(model,frame,'x',edges=[-2,-1,0,1,2],outcome_sd=1)
    assert result['status']=='unsupported_bins'
    assert result['standardized_contrast'] is None
    assert result['bin_counts']==[2,0,0,2]
    assert result['missing_reference_count']==1
    assert result['standardized_curve_sd'] is None
    assert result['magnitude_status']=='unsupported_bins'
    zero=accumulated_local_effects(model,frame,'x',edges=[-2,0,2],outcome_sd=0)
    assert zero['status']=='constant_outcome' and zero['standardized_contrast'] is None
    assert zero['standardized_curve_sd'] is None
    constant=accumulated_local_effects(model,pd.DataFrame({'x':[1,1,1]}),'x',edges=[1],outcome_sd=1)
    assert constant['status']=='constant_predictor'
    assert constant['standardized_curve_sd'] is None


@pytest.mark.parametrize('sd',[-1,float('nan'),float('inf'),True])
def test_invalid_scale_rejected(sd):
    from open.effects import accumulated_local_effects
    with pytest.raises(ValueError):
        accumulated_local_effects(FunctionModel(lambda X:X.x),pd.DataFrame({'x':[0,1]}),
            'x',edges=[0,1],outcome_sd=sd)


def comparison_fixture():
    rng=np.random.default_rng(8)
    x=rng.normal(size=72)
    frame=pd.DataFrame({'x':x,'z':x+.01*rng.normal(size=72),'y':2*x+rng.normal(0,.1,72)})
    config=OpenConfig(AnalysisConfig('y',13),performance_cv=PerformanceCVConfig(2,2),tuning_cv=TuningCVConfig(2))
    return frame,config,make_split_plan(frame,config)


def test_comparison_matched_folds_joint_removal_and_portable_export(tmp_path):
    from open.effects import compare_target_engines,save_engine_comparison
    frame,config,plan=comparison_fixture()
    result=compare_target_engines(frame,'y',('x','z'),config,plan,
        engines=('elastic_net','gbm'),joint_groups=(('x','z'),),bins=3)
    assert result['schema']=='open.engine_comparison'
    assert len(result['engines'])==2
    a,b=result['engines']
    assert a['heldout_keys']==b['heldout_keys']
    assert len(a['heldout_keys'])==144
    assert a['joint'][0]['delta_cv_r2'] > .8
    assert all(r['engine']=='gbm' and r['beta'] is None for r in b['effects'])
    assert all(r['beta'] is not None for r in a['effects'])
    assert a['effects'][0]['low']==b['effects'][0]['low']
    assert a['effects'][0]['outcome_sd']==b['effects'][0]['outcome_sd']
    assert a['ablations_retuned'] is False
    save_engine_comparison(result,tmp_path)
    loaded=json.loads((tmp_path/'comparison.json').read_text())
    assert loaded==result
    assert len(pd.read_csv(tmp_path/'effects.csv'))==4
    for engine in result['engines']:
        for effect in engine['effects']:
            levels=np.interp(frame[effect['source']],effect['edges'],effect['standardized_curve'])
            assert effect['standardized_curve_sd']==pytest.approx(np.std(levels,ddof=0))
    assert 'standardized_curve_sd' in pd.read_csv(tmp_path/'effects.csv')
    points=pd.read_csv(tmp_path/'percentile_points.csv')
    intervals=pd.read_csv(tmp_path/'interval_contrasts.csv')
    assert len(points)==44 and len(intervals)==40
    assert points['percentile'].tolist()[:11]==list(range(0,101,10))
    for engine in result['engines']:
        for effect in engine['effects']:
            assert len(effect['percentile_points'])==11
            assert len(effect['interval_contrasts'])==10
            assert all(len(f['interval_contrasts'])==10 for f in effect['folds'])
            assert sum(r['standardized_contrast'] for r in effect['interval_contrasts'])==pytest.approx(
                effect['standardized_curve'][-1]-effect['standardized_curve'][0])


def test_missing_recursive_target_not_imputed_and_constant_r2_undefined():
    from open.effects import compare_target_engines
    frame,config,_=comparison_fixture()
    config=replace(config,analysis=replace(config.analysis,root='x'))
    frame.loc[0:3,'y']=np.nan
    plan=make_split_plan(frame,config)
    result=compare_target_engines(frame,'y',('x','z'),config,plan,engines=('elastic_net',),bins=2)
    assert result['observed_target_rows']==68
    assert len(result['engines'][0]['heldout_keys'])==136
    assert not any(key[2]<4 for key in result['engines'][0]['heldout_keys'])
    frame['y']=1.
    result=compare_target_engines(frame,'y',('x',),config,plan,engines=('gbm',),bins=2)
    assert result['engines'][0]['cv_r2'] is None
    assert result['engines'][0]['effects'][0]['standardized_contrast'] is None


def test_invalid_joint_group_and_duplicate_engine_rejected():
    from open.effects import compare_target_engines
    frame,config,plan=comparison_fixture()
    with pytest.raises(ValueError,match='joint'):
        compare_target_engines(frame,'y',('x','z'),config,plan,joint_groups=(('x','foreign'),))
    with pytest.raises(ValueError,match='engines'):
        compare_target_engines(frame,'y',('x','z'),config,plan,engines=('gbm','gbm'))


def test_curve_aggregation_matches_contrast_with_unequal_fold_sizes():
    from open.effects import compare_target_engines
    frame,config,_=comparison_fixture()
    frame=frame.iloc[:-1].copy()
    plan=make_split_plan(frame,config)
    result=compare_target_engines(frame,'y',('x','z'),config,plan,engines=('gbm',),bins=2)
    effect=result['engines'][0]['effects'][0]
    curve=effect['standardized_curve']
    expected=np.interp(effect['high'],effect['edges'],curve)-np.interp(effect['low'],effect['edges'],curve)
    assert effect['standardized_contrast']==pytest.approx(expected,abs=1e-12)


def test_zero_width_contrast_still_has_curve_for_tied_predictor():
    from open.effects import accumulated_local_effects
    frame=pd.DataFrame({'x':[-2,0,0,0,0,0,0,2]})
    result=accumulated_local_effects(FunctionModel(lambda X:X.x**2),frame,'x',edges=[-2,0,2],outcome_sd=1)
    assert result['status']=='zero_width_contrast'
    assert result['standardized_contrast'] is None
    assert len(result['standardized_curve'])==3
    assert np.ptp(result['standardized_curve'])==4
    assert result['standardized_curve_sd']>0
    assert result['magnitude_status']=='available'


def test_flat_curve_zero_magnitude_and_curve_offsets_irrelevant():
    from open.effects import accumulated_local_effects,curve_magnitude
    frame=pd.DataFrame({'x':np.linspace(-2,2,101)})
    result=accumulated_local_effects(FunctionModel(lambda X:np.ones(len(X))),frame,'x',
        edges=[-2,-1,0,1,2],outcome_sd=1)
    assert result['standardized_curve_sd']==0
    assert result['magnitude_status']=='available'
    assert curve_magnitude([-2,0,2],[4,0,4],frame.x)==pytest.approx(
        curve_magnitude([-2,0,2],[104,100,104],frame.x))


def test_real_nonlinear_engines_recover_u_shape_without_surrogate_beta():
    import importlib.util
    from open.effects import compare_target_engines
    x=np.linspace(-2,2,120)
    frame=pd.DataFrame({'x':x,'y':x*x})
    config=OpenConfig(AnalysisConfig('y',13),performance_cv=PerformanceCVConfig(2,1),tuning_cv=TuningCVConfig(2))
    engines=('elastic_net','random_forest','gbm')
    if importlib.util.find_spec('lightgbm') is not None: engines+=('lightgbm',)
    result=compare_target_engines(frame,'y',('x',),config,make_split_plan(frame,config),engines=engines,bins=4)
    linear,*trees=result['engines']
    assert linear['cv_r2']<.1
    for engine in trees:
        effect=engine['effects'][0]
        assert engine['cv_r2']>.8
        assert effect['beta'] is None and effect['b'] is None
        assert np.ptp(effect['standardized_curve'])>1
        assert effect['status']=='available'


def test_outer_training_rows_only_are_supplied_to_tuning(monkeypatch):
    from open.effects import compare_target_engines
    from open.learners.elastic_net import ElasticNetPredictor
    frame,config,plan=comparison_fixture()
    allowed={tuple(split.train_indices) for split in plan.splits}
    original=ElasticNetPredictor.tune
    calls=[]
    def checked(self,X,y,cv,loss='mse'):
        assert tuple(X.index) in allowed
        assert len(X)<len(frame)
        calls.append(tuple(X.index))
        return original(self,X,y,cv,loss)
    monkeypatch.setattr(ElasticNetPredictor,'tune',checked)
    compare_target_engines(frame,'y',('x','z'),config,plan,engines=('elastic_net',),bins=2)
    assert len(calls)==len(plan.splits)


def test_percentile_contrasts_linear_levels_and_outcome_scaling():
    from open.effects import accumulated_local_effects,percentile_effects
    frame=pd.DataFrame({'x':np.linspace(0,10,101)})
    ale=accumulated_local_effects(FunctionModel(lambda X:2*X.x),frame,'x',
        edges=list(range(11)),outcome_sd=4)
    summary=percentile_effects(ale,list(range(11)))
    assert [p['predictor_value'] for p in summary['percentile_points']]==list(range(11))
    assert all(r['raw_contrast']==pytest.approx(2) for r in summary['interval_contrasts'])
    assert all(r['standardized_contrast']==pytest.approx(.5) for r in summary['interval_contrasts'])
    assert summary['percentile_points'][-1]['standardized_ale']-summary['percentile_points'][0]['standardized_ale']==pytest.approx(5)
    scaled=accumulated_local_effects(FunctionModel(lambda X:200*X.x),frame,'x',
        edges=list(range(11)),outcome_sd=400)
    assert [r['standardized_contrast'] for r in percentile_effects(scaled,list(range(11)))['interval_contrasts']]==pytest.approx([.5]*10)


def test_percentile_contrasts_show_u_shape_and_flat_rise_flat():
    from open.effects import accumulated_local_effects,percentile_effects
    frame=pd.DataFrame({'x':np.linspace(-2,2,101)})
    edges=np.linspace(-2,2,11)
    ale=accumulated_local_effects(FunctionModel(lambda X:X.x**2),frame,'x',edges=edges,outcome_sd=1)
    increments=[r['standardized_contrast'] for r in percentile_effects(ale,edges)['interval_contrasts']]
    assert increments==pytest.approx([-1.44,-1.12,-.8,-.48,-.16,.16,.48,.8,1.12,1.44])
    frame=pd.DataFrame({'x':np.linspace(0,10,101)})
    ale=accumulated_local_effects(FunctionModel(lambda X:np.clip(X.x-3,0,3)),frame,'x',edges=list(range(11)),outcome_sd=2)
    assert [r['standardized_contrast'] for r in percentile_effects(ale,list(range(11)))['interval_contrasts']]==pytest.approx([0,0,0,.5,.5,.5,0,0,0,0])


def test_percentile_intervals_do_not_interpolate_across_unsupported_bins():
    from open.effects import accumulated_local_effects,percentile_effects
    frame=pd.DataFrame({'x':[0,.5,9.5,10]})
    ale=accumulated_local_effects(FunctionModel(lambda X:X.x),frame,'x',edges=list(range(11)),outcome_sd=2)
    summary=percentile_effects(ale,list(range(11)))
    assert all(p['standardized_ale'] is None for p in summary['percentile_points'])
    assert summary['interval_contrasts'][0]['standardized_contrast']==pytest.approx(.5)
    assert summary['interval_contrasts'][-1]['standardized_contrast']==pytest.approx(.5)
    assert all(r['status']=='unsupported_bins' and r['standardized_contrast'] is None for r in summary['interval_contrasts'][1:-1])


def test_percentile_ties_and_constant_scales_remain_explicit():
    from open.effects import accumulated_local_effects,percentile_effects
    frame=pd.DataFrame({'x':[0,0,0,0,0,0,0,0,1,2]})
    positions=np.quantile(frame.x,np.linspace(0,1,11))
    ale=accumulated_local_effects(FunctionModel(lambda X:X.x),frame,'x',edges=[0,1,2],outcome_sd=1)
    summary=percentile_effects(ale,positions)
    assert summary['interval_contrasts'][0]['status']=='zero_width_contrast'
    assert summary['interval_contrasts'][0]['standardized_contrast'] is None
    assert summary['percentile_points'][0]['status']=='available'
    zero=accumulated_local_effects(FunctionModel(lambda X:X.x),frame,'x',edges=[0,1,2],outcome_sd=0)
    assert all(r['status']=='constant_outcome' and r['standardized_contrast'] is None for r in percentile_effects(zero,positions)['interval_contrasts'])
    constant=accumulated_local_effects(FunctionModel(lambda X:X.x),pd.DataFrame({'x':[1,1]}),'x',edges=[1],outcome_sd=1)
    assert all(r['status']=='constant_predictor' for r in percentile_effects(constant,[1]*11)['interval_contrasts'])
    for invalid in ([0,1],list(range(10))+[float('nan')],list(reversed(range(11)))):
        with pytest.raises(ValueError): percentile_effects(ale,invalid)


def test_adaptive_resolution_checks_all_heldout_folds_before_fitting():
    from open.effects import compare_target_engines
    frame,config,plan=comparison_fixture()
    result=compare_target_engines(frame,'y',('x','z'),config,plan,engines=('gbm',),bins=30,adaptive_bins=True)
    assert 2<=result['estimation_bins']<30
    assert result['requested_bins']==30
    assert all(f['magnitude_status']=='available' for e in result['engines'][0]['effects'] for f in e['folds'])
    assert all(len(e['percentile_points'])==11 for e in result['engines'][0]['effects'])


def test_absent_reference_percentiles_are_not_fabricated():
    from open.effects import accumulated_local_effects,percentile_effects
    frame=pd.DataFrame({'x':[np.nan,np.nan]})
    ale=accumulated_local_effects(FunctionModel(lambda X:X.x),frame,'x',edges=[0],outcome_sd=1)
    summary=percentile_effects(ale,[None]*11)
    assert all(p['predictor_value'] is None and p['status']=='no_reference_values' for p in summary['percentile_points'])
    assert all(r['low'] is None and r['high'] is None and r['status']=='no_reference_values' for r in summary['interval_contrasts'])


def test_effect_supplement_exports_effective_singleton_grid():
    from open.config import AnalysisConfig, OpenConfig, PerformanceCVConfig, TuningCVConfig
    from open.effects import compare_target_engines
    from open.splits import make_split_plan
    frame = pd.DataFrame({'x': np.linspace(-2, 2, 24)})
    frame['y'] = frame.x**2
    config = OpenConfig(AnalysisConfig('y', 17),
                        performance_cv=PerformanceCVConfig(2, 1),
                        tuning_cv=TuningCVConfig(3))
    result = compare_target_engines(frame, 'y', ('x',), config,
        make_split_plan(frame, config), engines=('random_forest',), bins=2)
    engine = result['engines'][0]
    assert engine['tuning_grid'] == {'min_samples_leaf': [2, 5], 'max_features': [.7]}
    assert engine['tuning_grid_size'] == 2
    assert engine['tuning_model_fit_count'] == 12
