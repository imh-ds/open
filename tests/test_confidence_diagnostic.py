import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import beta


def module():
    root=Path(__file__).parents[1];p=root/'examples/confidence_diagnostic.py'
    assert p.exists(), 'diagnostic helpers missing'
    sys.path.insert(0,str(root/'examples'))
    spec=importlib.util.spec_from_file_location('confidence_diagnostic',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_assurance_count_and_hypothetical_power():
    m=module();assert m.allowable_count(200)==2
    assert beta.ppf(1-.05/20,3,198)<.05<beta.ppf(1-.05/20,4,197)
    a=m.assurance(200,.01);assert a['single_pass']==pytest.approx(.6766786945356569)
    assert a['all_pass']==pytest.approx(.000405181917810057)
    assert a['assumption']=='independent strata with equal fixed true rates; hypothetical'
    assert m.assurance(200,.005)['all_pass']==pytest.approx(.18935307715142352)
    assert m.assurance(1000,.005)['all_pass']>m.assurance(200,.005)['all_pass']


def test_maximum_calibration_uncertainty_is_not_a_posterior():
    m=module();a=m.calibration_reference(100,.05)
    assert a['expected_exceedance']==pytest.approx(1/101)
    assert a['failure_per_class']==pytest.approx(.95**100)
    assert a['minimum_m_for_union_bound']==117
    assert a['union_failure_upper']> .05
    assert m.calibration_reference(117,.05)['union_failure_upper']<=.05


def test_matched_population_reference_and_unattainable_targets():
    m=module()
    for rho in (0.,.7):
        for q in (.01,.04):
            b=m.matched_coefficient(q,rho);c=dict(scenario='mixed',rho=rho,b=b)
            assert m.previous.models.oracle_gap(c)==pytest.approx(q)
    assert m.matched_coefficient(.04,.7)>m.matched_coefficient(.04,0)
    for q,rho in ((0.,0.),(.08,.7),(.04,.9),(float('nan'),0.),(.01,1.)):
        with pytest.raises(ValueError):m.matched_coefficient(q,rho)


def test_fresh_balanced_tasks_and_budget():
    m=module();t=m.tasks();assert len(t)==60 and len(m.shards())==6
    assert len({c['id'] for c in t})==60
    old=m.previous.rules.tasks('calibration')+m.previous.rules.tasks('confirmation')
    oldseeds={c[k] for c in old for k in ('seed','test_seed','split_seed')}
    newseeds={c[k] for c in t for k in ('seed','test_seed','split_seed')}
    assert len(newseeds)==180 and not newseeds&oldseeds
    assert sum(c['scenario']=='mixed' for c in t)==48
    assert all(sum((c['n'],c['rho'],c['target_gap'])==(n,r,q) for c in t)==6 for n in (300,600) for r in (0.,.7) for q in (.01,.04))
    assert sum(c['scenario']=='noise' for c in t)==6 and sum(c['scenario']=='additive' for c in t)==6
    assert m.BUDGET_MINUTES==125


def test_oracle_additive_projection_and_orthogonal_remainder():
    m=module();c=dict(scenario='mixed',rho=.7,b=m.matched_coefficient(.04,.7),p=5)
    X,y,truth=m.previous.models.generate(c,60000,708000001)
    a,j=m.population_predictions(X,c);np.testing.assert_allclose(j,truth)
    remainder=j-a
    assert np.mean(remainder**2)==pytest.approx(c['b']**2*(1-.7**2)**2/(1+.7**2),rel=.04)
    assert abs(np.mean(remainder*(X.x1**2-1)))<.06
    c['scenario']='additive';a,j=m.population_predictions(X,c);np.testing.assert_array_equal(a,j)
    c['scenario']='noise';a,j=m.population_predictions(X,c);assert np.array_equal(a,np.zeros(len(X)))


def test_raw_decision_attribution_keeps_nonpositive_and_fold_reasons():
    m=module();f=[dict(n=10,reference_mse=1.,baseline_mse=1.,joint_mse=.96,matched_additive_mse=1.) for _ in range(3)]
    with pytest.raises(ValueError):m.decision(f[:2],.05)
    d=m.decision(f,.05);assert d['eligible'] and not d['above_threshold'];assert d['score']==pytest.approx(.04)
    assert d['reasons']==['fixed_threshold_not_exceeded']
    f[0]['matched_additive_mse']=.95;f[1]['matched_additive_mse']=.95
    d=m.decision(f,.01);assert not d['eligible'] and not d['above_threshold'];assert 'insufficient_positive_folds' in d['reasons']
    for z in f:z['baseline_mse']=.9
    assert 'nonpositive_overall_gain' in m.decision(f,.01)['reasons']


def test_assurance_report_never_changes_frozen_decisions():
    m=module();a=m.analytical_report();assert a['historical_n']==200 and a['historical_allowable']==2
    assert a['accepted_method_revision'] is False
    assert len(a['prospective'])==20
