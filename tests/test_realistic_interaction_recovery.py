import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def module():
    assert (Path(__file__).parents[1]/'examples/realistic_interaction_recovery.py').exists(),'bounded recovery auditor missing'
    import realistic_interaction_recovery as r
    return r


def record():
    return dict(predicted=[1.,2.],mse=.3,surfaces=np.ones((2,5,5)).tolist(),ideal_additive='unassessed',scope='measured-input conditional teacher; diagnostic only')


def test_roundoff_accepted_but_frozen_exact_checker_reproduces_failure():
    r=module();s=r.study;c=next(c for c in s.d.stage_tasks('development') if c['cell']=='n150-g0.5-R0.8' and c['benchmark']=='product-b0.2')
    data=s.comparisons(c)[0];expected=s.population_record(c,data,{})
    saved=deepcopy(expected);saved['predicted'][0]=float(np.nextafter(saved['predicted'][0],np.inf))
    with pytest.raises(ValueError,match='teacher evidence'):s.audit_population(c,data,{},saved)
    result=r.validate_population(expected,saved)
    assert 0<result['max_absolute_difference']<=1e-12
    assert saved['predicted'][0]!=expected['predicted'][0]  # Raw evidence was not rewritten.


@pytest.mark.parametrize('field',['predicted','surfaces','mse'])
def test_meaningful_numerical_tampering_rejected(field):
    r=module();expected=record();saved=deepcopy(expected)
    if field=='mse':saved[field]+=1e-8
    elif field=='predicted':saved[field][0]+=1e-8
    else:saved[field][0][0][0]+=1e-8
    with pytest.raises(ValueError):r.validate_population(expected,saved)


@pytest.mark.parametrize('change',['nan','shape','metadata','extra','boolean','string','negative_mse'])
def test_schema_finiteness_and_metadata_remain_strict(change):
    r=module();expected=record();saved=deepcopy(expected)
    if change=='nan':saved['predicted'][0]=float('nan')
    elif change=='shape':saved['surfaces']=saved['surfaces'][:1]
    elif change=='metadata':saved['scope']='different teacher'
    elif change=='extra':saved['new']=1
    elif change=='boolean':saved['mse']=True
    elif change=='string':saved['predicted']=['1','2']
    else:saved['mse']=-1e-15
    with pytest.raises(ValueError):r.validate_population(expected,saved)


def test_recovery_override_is_temporary_and_forbids_fits():
    r=module();s=r.study;original=s.audit_population;compare=s.old.models.compare
    with r.numeric_audit() as stats:
        assert s.audit_population is not original
        with pytest.raises(ValueError,match='fit'):s.old.models.compare(None)
        expected=record();saved=deepcopy(expected);saved['mse']=np.nextafter(.3,np.inf)
        assert r.validate_population(expected,saved)['max_absolute_difference']<1e-12
    assert s.audit_population is original and s.old.models.compare is compare
