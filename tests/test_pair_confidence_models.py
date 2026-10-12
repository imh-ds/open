import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest


def module():
    root=Path(__file__).parents[1];p=root/'examples/pair_confidence_models.py'
    assert p.exists(),'confidence models missing'
    sys.path.insert(0,str(root/'examples'))
    s=importlib.util.spec_from_file_location('confidence_models',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def case():return dict(n=80,p=5,rho=.7,scenario='mixed',b=.8)


def test_generation_pure_scale_invariance_and_oracle():
    m=module();c=dict(case(),scenario='pure',b=.2);X,y,truth=m.generate(c,80,293000001)
    Z,z,t=m.generate(dict(c,b=.8),80,293000001)
    np.testing.assert_array_equal(X,Z);np.testing.assert_allclose(y*4,z)
    assert m.oracle_gap(c)==pytest.approx(m.oracle_gap(dict(c,b=.8)))
    assert m.oracle_gap(dict(c,scenario='additive'))==0.


def test_training_cost_audit_and_assessment_invariance():
    m=module();X,y,_=m.generate(case(),80,293000002);train=list(range(53));test=list(range(53,80))
    v=m.compare(X,y,train,test,4); assert v['fits']==173
    assert len(v['pairs'])==2 and len(v['pairs'][0]['selection']['fold_losses'])==24
    m.audit_comparison(v,X,y,train,test,4,refit=True)
    changed=y.copy();changed[test]+=100
    w=m.compare(X,changed,train,test,4)
    for a,b in zip(v['pairs'],w['pairs']):
        assert a['parameters']==b['parameters'] and a['selection']==b['selection']
        assert a['predicted']==b['predicted'] and a['curve']==b['curve']
        assert a['matched_parameters']=={k:a['parameters'][k] for k in ('curves__n_knots','ridge__alpha')}
        assert a['overall_gain']==pytest.approx(a['matched_gain']+a['marginal_gain'])


@pytest.mark.parametrize('field',['matched_gain','curve','parameters','selection'])
def test_raw_tampering_is_rejected(field):
    m=module();X,y,_=m.generate(case(),80,293000003);a=list(range(53));b=list(range(53,80))
    v=json.loads(json.dumps(m.compare(X,y,a,b,5)))
    r=v['pairs'][0]
    if field=='matched_gain':r[field]+=.1
    if field=='curve':r['curve']['contrast'][0][0]+=.1
    if field=='parameters':r['parameters']['ridge__alpha']=99.
    if field=='selection':r['selection']['fold_losses'][0][0]=float('nan')
    with pytest.raises((ValueError,AssertionError)):m.audit_comparison(v,X,y,a,b,5)


def test_supported_anchors_and_additive_contrast():
    m=module();axis=m.rules.GRID;a,b=np.meshgrid(axis,axis,indexing='ij')
    f=a*a+b**3
    np.testing.assert_allclose(m.anchored(f),0.,atol=1e-14)
    np.testing.assert_allclose(m.anchored(a*b),a*b)
    X=np.zeros((10,2));mask,counts=m.support(X)
    assert mask[2,2] and counts[2,2]==10 and not mask[0,0]
