import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def modules():
    assert (Path(__file__).parents[1]/'examples/realistic_interaction_truth.py').exists(), 'conditional truth module missing'
    import realistic_interaction_truth as t
    import realistic_interaction_design as d
    return t,d


def test_universe_and_collision_audit():
    t,d=modules();tasks=d.tasks()
    assert len(tasks)==14400 and len(d.stage_tasks('calibration'))==3600
    assert len(d.stage_tasks('development'))==10800
    assert len(d.shards('calibration'))==45 and len(d.shards('development'))==135
    assert len({c['id'] for c in tasks})==14400
    assert len(d.audit_seed_collisions(tasks,{}))>0
    with pytest.raises(ValueError):d.audit_seed_collisions(tasks,{'old':{'seed':708000000}})
    assert sum(c['benchmark']=='latent-additive' for c in tasks)==720
    assert all(c['reliability']==.8 for c in tasks if c['benchmark']=='latent-additive')


def case(t,d,benchmark='tanh-b0.4',r=.8):
    return next(c for c in d.tasks() if c['benchmark']==benchmark and c['gamma']==.5 and c['reliability']==r)


def test_tanh_truth_against_independent_conditional_integration():
    t,d=modules();c=case(t,d);z=np.array([[.3,-.8,0,0,0],[3.,-2.,0,0,0]])
    sl=np.array([[1,.5],[.5,1]]);sz=np.array([[1,.4],[.4,1]])
    k=np.sqrt(.8)*sl@np.linalg.inv(sz);v=sl-.8*sl@np.linalg.inv(sz)@sl
    s=np.sqrt(quad(lambda g:np.tanh(g)**2*np.exp(-g*g/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-12)[0])
    ehg=quad(lambda g:g*np.tanh(g)/s*np.exp(-g*g/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-12)[0]
    expected=[]
    for row in z:
        mu=k@row[:2]
        # Direct conditional integration: E[L1|L2,Z] times h(L2), not Stein's formula.
        f=lambda u:(mu[0]+v[0,1]/v[1,1]*(u-mu[1]))*np.tanh(u)/s*np.exp(-(u-mu[1])**2/(2*v[1,1]))/np.sqrt(2*np.pi*v[1,1])
        product=quad(f,mu[1]-12*np.sqrt(v[1,1]),mu[1]+12*np.sqrt(v[1,1]),epsabs=1e-12)[0]
        expected.append(mu[0]+.6*mu[1]+.4*(product-.5*ehg))
    np.testing.assert_allclose(t.teacher(c,z),expected,atol=1e-8,rtol=0)


def test_product_and_noiseless_truth():
    t,d=modules();z=np.array([[.3,-.8,0,0,0]])
    c=case(t,d,'product-b0.4');k,v=t.conditional(.5,.8);mu=z[:,:2]@k.T
    np.testing.assert_allclose(t.teacher(c,z),mu[:,0]+.6*mu[:,1]+.4*(mu[:,0]*mu[:,1]+v[0,1]-.5))
    c=case(t,d,'tanh-b0.4',1.);s=t.moments(.5,'tanh')['normalization']
    expectation=t.moments(.5,'tanh')['center']
    np.testing.assert_allclose(t.teacher(c,z),z[:,0]+.6*z[:,1]+.4*(z[:,0]*np.tanh(z[:,1])/s-expectation))


def test_measured_null_and_latent_diagnostic_are_different():
    t,d=modules();z=np.array([[1.,1.,0,0,0]])
    null=case(t,d,'observed-additive');diag=case(t,d,'latent-additive')
    np.testing.assert_allclose(t.teacher(null,z),[1.])
    k,v=t.conditional(.5,.8);mu=z[:,:2]@k.T
    np.testing.assert_allclose(t.teacher(diag,z),mu[:,0]+.6*(mu[:,1]**2+v[1,1]-1))
    assert not np.allclose(t.teacher(null,z),t.teacher(diag,z))


def test_generator_order_and_residual_variance():
    t,d=modules();c=case(t,d,'product-b0.4');seed=17;n=8
    x,y=t.generate(c,seed,n);rng=np.random.default_rng(seed)
    l=rng.normal(size=(n,5));l[:,1]=.5*l[:,0]+np.sqrt(.75)*l[:,1]
    e=rng.normal(size=(n,5));z=np.sqrt(.8)*l+np.sqrt(1-.8)*e
    f=l[:,0]+.6*l[:,1]+.4*(l[:,0]*l[:,1]-.5)
    var=1.36+1.2*.5+.4**2*1.25
    np.testing.assert_array_equal(x.to_numpy(),z)
    np.testing.assert_array_equal(y,f+rng.normal(scale=np.sqrt(var*2/3),size=n))


def test_quadrature_disagreement_stops(monkeypatch):
    t,d=modules();c=case(t,d);original=t._teacher
    def wrong(c,x,nodes):return original(c,x,nodes)+(1e-5 if nodes==128 else 0)
    monkeypatch.setattr(t,'_teacher',wrong)
    with pytest.raises(ValueError,match='quadrature'):t.teacher(c,np.zeros((1,5)))
