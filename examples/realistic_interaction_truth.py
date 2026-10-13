"""Measured-input conditional teachers and approved classical-error generators.

Truth is diagnostic only. No teacher value enters model selection or display.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import roots_hermitenorm


@lru_cache(None)
def quadrature(nodes):
    x,w=roots_hermitenorm(nodes)
    return x,w/np.sqrt(2*np.pi)


def agreement(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.max(np.abs(np.asarray(a)-b),initial=0)>=1e-8:
        raise ValueError('conditional truth quadrature disagreement')


@lru_cache(None)
def _moments(gamma,shape,nodes):
    if shape=='product':return dict(normalization=1.,center=gamma,interaction_variance=1+gamma**2)
    if shape!='tanh':raise ValueError('unsupported moderator shape')
    x,w=quadrature(nodes);s=float(np.sqrt(w@np.tanh(x)**2));h=np.tanh(x)/s
    gh=float(w@(x*h));variance=float((1-gamma**2)*(w@h**2)+gamma**2*(w@(x*x*h*h))-gamma**2*gh**2)
    return dict(normalization=s,center=gamma*gh,interaction_variance=variance)


def moments(gamma,shape):
    a=_moments(gamma,shape,256);b=_moments(gamma,shape,128)
    agreement(list(a.values()),list(b.values()));return dict(a)


def conditional(gamma,reliability):
    sl=np.array([[1.,gamma],[gamma,1.]])
    if reliability==1.:return np.eye(2),np.zeros((2,2))
    sz=np.array([[1.,reliability*gamma],[reliability*gamma,1.]])
    inv=np.linalg.inv(sz);k=np.sqrt(reliability)*sl@inv;v=sl-reliability*sl@inv@sl
    if np.linalg.eigvalsh(v).min()<-1e-12:raise ValueError('invalid conditional covariance')
    return k,v


def _teacher(c,x,nodes):
    z=np.asarray(x,float);a,b=z[:,0],z[:,1];bench=c['benchmark']
    if bench=='noise':return np.zeros(len(z))
    if bench=='observed-additive':return a+.6*(b*b-1)
    k,v=conditional(c['gamma'],c['reliability']);mu=z[:,:2]@k.T
    if bench=='latent-additive':return mu[:,0]+.6*(mu[:,1]**2+v[1,1]-1)
    m=_moments(c['gamma'],c['shape'],nodes)
    if c['shape']=='product':product=mu[:,0]*mu[:,1]+v[0,1]
    else:
        if v[1,1]==0:
            h=np.tanh(mu[:,1])/m['normalization'];product=mu[:,0]*h
        else:
            q,w=quadrature(nodes);u=mu[:,[1]]+np.sqrt(v[1,1])*q
            tanh=np.tanh(u);eh=(tanh/m['normalization'])@w;dh=((1-tanh**2)/m['normalization'])@w
            product=mu[:,0]*eh+v[0,1]*dh
    return mu[:,0]+.6*mu[:,1]+c['b']*(product-m['center'])


def teacher(c,x):
    a=_teacher(c,x,256);b=_teacher(c,x,128);agreement(a,b);return a


def generate(c,seed,n):
    rng=np.random.default_rng(seed);l=rng.normal(size=(n,5));g=c['gamma'];r=c['reliability']
    l[:,1]=g*l[:,0]+np.sqrt(1-g*g)*l[:,1]
    e=rng.normal(size=(n,5));z=np.sqrt(r)*l+np.sqrt(1-r)*e
    bench=c['benchmark'];a,b=l[:,0],l[:,1]
    if bench=='noise':f=np.zeros(n);variance=1.5  # residual variance=1
    elif bench=='observed-additive':f=z[:,0]+.6*(z[:,1]**2-1);variance=1.72
    elif bench=='latent-additive':f=a+.6*(b*b-1);variance=1.72
    else:
        m=moments(g,c['shape']);h=b if c['shape']=='product' else np.tanh(b)/m['normalization']
        f=a+.6*b+c['b']*(a*h-m['center']);variance=1.36+1.2*g+c['b']**2*m['interaction_variance']
    y=f+rng.normal(scale=np.sqrt(variance*2/3),size=n)
    return pd.DataFrame(z,columns=[f'x{i+1}' for i in range(5)]),y
