"""Frozen research design and decision arithmetic; no production confidence API."""
import math
import numpy as np
from scipy.stats import binomtest

CELLS = tuple((n,5,rho) for n in (80,150,300,600) for rho in (0.,.7))+((300,15,.7),(300,15,.9))
GRID = np.array([-1.2815516,-.5244005,0.,.5244005,1.2815516])
PAIRS = ((0,1),(0,2))
RULE = 'matched_positive_overall_two_folds_null_max.v1'


def cell_id(n,p,rho):
    return f'n{n}-p{p}-rho{rho:g}'


def tasks(stage):
    if stage not in ('calibration','confirmation'):
        raise ValueError('unknown study stage')
    result=[]; offset=308000000 if stage=='calibration' else 408000000
    for n,p,rho in CELLS:
        groups=[(s,0.,100 if stage=='calibration' else 200) for s in ('noise','additive')]
        if stage=='confirmation':groups += [(s,b,30) for s in ('pure','mixed') for b in (.2,.8)]
        for scenario,b,reps in groups:
            for rep in range(reps):
                i=len(result);cell=cell_id(n,p,rho)
                result.append(dict(id=f'confidence--{stage}--{cell}--{scenario}-b{b:g}-r{rep:03d}',
                    stage=stage,cell=cell,n=n,p=p,rho=rho,scenario=scenario,b=b,replicate=rep,
                    seed=offset+i,test_seed=offset+10000000+i,split_seed=offset+20000000+i))
    return result


def shards(stage):
    return [f'{i:03d}' for i in range(math.ceil(len(tasks(stage))/40))]


def pooled(folds,score):
    if score not in ('overall','matched'):raise ValueError('unknown score')
    numerator=denominator=0.
    for f in folds:
        if type(f['n']) is not int or f['n']<=0:raise ValueError('invalid assessment size')
        values=[f[k] for k in ('reference_mse','baseline_mse','joint_mse','matched_additive_mse')]
        if not np.isfinite(values).all() or min(values)<0 or values[0]<=0:
            raise ValueError('invalid prediction risks')
        baseline=f['baseline_mse'] if score=='overall' else f['matched_additive_mse']
        numerator += f['n']*(baseline-f['joint_mse']);denominator+=f['n']*f['reference_mse']
    if denominator<=0:raise ValueError('no assessment coverage')
    return float(numerator/denominator)


def pair_score(folds):
    if len(folds)!=3:raise ValueError('three outer folds required')
    overall=pooled(folds,'overall');matched=pooled(folds,'matched')
    positive=sum(f['matched_additive_mse']>f['joint_mse'] for f in folds)
    return max(0.,matched) if overall>0 and positive>=2 else 0.


def learn_thresholds(rows,planned):
    expected={c['id']:c for c in planned}
    if len(expected)!=len(planned) or len(rows)!=len(planned) or {r['id'] for r in rows}!=set(expected):
        raise ValueError('calibration requires complete unique planned identities')
    values={}
    for row in rows:
        c=expected[row['id']]
        if (row['cell']!=c['cell'] or row['scenario']!=c['scenario'] or
            c['scenario'] not in ('noise','additive') or isinstance(row['score'],bool) or
            not np.isfinite(row['score']) or row['score']<0):
            raise ValueError('invalid calibration score or identity')
        values.setdefault((c['cell'],c['scenario']),[]).append(row['score'])
    thresholds={}
    for cell in {c['cell'] for c in planned}:
        groups=[values.get((cell,s),[]) for s in ('noise','additive')]
        if any(len(g)!=100 for g in groups):raise ValueError('100 scores per null class required')
        thresholds[cell]=float(max(0.,max(groups[0]),max(groups[1])))
    return dict(sorted(thresholds.items()))


def qualifies(score,threshold):
    if not np.isfinite([score,threshold]).all() or score<0 or threshold<0:
        raise ValueError('invalid decision score')
    return bool(score>threshold)


def false_alarm_bound(k,n,strata=20):
    if not n:return 1.
    return float(binomtest(k,n,alternative='less').proportion_ci(
        confidence_level=1-.05/strata,method='exact').high)


def detection_bound(k,n,strata=1):
    if not n:return 0.
    return float(binomtest(k,n,alternative='greater').proportion_ci(
        confidence_level=1-.05/strata,method='exact').low)


def pattern_summary(curves):
    if len(curves)!=3:raise ValueError('three outer curves required')
    a=np.array([c['contrast'] for c in curves],float);m=np.array([c['support'] for c in curves])
    if a.shape!=(3,5,5) or m.shape!=(3,5,5) or m.dtype!=bool or not np.isfinite(a).all():
        raise ValueError('invalid curve records')
    mask=m.all(axis=0);mask[2,:]=False;mask[:,2]=False;count=int(mask.sum())
    result=dict(common_supported=count,consistent=False,min_cosine=None)
    if count<4:return dict(result,status='unresolved_support')
    v=a[:,mask];norm=np.linalg.norm(v,axis=1)
    if (norm<1e-8).any():return dict(result,status='unresolved_magnitude')
    similarities=[float(np.clip(v[i]@v[j]/(norm[i]*norm[j]),-1.,1.)) for i,j in ((0,1),(0,2),(1,2))]
    minimum=min(similarities)
    return dict(result,status='assessed',min_cosine=minimum,consistent=bool(minimum>=.8))
