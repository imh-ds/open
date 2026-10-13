"""Research-only diagnostics; no threshold update or production confidence rule."""
import math
import numpy as np
from scipy.stats import binom
import pair_confidence_study as previous

BUDGET_MINUTES=125


def allowable_count(n,strata=20,limit=.05):
    if type(n) is not int or n<=0 or type(strata) is not int or strata<=0 or not 0<limit<1:
        raise ValueError('invalid assurance design')
    tail=.05/strata;k=int(binom.ppf(tail,n,limit))
    if binom.cdf(k,n,limit)>tail:k-=1
    return k


def assurance(n,true_rate,strata=20):
    if not np.isfinite(true_rate) or not 0<=true_rate<=1:raise ValueError('invalid hypothetical rate')
    k=allowable_count(n,strata);p=float(binom.cdf(k,n,true_rate))
    return dict(n=n,true_rate=true_rate,strata=strata,allowable=k,single_pass=p,all_pass=p**strata,
        assumption='independent strata with equal fixed true rates; hypothetical')


def calibration_reference(m,limit=.05,strata=20):
    if type(m) is not int or m<=0 or not 0<limit<1 or type(strata) is not int or strata<=0:
        raise ValueError('invalid maximum calibration design')
    failure=(1-limit)**m
    return dict(m=m,limit=limit,strata=strata,expected_exceedance=1/(m+1),failure_per_class=failure,
        union_failure_upper=min(1.,strata*failure),minimum_m_for_union_bound=math.ceil(math.log(.05/strata)/math.log(1-limit)),
        scope='continuous specified-null maximum reference; per-class conservative bound for max of two classes; no posterior claim')


def analytical_report():
    return dict(historical_n=200,historical_allowable=allowable_count(200),accepted_method_revision=False,
        calibration=calibration_reference(100),prospective=[assurance(n,p) for n in (200,400,600,1000,2000) for p in (.001,.005,.01,.02)],
        population_references=[dict(rho=r,b=.8,distinct_gap=previous.models.oracle_gap(dict(scenario='mixed',rho=r,b=.8))) for r in (0.,.7,.9)],
        interpretation='Design calculations do not rescue the failed confirmation; no new threshold is learned.')


def matched_coefficient(q,rho):
    if not np.isfinite([q,rho]).all() or not 0<=rho<1 or q<=0:raise ValueError('invalid matched information target')
    A=1+rho*rho;C=(1-rho*rho)**2/A;D=1.36+1.2*rho;den=.6*C-q*A
    if den<=0:raise ValueError('unattainable target at fixed overall R-squared .6')
    return float(np.sqrt(q*D/den))


def tasks():
    groups=[(n,5,r,'mixed',q,matched_coefficient(q,r)) for n in (300,600) for r in (0.,.7) for q in (.01,.04)]
    groups += [(300,5,0.,'noise',0.,0.),(300,15,.7,'additive',0.,0.)]
    result=[]
    for n,p,rho,scenario,q,b in groups:
        for rep in range(6):
            i=len(result);cell=previous.rules.cell_id(n,p,rho)
            result.append(dict(id=f'diagnostic--{cell}--{scenario}-q{q:g}-r{rep:02d}',stage='diagnostic',cell=cell,
                n=n,p=p,rho=rho,scenario=scenario,target_gap=q,b=b,replicate=rep,
                seed=508000000+i,test_seed=518000000+i,split_seed=528000000+i))
    return result


def shards():return [f'{i:03d}' for i in range(6)]


def population_predictions(X,c):
    a=np.asarray(X.iloc[:,0]);b=np.asarray(X.iloc[:,1]);rho=c['rho'];coefficient=c['b']
    if c['scenario']=='noise':return np.zeros(len(X)),np.zeros(len(X))
    if c['scenario']=='additive':
        truth=a+.6*(b*b-1);return truth,truth
    if c['scenario'] not in ('pure','mixed'):raise ValueError('unknown oracle generator')
    base=a+.6*b if c['scenario']=='mixed' else np.zeros(len(X))
    joint=base+coefficient*(a*b-rho)
    additive=base+coefficient*rho/(1+rho*rho)*((a*a-1)+(b*b-1))
    return additive,joint


def decision(folds,threshold):
    if not np.isfinite(threshold) or threshold<0:raise ValueError('invalid frozen threshold')
    overall=previous.rules.pooled(folds,'overall');matched=previous.rules.pooled(folds,'matched')
    positive=sum(z['matched_additive_mse']>z['joint_mse'] for z in folds);reasons=[]
    if overall<=0:reasons.append('nonpositive_overall_gain')
    if matched<=0:reasons.append('nonpositive_matched_gain')
    if positive<2:reasons.append('insufficient_positive_folds')
    eligible=overall>0 and positive>=2;score=max(0.,matched) if eligible else 0.
    above=bool(score>threshold)
    if eligible and score<=threshold:reasons.append('fixed_threshold_not_exceeded')
    return dict(overall=overall,matched=matched,positive_folds=positive,eligible=bool(eligible),score=score,
        threshold=float(threshold),above_threshold=above,reasons=reasons)
