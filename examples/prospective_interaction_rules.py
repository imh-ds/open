"""Prospectively scoped benchmark bounds; no production confidence API."""
from scipy.stats import beta

ENDPOINTS={'false':(.02/36,.05,'upper'),'detection':(.02/24,.80,'lower'),'shape':(.01/24,.90,'lower')}


def bounds(k,n,endpoint):
    if endpoint not in ENDPOINTS or type(k) is not int or type(n) is not int or not 0<=k<=n:
        raise ValueError('invalid endpoint or binomial counts')
    alpha,target,direction=ENDPOINTS[endpoint]
    value=None
    if n:
        value=(1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))) if direction=='upper' else (
            0. if not k else float(beta.ppf(alpha,k,n-k+1)))
    passed=value is not None and (value<=target if direction=='upper' else value>=target)
    rate=k/n if n else None
    observed_meets=n and (rate<=target if direction=='upper' else rate>=target)
    explanation=('unresolved_no_observations' if not n else 'qualified' if passed else
                 'uncertain_despite_observed_rate_meeting_target' if observed_meets else 'observed_rate_misses_target')
    return dict(k=k,n=n,alpha=alpha,target=target,direction=direction,bound=value,
                observed_rate=rate,explanation=explanation,status='complete' if n else 'unresolved',**{'pass':bool(passed)})


def summarize_stratum(rows,planned,endpoint):
    ids=[c['id'] for c in planned];observed=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or len(set(observed))!=len(observed) or not set(observed)<=set(ids):
        raise ValueError('duplicate or foreign stratum identity')
    selected=rows if endpoint!='shape' else [r for r in rows if r['displayed']]
    key={'false':'control','detection':'detected','shape':'shape_correct'}.get(endpoint)
    if key is None:raise ValueError('unknown endpoint')
    result=bounds(sum(bool(r[key]) for r in selected),len(selected),endpoint)
    result.update(planned=len(planned),complete=len(rows),missing=len(planned)-len(rows),displayed=sum(bool(r.get('displayed')) for r in rows))
    if len(rows)!=len(planned):result.update(status='incomplete',explanation='unknown_incomplete_stratum',**{'pass':False})
    return result


def qualify_condition(nulls,signal):
    if len(nulls)!=2:raise ValueError('two null classes required')
    checks=[('null_noise',nulls[0]),('null_additive',nulls[1]),('control',signal['control']),('detection',signal['detection'])]
    reasons=[name for name,value in checks if not value['pass']]
    predictive=not reasons
    shape=predictive and signal['shape']['pass']
    return dict(predictive_support=bool(predictive),curve_pattern_support=bool(shape),
                reasons=reasons+([] if signal['shape']['pass'] else ['curve_pattern_unresolved_or_unsupported']))
