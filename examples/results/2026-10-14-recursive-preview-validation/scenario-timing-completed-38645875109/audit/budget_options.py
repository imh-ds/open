"""No-fit prospective budget/precision illustrations; no revised study is adopted."""
import csv
import json
import math
from pathlib import Path

from scipy.stats import beta,binom

BASE=Path(__file__).resolve().parents[1]


def main():
    rows=list(csv.DictReader((BASE/'audit/timing-summary.csv').open()))
    # Includes installation/audit/preservation and headroom over measured jobs.
    # The chain values are case computation only; no fresh single-case job was timed.
    caps=[14,19,40,40,25,30]
    measured=[dict(condition=f"{r['scenario']}-n{r['n']}",case_minutes=float(r['case_seconds'])/60,
        observed_job_minutes=float(r['job_seconds'])/60 if r['job_seconds'] else None,
        illustrative_job_cap_minutes=c) for r,c in zip(rows,caps)]
    options=[]
    for n in (50,54,55,60,80):
        max_wrong=max(k for k in range(n+1) if (1 if k==n else beta.ppf(.975,k+1,n-k))<=.10)
        min_exact=min(k for k in range(n+1) if (0 if k==0 else beta.ppf(.025,k,n-k+1))>=.80)
        def tail(p,lo,hi):
            return sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(lo,hi+1))
        assert tail(.1,0,max_wrong)<=.025<tail(.1,0,max_wrong+1)
        assert tail(.8,min_exact,n)<=.025<tail(.8,min_exact-1,n)
        options.append(dict(datasets_per_cell=n,total_datasets=6*n,
            prospective_reserved_minutes_reusing_6_pilots=250+256+150+1+(n-1)*sum(caps),
            prospective_reserved_minutes_fresh_only=250+256+150+1+n*sum(caps),
            # 1 prospective registration. Fresh-only qualification keeps all six pilots descriptive.
            max_incorrect_per_cell=max_wrong,min_exact_per_signal_cell=min_exact,
            wrong_upper_at_boundary=float(beta.ppf(.975,max_wrong+1,n-max_wrong)),
            exact_lower_at_boundary=float(beta.ppf(.025,min_exact,n-min_exact+1)),
            assumed_1pct_wrong_rate_probability_all6_wrong_bounds_pass=float(binom.cdf(max_wrong,n,.01)**6)))
    result=dict(status='PLANNING_ONLY',approved=False,ceiling=10000,
        integer_boundaries_checked_by_independent_binomial_tails=True,
        original_480_study_status='Incomplete; original identities and denominators unchanged',
        conditions=measured,options=options,
        caveats=['One timed dataset per cell is not a worst-case runtime bound.',
            'Preserve defaults and criteria; smaller n changes integer thresholds and precision.',
            'Choose costed scope prospectively, not from observed accuracy.',
            'Timeouts remain unknown and prevent positive qualification; do not retry automatically.',
            'Assurance probability is hypothetical, not observed method performance.',
            'Prefer fresh-only qualification after pilot-dependent budget planning; keep the six pilots descriptive.',
            'Any new identity list/reuse/stage gates need a frozen reviewed protocol before fits.'])
    (BASE/'audit/budget-options.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
