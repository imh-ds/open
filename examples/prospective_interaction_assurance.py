"""Data-independent protocol calculations; no fitting or candidate decisions.

Run with Python to write JSON to stdout. Reproduce the companion JSON using
the pinned study SciPy runtime. Probabilities use hypothetical fixed rates.
"""
import json
from scipy.stats import beta, binom


def upper(k, n, alpha):
    return 1.0 if k == n else float(beta.ppf(1-alpha, k+1, n-k))


def lower(k, n, alpha):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n-k+1))


def false_gate(n):
    alpha = .02/36
    k = max(i for i in range(n+1) if upper(i, n, alpha) <= .05)
    # Verify the independent CDF form of the same exact acceptance boundary.
    assert binom.cdf(k, n, .05) <= alpha
    assert binom.cdf(k+1, n, .05) > alpha
    return dict(n=n, max_false=k, upper_at_boundary=upper(k, n, alpha),
                upper_next=upper(k+1, n, alpha),
                hypothetical=[dict(true_rate=p, single_pass=float(binom.cdf(k, n, p)),
                    all_12_null_pass=float(binom.cdf(k, n, p)**12),
                    all_24_control_pass=float(binom.cdf(k, n, p)**24)) for p in (.005, .01, .02, .025)])


def positive_gate(n, alpha, required, hypothetical):
    possible = [i for i in range(n+1) if lower(i, n, alpha) >= required]
    if not possible:
        return dict(n=n, min_success=None, reason='no count can meet the bound')
    k = min(possible)
    assert binom.sf(k-1, n, required) <= alpha
    assert binom.sf(k-2, n, required) > alpha
    return dict(n=n, min_success=k, lower_at_boundary=lower(k, n, alpha),
                lower_previous=lower(k-1, n, alpha),
                hypothetical=[dict(true_rate=p, single_pass=float(binom.sf(k-1, n, p)))
                              for p in hypothetical])


def report():
    cells = [(300, 5, 0.), (300, 5, .7), (600, 5, 0.),
             (600, 5, .7), (300, 15, .7), (300, 15, .9)]
    coefficients = []
    for n, p, rho in cells:
        A = 1+rho*rho
        C = (1-rho*rho)**2/A
        D = 1.36+1.2*rho
        for q in ((.002, .005) if rho == .9 else (.01, .04)):
            denominator = .6*C-q*A
            assert denominator > 0
            b = (q*D/denominator)**.5
            recovered = .6*b*b*C/(D+b*b*A)
            assert abs(recovered-q) < 1e-14
            coefficients.append(dict(n=n, p=p, rho=rho, q=q, b=b,
                                     recovered_q=recovered, q_limit=.6*C/A))
    nulls = 6*2*1200
    signals = 6*4*450
    assert nulls % 80 == signals % 80 == 0
    shards_a, shards_b = nulls//80, signals//80
    assert max(shards_a, shards_b) <= 256
    return dict(status='proposal; no empirical observations or model fits',
        assumptions='All-grid power assumes independent equal-rate strata. Bounds use '
                    'independent datasets within each stratum; union coverage does not '
                    'require endpoint independence. Shape inference is conditional on '
                    'a fixed truth-blind display rule; zero selected is unresolved.',
        alpha=dict(false=.02, detection=.02, shape=.01, total=.05),
        families=dict(false=36, detection=24, shape=24),
        false_gates=[false_gate(n) for n in (300, 450, 600, 1000, 1200, 1500)],
        detection_gates=[positive_gate(n, .02/24, .8, (.9, .95)) for n in (100,250,300,450)],
        shape_gates=[positive_gate(n, .01/24, .9, (.99,)) for n in (100,150,200,250,300)],
        matched_information=coefficients,
        budget=dict(null_datasets=nulls, maximum_signal_datasets=signals,
                    maximum_datasets=nulls+signals, null_shards=shards_a,
                    maximum_signal_shards=shards_b, tasks_per_shard=80,
                    fit_tune_calls=(nulls+signals)*692,
                    saved_setting_refits=(nulls+signals)*20,
                    timeout_runner_minutes=(shards_a+shards_b)*20+2*(15+60)))


if __name__ == '__main__':
    print(json.dumps(report(), indent=2, allow_nan=False))
