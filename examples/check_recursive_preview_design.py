"""No-data, no-fit arithmetic/identity checks for the proposed reference study."""
import argparse
import hashlib
import json
from pathlib import Path

from scipy.stats import beta, binom


def seed_values(value, path=(), seed_context=False):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from seed_values(child, (*path, str(key)),
                                   seed_context or 'seed' in str(key).lower())
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from seed_values(child, (*path, str(index)), seed_context)
    elif seed_context and isinstance(value, int) and not isinstance(value, bool):
        yield path, value


def check(repo):
    protocol = repo / 'examples/recursive_preview_validation_protocol.md'
    manifest_files = sorted((repo / 'examples/study_manifests').glob('*.json'))
    historical = set()
    inventories = []
    for path in manifest_files:
        seeds = list(seed_values(json.loads(path.read_text(encoding='utf-8-sig'))))
        historical.update(value for _, value in seeds)
        inventories.append(dict(path=path.relative_to(repo).as_posix(),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                seed_fields=len(seeds)))
    cases = [dict(index=i, scenario=scenario, n=n, repetition=r,
                  generator_seed=1_600_000_000 + 1000*i,
                  analysis_seed=1_700_000_000 + 1000*i)
             for i, (scenario, n, r) in enumerate(
                 (s, n, r) for s in ('noise', 'chain', 'correlated')
                 for n in (80, 200) for r in range(80))]
    assert len(cases) == 480
    assert cases[160]['scenario'] == cases[240]['scenario'] == 'chain'
    assert (cases[160]['n'], cases[240]['n']) == (80, 200)
    proposed = [seed for c in cases for seed in
                (c['generator_seed'], *range(c['analysis_seed'], c['analysis_seed']+100))]
    assert len(proposed) == len(set(proposed)) == 48480
    assert min(proposed) >= 0 and max(proposed) < 2**32
    collisions = sorted(set(proposed) & historical)
    assert not collisions, collisions
    n = 80
    incorrect_pass = [k for k in range(n+1)
                      if (1 if k == n else beta.ppf(.975, k+1, n-k)) <= .10]
    recovery_pass = [k for k in range(n+1)
                     if (0 if k == 0 else beta.ppf(.025, k, n-k+1)) >= .80]
    assert incorrect_pass == [0, 1, 2] and recovery_pass == list(range(72, 81))
    upper = float(beta.ppf(.975, 1, n))
    lower = float(beta.ppf(.025, 72, n-72+1))
    assert abs(binom.cdf(0, n, upper)-.025) < 1e-12
    assert abs(binom.sf(71, n, lower)-.025) < 1e-12
    signal_var = .6**2 + .6**2 + 2*.6*.6*.4
    assert abs(signal_var-1.008) < 1e-12
    assert abs(signal_var/(signal_var+.672)-.60) < 1e-12
    assert abs(.6**2*(1-.4**2)/signal_var-.30) < 1e-12
    assert abs(.6**2+.8**2-1) < 1e-12
    assert abs(.8**2/(.8**2+.6**2)-.64) < 1e-12
    allocations = dict(registration=1, preflight=120, preparation=2*15,
                       aggregation=2*60, noise_shards=80*40, signal_shards=160*40)
    total = sum(allocations.values())
    assert total == 9871 and total <= 10000
    return dict(status='DESIGN_ARITHMETIC_PASS',
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                planned_datasets=len(cases), cases=cases,
                unique_reserved_seeds=len(proposed),
                historical_manifest_inventory=inventories,
                historical_explicit_seed_values=len(historical),
                historical_explicit_seed_collisions=collisions,
                seed_audit_scope='Explicit seed fields in inventoried historical manifests only; '
                                 'formula/receipt namespace audit remains a pre-freeze gate.',
                per_cell_wrong_finding_maximum=2,
                per_cell_exact_recovery_minimum=72,
                zero_of_80_upper=upper, one_of_80_upper=float(beta.ppf(.975, 2, 79)),
                two_of_80_upper=float(beta.ppf(.975, 3, 78)),
                three_of_80_upper=float(beta.ppf(.975, 4, 77)),
                recovery_72_of_80_lower=lower,
                recovery_71_of_80_lower=float(beta.ppf(.025, 71, 10)),
                chain_root_r_squared=.64, chain_recursive_r_squared=.36,
                correlated_predictor_r=.4, correlated_signal_variance=signal_var,
                correlated_root_r_squared=.60,
                incorrect_finding_gate_assurance=[
                    dict(assumed_true_rate=p, per_cell=float(binom.cdf(2, 80, p)),
                         all_six_cells=float(binom.cdf(2, 80, p)**6))
                    for p in (.005, .01, .02, .05)],
                rejected_five_percent_bound_all_cell_assurance_at_one_percent=.99**480,
                allocations=allocations, allocated_runner_minutes=total,
                approved_ceiling=10000, unallocated_minutes=10000-total,
                data_generated=0, model_fits=0, actions_dispatches=0,
                qualification='Not assessed; design checks do not validate the method.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = check(Path(__file__).resolve().parents[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('cases', 'historical_manifest_inventory')}, indent=2))


if __name__ == '__main__':
    main()
