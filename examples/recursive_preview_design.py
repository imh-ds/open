"""Approved reference-study identities, unchanged settings and dataset-level metrics."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from open.config import load_config
from open.layer_artifacts import portable

ROOT = Path(__file__).resolve().parents[1]
NAME = '2026-10-14-recursive-preview-validation'
PRODUCER = ROOT/f'examples/results/{NAME}/preflight-failure-38615814397'
RUNTIME = {'numpy':'1.26.4', 'pandas':'2.2.3', 'scipy':'1.11.2',
           'scikit-learn':'1.3.0', 'PyYAML':'6.0.2', 'matplotlib':'3.9.2'}


def digest(value):
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def specs():
    return [dict(id=f'{s}-n{n}-r{r:02}', index=i, scenario=s, n=n, repetition=r,
                 generator_seed=1_600_000_000+1000*i, analysis_seed=1_700_000_000+1000*i)
            for i, (s, n, r) in enumerate((s, n, r)
                for s in ('noise', 'chain', 'correlated') for n in (80, 200) for r in range(80))]


def shards(stage):
    if stage not in ('noise', 'signal'):
        raise ValueError('stage must be noise or signal')
    ids = list(range(160)) if stage == 'noise' else list(range(160, 480))
    return [ids[i:i+2] for i in range(0, len(ids), 2)]


def settings(spec):
    config = load_config(ROOT/'configs/default.yaml')
    return replace(config, analysis=replace(config.analysis, root='Y', random_seed=spec['analysis_seed']))


def generate(spec):
    x1, e2, ey, z1, z2 = np.random.default_rng(spec['generator_seed']).normal(size=(5, spec['n']))
    if spec['scenario'] == 'noise':
        x2, y = e2, ey
    elif spec['scenario'] == 'chain':
        x2 = .6*x1+.8*e2
        y = .8*x2+.6*ey
    elif spec['scenario'] == 'correlated':
        x2 = .4*x1+np.sqrt(.84)*e2
        y = .6*x1+.6*x2+np.sqrt(.672)*ey
    else:
        raise ValueError('unknown scenario')
    return pd.DataFrame(dict(Y=y, X1=x1, X2=x2, Z1=z1, Z2=z2))


def structure(spec, edges, groups, depths):
    expected = (set() if spec['scenario'] == 'noise' else
                {('X2', 'Y'), ('X1', 'X2')} if spec['scenario'] == 'chain' else
                {('X1', 'Y'), ('X2', 'Y')})
    intended = ({'Y':0} if spec['scenario'] == 'noise' else
                {'Y':0, 'X2':1, 'X1':2} if spec['scenario'] == 'chain' else
                {'Y':0, 'X1':1, 'X2':1})
    actual = {tuple(e) for e in edges}
    extras, missing = actual-expected, expected-actual
    unexpected_nodes = set(depths)-set(intended)
    matches = {name:depths.get(name) == depth for name, depth in intended.items() if name != 'Y'}
    incorrect = bool(extras or groups or unexpected_nodes)
    return dict(exact=not incorrect and not missing and dict(depths) == intended,
                incorrect=incorrect, missing=[list(e) for e in sorted(missing)],
                extra=[list(e) for e in sorted(extras)], groups=portable(groups),
                depth_matches=matches, nuisance_promoted=sorted(set(depths)&{'Z1','Z2'}),
                unexpected_nodes=sorted(unexpected_nodes))


def interval(k, n=80):
    return dict(count=k, lower=0. if not k else float(beta.ppf(.025, k, n-k+1)),
                upper=1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def summarize_cell(rows, *, signal):
    if len(rows) != 80:
        raise ValueError('every cell must retain 80 planned identities')
    complete = [r for r in rows if r['status'] == 'complete']
    unknown = 80-len(complete)
    incorrect = sum(r['structure']['incorrect'] for r in complete)
    recovered = sum(r['structure']['exact'] for r in complete)
    def bounds(k):
        return dict(count=k, lower=None, upper=None, possible_counts=[k, k+unknown],
                    inclusive_lower=interval(k)['lower'], inclusive_upper=interval(k+unknown)['upper'])
    wrong = bounds(incorrect) if unknown else interval(incorrect)
    recovery = bounds(recovered) if unknown else interval(recovered)
    return dict(planned=80, complete=len(complete), unknown=unknown,
                incorrect=wrong, recovery=recovery,
                qualified=not unknown and wrong['upper'] <= .10
                    and (not signal or recovery['lower'] >= .80))


def sources():
    files = [*sorted((ROOT/'src/open').rglob('*.py')),
             *sorted((ROOT/'examples').glob('recursive_preview_*.py')),
             ROOT/'examples/recursive_preview_validation_protocol.md',
             ROOT/'examples/recursive_preview_continuation_proposal.md',
             ROOT/'configs/default.yaml', ROOT/'pyproject.toml',
             ROOT/'.github/workflows/recursive-preview-validation.yml',
             PRODUCER/'raw/producer-manifest.json', PRODUCER/'raw/preflight.json',
             PRODUCER/'audit/raw-sha256.json', PRODUCER/'audit/diagnosis.json']
    seed_audit = ROOT/'examples/study_manifests/recursive-preview-seed-audit.json'
    if seed_audit.exists():
        files.append(seed_audit)
    return {p.relative_to(ROOT).as_posix():text_hash(p) for p in files}


def manifest():
    original = json.loads((PRODUCER/'raw/producer-manifest.json').read_text(encoding='utf-8-sig'))
    control = json.loads((PRODUCER/'raw/preflight.json').read_text(encoding='utf-8-sig'))
    hashes = json.loads((PRODUCER/'audit/raw-sha256.json').read_text(encoding='utf-8-sig'))
    diagnosis = json.loads((PRODUCER/'audit/diagnosis.json').read_text(encoding='utf-8-sig'))
    continuation = dict(producer_run_id=str(control['github_run']), producer_source=control['github_sha'],
        producer_manifest_digest=digest(original), archive_sha256=hashes['preflight-all-attempts.zip'],
        fixture_receipt_sha256=diagnosis['fixture_audit']['receipt_sha256'],
        prior_allocated_runner_minutes=9871, additional_job_minutes=45, additional_registration_minutes=1)
    return dict(schema='open.recursive_reference.v2', cases=specs(), sources=sources(),
                runtime=RUNTIME, workers=4, case_seconds=900, shard_minutes=40,
                max_parallel=20, allocated_runner_minutes=9917, approved_ceiling=10000,
                preflight_job_minutes=45, continuation=continuation,
                preflight_indices=[160,240], wrong_upper=.10, recovery_lower=.80,
                default_config=portable(load_config(ROOT/'configs/default.yaml')))


def validate_manifest(value):
    if value != manifest():
        raise ValueError('manifest differs from frozen source/design/settings/runtime')
