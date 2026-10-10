"""Development diagnosis of interaction selection; never reads outer test outcomes."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from open.analysis import analyze_target
from open.config import (AnalysisConfig, LearnerConfig, OpenConfig,
                         PerformanceCVConfig, StabilityConfig, TuningCVConfig)
from open.layer_artifacts import portable
from open.layer_stability import make_layer_sample_plans


def generated(n, seed, noise=False):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, n))
    y = rng.normal(size=n) if noise else 2*a*b+rng.normal(scale=.3, size=n)
    return pd.DataFrame(dict(y=y, a=a, b=b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Reuse completed conditions with matching configuration and input')
    args = parser.parse_args()
    output = ROOT/'examples/results/2026-10-10-interaction-investigation'
    output.mkdir(exist_ok=args.resume)
    source = ROOT/'examples/results/2026-10-09-procedure-evaluation'
    previous = json.loads((source/'pilot/interaction-gbm/evaluation.json').read_text())
    original = pd.read_csv(source/'interaction.csv', float_precision='round_trip')
    config = OpenConfig(AnalysisConfig('y', 613, 1),
        learner=LearnerConfig(library=('gbm',)), tuning_cv=TuningCVConfig(2),
        stability=StabilityConfig(n_resamples=10, scope='layer_consensus'))
    (output/'protocol.md').write_text((ROOT/'examples/interaction-investigation-protocol.md').read_text(), encoding='utf-8')
    conditions = []
    for fold in previous['folds']:
        data = original.iloc[fold['train_positions']].reset_index(drop=True)
        for k in (2, 5):
            conditions.append((f'original-fold-{fold["fold"]}-k{k}', data, k, fold['fold']))
    for seed in (10401, 10402):
        for n in (54, 216):
            conditions.append((f'fresh-interaction-{seed}-n{n}', generated(n, seed), 5, None))
    conditions.append(('fresh-noise-10403-n216', generated(216, 10403, True), 5, None))
    rows, summaries = [], []
    started = time.perf_counter()
    for name, data, k, original_fold in conditions:
        local_config = replace(config, performance_cv=PerformanceCVConfig(k, 1),
            analysis=replace(config.analysis, random_seed=613+(original_fold or 0)))
        cached = output/f'{name}.json'
        if args.resume and cached.exists():
            saved = json.loads(cached.read_text())
            assert saved['config'] == portable(asdict(local_config))
            np.testing.assert_allclose(pd.read_csv(output/f'{name}.csv', float_precision='round_trip'), data, rtol=0, atol=0)
            assert len(saved['records']) == 10
            records = saved['records']
            rows.extend(r['summary'] for r in records)
        else:
            records = []
            data.to_csv(output/f'{name}.csv', index=False)
        for plan in make_layer_sample_plans(data, local_config):
            if cached.exists() and args.resume:
                break
            sampled = data.iloc[list(plan.sampled_positions)]
            run_config = replace(local_config, analysis=replace(local_config.analysis, random_seed=plan.seed))
            local = analyze_target(sampled, 'y', ('a', 'b'), run_config, plan.split_plan)
            record = dict(condition=name, run=plan.run_index, seed=plan.seed,
                input_n=len(data), sample_n=len(sampled), folds=k,
                min_fit_n=min(len(s.train_indices) for s in plan.split_plan.splits),
                max_fit_n=max(len(s.train_indices) for s in plan.split_plan.splits),
                baseline=local.relevance.baseline_risk, full=local.relevance.full_risk,
                gain=local.total_gain_relative, stop=local.reason,
                a_promoted='a' in local.promoted, b_promoted='b' in local.promoted,
                a_unique=local.relevance.unique['a'].normalized,
                b_unique=local.relevance.unique['b'].normalized,
                a_broad=local.relevance.broad['a'].normalized,
                b_broad=local.relevance.broad['b'].normalized)
            rows.append(record)
            records.append(dict(summary=record, sample=portable(plan), local=portable(local)))
            if original_fold is not None and k == 2 and plan.run_index < 3:
                ledger = json.loads((source/f'pilot/interaction-gbm/fold_{original_fold}/assessment_ledger.json').read_text())
                old = next(a['local'] for a in ledger['assessments'] if a['depth'] == 0 and a['run_index'] == plan.run_index)
                np.testing.assert_allclose(local.relevance.full_risk, old['relevance']['full_risk'], rtol=1e-12)
                assert sorted(local.promoted) == old['promoted']
        (output/f'{name}.json').write_text(json.dumps(dict(config=asdict(local_config), records=records), indent=2)+'\n')
        for budget in (3, 10):
            subset = records[:budget]
            counts = {v: sum(r['summary'][v+'_promoted'] for r in subset) for v in ('a', 'b')}
            summaries.append(dict(condition=name, budget=budget, selected=counts,
                retained=[v for v, count in counts.items() if count/budget >= .7],
                low_gain=sum(r['summary']['stop'] is not None for r in subset),
                median_gain=float(np.median([r['summary']['gain'] for r in subset]))))
        print(name, summaries[-2:], flush=True)
    with (output/'assessments.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(summaries=summaries,
        assessments=len(rows), reproduced_original_assessments=9,
        elapsed_seconds=time.perf_counter()-started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        claim='development diagnosis only; no held-out model selection or scientific calibration'), indent=2)+'\n')


if __name__ == '__main__':
    main()
