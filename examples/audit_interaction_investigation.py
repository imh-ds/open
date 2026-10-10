"""Offline reconciliation of every saved interaction diagnostic decision."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from open.config import parse_config
from open.layer_artifacts import portable
from open.layer_stability import make_layer_sample_plans


def main():
    base = ROOT/'examples/results/2026-10-10-interaction-investigation'
    original = ROOT/'examples/results/2026-10-09-procedure-evaluation'
    summary = json.loads((base/'summary.json').read_text())
    assert summary['source_sha256'] == hashlib.sha256((ROOT/'examples/investigate_interaction_selection.py').read_bytes()).hexdigest()
    n = replays = 0
    for path in sorted(base.glob('*.json')):
        if path.name in ('summary.json', 'audit.json', 'startup_failures.json'):
            continue
        saved = json.loads(path.read_text())
        config = parse_config(saved['config'])
        data = pd.read_csv(path.with_suffix('.csv'), float_precision='round_trip')
        plans = make_layer_sample_plans(data, config)
        assert len(saved['records']) == len(plans) == 10
        for record, plan in zip(saved['records'], plans):
            assert record['sample'] == portable(plan)
            r, local = record['summary'], record['local']
            rel = local['relevance']
            gain = rel['baseline_risk']-rel['full_risk']
            np.testing.assert_allclose(gain, local['total_gain'], rtol=1e-12)
            np.testing.assert_allclose(gain/rel['baseline_risk'], r['gain'], rtol=1e-12)
            stop = gain <= 0 or r['gain'] < config.thresholds.min_total_gain_relative
            assert stop == local['stop'] == (r['stop'] == 'LOW_TOTAL_GAIN')
            expected = []
            for v in ('a', 'b'):
                u = rel['unique'][v]
                if stop:
                    assert u['normalized'] is None
                else:
                    np.testing.assert_allclose(u['raw']/gain, u['normalized'], rtol=1e-12)
                    if u['normalized'] >= config.thresholds.unique_relative:
                        expected.append(v)
                assert r[v+'_promoted'] == (v in local['promoted'])
            # These independent-predictor scenarios have no retained shared groups.
            assert not local['retained_pairs'] and not any(g['validated'] for g in local['shared_components'])
            assert sorted(local['promoted']) == expected
            assert r['min_fit_n'] == min(len(s.train_indices) for s in plan.split_plan.splits)
            assert r['sample_n'] == len(plan.sampled_positions)
            if path.stem.startswith('original-fold-') and r['folds'] == 2 and r['run'] < 3:
                fold = int(path.stem.split('-')[2])
                ledger = json.loads((original/f'pilot/interaction-gbm/fold_{fold}/assessment_ledger.json').read_text())
                old = next(a['local'] for a in ledger['assessments'] if a['depth'] == 0 and a['run_index'] == r['run'])
                np.testing.assert_allclose(rel['full_risk'], old['relevance']['full_risk'], rtol=1e-12)
                assert local['promoted'] == old['promoted']
                replays += 1
            n += 1
        for budget in (3, 10):
            records = saved['records'][:budget]
            counts = {v: sum(v in r['local']['promoted'] for r in records) for v in ('a', 'b')}
            reported = next(s for s in summary['summaries'] if s['condition'] == path.stem and s['budget'] == budget)
            assert reported['selected'] == counts
            assert reported['retained'] == [v for v in ('a', 'b') if counts[v]/budget >= .7]
            assert reported['low_gain'] == sum(r['local']['stop'] for r in records)
    assert n == summary['assessments'] == 110 and replays == 9
    assert len(pd.read_csv(base/'assessments.csv')) == n
    result = dict(assessments=n, exact_original_replays=replays, conditions=11,
                  arithmetic_selection_sample_splits_and_counts='verified',
                  scope='offline evidence audit, not independent refitting or scientific calibration')
    (base/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
