"""No-tracing timing of the already saved B=1 assessment identities.

This diagnostic does not change or validate the shipped B=100 workflow.
It compares complete local payloads, including risks and fit counts.
"""
import argparse
import json
import pathlib
import time
from dataclasses import replace

import pandas as pd
import open
from open.analysis import analyze_target
from open.config import load_config
from open.layer_artifacts import portable
from open.layer_stability import make_layer_sample_plans

p = argparse.ArgumentParser()
p.add_argument('--root', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
a = p.parse_args()
assert 'site-packages' in pathlib.Path(open.__file__).parts
c = load_config(a.root / 'configs/default.yaml')
d = pd.read_csv(a.root / 'examples/results/2026-10-08-layer-fixtures/chain/sample.csv')
saved = json.loads((a.root / 'examples/results/2026-10-13-research-preview-readiness/default-timing/bundle/assessment_ledger.json').read_text())
plan, = make_layer_sample_plans(d, c, n_resamples=1)
rows = []
for record in saved['assessments']:
    start = time.perf_counter()
    result = analyze_target(d.iloc[list(plan.sampled_positions)], record['target'],
                            tuple(record['candidates']),
                            replace(c, analysis=replace(c.analysis, random_seed=plan.seed)),
                            plan.split_plan)
    elapsed = time.perf_counter() - start
    payload = portable(result)
    assert payload == record['local'], 'local payload differs from saved traced run'
    rows.append(dict(target=record['target'], seconds=elapsed,
                     outer_fits=result.outer_model_fit_count,
                     tuning_fits=result.tuning_model_fit_count,
                     exact_saved_local_payload=True))
proof = dict(scope='timing diagnostic only; B=1 identities, no full-budget validation',
             installed_module=open.__file__, assessments=rows,
             seconds=sum(r['seconds'] for r in rows),
             no_tracemalloc=True, production_code_changed=False)
a.out.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
