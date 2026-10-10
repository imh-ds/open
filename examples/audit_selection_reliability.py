"""Recompute selection-study evidence without model fitting or test-set selection."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'examples'))
from open.config import parse_config
from open.layer_stability import make_layer_sample_plans
from run_selection_reliability import fingerprint, generate


def wilson(successes, total):
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., float(center-half)), min(1., float(center+half))]


def audit_case(base, spec, settings):
    case = base/spec['id']
    result = json.loads(gzip.decompress((case/'evidence.json.gz').read_bytes()))
    assert result['spec'] == spec
    assert parse_config(result['config']) == settings
    train = pd.read_csv(case/'train.csv', float_precision='round_trip')
    test = pd.read_csv(case/'test.csv', float_precision='round_trip')
    np.testing.assert_array_equal(train.to_numpy(), generate(spec['scenario'], spec['n'], spec['seed']).to_numpy())
    np.testing.assert_array_equal(test.to_numpy(), generate(spec['scenario'], 1000, spec['seed']+1000000).to_numpy())
    plans = make_layer_sample_plans(train, settings)
    records = result['records']
    assert len(records) == len(plans) == settings.stability.n_resamples
    relations, promotions, classes = Counter(), Counter(), {'a': Counter(), 'b': Counter()}
    stop_count = fit_count = tuning_count = warnings_count = completed = 0
    fit_sizes, inner_sizes, gains = [], [], []
    for r, plan in zip(records, plans):
        assert r['run'] == plan.run_index and r['seed'] == plan.seed
        assert r['sample_hash'] == fingerprint(plan.sampled_positions)
        assert r['split_hash'] == fingerprint(plan.split_plan)
        assert r['sample_n'] == len(plan.sampled_positions)
        warnings_count += len(r['warnings'])
        if r['status'] != 'completed':
            assert r['failure_type'] and r['failure_message']
            continue
        completed += 1
        fit_sizes.extend(r['fit_n'])
        inner_sizes.extend(r['tuning_n'])
        assert r['fit_n'] == [len(s.train_indices) for s in plan.split_plan.splits]
        assert r['tuning_n'] == [len(tr) for s in plan.split_plan.splits for tr, _ in
            KFold(settings.tuning_cv.folds, shuffle=True, random_state=plan.seed).split(s.train_indices)]
        rel = r['relevance']
        gain = rel['baseline_risk']-rel['full_risk']
        np.testing.assert_allclose(gain, rel['total_gain'], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(gain/rel['baseline_risk'], rel['relative_gain'], rtol=1e-12, atol=1e-14)
        gains.append(rel['relative_gain'])
        stopped = gain <= 0 or gain/rel['baseline_risk'] < settings.thresholds.min_total_gain_relative
        assert stopped == (r['stop'] == 'LOW_TOTAL_GAIN')
        stop_count += stopped
        fit_count += r['outer_fits']
        tuning_count += r['tuning_fits']
        expected_promoted = []
        for v in ('a', 'b'):
            for contrast in (rel['broad'][v], rel['unique'][v]):
                if stopped:
                    assert contrast['normalized'] is None
                else:
                    np.testing.assert_allclose(contrast['raw']/gain, contrast['normalized'], rtol=1e-12, atol=1e-14)
            kind = 'LOW_TOTAL_GAIN' if stopped else r['classes'][v]['kind']
            classes[v][kind] += 1
            if kind in ('PROXIMAL_UNIQUE', 'SHARED_GROUP_MEMBER'):
                expected_promoted.append(v)
            if kind == 'PROXIMAL_UNIQUE':
                assert rel['unique'][v]['normalized'] >= settings.thresholds.unique_relative
                relations[(v, kind, (), ())] += 1
            elif kind in ('SCREENED_BY_PARTNER', 'SCREENED_BY_SET', 'SCREENED_GENERAL'):
                vr = r['classes'][v]
                screened = (vr['screened_by_partner'] if kind == 'SCREENED_BY_PARTNER' else
                            vr['screened_by_set'] if kind == 'SCREENED_BY_SET' else [])
                relations[(v, kind, (), tuple(sorted(screened)))] += 1
            promotions[v] += v in r['promoted']
        assert r['promoted'] == expected_promoted
        for identity, group in r['joint'].items():
            members = tuple(identity.split(','))
            assert members == ('a', 'b') and not stopped
            assert all(r['classes'][v]['kind'] == 'SHARED_GROUP_MEMBER' for v in members)
            np.testing.assert_allclose(group['raw']/gain, group['normalized'], rtol=1e-12)
            assert group['normalized'] >= settings.thresholds.group_relative
            relations[(None, 'SHARED_GROUP', members, ())] += 1
    evidence = result['evidence']
    assert evidence['complete'] == (completed == len(plans))
    for c in evidence['candidates']:
        assert c['promotion'] == dict(selected=promotions[c['source']], planned=len(plans), completed=completed)
        assert {k:v['selected'] for k, v in c['classes'].items()} == dict(classes[c['source']])
    actual = {}
    retained = set()
    shared = False
    for r in evidence['relations']:
        identity = (r['source'], r['relation'], tuple(r['members']), tuple(r['screened_by']))
        assert identity not in actual
        actual[identity] = r['frequency']['selected']
        assert r['frequency']['planned'] == len(plans) and r['frequency']['completed'] == completed
        qualifies = (completed == len(plans) and r['relation'] in ('PROXIMAL_UNIQUE', 'SHARED_GROUP')
                     and actual[identity] > 0 and actual[identity]/len(plans) >= settings.thresholds.stability)
        assert r['qualifies'] == qualifies
        if qualifies:
            retained.update(r['members'] or [r['source']])
            shared |= r['relation'] == 'SHARED_GROUP'
    assert actual == dict(relations)
    row = dict(**spec, status=result['status'], planned=len(plans), completed=completed,
        low_gain=stop_count, median_local_gain=float(np.median(gains)) if gains else None,
        a_promoted=promotions['a'], b_promoted=promotions['b'], shared_count=relations[(None, 'SHARED_GROUP', ('a', 'b'), ())],
        min_fit_n=min(fit_sizes) if fit_sizes else None, max_fit_n=max(fit_sizes) if fit_sizes else None,
        min_tuning_n=min(inner_sizes) if inner_sizes else None, max_tuning_n=max(inner_sizes) if inner_sizes else None,
        outer_fits=fit_count, tuning_fits=tuning_count, warnings=warnings_count+len(result['warnings']),
        retained=','.join(sorted(retained)), shared_retained=shared,
        full_skill=None, retained_skill=None, skill_loss=None, root_recovered=None, false_input=None,
        loss_over_point05=None)
    if result['status'] != 'complete':
        assert result['scores'] is None
        return row
    assert completed == len(plans) and result['retained'] == sorted(retained)
    predictions = pd.read_csv(case/'predictions.csv', float_precision='round_trip')
    assert len(predictions) == 1000 and np.isfinite(predictions).all().all()
    np.testing.assert_array_equal(predictions.observed.to_numpy(), test.y.to_numpy())
    np.testing.assert_allclose(predictions.baseline, train.y.mean(), rtol=0, atol=1e-15)
    denominator = float(np.square(predictions.observed-predictions.baseline).sum())
    for name in ('baseline', 'full', 'retained'):
        mse = float(np.square(predictions.observed-predictions[name]).mean())
        expected = dict(mse=mse, rmse=float(np.sqrt(mse)), skill=1-mse*len(predictions)/denominator)
        for metric, value in expected.items():
            np.testing.assert_allclose(value, result['scores'][name][metric], rtol=1e-12, atol=1e-14)
    if retained == {'a', 'b'}:
        np.testing.assert_array_equal(predictions.full.to_numpy(), predictions.retained.to_numpy())
    if not retained:
        np.testing.assert_array_equal(predictions.baseline.to_numpy(), predictions.retained.to_numpy())
    scenario = spec['scenario']
    recovery = (not retained if scenario == 'noise' else 'a' in retained if scenario == 'ordinary'
                else bool(retained) if scenario == 'redundant' else retained == {'a', 'b'})
    false = bool(retained) if scenario == 'noise' else 'b' in retained if scenario == 'ordinary' else False
    full = result['scores']['full']['skill']
    selected = result['scores']['retained']['skill']
    row.update(full_skill=full, retained_skill=selected, skill_loss=full-selected,
               root_recovered=recovery, false_input=false, loss_over_point05=full-selected > .05)
    return row


def main():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base = args.output
    manifest = json.loads((base/'manifest.json').read_text())
    for source, expected in manifest['source_hashes'].items():
        actual = hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        if actual != expected and source == 'examples/run_selection_reliability.py' and manifest.get('execution_sources'):
            assert hashlib.sha256((base/'runner-at-initial-dispatch.py').read_bytes()).hexdigest() == expected
            assert actual == manifest['execution_sources'][-1]['runner_hash']
        else:
            assert actual == expected, source
    assert hashlib.sha256((base/'protocol.md').read_bytes()).hexdigest() == manifest['protocol_hash']
    settings = parse_config(manifest['config'])
    rows = [audit_case(base, spec, settings) for spec in manifest['specs']]
    cells = []
    for scenario in sorted({r['scenario'] for r in rows}):
        for n in sorted({r['n'] for r in rows if r['scenario'] == scenario}):
            cell = [r for r in rows if r['scenario'] == scenario and r['n'] == n]
            complete = [r for r in cell if r['status'] == 'complete']
            summary = dict(scenario=scenario, n=n, planned=len(cell), completed=len(complete))
            if len(complete) == len(cell):
                for label in ('root_recovered', 'false_input', 'shared_retained', 'loss_over_point05'):
                    count = sum(r[label] for r in complete)
                    summary[label] = dict(count=count, planned=len(cell), wilson95=wilson(count, len(cell)))
                losses = np.array([r['skill_loss'] for r in complete])
                summary.update(mean_skill_loss=float(losses.mean()),
                    mcse_skill_loss=float(losses.std(ddof=1)/np.sqrt(len(losses))) if len(losses) > 1 else None,
                    min_skill_loss=float(losses.min()), max_skill_loss=float(losses.max()),
                    mean_full_skill=float(np.mean([r['full_skill'] for r in complete])),
                    mean_retained_skill=float(np.mean([r['retained_skill'] for r in complete])),
                    mean_low_gain_fraction=float(np.mean([r['low_gain']/r['planned'] for r in complete])))
            cells.append(summary)
    with (base/'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (base/'summary.json').write_text(json.dumps(dict(cells=cells, cases=rows), indent=2)+'\n')
    result = dict(planned_cases=len(rows), completed_cases=sum(r['status'] == 'complete' for r in rows),
        planned_assessments=sum(r['planned'] for r in rows), completed_assessments=sum(r['completed'] for r in rows),
        independent_test_predictions=1000*sum(r['status'] == 'complete' for r in rows),
        warnings=sum(r['warnings'] for r in rows), local_outer_fits=sum(r['outer_fits'] for r in rows),
        local_tuning_fits=sum(r['tuning_fits'] for r in rows),
        checks='source/protocol, regenerated inputs/samples/splits, training sizes, local arithmetic, classes/promotions, relation counts, retention, test scores and dataset summaries',
        limit='Does not independently refit every local model or establish scientific calibration')
    (base/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
