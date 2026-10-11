"""Fresh paired learner screen; never modifies production learners or old studies."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'examples'))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
import sample_size_pilot as pilot
import selection_investigation as fitting
from open.splits import make_split_plan

VARIANTS = ('gbm_shipped', 'gbm_deeper', 'rf_shipped', 'benchmark')
MANIFEST = ROOT / 'examples/study_manifests/fresh-learning.json'


def cases():
    result = []
    for scenario in ('noise', 'additive', 'interaction', 'chain'):
        for n, p in ((80, 5), (300, 5), (300, 15)):
            for r in range(20):
                i = len(result)
                result.append(dict(id=f'{scenario}-n{n}-p{p}-r{r:02d}',
                    scenario=scenario, n=n, p=p, replicate=r, seed=8000000+i,
                    test_seed=18000000+i, discovery_seed=28000000+i))
    return result


def tasks():
    return [dict(id=f"{case['id']}--{variant}", case=case, variant=variant)
            for case in cases() for variant in VARIANTS]


def shards():
    return [f'{i:03d}' for i in range(120)]


def shard_tasks(identity):
    if identity not in shards():
        raise ValueError('foreign shard')
    i = int(identity)
    return tasks()[8*i:8*(i+1)]


def manifest():
    paths = list((ROOT / 'src/open').rglob('*.py')) + [
        ROOT / 'examples' / name for name in ('fresh_learning.py',
        'fresh_learning_protocol.md', 'sample_size_pilot.py',
        'selection_investigation.py', 'run_selection_reliability.py')]
    paths.append(ROOT / '.github/workflows/fresh-learning.yml')
    paths.append(ROOT / '.github/actions/investigation-setup/action.yml')
    return dict(schema=1, tasks=tasks(), shards=shards(),
        sources={p.relative_to(ROOT).as_posix():pilot.text_hash(p.read_bytes())
                 for p in sorted(paths)}, budget_runner_minutes=7275,
        environment=dict(python='3.11.9', numpy='1.26.4', pandas='2.2.3',
            scipy='1.11.2', sklearn='1.3.0', PyYAML='6.0.2'))


def freeze(check=True):
    value = manifest()
    if check:
        if json.loads(MANIFEST.read_text()) != value:
            raise ValueError('frozen manifest/source differs')
    else:
        pilot.write_json(MANIFEST, value)
    return value


def context(task):
    return pilot.fingerprint(dict(task=task, manifest=manifest()))


def features(frame, variant, scenario):
    if variant != 'benchmark':
        return frame.drop(columns='y')
    if scenario == 'interaction':
        return pd.DataFrame({'known_product':frame.x1*frame.x2})
    if scenario == 'noise':
        return pd.DataFrame({'constant':np.zeros(len(frame))})
    return frame[['x1']]


def data(task):
    c = task['case']
    train, _ = pilot.generate(c['scenario'], c['n'], c['p'], c['seed'])
    test, oracle = pilot.generate(c['scenario'], 2000, c['p'], c['test_seed'])
    config = pilot.settings(c)
    X = features(train, task['variant'], c['scenario'])
    T = features(test, task['variant'], c['scenario'])
    return config, train, test, oracle, X, T, make_split_plan(train, config)


def fit_variant(task):
    return 'linear' if task['variant'] == 'benchmark' else task['variant']


def check_parameters(task, params):
    if params not in list(ParameterGrid(fitting.model_grid(fit_variant(task)))):
        raise ValueError('parameters outside declared grid')


def inner_sizes(n, config):
    return [len(a) for a,b in KFold(3,shuffle=True,
        random_state=config.analysis.random_seed).split(np.arange(n))]


def check_result(task, folder, refit=False):
    """Regenerate identities/data/splits and arithmetic; optional saved-setting refits."""
    result = json.loads((folder / 'result.json').read_text())
    if result.get('context') != context(task) or result.get('task') != task:
        raise ValueError('result context/identity differs')
    if result['status'] != 'complete':
        raise ValueError('incomplete result')
    config, train, test, oracle, X, T, plan = data(task)
    if len(result['folds']) != len(plan.splits):
        raise ValueError('missing/duplicate folds')
    variant = fit_variant(task)
    for saved, split in zip(result['folds'], plan.splits):
        check_parameters(task,saved['parameters'])
        if saved['inner_training_n'] != inner_sizes(len(split.train_indices),config):
            raise ValueError('inner training sizes differ')
        if (saved['repeat'], saved['fold'], saved['train'], saved['test']) != (
                split.repeat, split.fold, list(split.train_indices), list(split.test_indices)):
            raise ValueError('fold split differs')
        observed = train.y.iloc[saved['test']].to_numpy()
        np.testing.assert_allclose(saved['observed'], observed, rtol=1e-10, atol=1e-10)
        baseline = np.full(len(observed), train.y.iloc[saved['train']].mean())
        np.testing.assert_allclose(saved['baseline'], baseline)
        np.testing.assert_allclose(saved['mse'], np.square(observed-saved['predicted']).mean())
        if refit:
            model = fitting.model_pipeline(variant, config, saved['parameters'])
            model.fit(X.iloc[saved['train']], train.y.iloc[saved['train']])
            np.testing.assert_allclose(model.predict(X.iloc[saved['test']]), saved['predicted'],
                                       rtol=1e-9, atol=1e-9)
    preds = pd.read_csv(folder / 'predictions.csv')
    if list(preds) != ['observed', 'predicted', 'baseline', 'oracle'] or len(preds) != 2000:
        raise ValueError('prediction schema differs')
    np.testing.assert_allclose(preds.observed, test.y, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(preds.baseline, train.y.mean())
    np.testing.assert_allclose(preds.oracle, oracle, rtol=1e-10, atol=1e-10)
    if not np.isfinite(preds.to_numpy()).all():
        raise ValueError('nonfinite predictions')
    check_parameters(task,result['final_parameters'])
    if result['final_inner_training_n'] != inner_sizes(len(train),config):
        raise ValueError('final inner training sizes differ')
    if refit:
        model = fitting.model_pipeline(variant, config, result['final_parameters']).fit(X, train.y)
        np.testing.assert_allclose(model.predict(T), preds.predicted, rtol=1e-9, atol=1e-9)
    obs = np.concatenate([f['observed'] for f in result['folds']])
    predicted = np.concatenate([f['predicted'] for f in result['folds']])
    baseline = np.concatenate([f['baseline'] for f in result['folds']])
    for name, scores in [('cv', fitting.skill(obs, predicted, baseline)),
                         ('test', fitting.skill(test.y, preds.predicted, preds.baseline)),
                         ('oracle', fitting.skill(test.y, oracle, preds.baseline))]:
        for key, value in scores.items():
            np.testing.assert_allclose(result[name][key], value, rtol=1e-9, atol=1e-9)
    return result


def run_task(task, output):
    if task not in tasks():
        raise ValueError('unplanned task')
    folder = Path(output) / task['id']
    identity = context(task)
    if (folder / 'result.json').exists():
        result = check_result(task, folder, refit=True)
        pilot.write_json(folder / 'verified.json', dict(context=identity, saved_setting_refits=True))
        return result
    config, train, test, oracle, X, T, plan = data(task)
    pilot.write_json(folder / 'receipt.json', dict(task=task, context=identity,
        splits=json.loads(plan.to_json()), features=list(X), config=pilot.portable(config)))
    started = time.monotonic()
    folds = []
    for split in plan.splits:
        path = folder / f'fold-{split.repeat}-{split.fold}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['context'] != identity:
                raise ValueError('fold context differs')
            value = saved['result']
        else:
            value = fitting.fold_fit(dict(variant=fit_variant(task)), config, X, train.y, split)
            pilot.write_json(path, dict(context=identity, result=value))
        folds.append(value)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model, params, mse, sizes = fitting.tuned_model(fit_variant(task), config, X, train.y)
        pred = model.predict(T)
    predictions = pd.DataFrame(dict(observed=test.y, predicted=pred,
        baseline=np.full(len(test), train.y.mean()), oracle=oracle))
    temporary = folder / 'predictions.tmp.csv'
    predictions.to_csv(temporary, index=False)
    temporary.replace(folder / 'predictions.csv')
    obs = np.concatenate([f['observed'] for f in folds])
    predicted = np.concatenate([f['predicted'] for f in folds])
    baseline = np.concatenate([f['baseline'] for f in folds])
    result = dict(task=task, context=identity, status='complete', folds=folds,
        final_parameters=params, final_inner_mse=mse, final_inner_training_n=sizes,
        cv=fitting.skill(obs, predicted, baseline),
        test=fitting.skill(test.y, pred, predictions.baseline),
        oracle=fitting.skill(test.y, oracle, predictions.baseline),
        elapsed_seconds=time.monotonic()-started,
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured])
    pilot.write_json(folder / 'result.json', result)
    try:
        check_result(task, folder, refit=True)
    except Exception:
        # A failed verification is never a completed task.
        (folder / 'result.json').replace(folder / 'invalid-result.json')
        raise
    pilot.write_json(folder / 'verified.json', dict(context=identity, saved_setting_refits=True))
    return result


def run_shard(identity, output):
    freeze()
    failures = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_task, t, output):t for t in shard_tasks(identity)}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(task['id'])
                pilot.write_json(Path(output)/task['id']/'error.json',
                    dict(task=task, context=context(task), error=repr(error)))
    if failures:
        raise RuntimeError(f'failed tasks retained: {failures}')


def wilson(success, n):
    z = 1.959963984540054
    p = success/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(centre-half), float(centre+half)]


def nominees(cells):
    accepted = []
    for variant in ('gbm_deeper','rf_shipped'):
        subset = [c for c in cells if c['variant'] == variant]
        if len(subset) != 12 or any(c['complete'] != 20 or c['paired_n'] != 20 for c in subset):
            continue
        if (all(c['mean_paired_difference'] >= .05 for c in subset
                if c['scenario'] == 'interaction' and c['n'] == 300)
            and all(c['mean_paired_difference'] >= -.05 for c in subset
                if c['scenario'] in ('additive','chain'))):
            accepted.append(variant)
    return accepted


def aggregate(output, verify=True):
    output = Path(output)
    planned = tasks()
    expected = {t['id'] for t in planned}
    if {p.name for p in output.iterdir() if p.is_dir()} - expected:
        raise ValueError('foreign evidence directory')
    results, errors = {}, []
    for task in planned:
        folder = output / task['id']
        if (folder / 'result.json').exists():
            try:
                value = check_result(task, folder)
                verified = json.loads((folder/'verified.json').read_text()) if verify else {}
                if verify and verified != dict(context=context(task), saved_setting_refits=True):
                    raise ValueError('missing saved-setting verification')
                if value['status'] != 'complete':
                    raise ValueError('incomplete result')
                results[task['id']] = value
            except Exception as error:
                errors.append(dict(id=task['id'], error=repr(error)))
    cells = []
    for scenario in ('noise', 'additive', 'interaction', 'chain'):
        for n, p in ((80,5), (300,5), (300,15)):
            for variant in VARIANTS:
                ts = [t for t in planned if (t['case']['scenario'], t['case']['n'],
                    t['case']['p'], t['variant']) == (scenario,n,p,variant)]
                rs = [results[t['id']] for t in ts if t['id'] in results]
                values = [r['test']['skill'] for r in rs]
                successes = sum(v > 0 for v in values)
                differences = []
                for t in ts:
                    base = t['case']['id']+'--gbm_shipped'
                    if t['id'] in results and base in results:
                        differences.append(results[t['id']]['test']['skill']-results[base]['test']['skill'])
                cells.append(dict(scenario=scenario,n=n,p=p,variant=variant,planned=20,
                    complete=len(rs),positive_skill_lower_bound_count=successes,
                    positive_skill_count_bounds=[successes,successes+20-len(rs)],
                    positive_skill_wilson_bounds=[wilson(successes,20)[0],
                        wilson(successes+20-len(rs),20)[1]],
                    mean_test_skill=float(np.mean(values)) if values else None,
                    test_skill_values=values,paired_n=len(differences),
                    paired_differences=differences,
                    mean_paired_difference=float(np.mean(differences)) if differences else None))
    report = dict(planned=960,complete=len(results),missing=960-len(results),
        audit_errors=errors,cells=cells,
        development_nominees=nominees(cells) if len(results) == 960 else [],
        failed_tasks=[t['id'] for t in planned if (output/t['id']/'error.json').exists()],
        scope='development prediction screen; no network claims')
    pilot.write_json(output/'audit.json',report)
    lines = ['# Fresh independent learner screen', '',
        f"Completed {len(results)}/960 planned method fits; audit errors {len(errors)}.",
        'Prediction screen only: network reliability and minimum sample size remain unvalidated.', '',
        f"Development nominees (not confirmed): {', '.join(report['development_nominees']) or 'none'}.", '',
        '|Scenario|N|p|Method|Complete / 20|Mean test skill|Paired difference to GBM|',
        '|---|---:|---:|---|---:|---:|---:|']
    for c in cells:
        score = 'NA' if c['mean_test_skill'] is None else f"{c['mean_test_skill']:.3f}"
        diff = 'NA' if c['mean_paired_difference'] is None else f"{c['mean_paired_difference']:.3f} (n={c['paired_n']})"
        lines.append(f"|{c['scenario']}|{c['n']}|{c['p']}|{c['variant']}|{c['complete']}|{score}|{diff}|")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze','matrix','shard','aggregate','smoke'])
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--id')
    parser.add_argument('--output',type=Path,default=ROOT/'.superpowers/fresh-learning')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(not args.write)
        print('frozen manifest verified')
    elif args.command == 'matrix':
        freeze()
        print(json.dumps(shards()))
    elif args.command == 'shard':
        run_shard(args.id,args.output)
    elif args.command == 'smoke':
        freeze()
        run_task(tasks()[0],args.output)
    else:
        freeze()
        args.output.mkdir(parents=True,exist_ok=True)
        result = aggregate(args.output)
        print(json.dumps({k:result[k] for k in ('planned','complete','missing')}))
        if result['audit_errors'] or result['missing']:
            sys.exit(1)


if __name__ == '__main__':
    main()
