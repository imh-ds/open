"""Fresh, bounded development evidence for the existing post-fit pair screen."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import time
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from open.interactions import PairScreenConfig, screen_pair_interactions, aggregate_pair_screens
from open.learners.trees import TreePredictor, GRIDS

SCENARIOS = ('noise', 'additive', 'pure', 'mixed', 'correlated_additive', 'correlated_mixed')
INTERACTION_SCENARIOS = ('pure', 'mixed', 'correlated_mixed')
POINTS = ((80, 5), (300, 5), (600, 5), (300, 15))
ENGINES = ('random_forest', 'gbm')
CUTOFFS = (.05, .10, .20, .30, .40, .50)
MANIFEST = ROOT/'examples/study_manifests/pair-screen.json'
SCREEN = PairScreenConfig()
EXPECTED_ENV = dict(python='3.11.9', numpy='1.26.4', pandas='2.2.3',
    scipy='1.11.2', sklearn='1.3.0', matplotlib='3.9.2', PyYAML='6.0.2')


def environment():
    return dict(python=platform.python_version(), **{key:importlib.metadata.version(
        'scikit-learn' if key=='sklearn' else key) for key in EXPECTED_ENV if key!='python'})


def validate_environment():
    if environment() != EXPECTED_ENV:
        raise ValueError('runtime environment differs from frozen versions')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def cases():
    result = []
    for scenario in SCENARIOS:
        for n, p in POINTS:
            for r in range(20):
                i = len(result)
                result.append(dict(id=f'{scenario}-n{n}-p{p}-r{r:02d}', scenario=scenario,
                    n=n, p=p, replicate=r, seed=38000000+i, test_seed=48000000+i,
                    split_seed=58000000+i))
    return result


def tasks():
    return [dict(id=c['id']+'--'+e, case=c, engine=e) for c in cases() for e in ENGINES]


def shards():
    return [f'{i:03d}' for i in range(240)]


def shard_tasks(identity):
    if identity not in shards():
        raise ValueError('foreign shard')
    i = int(identity)
    return tasks()[4*i:4*(i+1)]


@lru_cache(maxsize=1)
def manifest():
    paths = list((ROOT/'src/open').rglob('*.py')) + [ROOT/name for name in (
        'examples/pair_screen_study.py', 'examples/pair_screen_protocol.md',
        '.github/workflows/pair-screen.yml', '.github/actions/pair-screen-setup/action.yml')]
    return dict(schema='open.pair_screen_study.v1', tasks=tasks(), shards=shards(),
        config=asdict(SCREEN), cutoffs=list(CUTOFFS), budget_runner_minutes=7275,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(
            p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)},
        environment=EXPECTED_ENV)


def freeze(check=True):
    validate_environment()
    manifest.cache_clear()
    value = manifest()
    if check:
        if read(MANIFEST) != value:
            raise ValueError('frozen source/protocol/task context differs')
    else:
        write(MANIFEST, value)
    return value


def context(task):
    return fingerprint(dict(task=task, manifest=manifest()))


def generate(case, n, seed):
    if case['scenario'] not in SCENARIOS or case['p'] < 2:
        raise ValueError('invalid scenario/p')
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, case['p']))
    rho = .7 if case['scenario'].startswith('correlated_') else 0.
    if rho:
        x[:, 1] = rho*x[:, 0]+np.sqrt(1-rho*rho)*x[:, 1]
    scenario = case['scenario']
    truth = ['x1', 'x2'] if scenario in INTERACTION_SCENARIOS else None
    if scenario == 'noise':
        mean, variance = np.zeros(n), 0.
    elif scenario.endswith('additive'):
        mean, variance = x[:, 0]+.6*(x[:, 1]**2-1), 1.72
    elif scenario == 'pure':
        mean, variance = x[:, 0]*x[:, 1], 1.
    else:
        mean = x[:, 0]+.6*x[:, 1]+.8*(x[:, 0]*x[:, 1]-rho)
        variance = 1+.36+1.2*rho+.64*(1+rho*rho)
    y = mean+rng.normal(scale=1 if not variance else np.sqrt(variance*2/3), size=n)
    return pd.DataFrame(x, columns=[f'x{i+1}' for i in range(case['p'])]), y, mean, truth


def partitions(case, size=None):
    return list(KFold(3, shuffle=True, random_state=case['split_seed']).split(
        np.arange(case['n'] if size is None else size)))


def skill(observed, predicted, baseline):
    risk = float(np.square(np.asarray(observed)-predicted).mean())
    base = float(np.square(np.asarray(observed)-baseline).mean())
    return dict(mse=risk, baseline_mse=base, skill=1-risk/base)


def fit(engine, case, X, y, params=None):
    model = TreePredictor(engine, random_seed=case['split_seed'])
    if params is None:
        params = model.tune(X, y, partitions(case, len(X)))
    return model.fit(X, y, params), params


def dataset(task):
    c = task['case']
    return (generate(c, c['n'], c['seed']), generate(c, 1000, c['test_seed']))


def verification(task, folder):
    return dict(context=context(task), saved_setting_refits=True,
        result_sha256=fingerprint(read(folder/'result.json')),
        receipt_sha256=fingerprint(read(folder/'receipt.json')),
        predictions_sha256=hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest())


def stats(result):
    pairs = result['screen']['pairs']
    ranked = sorted([p for p in pairs if p['mean_screen_rms'] is not None],
                    key=lambda p: (-p['mean_screen_rms'], p['members']))
    truth = result['truth']
    rank = next((i+1 for i, p in enumerate(ranked) if p['members']==truth), None)
    shown = [p for p in pairs if p['mean_screen_rms'] is not None
             and p['shortlisted_folds'] and p['supported_detail_folds']]
    sensitivity = []
    for cutoff in CUTOFFS:
        exceeds = [p for p in ranked if p['mean_screen_rms'] >= cutoff]
        sensitivity.append(dict(cutoff=cutoff,
            true_exceeds=any(p['members']==truth for p in exceeds) if truth and rank is not None else None,
            any_incidental_exceeds=any(p['members']!=truth for p in exceeds)))
    return dict(true_rank=rank, true_assessed=rank is not None if truth else None,
        true_top1=rank==1 if truth and rank is not None else None,
        true_top5=rank<=5 if truth and rank is not None else None,
        true_detail=any(p['members']==truth for p in shown) if truth else None,
        incidental_details=sum(p['members']!=truth for p in shown), any_displayed=bool(shown),
        assessed_pairs=len(ranked), planned_pairs=len(pairs), sensitivity=sensitivity)


def check_result(task, folder, refit=False):
    value = read(folder/'result.json')
    if value['task'] != task or value['context'] != context(task) or value['status'] != 'complete':
        raise ValueError('task/context/status differs')
    receipt = read(folder/'receipt.json')
    if (receipt.get('environment') != EXPECTED_ENV or receipt.get('context') != context(task)
        or receipt.get('task') != task or receipt.get('screen_config') != asdict(SCREEN)):
        raise ValueError('receipt runtime/context/config differs')
    if refit:
        validate_environment()
    (X, y, mean, truth), (T, observed, oracle, _) = dataset(task)
    if value['truth'] != truth or len(value['folds']) != 3:
        raise ValueError('truth/planned folds differ')
    grid = list(ParameterGrid(GRIDS[task['engine']]))
    for f, (train, test) in zip(value['folds'], partitions(task['case'])):
        if f['train'] != train.tolist() or f['test'] != test.tolist() or f['params'] not in grid:
            raise ValueError('split/parameters differ')
        sizes = [len(a) for a,b in partitions(task['case'], len(train))]
        if f['inner_training_n'] != sizes:
            raise ValueError('inner sizes differ')
        np.testing.assert_allclose(f['observed'], y[test])
        baseline = np.full(len(test), y[train].mean())
        np.testing.assert_allclose(f['baseline'], baseline)
        np.testing.assert_allclose(f['mse'], np.square(y[test]-f['predicted']).mean())
        screen = f['screen']
        expected_pairs = [list(p) for p in __import__('itertools').combinations(sorted(X.columns), 2)]
        if (screen['config'] != asdict(SCREEN) or screen['additional_model_fits'] != 0
            or screen['prediction_rows'] > SCREEN.prediction_rows
            or [p['members'] for p in screen['pairs']] != expected_pairs):
            raise ValueError('pair plan/config/budget differs')
        if refit:
            model, _ = fit(task['engine'], task['case'], X.iloc[train], y[train], f['params'])
            np.testing.assert_allclose(model.predict(X.iloc[test]), f['predicted'], rtol=1e-9, atol=1e-9)
            actual = screen_pair_interactions(model, X.iloc[test], tuple(X.columns),
                outcome_sd=float(np.std(y, ddof=0)), config=SCREEN)
            if fingerprint(actual) != fingerprint(screen):
                raise ValueError('refitted pair screen differs')
    summary = aggregate_pair_screens([f['screen'] for f in value['folds']])
    if fingerprint(summary) != fingerprint(value['screen']):
        raise ValueError('aggregate pair arithmetic differs')
    if fingerprint(stats(value)) != fingerprint(value['metrics']):
        raise ValueError('summary metrics differ')
    cv = skill(np.concatenate([f['observed'] for f in value['folds']]),
        np.concatenate([f['predicted'] for f in value['folds']]),
        np.concatenate([f['baseline'] for f in value['folds']]))
    np.testing.assert_allclose(list(cv.values()), list(value['cv'][k] for k in cv))
    preds = pd.read_csv(folder/'predictions.csv')
    if list(preds) != ['observed','predicted','baseline','oracle'] or len(preds)!=1000:
        raise ValueError('independent prediction schema differs')
    if not np.isfinite(preds.to_numpy()).all():
        raise ValueError('nonfinite predictions')
    np.testing.assert_allclose(preds.observed, observed)
    np.testing.assert_allclose(preds.oracle, oracle)
    np.testing.assert_allclose(preds.baseline, y.mean())
    if value['final_params'] not in grid:
        raise ValueError('final settings differ')
    if value['final_inner_training_n'] != [len(a) for a,b in partitions(task['case'])]:
        raise ValueError('final inner sizes differ')
    testskill = skill(observed, preds.predicted, preds.baseline)
    np.testing.assert_allclose(list(testskill.values()), [value['test'][k] for k in testskill])
    if refit:
        model, _ = fit(task['engine'], task['case'], X, y, value['final_params'])
        np.testing.assert_allclose(model.predict(T), preds.predicted, rtol=1e-9, atol=1e-9)
    return value


def run_task(task, output):
    if task not in tasks():
        raise ValueError('unplanned task')
    validate_environment()
    folder = Path(output)/task['id']
    identity = context(task)
    if (folder/'result.json').exists():
        (folder/'verified.json').unlink(missing_ok=True)
        try:
            value = check_result(task, folder, refit=True)
        except Exception:
            (folder/'result.json').replace(folder/'invalid-result.json')
            raise
        write(folder/'verified.json', verification(task, folder))
        return value
    (X, y, mean, truth), (T, test_y, oracle, _) = dataset(task)
    write(folder/'receipt.json', dict(task=task, context=identity, screen_config=asdict(SCREEN),
        environment=environment()))
    started = time.monotonic()
    folds = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for i, (train, test) in enumerate(partitions(task['case'])):
            path = folder/f'fold-{i}.json'
            if path.exists():
                saved = read(path)
                if saved['context'] != identity:
                    raise ValueError('fold context differs')
                f = saved['result']
            else:
                model, params = fit(task['engine'], task['case'], X.iloc[train], y[train])
                predicted = model.predict(X.iloc[test])
                f = dict(train=train.tolist(), test=test.tolist(), params=params,
                    observed=y[test].tolist(), predicted=predicted.tolist(),
                    baseline=np.full(len(test), y[train].mean()).tolist(),
                    inner_training_n=[len(a) for a,b in partitions(task['case'], len(train))],
                    mse=float(np.square(y[test]-predicted).mean()),
                    screen=screen_pair_interactions(model, X.iloc[test], tuple(X.columns),
                        outcome_sd=float(np.std(y)), config=SCREEN))
                write(path, dict(context=identity, result=f))
            folds.append(f)
        model, params = fit(task['engine'], task['case'], X, y)
        preds = pd.DataFrame(dict(observed=test_y, predicted=model.predict(T),
            baseline=np.full(len(T), y.mean()), oracle=oracle))
    temporary = folder/'predictions.tmp.csv'
    preds.to_csv(temporary, index=False)
    temporary.replace(folder/'predictions.csv')
    value = dict(task=task, context=identity, status='complete', truth=truth, folds=folds,
        screen=aggregate_pair_screens([f['screen'] for f in folds]), final_params=params,
        final_inner_training_n=[len(a) for a,b in partitions(task['case'])],
        cv=skill(np.concatenate([f['observed'] for f in folds]),
            np.concatenate([f['predicted'] for f in folds]), np.concatenate([f['baseline'] for f in folds])),
        test=skill(test_y, preds.predicted, preds.baseline),
        fitting_screen_elapsed_seconds=time.monotonic()-started,
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured])
    value['metrics'] = stats(value)
    write(folder/'result.json', value)
    try:
        check_result(task, folder, refit=True)
    except Exception:
        (folder/'result.json').replace(folder/'invalid-result.json')
        raise
    write(folder/'verified.json', verification(task, folder))
    write(folder/'timing.json', dict(total_seconds=time.monotonic()-started))
    return value


def run_shard(identity, output):
    freeze()
    write(Path(output)/f'shard-{identity}.json', dict(id=identity, tasks=shard_tasks(identity),
        manifest=fingerprint(manifest()), status='started'))
    failures = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_task, t, output):t for t in shard_tasks(identity)}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(task['id'])
                write(Path(output)/task['id']/'error.json', dict(task=task,
                    context=context(task), error=repr(error)))
    write(Path(output)/f'shard-{identity}.json', dict(id=identity, tasks=shard_tasks(identity),
        manifest=fingerprint(manifest()), status='failed' if failures else 'complete'))
    if failures:
        raise RuntimeError(f'failed tasks retained: {failures}')


def merge_shards(inputs, destination):
    destination = Path(destination)
    for folder in inputs:
        for source in sorted(Path(folder).rglob('*')):
            if not source.is_file():
                continue
            target = destination/source.relative_to(folder)
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f'conflicting artifact: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)


def wilson(success, n):
    z = 1.959963984540054
    centre = (success/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(success/n*(1-success/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(centre-half), float(centre+half)]


def aggregate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    planned = tasks()
    if {p.name for p in output.iterdir() if p.is_dir()} - {t['id'] for t in planned}:
        raise ValueError('foreign evidence directory')
    results, errors = {}, []
    for task in planned:
        folder = output/task['id']
        if (folder/'result.json').exists():
            try:
                r = check_result(task, folder)
                if read(folder/'verified.json') != verification(task, folder):
                    raise ValueError('saved-setting verification missing')
                results[task['id']] = r
            except Exception as error:
                errors.append(dict(id=task['id'], error=repr(error)))
    cells = []
    cutoff_rows = []
    for scenario in SCENARIOS:
        for n,p in POINTS:
            for engine in ENGINES:
                ts = [t for t in planned if (t['case']['scenario'],t['case']['n'],t['case']['p'],t['engine'])
                      == (scenario,n,p,engine)]
                rs = [results[t['id']] for t in ts if t['id'] in results]
                count = len(rs)
                c = dict(scenario=scenario,n=n,p=p,engine=engine,planned=20,complete=count,
                    missing=20-count,mean_test_skill=float(np.mean([r['test']['skill'] for r in rs])) if rs else None,
                    mean_incidental_details=float(np.mean([r['metrics']['incidental_details'] for r in rs])) if rs else None,
                    mean_assessed_pair_fraction=float(np.mean([r['metrics']['assessed_pairs']/r['metrics']['planned_pairs'] for r in rs])) if rs else None,
                    mean_fit_screen_seconds=float(np.mean([r['fitting_screen_elapsed_seconds'] for r in rs])) if rs else None)
                for key in ('true_top1','true_top5','true_detail','any_displayed'):
                    successes = sum(r['metrics'][key] is True for r in rs)
                    unknown = sum(r['metrics'][key] is None for r in rs)
                    c[key+'_bounds'] = [successes,successes+20-count+unknown]
                    c[key+'_wilson'] = wilson(successes,20) if count==20 and not unknown else None
                c['true_assessed'] = sum(r['metrics']['true_assessed'] is True for r in rs)
                cells.append(c)
                for i, cutoff in enumerate(CUTOFFS):
                    true_count = sum(r['metrics']['sensitivity'][i]['true_exceeds'] is True for r in rs)
                    true_unknown = sum(r['metrics']['sensitivity'][i]['true_exceeds'] is None for r in rs)
                    incidental_count = sum(r['metrics']['sensitivity'][i]['any_incidental_exceeds'] for r in rs)
                    incidental_unknown = sum(not r['metrics']['sensitivity'][i]['any_incidental_exceeds']
                        and r['metrics']['assessed_pairs']<r['metrics']['planned_pairs'] for r in rs)
                    cutoff_rows.append(dict(scenario=scenario,n=n,p=p,engine=engine,cutoff=cutoff,
                        planned=20,complete=count,missing=20-count,
                        true_assessed=c['true_assessed'],
                        true_exceeds=true_count, true_exceeds_bounds=[true_count,true_count+20-count+true_unknown]
                            if scenario in INTERACTION_SCENARIOS else None,
                        any_incidental_exceeds=incidental_count,
                        incidental_exceeds_bounds=[incidental_count,incidental_count+20-count+incidental_unknown],
                        partially_assessed=sum(r['metrics']['assessed_pairs']<r['metrics']['planned_pairs'] for r in rs)))
    report = dict(planned=960,complete=len(results),missing=960-len(results),audit_errors=errors,
        failed_tasks=[t['id'] for t in planned if (output/t['id']/'error.json').exists()],
        cells=cells,cutoffs=cutoff_rows,acceptance='none: development only',
        prediction_rows=sum(r['screen']['prediction_rows'] for r in results.values()),
        warning_count=sum(len(r['warnings']) for r in results.values()))
    write(output/'audit.json', report)
    pd.DataFrame(cells).to_csv(output/'summary.csv', index=False)
    pd.DataFrame(cutoff_rows).to_csv(output/'cutoff_sensitivity.csv', index=False)
    lines = ['# Pair-screen development results', '',
        f"Completed {len(results)}/960 planned tasks; audit errors {len(errors)}.",
        'Development only. No cutoff or interaction-retention rule is accepted.', '',
        'Counts retain 20 planned independent datasets per cell. Rank bounds include unassessed pairs and missing tasks.',
        'True detail / any displayed are operational display coverage, not population detection or absence.', '',
        '|Scenario|N|p|Engine|Complete|True top one bounds|True detail bounds|Any displayed bounds|Mean test skill|',
        '|---|---:|---:|---|---:|---|---|---|---:|']
    for c in cells:
        truth = c['scenario'] in INTERACTION_SCENARIOS
        score = 'NA' if c['mean_test_skill'] is None else f"{c['mean_test_skill']:.3f}"
        lines.append(f"|{c['scenario']}|{c['n']}|{c['p']}|{c['engine']}|{c['complete']}/20|"
            f"{c['true_top1_bounds'] if truth else 'NA'}|{c['true_detail_bounds'] if truth else 'NA'}|"
            f"{c['any_displayed_bounds']}|{score}|")
    lines += ['', '![Prespecified summary plots](SUMMARY.png)', '',
        'See summary.csv for assessed-pair fractions and Wilson intervals; cutoff_sensitivity.csv',
        'retains assessed/missing counts and unknown-inclusive exceedance bounds. No cutoff is selected.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = read(Path(output)/'audit.json')
    fig, axes = plt.subplots(1,2,figsize=(12,5), layout='constrained')
    for engine in ENGINES:
        for ax,scenario,key in ((axes[0],'mixed','true_top1_bounds'),(axes[1],'additive','any_displayed_bounds')):
            cells = [c for c in report['cells'] if c['engine']==engine and c['scenario']==scenario and c['p']==5]
            cells.sort(key=lambda c:c['n'])
            ax.plot([c['n'] for c in cells], [c[key][0]/20 for c in cells],marker='o',label=engine)
            ax.fill_between([c['n'] for c in cells],[c[key][0]/20 for c in cells],
                [c[key][1]/20 for c in cells],alpha=.12)
            ax.set(xlabel='Discovery sample size',ylabel='Fraction of 20 planned datasets',ylim=(-.03,1.03))
            ax.legend()
    axes[0].set_title('Known mixed pair ranks first (p=5)')
    axes[1].set_title('Any incidental displayed candidate: additive (p=5)')
    fig.suptitle('Development screen: no validated detection threshold; shading = missing/unassessed bounds')
    fig.savefig(Path(output)/'SUMMARY.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','matrix','shard','aggregate','smoke'])
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--id')
    parser.add_argument('--output',type=Path,default=ROOT/'.superpowers/pair-screen')
    parser.add_argument('--input',type=Path)
    args = parser.parse_args()
    if args.command=='freeze':
        freeze(not args.write)
        print('frozen pair-screen context verified')
        return
    freeze()
    if args.command=='matrix':
        print(json.dumps(shards()))
    elif args.command=='shard':
        run_shard(args.id,args.output)
    elif args.command=='smoke':
        for engine in ENGINES:
            task=next(t for t in tasks() if t['case']['scenario']=='mixed' and
                t['case']['n']==80 and t['case']['replicate']==0 and t['engine']==engine)
            run_task(task,args.output)
    else:
        if args.input:
            merge_shards([p for p in args.input.iterdir() if p.is_dir()],args.output)
        report=aggregate(args.output)
        plots(args.output)
        print(json.dumps({k:report[k] for k in ('planned','complete','missing')}))
        if report['missing'] or report['audit_errors']:
            sys.exit(1)


if __name__=='__main__':
    main()
