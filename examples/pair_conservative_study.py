"""Fresh one-SE-style joint selection study; production and frozen studies unchanged."""
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
sys.path.insert(0, str(ROOT/'examples'))
import pair_screen_study as io
import spline_stability_smoke as stability
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from open.pair_validation import PairValidationConfig, _estimator, validate_pairs
from pair_correlation_study import WeightedSplineFeatures, estimator, generate, oracle_gap, diagnostics

MANIFEST = ROOT/'examples/study_manifests/pair-conservative.json'
PAIRS = (('x1', 'x2'), ('x1', 'x3'))
WEIGHTS = (1., 10., 100.)
POINTS = ((80, 5), (300, 5), (600, 5), (300, 15))
CUTOFFS = (0., .01, .02, .05)
FIT_CAP = 224


def choose_one_se(parameters, fold_losses):
    """Research-only penalty preference; fold uncertainty is a heuristic, not a CI."""
    losses = np.asarray(fold_losses, dtype=float)
    if (not parameters or losses.ndim != 2 or losses.shape[0] != len(parameters) or
        losses.shape[1] < 2 or not np.isfinite(losses).all() or (losses < 0).any()):
        raise ValueError('CV losses must be a finite nonnegative candidate-by-fold matrix')
    required = {'curves__joint_weight', 'curves__n_knots', 'ridge__alpha'}
    for p in parameters:
        if (set(p) != required or any(isinstance(v,bool) for v in p.values()) or
            p['curves__joint_weight'] not in WEIGHTS or p['curves__n_knots'] not in (3,5) or
            p['ridge__alpha'] not in (.1,1.,10.,100.)):
            raise ValueError('candidate outside research grid')
    if len({tuple(sorted(p.items())) for p in parameters}) != len(parameters):
        raise ValueError('duplicate candidates')
    means = losses.mean(axis=1)
    best = int(np.argmin(means))
    se = float(np.std(losses[best], ddof=1)/np.sqrt(losses.shape[1]))
    threshold = float(means[best]+se)
    eligible = np.flatnonzero(means <= threshold).tolist()
    def preference(i):
        p=parameters[i]
        return (-p['ridge__alpha']/p['curves__joint_weight']**2,
            p['curves__n_knots'], -p['ridge__alpha'], p['curves__joint_weight'])
    selected = min(eligible, key=preference)
    return dict(rule='joint_penalty_one_se.v1', best_index=best, selected_index=selected,
        mean_losses=means.tolist(), best_se=se, threshold=threshold, eligible_indices=eligible)


def tasks():
    result = []
    for rho in (0., .7, .9):
        for scenario in ('noise', 'additive', 'pure', 'mixed'):
            for n, p in POINTS:
                for rep in range(5):
                    i = len(result)
                    result.append(dict(id=f'one-se--{scenario}-rho{rho:g}-n{n}-p{p}-r{rep:02d}',
                        scenario=scenario, rho=rho, n=n, p=p, replicate=rep,
                        seed=178000000+i, test_seed=188000000+i, split_seed=198000000+i))
    return result


def shards():
    return [f'{i:03d}' for i in range(120)]


def shard_tasks(identity):
    if identity not in shards():
        raise ValueError('foreign shard')
    i = int(identity)
    return tasks()[2*i:2*i+2]


def manifest():
    paths = list((ROOT/'src/open').rglob('*.py'))+[ROOT/n for n in (
        'examples/pair_conservative_study.py', 'examples/pair_conservative_protocol.md',
        'examples/pair_correlation_study.py',
        'examples/pair_screen_study.py', 'examples/spline_stability_smoke.py',
        '.github/workflows/pair-conservative.yml', '.github/actions/pair-conservative-setup/action.yml')]
    return dict(schema='open.pair_conservative_study.v1', tasks=tasks(), shards=shards(),
        pairs=[list(p) for p in PAIRS], joint_weights=list(WEIGHTS), fit_cap=FIT_CAP,
        inner_folds=3, outer_folds=3, knots=[3, 5], alphas=[.1, 1., 10., 100.],
        selector_rule='joint_penalty_one_se.v1', independent_test_n=1000, cutoffs=list(CUTOFFS), environment=io.EXPECTED_ENV,
        budget_runner_minutes=3675,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(
            p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)})


def freeze(check=True):
    io.validate_environment()
    value = manifest()
    if check:
        if io.read(MANIFEST) != value:
            raise ValueError('frozen correlation context differs')
    else:
        io.write(MANIFEST, value)
    return value


def context(task):
    return io.fingerprint(dict(task=task, manifest=manifest()))


def splits(c):
    return list(KFold(3, shuffle=True, random_state=c['split_seed']).split(range(c['n'])))


def diagnostics(predicted, observed, sd):
    squared = np.square(observed-predicted)
    return dict(max_absolute_prediction_sd=float(np.max(np.abs(predicted))/sd),
        worst_error_share=float(squared.max()/squared.sum()) if squared.sum()>0 else 0.)


def compare(X, y, train, test, seed):
    train, test = list(map(int, train)), list(map(int, test))
    v2 = validate_pairs(X, y, tuple(train), tuple(test), PAIRS,
        config=PairValidationConfig(seed=seed))
    assert all(p['status']=='available' for p in v2['pairs'])
    inner = list(KFold(3, shuffle=True, random_state=seed).split(train))
    research, conservative, selection = [], [], []
    sd = float(np.std(y[train]))
    for pair in PAIRS:
        position = tuple(X.columns.get_loc(p) for p in pair)
        search = GridSearchCV(estimator(position), dict(curves__n_knots=[3, 5],
            ridge__alpha=[.1, 1., 10., 100.], curves__joint_weight=list(WEIGHTS)),
            cv=inner, scoring='neg_mean_squared_error', error_score='raise',
            n_jobs=1, refit=True).fit(X.iloc[train], y[train])
        pred = search.predict(X.iloc[test])
        mse = float(np.square(y[test]-pred).mean())
        research.append(dict(members=list(pair), parameters=search.best_params_,
            predicted=pred.tolist(), joint_mse=mse,
            gain=(v2['baseline_mse']-mse)/v2['reference_mse'],
            **diagnostics(pred, y[test], sd)))
        parameters = list(search.cv_results_['params'])
        losses = [[-float(search.cv_results_[f'split{fold}_test_score'][i])
            for fold in range(3)] for i in range(len(parameters))]
        chosen = choose_one_se(parameters, losses)
        if parameters[chosen['best_index']] != search.best_params_:
            raise ValueError('minimum-CV choice disagrees with retained fold scores')
        settings = parameters[chosen['selected_index']]
        model = estimator(position).set_params(**settings).fit(X.iloc[train], y[train])
        cpred = model.predict(X.iloc[test])
        cmse = float(np.square(y[test]-cpred).mean())
        conservative.append(dict(members=list(pair), parameters=settings, predicted=cpred.tolist(),
            joint_mse=cmse, gain=(v2['baseline_mse']-cmse)/v2['reference_mse'],
            **diagnostics(cpred, y[test], sd)))
        selection.append(dict(**chosen, candidate_parameters=parameters, fold_losses=losses))
    fits = v2['additional_fits']+2*(2*4*3*3+2)
    if fits>FIT_CAP:
        raise ValueError('research fit budget exceeded')
    return dict(v2=v2, research=research, conservative=conservative, selection=selection, additional_fits=fits,
        train_indices=train, assessment_indices=test)


def check_comparison(value, X, y, train, test, seed, refit=False):
    if value['train_indices']!=list(train) or value['assessment_indices']!=list(test):
        raise ValueError('comparison split differs')
    v = value['v2']
    expected = json.loads(json.dumps(PairValidationConfig(seed=seed).__dict__))
    if (v['schema']!='open.experimental_pair_validation.v2' or json.loads(json.dumps(v['config']))!=expected or
        v['train_indices']!=list(train) or v['assessment_indices']!=list(test) or
        v['additional_fits']!=75 or value['additional_fits']!=223 or
        v['feature_scaling']!='standardized_marginals_natural_joint' or
        v['solver']!=dict(name='lsqr', tol=1e-8)):
        raise ValueError('comparison configuration differs')
    observed = y[test]
    baseline_params = v['baseline_parameters']
    if (set(baseline_params)!={'curves__n_knots', 'ridge__alpha'} or
        baseline_params['curves__n_knots'] not in (3, 5) or
        baseline_params['ridge__alpha'] not in (.1, 1., 10., 100.)):
        raise ValueError('baseline parameters outside frozen grid')
    inner = list(KFold(3, shuffle=True, random_state=seed).split(train))
    if v['inner_training_n']!=[len(a) for a, b in inner]:
        raise ValueError('inner training sizes differ')
    reference = float(np.square(observed-y[train].mean()).mean())
    np.testing.assert_allclose(v['observed'], observed, rtol=0, atol=0)
    if [r['members'] for r in v['pairs']]!=[list(p) for p in PAIRS] or [
        r['members'] for r in value['research']]!=[list(p) for p in PAIRS] or [
        r['members'] for r in value['conservative']]!=[list(p) for p in PAIRS]:
        raise ValueError('locked pairs differ')
    if refit:
        stability.verify_comparison(v, X, y)
    common_base = np.asarray(v['pairs'][0]['baseline_predicted'])
    for production, research, conservative in zip(v['pairs'], value['research'], value['conservative']):
        if production['status']!='available':
            raise ValueError('production comparison unavailable')
        base = np.asarray(production['baseline_predicted'])
        if base.shape!=observed.shape or not np.isfinite(base).all():
            raise ValueError('invalid baseline predictions')
        np.testing.assert_allclose(base, common_base, rtol=0, atol=0)
        mse_base = float(np.square(observed-base).mean())
        np.testing.assert_allclose([v['baseline_mse'], v['reference_mse']], [mse_base, reference])
        np.testing.assert_allclose([production['baseline_mse'], production['reference_mse']],
            [mse_base, reference], rtol=1e-9, atol=1e-10)
        for record in (production, research, conservative):
            params = record['parameters']
            keys = {'curves__n_knots', 'ridge__alpha'} | (
                {'curves__joint_weight'} if record is not production else set())
            if (set(params)!=keys or params['curves__n_knots'] not in (3, 5) or
                params['ridge__alpha'] not in (.1, 1., 10., 100.) or
                (record is not production and params['curves__joint_weight'] not in WEIGHTS)):
                raise ValueError('selected parameters outside frozen grid')
            predicted = np.asarray(record['predicted'])
            if predicted.shape!=observed.shape or not np.isfinite(predicted).all():
                raise ValueError('invalid predictions')
            mse = float(np.square(observed-predicted).mean())
            np.testing.assert_allclose([record['joint_mse'], record['gain']],
                [mse, (mse_base-mse)/reference], rtol=1e-9, atol=1e-10)
            if record is not production:
                if record | diagnostics(predicted, observed, np.std(y[train])) != record:
                    raise ValueError('research diagnostics differ')
                if refit:
                    position = tuple(X.columns.get_loc(p) for p in record['members'])
                    model = estimator(position).set_params(**params).fit(X.iloc[train], y[train])
                    np.testing.assert_allclose(model.predict(X.iloc[test]), predicted, rtol=1e-9, atol=1e-9)
    if len(value['selection']) != 2:
        raise ValueError('selection record count differs')
    from sklearn.model_selection import ParameterGrid
    expected_grid = list(ParameterGrid(dict(curves__n_knots=[3,5], ridge__alpha=[.1,1.,10.,100.],
        curves__joint_weight=list(WEIGHTS))))
    for minimum, restrained, saved in zip(value['research'], value['conservative'], value['selection']):
        if saved['candidate_parameters'] != expected_grid or np.asarray(saved['fold_losses']).shape != (24,3):
            raise ValueError('retained CV grid differs')
        derived = choose_one_se(saved['candidate_parameters'], saved['fold_losses'])
        if saved != dict(**derived, candidate_parameters=saved['candidate_parameters'], fold_losses=saved['fold_losses']):
            raise ValueError('retained selection arithmetic differs')
        if (minimum['parameters'] != expected_grid[derived['best_index']] or
            restrained['parameters'] != expected_grid[derived['selected_index']]):
            raise ValueError('selected model settings differ')


def verification(c, folder):
    return dict(context=context(c), saved_setting_refits=28,
        **{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
           for name in ('result.json', 'receipt.json', 'timing.json', 'predictions.csv',
                        *(f'comparison-{i}.json' for i in range(4)))})


def check_result(c, folder, refit=False):
    v = io.read(folder/'result.json')
    if v['task']!=c or v['context']!=context(c) or v['status']!='complete':
        raise ValueError('result identity differs')
    if io.read(folder/'receipt.json')!=dict(task=c, context=context(c), environment=io.EXPECTED_ENV):
        raise ValueError('receipt differs')
    X, y, _, _ = generate(c, c['n'], c['seed'])
    T, ty, full, add = generate(c, 1000, c['test_seed'])
    if len(v['folds'])!=3:
        raise ValueError('outer fold coverage differs')
    for f, (train, test) in zip(v['folds'], splits(c)):
        check_comparison(f, X, y, train.tolist(), test.tolist(), c['split_seed'], refit)
    combined = pd.concat([X, T], ignore_index=True)
    cy = np.concatenate([y, ty])
    check_comparison(v['independent'], combined, cy, list(range(c['n'])),
        list(range(c['n'], c['n']+1000)), c['split_seed'], refit)
    pred = pd.read_csv(folder/'predictions.csv')
    expected = prediction_table(v, ty, full, add)
    if list(pred)!=list(expected):
        raise ValueError('prediction columns differ')
    np.testing.assert_allclose(pred.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v['oracle_gap'], oracle_gap(c), rtol=0, atol=0)
    np.testing.assert_allclose([v['oracle_full_mse'], v['oracle_additive_mse']],
        [np.square(ty-full).mean(), np.square(ty-add).mean()], rtol=1e-12)
    timing = io.read(folder/'timing.json')
    if (timing['fitting_tuning_fits']!=892 or timing['saved_setting_refits']!=28 or
        type(timing['resumed_comparisons']) is not int or not 0<=timing['resumed_comparisons']<=4 or
        timing['fitting_tuning_fits_this_attempt']!=223*(4-timing['resumed_comparisons']) or
        not all(np.isfinite(timing[k]) and timing[k]>=0 for k in (
            'total_seconds', 'verification_seconds', 'retained_fitting_seconds'))):
        raise ValueError('cost evidence differs')
    return v


def prediction_table(v, observed, full, additive):
    data = dict(observed=observed, oracle_full=full, oracle_additive=additive,
        baseline=v['independent']['v2']['pairs'][0]['baseline_predicted'])
    for i in range(2):
        data[f'v2_pair{i}'] = v['independent']['v2']['pairs'][i]['predicted']
        data[f'research_pair{i}'] = v['independent']['research'][i]['predicted']
        data[f'conservative_pair{i}'] = v['independent']['conservative'][i]['predicted']
    return pd.DataFrame(data)


def run_task(c, output):
    io.validate_environment()
    if c not in tasks():
        raise ValueError('foreign task')
    start = time.monotonic()
    folder = Path(output)/c['id']
    folder.mkdir(parents=True, exist_ok=True)
    receipt = dict(task=c, context=context(c), environment=io.environment())
    if (folder/'receipt.json').exists() and io.read(folder/'receipt.json')!=receipt:
        raise ValueError('existing receipt differs')
    io.write(folder/'receipt.json', receipt)
    if (folder/'verified.json').exists():
        value = check_result(c, folder)
        if io.read(folder/'verified.json')!=verification(c, folder):
            raise ValueError('existing verification differs')
        return value
    X, y, _, _ = generate(c, c['n'], c['seed'])
    T, ty, full, add = generate(c, 1000, c['test_seed'])
    combined = pd.concat([X, T], ignore_index=True)
    cy = np.concatenate([y, ty])
    comparisons = [(X, y, a.tolist(), b.tolist()) for a, b in splits(c)]+[
        (combined, cy, list(range(c['n'])), list(range(c['n'], c['n']+1000)))]
    results, recorded_warnings = [], []
    resumed = 0
    for i, (data, outcome, train, test) in enumerate(comparisons):
        checkpoint = folder/f'comparison-{i}.json'
        if checkpoint.exists():
            saved = io.read(checkpoint)
            if saved['context']!=context(c):
                raise ValueError('checkpoint context differs')
            # Arithmetic now, saved-setting refits once in final verification.
            check_comparison(saved['result'], data, outcome, train, test, c['split_seed'])
            resumed += 1
        else:
            fitting_start = time.monotonic()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                result = compare(data, outcome, train, test, c['split_seed'])
            saved = dict(context=context(c), result=result, fitting_seconds=time.monotonic()-fitting_start, warnings=[
                dict(category=w.category.__name__, message=str(w.message)) for w in captured])
            io.write(checkpoint, saved)
        results.append(saved['result'])
        recorded_warnings.extend(saved['warnings'])
    value = dict(task=c, context=context(c), status='complete', folds=results[:3], independent=results[3],
        oracle_gap=oracle_gap(c), oracle_full_mse=float(np.square(ty-full).mean()),
        oracle_additive_mse=float(np.square(ty-add).mean()), warnings=recorded_warnings)
    io.write(folder/'result.json', value)
    prediction_table(value, ty, full, add).to_csv(folder/'predictions.csv', index=False)
    verification_start = time.monotonic()
    # Provisional timing allows the same validator to check both construction and final evidence.
    fitting_seconds = sum(io.read(folder/f'comparison-{i}.json')['fitting_seconds'] for i in range(4))
    cost = dict(fitting_tuning_fits=892, saved_setting_refits=28, resumed_comparisons=resumed,
        fitting_tuning_fits_this_attempt=223*(4-resumed), retained_fitting_seconds=fitting_seconds)
    io.write(folder/'timing.json', dict(**cost,
        total_seconds=time.monotonic()-start, verification_seconds=0.))
    check_result(c, folder, refit=True)
    io.write(folder/'timing.json', dict(**cost,
        total_seconds=time.monotonic()-start, verification_seconds=time.monotonic()-verification_start))
    io.write(folder/'verified.json', verification(c, folder))
    return value


def pooled_gain(v, mode, pair):
    records = [f['v2']['pairs'][pair] if mode=='v2' else f[mode][pair] for f in v['folds']]
    return sum((f['v2']['baseline_mse']-r['joint_mse'])*len(r['predicted'])
        for f, r in zip(v['folds'], records))/sum(
        f['v2']['reference_mse']*len(r['predicted']) for f, r in zip(v['folds'], records))


def aggregate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    planned = tasks()
    if {p.name for p in output.iterdir() if p.is_dir()}-{c['id'] for c in planned}:
        raise ValueError('foreign task directories')
    results, errors, failed = {}, [], []
    rows = []
    for c in planned:
        folder = output/c['id']
        if (folder/'error.json').exists():
            error = io.read(folder/'error.json')
            if error['task']!=c or error['context']!=context(c):
                raise ValueError('failure context differs')
            failed.append(dict(id=c['id'], error=error['error']))
        if not (folder/'result.json').exists():
            continue
        try:
            v = check_result(c, folder)
            if io.read(folder/'verified.json')!=verification(c, folder):
                raise ValueError('verification digest differs')
            results[c['id']] = v
            for mode in ('v2', 'research', 'conservative'):
                for pair in range(2):
                    r = v['independent']['v2']['pairs'][pair] if mode=='v2' else v['independent'][mode][pair]
                    rows.append(dict(id=c['id'], scenario=c['scenario'], rho=c['rho'], n=c['n'], p=c['p'],
                        mode=mode, pair=':'.join(PAIRS[pair]), cv_gain=pooled_gain(v, mode, pair),
                        independent_gain=r['gain'], target_oracle_gap=v['oracle_gap'],
                        oracle_gap=v['oracle_gap'] if pair==0 else 0.,
                        oracle_full_mse=v['oracle_full_mse'], oracle_additive_mse=v['oracle_additive_mse'],
                        **diagnostics(np.asarray(r['predicted']), np.asarray(v['independent']['v2']['observed']),
                            np.std(generate(c, c['n'], c['seed'])[1]))))
        except Exception as e:
            results.pop(c['id'], None)
            rows = [r for r in rows if r['id']!=c['id']]
            errors.append(dict(id=c['id'], error=repr(e)))
    retained = dict(comparisons=0, fitting_tuning_fits=0, fitting_seconds=0., warning_count=0)
    partial = retained.copy()
    unknown_cost = []
    for c in planned:
        folder = output/c['id']
        if not folder.exists():
            continue
        if c['id'] not in results:
            unknown_cost.append(c['id'])
        checkpoints = list(folder.glob('comparison-*.json'))
        if not checkpoints:
            continue
        X, y, _, _ = generate(c, c['n'], c['seed'])
        T, ty, _, _ = generate(c, 1000, c['test_seed'])
        combined = pd.concat([X, T], ignore_index=True)
        cy = np.concatenate([y, ty])
        comparisons = [(X, y, a.tolist(), b.tolist()) for a, b in splits(c)]+[
            (combined, cy, list(range(c['n'])), list(range(c['n'], c['n']+1000)))]
        for checkpoint in checkpoints:
            try:
                i = int(checkpoint.stem.split('-')[1])
                if checkpoint.name != f'comparison-{i}.json' or not 0<=i<4:
                    raise ValueError('foreign checkpoint index')
                saved = io.read(checkpoint)
                if (saved['context'] != context(c) or not isinstance(saved['warnings'], list) or
                    not np.isfinite(saved['fitting_seconds']) or saved['fitting_seconds']<0):
                    raise ValueError('checkpoint cost context differs')
                data, outcome, train, test = comparisons[i]
                check_comparison(saved['result'], data, outcome, train, test, c['split_seed'])
                for bucket in ([retained, partial] if c['id'] not in results else [retained]):
                    bucket['comparisons'] += 1
                    bucket['fitting_tuning_fits'] += 223
                    bucket['fitting_seconds'] += saved['fitting_seconds']
                    bucket['warning_count'] += len(saved['warnings'])
            except Exception as e:
                errors.append(dict(id=c['id'], checkpoint=checkpoint.name, error=repr(e)))
    current_failures = [f for f in failed if f['id'] not in results]
    historical_failures = [f for f in failed if f['id'] in results]
    cells, sensitivities = [], []
    for rho in (0., .7, .9):
        for scenario in ('noise', 'additive', 'pure', 'mixed'):
            for n, p in POINTS:
                group = [c for c in planned if (c['rho'], c['scenario'], c['n'], c['p'])==(rho, scenario, n, p)]
                complete = sum(c['id'] in results for c in group)
                cells.append(dict(rho=rho, scenario=scenario, n=n, p=p, planned=5,
                    complete=complete, missing=5-complete, oracle_gap=oracle_gap(group[0])))
                for mode in ('v2', 'research', 'conservative'):
                    for pair in PAIRS:
                        rs = [r for r in rows if (r['rho'], r['scenario'], r['n'], r['p'], r['mode'], r['pair'])==(
                            rho, scenario, n, p, mode, ':'.join(pair))]
                        for cutoff in CUTOFFS:
                            k = sum(r['cv_gain']>cutoff for r in rs)
                            sensitivities.append(dict(rho=rho, scenario=scenario, n=n, p=p, mode=mode,
                                pair=':'.join(pair), cutoff=cutoff, planned=5, assessed=len(rs), above=k,
                                unknown=5-len(rs), lower=k, upper=k+5-len(rs)))
    report = dict(planned=240, complete=len(results), missing=240-len(results),
        missing_ids=[c['id'] for c in planned if c['id'] not in results],
        acceptance='none: development only', audit_errors=errors, failed_tasks=current_failures,
        historical_failed_tasks=historical_failures, cells=cells,
        cost_scope='top-level costs cover verified complete tasks only; checkpoint costs overlap them',
        retained_checkpoint_cost=retained, partial_checkpoint_cost=partial,
        unfinished_attempt_cost_unknown_ids=unknown_cost,
        warning_count=sum(len(v['warnings']) for v in results.values()),
        fitting_tuning_fits=892*len(results), saved_setting_refits=28*len(results),
        fitting_tuning_fits_this_attempt=sum(io.read(output/identity/'timing.json')['fitting_tuning_fits_this_attempt'] for identity in results),
        retained_fitting_seconds=sum(io.read(output/identity/'timing.json')['retained_fitting_seconds'] for identity in results),
        total_seconds=sum(io.read(output/identity/'timing.json')['total_seconds'] for identity in results),
        sources=manifest()['sources'])
    io.write(output/'audit.json', report)
    pd.DataFrame(rows).to_csv(output/'case_metrics.csv', index=False)
    pd.DataFrame(cells).to_csv(output/'summary.csv', index=False)
    pd.DataFrame(sensitivities).to_csv(output/'gain_sensitivity.csv', index=False)
    lines = ['# Conservative joint-model selection development', '',
        f"Complete {len(results)}/240; missing {240-len(results)}; audit errors {len(errors)}.",
        'Production defaults unchanged. No threshold or model adopted. Five planned datasets per cell.',
        'Oracle gap is a Gaussian population reference, not a bound on fitted gains.',
        'Fixed pair comparisons do not establish network recovery or all-pair false-positive rates.', '',
        'Top-level costs cover verified complete tasks. Audit JSON separately retains completed-checkpoint costs,',
        'including partial tasks. These totals overlap: do not add them together. Interrupted work without a',
        'completed checkpoint has unknown cost; saved costs are a lower bound on actual computation.', '',
        '|Correlation|Signal|N|p|Complete / 5|Population gap|', '|---:|---|---:|---:|---:|---:|']
    lines += [f"|{c['rho']}|{c['scenario']}|{c['n']}|{c['p']}|{c['complete']}|{c['oracle_gap']:.4f}|" for c in cells]
    if rows:
        plot(output, pd.DataFrame(rows))
        lines += ['', '![Independent assessment](conservative-summary.png)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def plot(output, rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained', sharey=True)
    for ax, rho in zip(axes, (0., .7, .9)):
        for j, scenario in enumerate(('noise', 'additive', 'pure', 'mixed')):
            for mode, shift, color in (('v2', -.15, '#708090'), ('research', 0., '#2166ac'), ('conservative', .15, '#c78b36')):
                data = rows[(rows.rho==rho)&(rows.scenario==scenario)&(rows['mode']==mode)&(rows.pair=='x1:x2')]
                ax.scatter(data.independent_gain, [j+shift]*len(data), s=15, alpha=.6,
                    color=color, label=mode if j==0 else None)
            c = next(c for c in tasks() if c['rho']==rho and c['scenario']==scenario)
            ax.scatter([oracle_gap(c)], [j], marker='|', s=220, color='black', label='population reference' if j==0 else None)
        ax.axvline(0, color='black', lw=.6)
        ax.set_title(f'Input correlation {rho}')
        ax.set_yticks(range(4), ('noise', 'additive', 'pure', 'mixed'))
        ax.set_xlabel('Independent-test incremental skill')
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed x1:x2 comparisons; each dot is one dataset across N/p conditions\nOne-SE-style heuristic, not a confidence interval; no cutoff accepted')
    fig.savefig(output/'conservative-summary.png', dpi=150)
    plt.close(fig)


def run_shard(identity, output):
    freeze()
    io.write(Path(output)/f'shard-{identity}.json', dict(id=identity, tasks=shard_tasks(identity)))
    errors = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_task, c, output):c for c in shard_tasks(identity)}
        for future in as_completed(futures):
            c = futures[future]
            try:
                future.result()
            except Exception as e:
                io.write(Path(output)/c['id']/'error.json', dict(task=c, context=context(c), error=repr(e)))
                errors.append(c['id'])
    if errors:
        raise RuntimeError(f'failed tasks retained: {errors}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'matrix', 'shard', 'aggregate', 'smoke'))
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--id')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'.superpowers/pair-conservative')
    a = parser.parse_args()
    if a.command=='freeze':
        freeze(not a.write)
        print('frozen correlation context verified')
        return
    freeze()
    if a.command=='matrix':
        print(json.dumps(shards()))
    elif a.command=='shard':
        run_shard(a.id, a.output)
    elif a.command=='smoke':
        for index in (0, 145):
            run_task(tasks()[index], a.output)
        result = aggregate(a.output)
        if result['complete']!=2 or result['audit_errors']:
            raise RuntimeError('construction preflight failed')
        print(json.dumps({k:result[k] for k in ('complete', 'missing', 'warning_count')}))
    else:
        if a.input:
            io.merge_shards([p for p in a.input.iterdir() if p.is_dir()], a.output)
        result = aggregate(a.output)
        print(json.dumps({k:result[k] for k in ('planned', 'complete', 'missing')}))
        if result['missing'] or result['audit_errors']:
            sys.exit(1)


if __name__=='__main__':
    main()
