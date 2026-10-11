"""Frozen development diagnostics and checkpoint-sharded recursive continuation.

Nothing in this module changes production selection or the archived pilot.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings
import zipfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
import sample_size_pilot as pilot
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from open.layer_consensus import aggregate_target_assessments
from open.learners.trees import TreePredictor, GRIDS
from open.preprocessing import make_preprocessor
from open.risk import evaluate_risk
from open.splits import make_split_plan

NAME = '2026-10-10-selection-investigation'
OUTPUT = ROOT/'examples/results'/NAME
PROTOCOL = Path(__file__).with_name('selection_investigation_protocol.md')
MANIFEST = ROOT/'examples/study_manifests/selection-investigation.json'
ARCHIVE = ROOT/'examples/results/2026-10-10-sample-size-pilot/actions-evidence.zip'
ARCHIVE_HASH = 'a7d9c6a27d0318a52f8b51f293ab33212bbc8431eef6ad274dc59add96a5d13a'
SMOKE_ARCHIVE = ROOT/'examples/study_transfers/selection-investigation-smoke.zip'
SMOKE_HASH = '45cd4e528d9bd8fbb46b423a57833a1de629e79e603f96a7674a42ba3cb43406'
CHAIN = 'chain-n300-p15'


def diagnostic_tasks():
    tasks = []
    for spec in pilot.specs():
        if spec['scenario'] == 'interaction':
            variants = [(engine, inputs) for engine in ('gbm_shipped', 'gbm_deeper', 'rf_shipped')
                        for inputs in ('all', 'signal')]+[('linear', 'product')]
        elif spec['id'] == 'chain-n80-p5':
            variants = [('gbm_shipped', x) for x in ('all', 'signal', 'signal_noise', 'noise', 'without_noise')]
            variants += [('rf_shipped', x) for x in ('all', 'signal_noise')]
            variants += [('linear', x) for x in ('all', 'signal', 'signal_noise', 'noise')]
        else:
            continue
        for scope in ('full', 'sample0'):
            for variant, inputs in variants:
                tasks.append(dict(id=f"{spec['id']}-{scope}-{variant}-{inputs}", kind='model',
                    case=spec['id'], scope=scope, variant=variant, inputs=inputs))
    tasks += [dict(id=f'chain-n80-p5-loco-{i:03d}', kind='loco', case='chain-n80-p5', run=i)
              for i in range(10)]
    return tasks


def assessment_tasks(comparisons):
    tasks = []
    for c in comparisons:
        missing = c['missing']
        if missing != sorted(set(missing)) or any(type(i) is not int or not 0 <= i < 100 for i in missing):
            raise ValueError('invalid assessment identities')
        for start in range(0, 100, 10):
            indices = [i for i in missing if start <= i < start+10]
            if indices:
                tasks.append(dict(id=f"d{c['depth']}-{c['target']}-{start:03d}",
                    depth=c['depth'], target=c['target'], candidates=c['candidates'], indices=indices))
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('overlapping comparison identities')
    return tasks


def make_manifest():
    return dict(schema='open.selection_investigation.v1', tasks=diagnostic_tasks(),
        sources={**pilot.sources(), **{name:pilot.text_hash((ROOT/name).read_bytes()) for name in
            ('examples/selection_investigation.py','.github/workflows/selection-investigation.yml',
             '.github/actions/investigation-setup/action.yml')}},
        protocol_hash=pilot.text_hash(PROTOCOL.read_bytes()), archive_sha256=ARCHIVE_HASH,
        smoke_archive_sha256=SMOKE_HASH,
        continuation=CHAIN, assessment_bucket=10, workers=4, max_parallel=256,
        hard_job_minutes=360, test_n=2000, automatic_reruns=False)


def validate_manifest(value):
    if value != make_manifest():
        raise ValueError('investigation manifest differs from frozen tasks/settings/sources/protocol')


def merge_files(inputs, destination):
    destination = Path(destination)
    for folder in inputs:
        folder = Path(folder)
        for source in sorted(folder.rglob('*')):
            if not source.is_file():
                continue
            target = destination/source.relative_to(folder)
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise ValueError(f'conflicting duplicate artifact: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


def case_spec(case):
    return next(s for s in pilot.specs() if s['id'] == case)


def case_data(case):
    spec = case_spec(case)
    train, _ = pilot.generate(spec['scenario'], spec['n'], spec['p'], spec['seed'])
    test, oracle = pilot.generate(spec['scenario'], 2000, spec['p'], spec['test_seed'])
    return spec, pilot.settings(spec), train, test, oracle


def pending_comparisons(data, config, directory):
    """Derive the next layer only from complete, context-validated production evidence."""
    plans = pilot.engine.make_layer_sample_plans(data, config)
    depths, frontier = {config.analysis.root:0}, (config.analysis.root,)
    while frontier:
        depth = depths[frontier[0]]
        candidates = tuple(n for n in data if n not in depths)
        if depth >= config.analysis.max_depth or not candidates:
            return []
        pending, evidence = [], []
        for target in sorted(frontier):
            context = pilot.checkpoint_context(data, depth, target, candidates, config)
            folder = Path(directory)/f'd{depth}-{target}'
            expected = {f'{p.run_index:03d}.json.gz' for p in plans}
            if any(p.name not in expected for p in folder.glob('*.json.gz')):
                raise ValueError('foreign assessment identity')
            missing, records = [], []
            for plan in plans:
                path = folder/f'{plan.run_index:03d}.json.gz'
                if not path.exists():
                    missing.append(plan.run_index)
                else:
                    record = pilot.load_checkpoint(path, plan, context)
                    if (record.depth, record.target, record.candidates) != (depth, target, candidates):
                        raise ValueError('assessment comparison identity differs')
                    if record.status != 'completed':
                        raise ValueError('failed assessment retained; layer cannot be finalized')
                    records.append(record)
            if missing:
                pending.append(dict(depth=depth, target=target, candidates=list(candidates), missing=missing))
            else:
                evidence.append(aggregate_target_assessments(tuple(records), planned_count=len(plans),
                    pi_min=config.thresholds.stability))
        if pending:
            return pending
        if not all(t.complete for t in evidence):
            raise ValueError('incomplete layer cannot advance')
        promoted = {source for t in evidence for r in t.relations if r.qualifies
                    for source in r.members or (r.source,)}
        frontier = tuple(n for n in data if n in promoted)
        depths.update((n, depth+1) for n in frontier)
    return []


def extract_original(output):
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_HASH:
        raise ValueError('archived pilot checksum differs')
    destination = Path(output)/'original'
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe archive path')
            if member.is_dir():
                continue
            target = destination/relative
            value = archive.read(member)
            if target.exists() and target.read_bytes() != value:
                raise ValueError('original evidence changed')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
    pilot.validate_manifest(json.loads((destination/'manifest.json').read_text()))
    return destination


def prepare(output):
    original = extract_original(output)
    case = Path(output)/'continuation'/CHAIN
    # Preserve the original result/stop separately; continuation gets checkpoints only.
    merge_files([original/CHAIN/'checkpoints'], case/'checkpoints')
    if hashlib.sha256(SMOKE_ARCHIVE.read_bytes()).hexdigest() != SMOKE_HASH:
        raise ValueError('smoke checkpoint archive checksum differs')
    with zipfile.ZipFile(SMOKE_ARCHIVE) as archive:
        receipt=json.loads(archive.read('receipt.json'))
        observed_manifest=json.loads(archive.read('source-manifest.json'))
        if receipt['manifest'] != pilot.fingerprint(observed_manifest):
            raise ValueError('smoke receipt differs from recorded manifest')
        if any(observed_manifest['sources'].get(k)!=v for k,v in pilot.sources().items()):
            raise ValueError('smoke production sources differ')
        _,config,train,_,_=case_data(CHAIN)
        # Transfer is fixed to the four previously missing original run identities.
        task=receipt['task']
        fixed=dict(id='d1-x1-010',depth=1,target='x1',
            candidates=[n for n in train if n not in ('y','x1')],indices=[16,17,18,19])
        if task!=fixed:
            raise ValueError('smoke comparison identity differs')
        names={f"checkpoints/d1-x1/{i:03d}.json.gz" for i in task['indices']}
        names|={f"checkpoints/d1-x1/{i:03d}.json.metrics.json" for i in task['indices']}
        if set(archive.namelist())!=names|{'receipt.json','source-manifest.json'}:
            raise ValueError('foreign smoke checkpoint files')
        for name in sorted(names):
            target=case/name
            value=archive.read(name)
            if target.exists() and target.read_bytes()!=value:
                raise ValueError('conflicting duplicate smoke checkpoint')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(value)
    return stage(output)


def stage(output):
    _, config, train, _, _ = case_data(CHAIN)
    case = Path(output)/'continuation'/CHAIN
    pilot.audit_checkpoints(train, config, case)
    tasks = assessment_tasks(pending_comparisons(train, config, case/'checkpoints'))
    pilot.write_json(Path(output)/'pending.json', tasks)
    return tasks


def run_assessments(task, output):
    # Matrix commands may select only a task derived from the entire current frontier.
    planned = stage(output)
    if task not in planned:
        raise ValueError('unplanned or overlapping assessment shard')
    _, config, train, _, _ = case_data(CHAIN)
    plans = pilot.engine.make_layer_sample_plans(train, config)
    context = pilot.checkpoint_context(train, task['depth'], task['target'], tuple(task['candidates']), config)
    shard = Path(output)/'shard'
    folder = shard/'checkpoints'/f"d{task['depth']}-{task['target']}"
    # Workers write directly to the uploaded directory, including on interruption.
    pilot.write_json(shard/'receipt.json', dict(task=task, manifest=pilot.fingerprint(make_manifest())))
    paths = [folder/f'{i:03d}.json.gz' for i in task['indices']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(pilot.one_assessment, train, task['depth'], task['target'],
            tuple(task['candidates']), plans[i], config, path, context) for i, path in zip(task['indices'], paths)]
        for future in as_completed(futures):
            future.result()


def merge_shards(output, downloads):
    output = Path(output)
    tasks = json.loads((output/'pending.json').read_text())
    expected = {t['id']:t for t in tasks}
    seen = set()
    partial = []
    for folder in sorted(Path(downloads).glob('completion-*')):
        receipt = json.loads((folder/'receipt.json').read_text())
        task = receipt['task']
        if task['id'] in seen or expected.get(task['id']) != task:
            raise ValueError('duplicate/foreign assessment shard identity')
        if receipt['manifest'] != pilot.fingerprint(make_manifest()):
            raise ValueError('shard source/settings differ')
        actual = {p.relative_to(folder/'checkpoints').as_posix() for p in (folder/'checkpoints').rglob('*') if p.is_file()}
        prefix = f"d{task['depth']}-{task['target']}"
        required = {f'{prefix}/{i:03d}.json.gz' for i in task['indices']}
        required |= {f'{prefix}/{i:03d}.json.metrics.json' for i in task['indices']}
        if not actual <= required:
            raise ValueError('foreign assessment files')
        merge_files([folder/'checkpoints'], output/'continuation'/CHAIN/'checkpoints')
        if actual != required:
            partial.append(task['id'])
        seen.add(task['id'])
    if seen != set(expected) or partial:
        raise ValueError('missing assessment shard/files; partial checkpoints preserved; layer remains incomplete')
    return stage(output)


class SavedAssessor:
    """No fits, no soft budget: actual recursive engine reads complete saved comparisons."""
    def __init__(self, directory, deadline, workers):
        self.directory = Path(directory)

    def __call__(self, data, *, depth, target, candidates, plans, config):
        context = pilot.checkpoint_context(data, depth, target, candidates, config)
        return tuple(pilot.load_checkpoint(self.directory/f'd{depth}-{target}'/
            f'{p.run_index:03d}.json.gz', p, context) for p in plans)


def finalize(output):
    if stage(output):
        raise ValueError('unfinished comparisons; no final graph may be reported')
    native = pilot.CheckpointAssessor
    pilot.CheckpointAssessor = SavedAssessor
    try:
        result = pilot.run_case(case_spec(CHAIN), Path(output)/'continuation', seconds=21600)
    finally:
        pilot.CheckpointAssessor = native
    audited = pilot.audit_case(case_spec(CHAIN), Path(output)/'continuation'/CHAIN)
    pilot.write_json(Path(output)/'continuation-audit.json', audited)
    if result['status'] != 'complete':
        raise ValueError('continuation did not complete; evidence retained')


def features(frame, inputs):
    if inputs == 'product':
        return pd.DataFrame({'x1_times_x2':frame.x1*frame.x2}, index=frame.index)
    names = [n for n in frame if n != 'y']
    choices = dict(all=names, signal=['x1'],
        signal_noise=['x1','x4'], noise=['x4'], without_noise=[n for n in names if n != 'x4'])
    return frame[choices[inputs]].copy()


def model_data(task):
    spec, config, train, test, oracle = case_data(task['case'])
    positions = tuple(range(len(train)))
    split = make_split_plan(train, config)
    if task['scope'] == 'sample0':
        plan = pilot.engine.make_layer_sample_plans(train, config)[0]
        positions, split = plan.sampled_positions, plan.split_plan
        train = train.iloc[list(positions)].reset_index(drop=True)
    if task['inputs'] == 'signal' and spec['scenario'] == 'interaction':
        X, T = train[['x1','x2']], test[['x1','x2']]
    else:
        X, T = features(train, task['inputs']), features(test, task['inputs'])
    return config, train, test, oracle, X, T, split, positions


def model_grid(variant):
    if variant == 'linear':
        return {}
    if variant == 'gbm_deeper':
        return dict(max_depth=[2,3,5], learning_rate=[.05,.1])
    return GRIDS['random_forest' if variant == 'rf_shipped' else 'gbm']


def model_pipeline(variant, config, params):
    if variant == 'linear':
        return Pipeline([('preprocessor', make_preprocessor(config.preprocessing)),
                         ('regressor', LinearRegression())])
    return TreePredictor('random_forest' if variant == 'rf_shipped' else 'gbm',
        config.preprocessing, config.analysis.random_seed)._new_pipeline(params)


def tuned_model(variant, config, X, y):
    grid = model_grid(variant)
    initial = {k:v[0] for k,v in grid.items()}
    model = model_pipeline(variant, config, initial)
    inner = tuple(KFold(3, shuffle=True, random_state=config.analysis.random_seed).split(X))
    search = GridSearchCV(model, {'regressor__'+k:v for k,v in grid.items()},
        cv=inner, scoring='neg_mean_squared_error', refit=False, n_jobs=1, error_score='raise')
    search.fit(X, y)
    params = {k.removeprefix('regressor__'):v for k,v in search.best_params_.items()}
    fitted = model_pipeline(variant, config, params).fit(X, y)
    return fitted, params, float(-search.best_score_), [len(a) for a,b in inner]


def fold_fit(task, config, X, y, split):
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model, params, inner_mse, sizes = tuned_model(task['variant'], config,
            X.iloc[list(split.train_indices)], y.iloc[list(split.train_indices)])
        pred = model.predict(X.iloc[list(split.test_indices)])
    observed = y.iloc[list(split.test_indices)].to_numpy()
    baseline = float(y.iloc[list(split.train_indices)].mean())
    return dict(repeat=split.repeat, fold=split.fold, train=list(split.train_indices),
        test=list(split.test_indices), parameters=params, inner_mse=inner_mse,
        inner_training_n=sizes, observed=observed.tolist(), predicted=pred.tolist(),
        baseline=[baseline]*len(pred), mse=float(np.square(observed-pred).mean()),
        elapsed_seconds=time.monotonic()-started, peak_bytes=pilot.process_peak_bytes(),
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured])


def skill(observed, predicted, baseline):
    mse = float(np.square(np.asarray(observed)-predicted).mean())
    risk = float(np.square(np.asarray(observed)-baseline).mean())
    return dict(mse=mse, baseline_mse=risk, skill=float(1-mse/risk) if risk else None)


def task_context(task):
    return pilot.fingerprint(dict(task=task, manifest=make_manifest()))


def run_model(task, output):
    folder = Path(output)/'diagnostics'/task['id']
    folder.mkdir(parents=True, exist_ok=True)
    config, train, test, oracle, X, T, plan, positions = model_data(task)
    context = task_context(task)
    pilot.write_json(folder/'plan.json', dict(context=context, task=task, positions=list(positions),
        splits=json.loads(plan.to_json()), features=list(X), config=pilot.portable(config)))
    folds = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {}
        for split in plan.splits:
            path = folder/f'fold-{split.repeat}-{split.fold}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['context'] != context:
                    raise ValueError('fold checkpoint source/context differs')
                folds.append(saved['result'])
            else:
                futures[pool.submit(fold_fit, task, config, X, train.y, split)] = path
        for future in as_completed(futures):
            value = future.result()
            pilot.write_json(futures[future], dict(context=context, result=value))
            folds.append(value)
    folds.sort(key=lambda f:(f['repeat'],f['fold']))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model, params, inner_mse, sizes = tuned_model(task['variant'], config, X, train.y)
        pred = model.predict(T)
    predictions = pd.DataFrame(dict(observed=test.y, predicted=pred,
        baseline=np.full(len(test), train.y.mean()), oracle=oracle))
    predictions.to_csv(folder/'predictions.csv', index=False)
    observed = np.concatenate([f['observed'] for f in folds])
    predicted = np.concatenate([f['predicted'] for f in folds])
    baseline = np.concatenate([f['baseline'] for f in folds])
    result = dict(task=task, context=context, status='complete', folds=folds,
        discovery_n=len(train), final_parameters=params, final_inner_mse=inner_mse,
        final_inner_training_n=sizes, cv=skill(observed,predicted,baseline),
        test=skill(test.y,pred,predictions.baseline), oracle=skill(test.y,oracle,predictions.baseline),
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])
    pilot.write_json(folder/'result.json', result)
    return result


def run_loco(task, output):
    original = extract_original(output)
    _, config, train, _, _ = case_data(task['case'])
    plan = pilot.engine.make_layer_sample_plans(train, config)[task['run']]
    config = replace(config, analysis=replace(config.analysis, random_seed=plan.seed))
    candidates = tuple(n for n in train if n != 'y')
    original_config = pilot.settings(case_spec(task['case']))
    context = pilot.checkpoint_context(train, 0, 'y', candidates, original_config)
    saved = pilot.load_checkpoint(original/task['case']/'checkpoints/d0-y'/
        f"{task['run']:03d}.json.gz", plan, context)
    sampled = train.iloc[list(plan.sampled_positions)]
    common = dict(data=sampled, target='y', candidate_set=candidates, split_plan=plan.split_plan,
                  tuning_folds=3)
    predictor = pilot.make_predictor(config)
    risks = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        risks['baseline'] = evaluate_risk(**common, predictors=())
        risks['full'] = evaluate_risk(**common, predictors=candidates, predictor=predictor)
        params = {(f.repeat_index,f.fold_index):f.fitted_parameters for f in risks['full'].folds}
        ablated = tuple(n for n in candidates if n != 'x4')
        risks['fixed_ablation'] = evaluate_risk(**common, predictors=ablated, predictor=predictor,
                                               fitted_parameters=params)
        risks['retuned_ablation'] = evaluate_risk(**common, predictors=ablated, predictor=predictor)
        risks['standalone'] = evaluate_risk(**common, predictors=('x4',), predictor=predictor)
    values = {k:v.aggregate_risk for k,v in risks.items()}
    rel = saved.local.relevance
    np.testing.assert_allclose([values['baseline'],values['full'],
        values['fixed_ablation']-values['full'],values['baseline']-values['standalone']],
        [rel.baseline_risk,rel.full_risk,rel.unique['x4'].raw,rel.broad['x4'].raw], rtol=1e-12, atol=1e-12)
    gain = values['baseline']-values['full']
    contrasts = {}
    for name in ('fixed_ablation','retuned_ablation'):
        raw = values[name]-values['full']
        contrasts[name] = dict(raw=raw, baseline_relative=raw/values['baseline'],
            gain_relative=raw/gain if gain > 0 else None)
    result = dict(task=task, context=task_context(task), status='complete',
        sample=pilot.portable(plan), risks={k:pilot.portable(v) for k,v in risks.items()},
        values=values, contrasts=contrasts, original_class=pilot.portable(saved.local.classes['x4']),
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured])
    pilot.write_json(Path(output)/'diagnostics'/task['id']/'result.json', result)
    return result


def run_task(task, output):
    started = time.monotonic()
    folder = Path(output)/'diagnostics'/task['id']
    if (folder/'result.json').exists():
        previous = json.loads((folder/'result.json').read_text())
        if previous['context'] != task_context(task):
            raise ValueError('task context differs')
        if previous['status'] != 'complete':
            raise ValueError('failed result retained; no automatic statistical rerun')
        audit_task(task, output)
        return
    try:
        result = run_model(task,output) if task['kind']=='model' else run_loco(task,output)
        result['elapsed_seconds'] = time.monotonic()-started
        pilot.write_json(folder/'result.json', result)
    except Exception as exc:
        pilot.write_json(folder/'result.json', dict(task=task,context=task_context(task),status='failed',
            reason=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-started))
        raise


def audit_task(task, output):
    folder = Path(output)/'diagnostics'/task['id']
    result = json.loads((folder/'result.json').read_text())
    if result['task'] != task or result['context'] != task_context(task):
        raise ValueError('diagnostic identity/source differs')
    if result['status'] != 'complete':
        return result
    if task['kind'] == 'loco':
        # Strictly reconstruct all predictions, original plans and fixed contrasts.
        _, cfg, train, _, _ = case_data(task['case'])
        plan = pilot.engine.make_layer_sample_plans(train,cfg)[task['run']]
        if result['sample'] != pilot.portable(plan):
            raise ValueError('LOCO sample/splits differ')
        sampled = train.iloc[list(plan.sampled_positions)]
        config = replace(cfg,analysis=replace(cfg.analysis,random_seed=plan.seed))
        full_params = None
        candidates = tuple(n for n in train if n != 'y')
        for name in ('baseline','full','fixed_ablation','retuned_ablation','standalone'):
            risk = result['risks'][name]
            inputs = (() if name=='baseline' else ('x4',) if name=='standalone' else
                tuple(n for n in candidates if n!='x4') if 'ablation' in name else candidates)
            losses = []
            if len(risk['folds']) != len(plan.split_plan.splits) or risk['predictors'] != list(inputs):
                raise ValueError('LOCO folds/predictors differ')
            for f,s in zip(risk['folds'],plan.split_plan.splits):
                if (f['repeat_index'],f['fold_index']) != (s.repeat,s.fold):
                    raise ValueError('LOCO fold identity differs')
                params = f['fitted_parameters']
                if inputs:
                    if params not in list(ParameterGrid(GRIDS['gbm'])):
                        raise ValueError('LOCO parameters outside original grid')
                    pred = pilot.make_predictor(config).fit(sampled.iloc[list(s.train_indices)][list(inputs)],
                        sampled.y.iloc[list(s.train_indices)],params).predict(sampled.iloc[list(s.test_indices)][list(inputs)])
                else:
                    pred = np.full(len(s.test_indices),sampled.y.iloc[list(s.train_indices)].mean())
                obs = f['observations']
                if [o['row_position'] for o in obs] != list(s.test_indices):
                    raise ValueError('LOCO held-out row identities differ')
                np.testing.assert_allclose([o['predicted'] for o in obs],pred,rtol=1e-12,atol=1e-12)
                observed = sampled.y.iloc[list(s.test_indices)].to_numpy()
                np.testing.assert_array_equal([o['observed'] for o in obs],observed)
                loss = np.square(observed-pred)
                np.testing.assert_allclose([o['loss'] for o in obs],loss,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(f['risk'],loss.mean(),rtol=1e-12,atol=1e-12)
                losses.extend([o['loss'] for o in obs])
            np.testing.assert_allclose([risk['aggregate_risk'],result['values'][name]],np.mean(losses),rtol=1e-12,atol=1e-12)
            if name=='full':
                full_params = [f['fitted_parameters'] for f in risk['folds']]
            if name=='fixed_ablation' and [f['fitted_parameters'] for f in risk['folds']] != full_params:
                raise ValueError('fixed ablation did not use full-model settings')
        v = result['values']
        for name in ('fixed_ablation','retuned_ablation'):
            raw=v[name]-v['full']; gain=v['baseline']-v['full']
            expected=dict(raw=raw,baseline_relative=raw/v['baseline'],gain_relative=raw/gain if gain>0 else None)
            if result['contrasts'][name] != expected:
                raise ValueError('LOCO contrast arithmetic differs')
        original=extract_original(output)
        saved=pilot.load_checkpoint(original/task['case']/'checkpoints/d0-y'/f"{task['run']:03d}.json.gz",
            plan,pilot.checkpoint_context(train,0,'y',candidates,cfg))
        rel=saved.local.relevance
        np.testing.assert_allclose([v['baseline'],v['full'],v['fixed_ablation']-v['full'],v['baseline']-v['standalone']],
            [rel.baseline_risk,rel.full_risk,rel.unique['x4'].raw,rel.broad['x4'].raw],rtol=1e-12,atol=1e-12)
        if result['original_class'] != pilot.portable(saved.local.classes['x4']):
            raise ValueError('original classification differs')
        return result
    config,train,test,oracle,X,T,plan,positions = model_data(task)
    expected_plan = dict(context=task_context(task),task=task,positions=list(positions),
        splits=json.loads(plan.to_json()),features=list(X),config=pilot.portable(config))
    if json.loads((folder/'plan.json').read_text()) != expected_plan:
        raise ValueError('diagnostic data/splits/config differ')
    allowed = list(ParameterGrid(model_grid(task['variant'])))
    if len(result['folds']) != len(plan.splits) or result['discovery_n'] != len(train):
        raise ValueError('diagnostic folds/size differ')
    for f,s in zip(result['folds'],plan.splits):
        if (f['repeat'],f['fold'],f['train'],f['test']) != (s.repeat,s.fold,list(s.train_indices),list(s.test_indices)):
            raise ValueError('diagnostic fold identity differs')
        if f['parameters'] not in allowed:
            raise ValueError('diagnostic parameters outside declared grid')
        inner = tuple(KFold(3,shuffle=True,random_state=config.analysis.random_seed).split(s.train_indices))
        if f['inner_training_n'] != [len(a) for a,b in inner]:
            raise ValueError('diagnostic tuning sizes differ')
        model=model_pipeline(task['variant'],config,f['parameters']).fit(X.iloc[f['train']],train.y.iloc[f['train']])
        pred=model.predict(X.iloc[f['test']])
        np.testing.assert_allclose(f['predicted'],pred,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(f['observed'],train.y.iloc[f['test']])
        np.testing.assert_array_equal(f['baseline'],np.full(len(f['test']),train.y.iloc[f['train']].mean()))
        np.testing.assert_allclose(f['mse'],np.square(np.asarray(f['observed'])-pred).mean(),rtol=1e-12,atol=1e-12)
    expected_cv = skill(np.concatenate([f['observed'] for f in result['folds']]),
        np.concatenate([f['predicted'] for f in result['folds']]),np.concatenate([f['baseline'] for f in result['folds']]))
    if result['cv'] != expected_cv or result['final_parameters'] not in allowed:
        raise ValueError('diagnostic CV arithmetic/settings differ')
    inner = tuple(KFold(3,shuffle=True,random_state=config.analysis.random_seed).split(X))
    if result['final_inner_training_n'] != [len(a) for a,b in inner]:
        raise ValueError('final tuning sizes differ')
    predictions=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
    if len(predictions)!=2000 or not np.isfinite(predictions.to_numpy()).all():
        raise ValueError('diagnostic test coverage differs')
    model=model_pipeline(task['variant'],config,result['final_parameters']).fit(X,train.y)
    np.testing.assert_allclose(predictions.predicted,model.predict(T),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(predictions.observed,test.y)
    np.testing.assert_array_equal(predictions.oracle,oracle)
    np.testing.assert_array_equal(predictions.baseline,np.full(len(test),train.y.mean()))
    if result['test'] != skill(test.y,predictions.predicted,predictions.baseline) or result['oracle'] != skill(test.y,oracle,predictions.baseline):
        raise ValueError('diagnostic test arithmetic differs')
    return result


def audit(output):
    rows = []
    for task in diagnostic_tasks():
        try:
            rows.append(audit_task(task,output))
        except FileNotFoundError as exc:
            rows.append(dict(task=task,status='missing',reason=str(exc)))
        except Exception as exc:
            rows.append(dict(task=task,status='audit_failed',reason=type(exc).__name__+': '+str(exc)))
    try:
        continuation=pilot.audit_case(case_spec(CHAIN),Path(output)/'continuation'/CHAIN)
    except Exception as exc:
        continuation=dict(status='unavailable',reason=type(exc).__name__+': '+str(exc))
    complete=sum(r['status']=='complete' for r in rows)
    report=dict(planned=74,complete=complete,incomplete=74-complete,
        audit_errors=sum(r['status']=='audit_failed' for r in rows),diagnostics=rows,
        continuation=continuation,historical_pilot_complete=11,historical_pilot_planned=12)
    pilot.write_json(Path(output)/'audit.json',report)
    lines=['# Selection investigation', '',f'Diagnostics completed/audited: {complete}/74. Continuation: {continuation["status"]}.',
        '', 'Development diagnosis on previously inspected datasets; no minimum-N guidance or policy validation.',
        'Original pilot remains historically 11/12 complete. Continuation is reported separately.', '',
        '| Case | Scope | Model | Inputs | CV skill | Test skill |', '|---|---|---|---|---:|---:|']
    for r in rows:
        t=r['task']
        if t['kind']=='model':
            lines.append(f"| {t['case']} | {t['scope']} | {t['variant']} | {t['inputs']} | {r.get('cv',{}).get('skill','unavailable')} | {r.get('test',{}).get('skill','unavailable')} |")
    lines += ['', '| LOCO sample | Fixed-setting x4 contribution / baseline | Retuned contribution / baseline |', '|---|---:|---:|']
    for r in rows:
        if r['task']['kind']=='loco':
            c=r.get('contrasts',{})
            lines.append(f"| {r['task']['run']} | {c.get('fixed_ablation',{}).get('baseline_relative','unavailable')} | {c.get('retuned_ablation',{}).get('baseline_relative','unavailable')} |")
    lines += ['', 'Missing/failed tasks keep their planned identities. Negative skill means worse than the training-mean baseline.',
        'CV tests training-only tuning; the same fixed development test rows compare refitted models.',
        'Predictive improvements do not certify a link or justify changing thresholds. LOCO samples overlap.',
        '', 'Continuation structural results: '+json.dumps(continuation.get('structural')),
        'Continuation prediction results: '+json.dumps(continuation.get('scores'))]
    (Path(output)/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(planned=74,complete=complete,incomplete=74-complete,
        continuation=continuation['status'],audit_errors=report['audit_errors'])))
    return complete==74 and report['audit_errors']==0 and continuation['status']=='complete'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('freeze','prepare','matrix','stage','assessment','merge','finalize','task','audit'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--id')
    parser.add_argument('--downloads',type=Path)
    args=parser.parse_args()
    if args.command=='freeze':
        if MANIFEST.exists():
            validate_manifest(json.loads(MANIFEST.read_text()))
        else:
            pilot.write_json(MANIFEST,make_manifest())
        return
    validate_manifest(json.loads(MANIFEST.read_text()))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':
        print(json.dumps(prepare(args.output),separators=(',',':')))
    elif args.command=='matrix':
        print(json.dumps([t['id'] for t in diagnostic_tasks()],separators=(',',':')))
    elif args.command=='stage':
        print(json.dumps(stage(args.output),separators=(',',':')))
    elif args.command=='assessment':
        task=next(t for t in json.loads((args.output/'pending.json').read_text()) if t['id']==args.id)
        run_assessments(task,args.output)
    elif args.command=='merge':
        print(json.dumps(merge_shards(args.output,args.downloads),separators=(',',':')))
    elif args.command=='finalize':
        finalize(args.output)
    elif args.command=='task':
        task=next(t for t in diagnostic_tasks() if t['id']==args.id)
        run_task(task,args.output)
    elif args.command=='audit':
        if args.downloads:
            for folder in sorted(args.downloads.glob('diagnostic-*')):
                merge_files([folder],args.output/'diagnostics'/folder.name.removeprefix('diagnostic-'))
        if not audit(args.output):
            raise SystemExit(1)


if __name__=='__main__':
    main()
