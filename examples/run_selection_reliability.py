"""Fixed development study of production root selection with independent tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
import traceback
import warnings

# Each case is parallel; each estimator and numerical runtime remains single-threaded.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from open.config import AnalysisConfig, LearnerConfig, OpenConfig, StabilityConfig
from open.layer_artifacts import decode, portable
from open.layer_types import LocalAssessment
from open.layer_consensus import aggregate_target_assessments, selected_joint_relations
from open.layer_stability import assess_layer_target, make_layer_sample_plans
from open.learners import make_predictor

SCENARIOS = ('noise', 'ordinary', 'redundant', 'interaction')
SIZES = (80, 180, 400)


def save_checkpoint(path, assessment, context):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
            compressed.write(json.dumps(dict(context=context, assessment=portable(assessment)),
                separators=(',', ':'), allow_nan=False).encode())
    temporary.replace(path)


def load_checkpoint(path, sample, context):
    payload = json.loads(gzip.decompress(path.read_bytes()))
    if payload['context'] != context:
        raise ValueError('checkpoint context differs from current data/settings/source')
    assessment = decode(LocalAssessment, payload['assessment'])
    if assessment.sample != sample:
        raise ValueError('checkpoint sample differs from planned assessment')
    return assessment


def generate(scenario, n, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, n))
    if scenario == 'noise':
        y = rng.normal(size=n)
    else:
        if scenario == 'redundant':
            b = a+rng.normal(scale=.03, size=n)
        error = rng.normal(scale=.3, size=n)
        y = 2*a*b+error if scenario == 'interaction' else 2*a+error
    return pd.DataFrame(dict(y=y, a=a, b=b))


def fingerprint(value):
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def config():
    return OpenConfig(AnalysisConfig('y', 613, 1),
        learner=LearnerConfig(library=('gbm',)),
        stability=StabilityConfig(scope='layer_consensus'))


def fit_predictions(train, test, inputs, settings):
    if not inputs:
        return np.full(len(test), train.y.mean()), None
    model = make_predictor(settings)
    inner = KFold(settings.tuning_cv.folds, shuffle=True,
                  random_state=settings.analysis.random_seed)
    params = model.tune(train[list(inputs)], train.y, inner)
    model.fit(train[list(inputs)], train.y, params)
    prediction = np.asarray(model.predict(test[list(inputs)]))
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError('invalid independent-test predictions')
    return prediction, params


def scores(predictions):
    y = predictions.observed.to_numpy()
    baseline_sse = np.square(y-predictions.baseline).sum()
    result = {}
    for name in ('baseline', 'full', 'retained'):
        mse = float(np.square(y-predictions[name]).mean())
        result[name] = dict(mse=mse, rmse=float(np.sqrt(mse)),
            skill=float(1-mse*len(y)/baseline_sse) if baseline_sse > 0 else None)
    return result


def run_case(spec, output, settings, resume=False):
    started = time.perf_counter()
    case = Path(output)/spec['id']
    case.mkdir(exist_ok=resume)
    result = dict(spec=spec, config=portable(asdict(settings)), status='failed',
                  records=[], warnings=[], scores=None, retained=None)
    try:
        train = generate(spec['scenario'], spec['n'], spec['seed'])
        test = generate(spec['scenario'], 1000, spec['seed']+1000000)
        context = fingerprint(dict(spec=spec, config=portable(settings), train=train.to_dict('list'),
            sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT/'src/open').rglob('*.py'))}))
        result['checkpoint_context'] = context
        if resume and (case/'evidence.json.gz').exists():
            previous = json.loads(gzip.decompress((case/'evidence.json.gz').read_bytes()))
            if previous.get('checkpoint_context') != context:
                raise ValueError('completed case context differs; refusing reuse')
            return dict(id=spec['id'], status=previous['status'], retained=previous['retained'],
                scores=previous['scores'], elapsed_seconds=previous['elapsed_seconds'], reused=True,
                failure=previous.get('failure_message'))
        train.to_csv(case/'train.csv', index=False)
        test.to_csv(case/'test.csv', index=False)
        plans = make_layer_sample_plans(train, settings)
        assessments = []
        checkpoint_dir = ROOT/'.superpowers/selection-checkpoints'/Path(output).name/spec['id']
        for plan in plans:
            # Singleton calls permit progress reporting. Run IDs are bookkeeping;
            # retain the original seed, sample, split and restored ID for consensus.
            checkpoint = checkpoint_dir/f'{plan.run_index:03d}.json.gz'
            if resume and checkpoint.exists():
                assessment = load_checkpoint(checkpoint, plan, context)
            else:
                assessment = assess_layer_target(train, depth=0, target='y', candidates=('a', 'b'),
                    plans=(replace(plan, run_index=0),), config=settings)[0]
                assessment = replace(assessment, sample=plan)
                save_checkpoint(checkpoint, assessment, context)
            assessments.append(assessment)
            if len(assessments) % 10 == 0:
                (case/'progress.json').write_text(json.dumps(dict(completed=len(assessments),
                    planned=len(plans), low_gain=sum(a.local is not None and a.local.stop for a in assessments),
                    failures=sum(a.status != 'completed' for a in assessments),
                    elapsed_seconds=time.perf_counter()-started))+'\n')
        assessments = tuple(assessments)
        if spec['id'].startswith('smoke'):
            together = assess_layer_target(train, depth=0, target='y', candidates=('a', 'b'),
                plans=plans, config=settings)
            if fingerprint(assessments) != fingerprint(together):
                raise AssertionError('singleton assessment orchestration changed production evidence')
        for a in assessments:
            plan = a.sample
            item = dict(run=plan.run_index, seed=plan.seed, status=a.status,
                failure_type=a.failure_type, failure_message=a.failure_message,
                warnings=portable(a.warnings), sample_n=len(plan.sampled_positions),
                sample_hash=fingerprint(plan.sampled_positions),
                split_hash=fingerprint(plan.split_plan))
            if a.local is not None:
                local = a.local
                inner = [len(training) for s in plan.split_plan.splits for training, _ in
                    KFold(settings.tuning_cv.folds, shuffle=True, random_state=plan.seed).split(s.train_indices)]
                item.update(stop=local.reason, relevance=portable(local.relevance),
                    classes=portable(local.classes), promoted=sorted(local.promoted),
                    joint={','.join(members):dict(raw=raw, normalized=normalized)
                           for members, (raw, normalized, _) in selected_joint_relations(local).items()},
                    fit_n=[len(s.train_indices) for s in plan.split_plan.splits],
                    tuning_n=inner, outer_fits=local.outer_model_fit_count,
                    tuning_fits=local.tuning_model_fit_count)
            result['records'].append(item)
        evidence = aggregate_target_assessments(assessments,
            planned_count=settings.stability.n_resamples, pi_min=settings.thresholds.stability)
        result['evidence'] = dict(
            complete=evidence.complete, candidates=portable(evidence.candidates_evidence),
            relations=portable(evidence.relations))
        if not evidence.complete:
            result['status'] = 'incomplete'
        else:
            retained = sorted({v for r in evidence.relations if r.qualifies
                               for v in (r.members or (r.source,))})
            result['retained'] = retained
            baseline = np.full(len(test), train.y.mean())
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                full, full_params = fit_predictions(train, test, ('a', 'b'), settings)
                selected, selected_params = fit_predictions(train, test, tuple(retained), settings)
            result['warnings'] = [dict(category=w.category.__name__, message=str(w.message)) for w in captured]
            predictions = pd.DataFrame(dict(observed=test.y, baseline=baseline, full=full, retained=selected))
            predictions.to_csv(case/'predictions.csv', index=False)
            result.update(status='complete', scores=scores(predictions),
                          full_parameters=full_params, retained_parameters=selected_params)
    except Exception as exc:
        if resume and (case/'evidence.json.gz').exists():
            raise  # Preserve an existing case when refusing an incompatible resume.
        result.update(failure_type=type(exc).__name__, failure_message=str(exc),
                      traceback=traceback.format_exc())
    result['elapsed_seconds'] = time.perf_counter()-started
    # Fixed mtime makes compressed evidence stable across reproduction.
    with (case/'evidence.json.gz').open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
            compressed.write(json.dumps(result, separators=(',', ':'), allow_nan=False).encode())
    return dict(id=spec['id'], status=result['status'], retained=result['retained'],
                scores=result['scores'], elapsed_seconds=result['elapsed_seconds'],
                failure=result.get('failure_message'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='One tiny plumbing check, excluded from study')
    parser.add_argument('--resume', action='store_true', help='Verify saved protocol/settings and reuse completed assessments')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    args.output.mkdir(exist_ok=args.resume, parents=True)
    settings = config()
    specs = [dict(id=f'{scenario}-n{n}-r{rep}', scenario=scenario, n=n, repetition=rep,
                  seed=171001+si*10000+ni*100+rep)
             for si, scenario in enumerate(SCENARIOS) for ni, n in enumerate(SIZES) for rep in range(4)]
    if args.smoke:
        settings = replace(settings, performance_cv=replace(settings.performance_cv, folds=2, repeats=1),
            tuning_cv=replace(settings.tuning_cv, folds=2), stability=replace(settings.stability, n_resamples=2))
        specs = [dict(id='smoke-redundant', scenario='redundant', n=24, repetition=0, seed=99101)]
    sources = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__), *sorted((ROOT/'src/open').rglob('*.py'))]}
    manifest = dict(scope='smoke' if args.smoke else 'development_root_selection',
        config=portable(asdict(settings)), specs=specs, workers=args.workers,
        source_hashes=sources, protocol_hash=hashlib.sha256((ROOT/'examples/selection_reliability_protocol.md').read_bytes()).hexdigest())
    if args.resume:
        original = json.loads((args.output/'manifest.json').read_text())
        for field in ('scope', 'config', 'specs', 'protocol_hash'):
            if original[field] != manifest[field]:
                raise ValueError(f'resume changed frozen {field}')
        runner_source = Path(__file__).relative_to(ROOT).as_posix()
        for source, expected in original['source_hashes'].items():
            if source != runner_source and sources[source] != expected:
                raise ValueError(f'production source changed: {source}')
        archive = args.output/'runner-at-initial-dispatch.py'
        if not archive.exists():
            initial = (Path(__file__).read_bytes() if sources[runner_source] == original['source_hashes'][runner_source]
                       else subprocess.run(['git', 'show', '2218755:'+runner_source], cwd=ROOT,
                                           capture_output=True, check=True).stdout)
            if hashlib.sha256(initial).hexdigest() != original['source_hashes'][runner_source]:
                raise ValueError('initial runner archive does not match original fingerprint')
            archive.write_bytes(initial)
        original.setdefault('execution_sources', []).append(dict(runner_hash=sources[runner_source],
            workers=args.workers, dispatch='repetition/sample-size/scenario',
            reason='Resume after interruption; checkpointing and dispatch scheduling only; fixed statistical settings'))
        (args.output/'manifest.json').write_text(json.dumps(original, indent=2)+'\n')
        old_progress = {p.parent.name:json.loads(p.read_text()) for p in args.output.glob('*/progress.json')}
        (args.output/f'interruption-{time.time_ns()}.json').write_text(json.dumps(dict(
            checkpoints=old_progress, note='Initial runner saved progress counts but no recoverable local evidence; same fixed samples replayed'), indent=2)+'\n')
    else:
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        shutil.copyfile(ROOT/'examples/selection_reliability_protocol.md', args.output/'protocol.md')
    completed, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        dispatch = sorted(specs, key=lambda s:(s['repetition'], s['n'], SCENARIOS.index(s['scenario'])))
        futures = {pool.submit(run_case, spec, str(args.output.resolve()), settings, args.resume):spec for spec in dispatch}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(id=spec['id'], status='worker_failed', failure=str(exc))
            completed.append(result)
            (args.output/'progress.json').write_text(json.dumps(dict(completed=completed,
                planned=len(specs), elapsed_seconds=time.perf_counter()-started), indent=2)+'\n')
            print(f"{len(completed)}/{len(specs)} {result['id']}: {result['status']}, retained={result.get('retained')}", flush=True)
    (args.output/'execution.json').write_text(json.dumps(dict(planned=len(specs), completed=len(completed),
        failed=sum(r['status'] != 'complete' for r in completed), elapsed_seconds=time.perf_counter()-started), indent=2)+'\n')


if __name__ == '__main__':
    main()
