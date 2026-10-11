"""Bounded feasibility pilot; production layer recursion and selection are unchanged."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'examples'))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from open.config import AnalysisConfig, LearnerConfig, OpenConfig, StabilityConfig
from open.layer_artifacts import build_payloads, reconstruct_payloads, portable, decode, validate_local
from open.layer_types import LocalAssessment
import open.layer_stability as engine
from open.metadata import validate_metadata
from open.learners import make_predictor
from open.learners.trees import GRIDS
from run_selection_reliability import fingerprint, fit_predictions, load_checkpoint, save_checkpoint

NAME = '2026-10-10-sample-size-pilot'
OUTPUT = ROOT/'examples/results'/NAME
PROTOCOL = ROOT/'examples/sample_size_pilot_protocol.md'
NATIVE_ASSESS = engine.assess_layer_target


class BudgetExceeded(RuntimeError):
    pass


def text_hash(value):
    return hashlib.sha256(value.replace(b'\r\n', b'\n')).hexdigest()


def sources():
    files = [*sorted((ROOT/'src/open').rglob('*.py')),
             Path(__file__), ROOT/'examples/run_selection_reliability.py']
    return {p.relative_to(ROOT).as_posix():text_hash(p.read_bytes()) for p in files}


def specs():
    result = []
    for scenario in ('noise', 'additive', 'interaction', 'chain'):
        for n, p in ((80, 5), (300, 5), (300, 15)):
            index = len(result)
            result.append(dict(id=f'{scenario}-n{n}-p{p}', scenario=scenario, n=n, p=p,
                seed=410000+index*1000, test_seed=7410000+index*1000,
                discovery_seed=610000+index*1000))
    return result


def settings(spec):
    return OpenConfig(AnalysisConfig('y', spec['discovery_seed'],
        3 if spec['scenario'] == 'chain' else 1),
        learner=LearnerConfig(library=('gbm',)),
        stability=StabilityConfig(scope='layer_consensus'))


def generate(scenario, n, p, seed):
    if scenario not in ('noise', 'additive', 'interaction', 'chain') or p < 3:
        raise ValueError('unknown scenario or insufficient candidates')
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    if scenario == 'chain':
        x[:, 1] = .6*x[:, 2]+.8*rng.normal(size=n)
        x[:, 0] = .6*x[:, 1]+.8*rng.normal(size=n)
    mean = (np.zeros(n) if scenario == 'noise' else
            x[:, 0]*x[:, 1] if scenario == 'interaction' else x[:, 0])
    y = mean+rng.normal(scale=1 if scenario == 'noise' else np.sqrt(2/3), size=n)
    return pd.DataFrame({'y':y, **{f'x{i+1}':x[:, i] for i in range(p)}}), mean


def make_manifest():
    return dict(schema='open.sample_size_feasibility.v1', cases=specs(),
        configs={s['id']:portable(settings(s)) for s in specs()},
        source_hashes=sources(), protocol_hash=text_hash(PROTOCOL.read_bytes()),
        test_n=2000, workers=4, soft_seconds=5400, max_parallel=3,
        hard_job_minutes=110, maximum_runner_minutes=1365)


def validate_manifest(manifest):
    if manifest != make_manifest():
        raise ValueError('pilot manifest differs from fixed identities/settings/sources/protocol')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def structural_scores(spec, parents, depths):
    required_root = set() if spec['scenario'] == 'noise' else (
        {'x1', 'x2'} if spec['scenario'] == 'interaction' else {'x1'})
    expected = {(v, 'y') for v in required_root}
    expected_depth = {'y':0, **{v:1 for v in required_root}}
    if spec['scenario'] == 'chain':
        expected |= {('x2', 'x1'), ('x3', 'x2')}
        expected_depth.update(x2=2, x3=3)
    actual = {(source, target) for source, targets in parents.items() for target in targets}
    retained = {s for s, t in actual if t == 'y'}
    return dict(root_recovered=required_root <= retained,
        false_root_inputs=sorted(retained-required_root),
        required_edges_found=len(actual & expected), required_edges_total=len(expected),
        missing_edges=[list(e) for e in sorted(expected-actual)],
        extra_edges=[list(e) for e in sorted(actual-expected)],
        path_recovered=expected <= actual,
        required_depth_matches={v:depths.get(v) == d for v, d in expected_depth.items()})


def prediction_scores(frame):
    baseline = float(np.square(frame.observed-frame.baseline).mean())
    result = {}
    for name in ('full', 'retained', 'oracle'):
        mse = float(np.square(frame.observed-frame[name]).mean())
        result[name] = dict(mse=mse, skill=float(1-mse/baseline) if baseline > 0 else None)
    result['skill_loss'] = (result['full']['skill']-result['retained']['skill']
        if baseline > 0 else None)
    return result


def checkpoint_context(data, depth, target, candidates, config):
    return fingerprint(dict(data=data.to_dict('list'), depth=depth, target=target,
        candidates=candidates, config=config, sources=sources(),
        protocol=text_hash(PROTOCOL.read_bytes())))


def process_peak_bytes():
    """OS process-lifetime high-water mark, not a whole-job concurrent peak."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak*1024)


def one_assessment(data, depth, target, candidates, plan, config, checkpoint, context):
    if checkpoint.exists():
        load_checkpoint(checkpoint, plan, context)
        return 'reused'
    started = time.monotonic()
    result = NATIVE_ASSESS(data, depth=depth, target=target, candidates=candidates,
        plans=(replace(plan, run_index=0),), config=config)[0]
    result = replace(result, sample=plan)
    save_checkpoint(checkpoint, result, context)
    write_json(checkpoint.with_suffix('.metrics.json'), dict(context=context,
        elapsed_seconds=time.monotonic()-started, process_peak_bytes=process_peak_bytes(),
        scope='worker process lifetime; includes imports; not simultaneous total job memory'))
    return 'computed'


class CheckpointAssessor:
    def __init__(self, directory, deadline, workers=4):
        self.directory, self.deadline, self.workers = Path(directory), deadline, workers

    def __call__(self, data, *, depth, target, candidates, plans, config):
        context = checkpoint_context(data, depth, target, candidates, config)
        folder = self.directory/f'd{depth}-{target}'
        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(plans), self.workers):
                if time.monotonic() >= self.deadline:
                    raise BudgetExceeded('case soft budget exhausted; checkpoints retained')
                batch = plans[start:start+self.workers]
                futures = [pool.submit(one_assessment, data, depth, target, candidates,
                    plan, config, folder/f'{plan.run_index:03d}.json.gz', context)
                    for plan in batch]
                for future, plan in zip(futures, batch):
                    future.result()
                    results.append(load_checkpoint(folder/f'{plan.run_index:03d}.json.gz', plan, context))
                write_json(folder/'progress.json', dict(depth=depth, target=target,
                    candidates=list(candidates), completed=len(results), planned=len(plans)))
                print(f'{target} depth {depth}: {len(results)}/{len(plans)}', flush=True)
        return tuple(results)


def row_diagnostics(network, config):
    records = []
    for layer in network.layers:
        for target in layer.evidence:
            for a in target.assessments:
                plan = a.sample
                splits = plan.split_plan.splits if plan.split_plan is not None else ()
                records.append(dict(depth=a.depth, target=a.target, run=plan.run_index,
                    candidates=list(a.candidates), status=a.status,
                    sample_n=len(plan.sampled_positions),
                    fit_n=[len(s.train_indices) for s in splits],
                    validation_n=[len(s.test_indices) for s in splits],
                    tuning_n=[len(tr) for s in splits for tr, _ in KFold(
                        config.tuning_cv.folds, shuffle=True, random_state=plan.seed).split(s.train_indices)],
                    stop=a.local.reason if a.local is not None else None,
                    relative_gain=a.local.total_gain_relative if a.local is not None else None,
                    outer_fits=a.local.outer_model_fit_count if a.local is not None else 0,
                    tuning_fits=a.local.tuning_model_fit_count if a.local is not None else 0))
    return records


def run_case(spec, output, config=None, seconds=5400, workers=4, test_n=2000):
    config = settings(spec) if config is None else config
    started = time.monotonic()
    case = Path(output)/spec['id']
    case.mkdir(parents=True, exist_ok=True)
    context = fingerprint(dict(spec=spec, config=config, test_n=test_n,
        sources=sources(), protocol=text_hash(PROTOCOL.read_bytes())))
    if (case/'result.json').exists():
        previous = json.loads((case/'result.json').read_text())
        if previous['context'] != context:
            raise ValueError('case context changed; preserve old evidence in a separate output')
        if previous['status'] == 'complete':
            return previous
    result = dict(spec=spec, config=portable(config), context=context, status='incomplete',
        structural=None, scores=None, diagnostics=[], warnings=[], execution=dict(
            workers=workers, soft_seconds=seconds, test_n=test_n, sources=sources(),
            protocol_hash=text_hash(PROTOCOL.read_bytes()), python=platform.python_version(),
            platform=platform.platform(), dependencies={n:importlib.metadata.version(n)
                for n in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
            git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            github_run=os.environ.get('GITHUB_RUN_ID')))
    write_json(case/'result.json', result)
    train, _ = generate(spec['scenario'], spec['n'], spec['p'], spec['seed'])
    test, oracle = generate(spec['scenario'], test_n, spec['p'], spec['test_seed'])
    train.to_csv(case/'train.csv', index=False)
    test.to_csv(case/'test.csv', index=False)
    native = engine.assess_layer_target
    engine.assess_layer_target = CheckpointAssessor(case/'checkpoints', started+seconds, workers)
    try:
        network = engine.fit_layer_consensus(train, 'y', tuple(train.columns), config)
        metadata = validate_metadata(tuple(train.columns), [dict(name=n, display_label=n,
            type='continuous', role='root' if n == 'y' else 'candidate') for n in train], 'y')
        payloads = build_payloads(network, metadata)
        (case/'network.json.gz').write_bytes(gzip.compress(
            json.dumps(payloads, allow_nan=False).encode(), mtime=0))
        result['diagnostics'] = row_diagnostics(network, config)
        result['warnings'] = [portable(w) for l in network.layers for t in l.evidence
            for a in t.assessments for w in a.warnings]
        if network.status != 'COMPLETE':
            result['reason'] = network.status
        else:
            result['structural'] = structural_scores(spec, network.parents, network.assigned_depth)
            if time.monotonic() >= started+seconds:
                raise BudgetExceeded('no budget remaining for test refits')
            inputs = tuple(n for n in train if n != 'y')
            selected = tuple(n for n in inputs if 'y' in network.parents.get(n, ()))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                full, full_params = fit_predictions(train, test, inputs, config)
            result['warnings'].extend(dict(category=w.category.__name__, message=str(w.message)) for w in captured)
            if time.monotonic() >= started+seconds:
                raise BudgetExceeded('no budget remaining for retained-input refit')
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                retained, retained_params = fit_predictions(train, test, selected, config)
            result['warnings'].extend(dict(category=w.category.__name__, message=str(w.message)) for w in captured)
            predictions = pd.DataFrame(dict(observed=test.y, baseline=np.full(test_n, train.y.mean()),
                full=full, retained=retained, oracle=oracle))
            predictions.to_csv(case/'predictions.csv', index=False)
            result.update(status='complete', scores=prediction_scores(predictions),
                full_parameters=full_params, retained_parameters=retained_params)
    except Exception as exc:
        result.update(reason=type(exc).__name__+': '+str(exc), structural=None, scores=None)
    finally:
        engine.assess_layer_target = native
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(case/'result.json', result)
    return result


def audit_checkpoints(train, config, case):
    plans = engine.make_layer_sample_plans(train, config)
    count = 0
    for path in sorted((case/'checkpoints').rglob('*.json.gz')):
        payload = json.loads(gzip.decompress(path.read_bytes()))
        a = decode(LocalAssessment, payload['assessment'])
        if (a.sample.run_index >= len(plans) or a.target not in train
                or not set(a.candidates) <= set(train)-{a.target}
                or a.depth >= config.analysis.max_depth
                or path.parent.name != f'd{a.depth}-{a.target}'
                or path.name != f'{a.sample.run_index:03d}.json.gz'):
            raise ValueError('checkpoint target/sample identity differs')
        context = checkpoint_context(train, a.depth, a.target, a.candidates, config)
        load_checkpoint(path, plans[a.sample.run_index], context)
        if a.local is not None:
            validate_local(a.local, a.sample, config)
        count += 1
    return count


def verify_refits(train, test, predictions, network, result, config):
    inputs = tuple(n for n in train if n != 'y')
    selected = tuple(n for n in inputs if 'y' in network.parents.get(n, ()))
    allowed = list(ParameterGrid(GRIDS[config.learner.library[0]]))
    for label, variables in (('full', inputs), ('retained', selected)):
        params = result[label+'_parameters']
        if not variables:
            if params is not None:
                raise ValueError('baseline refit must have no tuned parameters')
            expected = np.full(len(test), train.y.mean())
        else:
            if params not in allowed:
                raise ValueError('saved refit settings outside shipped grid')
            model = make_predictor(config)
            model.fit(train[list(variables)], train.y, params)
            expected = model.predict(test[list(variables)])
        np.testing.assert_allclose(predictions[label], expected, rtol=1e-12, atol=1e-12)


def audit_case(spec, case, config=None, test_n=2000):
    result = json.loads((case/'result.json').read_text())
    config = settings(spec) if config is None else config
    expected_context = fingerprint(dict(spec=spec, config=config, test_n=test_n,
        sources=sources(), protocol=text_hash(PROTOCOL.read_bytes())))
    if result['spec'] != spec or result['config'] != portable(config) or result['context'] != expected_context:
        raise ValueError('case identity/settings/source mismatch')
    train, _ = generate(spec['scenario'], spec['n'], spec['p'], spec['seed'])
    test, oracle = generate(spec['scenario'], test_n, spec['p'], spec['test_seed'])
    for name, expected in (('train', train), ('test', test)):
        actual = pd.read_csv(case/f'{name}.csv', float_precision='round_trip')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    checkpoint_count = audit_checkpoints(train, config, case)
    result['audited_checkpoints'] = checkpoint_count
    if not (case/'network.json.gz').exists():
        if result['status'] == 'complete' or result['scores'] is not None or result['structural'] is not None:
            raise ValueError('missing network cannot publish final scores')
        return result
    payloads = json.loads(gzip.decompress((case/'network.json.gz').read_bytes()))
    network, _ = reconstruct_payloads(*(payloads[n] for n in (
        'analysis_metadata.json', 'layer_evidence.json', 'assessment_ledger.json')))
    if network.config != config or result['diagnostics'] != row_diagnostics(network, config):
        raise ValueError('network/row diagnostics differ')
    plans = engine.make_layer_sample_plans(train, config)
    if network.plans != plans:
        raise ValueError('realized samples/splits differ')
    for layer in network.layers:
        for target in layer.evidence:
            context = checkpoint_context(train, layer.depth, target.target, target.candidates, config)
            for a in target.assessments:
                saved = load_checkpoint(case/'checkpoints'/f'd{layer.depth}-{target.target}'/
                    f'{a.sample.run_index:03d}.json.gz', a.sample, context)
                if fingerprint(saved) != fingerprint(a):
                    raise ValueError('network differs from checkpoint evidence')
    if result['status'] != 'complete':
        if result['scores'] is not None or result['structural'] is not None:
            raise ValueError('incomplete case cannot publish final scores')
        return result
    if network.status != 'COMPLETE':
        raise ValueError('incomplete network cannot publish final scores')
    if result['structural'] != structural_scores(spec, network.parents, network.assigned_depth):
        raise ValueError('structural scores differ')
    predictions = pd.read_csv(case/'predictions.csv', float_precision='round_trip')
    if len(predictions) != test_n or not np.isfinite(predictions.to_numpy()).all():
        raise ValueError('invalid test prediction coverage')
    np.testing.assert_array_equal(predictions.observed, test.y)
    np.testing.assert_array_equal(predictions.oracle, oracle)
    np.testing.assert_array_equal(predictions.baseline, np.full(test_n, train.y.mean()))
    if result['scores'] != prediction_scores(predictions):
        raise ValueError('prediction arithmetic differs')
    verify_refits(train, test, predictions, network, result, config)
    return result


def audit(output):
    output = Path(output)
    validate_manifest(json.loads((output/'manifest.json').read_text()))
    rows = []
    for spec in specs():
        case = output/spec['id']
        if not (case/'result.json').exists():
            rows.append(dict(spec=spec, status='missing', reason='no case artifact'))
        else:
            try:
                rows.append(audit_case(spec, case))
            except Exception as exc:
                rows.append(dict(spec=spec, status='audit_failed', reason=type(exc).__name__+': '+str(exc)))
    complete = sum(r['status'] == 'complete' for r in rows)
    report = dict(planned=12, complete=complete, incomplete=12-complete, cases=rows,
        audit_errors=sum(r['status'] == 'audit_failed' for r in rows),
        all_complete=complete == 12, limit='Feasibility only; no minimum-N or independent local refit claim')
    write_json(output/'audit.json', report)
    lines = ['# Sample-size feasibility pilot', '',
        f'Completed {complete}/12 planned datasets. Incomplete or absent: {12-complete}.', '',
        'One dataset per condition: these results do not establish reliability or a minimum N.', '',
        '| Case | Status | Seconds | Root recovered | Required edges | False root inputs | Skill loss |',
        '|---|---|---:|---|---|---|---:|']
    for r in rows:
        st, sc = r.get('structural'), r.get('scores')
        elapsed = f"{r['elapsed_seconds']:.1f}" if 'elapsed_seconds' in r else 'unavailable'
        lines.append(f"| {r['spec']['id']} | {r['status']} | {elapsed} | "
            f"{st['root_recovered'] if st else 'unavailable'} | "
            f"{str(st['required_edges_found'])+'/'+str(st['required_edges_total']) if st else 'unavailable'} | "
            f"{len(st['false_root_inputs']) if st else 'unavailable'} | "
            f"{sc['skill_loss'] if sc else 'unavailable'} |")
    lines += ['', 'Full evidence includes actual fit/tuning sizes, checkpoints and independent test predictions.',
        'A missing downstream comparison remains a full-path miss; prediction does not certify the graph.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(planned=12, complete=complete, incomplete=12-complete)))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'matrix', 'case', 'audit', 'smoke'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.command == 'freeze':
        path = args.output/'manifest.json'
        if path.exists():
            validate_manifest(json.loads(path.read_text()))
        else:
            write_json(path, make_manifest())
    elif args.command == 'matrix':
        validate_manifest(json.loads((args.output/'manifest.json').read_text()))
        print(json.dumps([s['id'] for s in specs()]))
    elif args.command == 'case':
        validate_manifest(json.loads((args.output/'manifest.json').read_text()))
        spec = next(s for s in specs() if s['id'] == args.case)
        result = run_case(spec, args.output)
        print(json.dumps(dict(id=spec['id'], status=result['status'], reason=result.get('reason'))))
    elif args.command == 'audit':
        if audit(args.output)['audit_errors']:
            raise SystemExit(1)
    else:
        spec = dict(specs()[9], id='smoke-chain', n=120, p=3)
        config = settings(spec)
        config = replace(config, stability=replace(config.stability, n_resamples=1))
        result = run_case(spec, args.output, config, seconds=300, workers=1, test_n=50)
        print(json.dumps(dict(status=result['status'], reason=result.get('reason'))))
        if result['status'] != 'complete':
            raise SystemExit(1)
        audit_case(spec, args.output/spec['id'], config, test_n=50)
        print('Smoke evidence audit passed; this is not a reliability result.')


if __name__ == '__main__':
    main()
