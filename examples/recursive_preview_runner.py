"""Checkpointed execution of unchanged production assessments; no automatic refits."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import open as package
import open.analysis as analysis
import open.grouping as grouping
import open.layer_stability as engine
from open.layer_artifacts import build_payloads, decode, portable
from open.layer_types import LocalAssessment
from open.metadata import validate_metadata

from recursive_preview_design import digest, generate, settings

NATIVE_ASSESS = engine.assess_layer_target


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(portable(value), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(text, encoding='utf-8', newline='\n')
        temporary.replace(path)


def write_gzip(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(portable(value), sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    temporary.replace(path)


def read_gzip(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def file_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'receipt.json'}


def verified_receipt(directory, context):
    directory = Path(directory)
    receipt = json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
    if receipt['context'] != context:
        raise ValueError('completed case context differs')
    if receipt['status'] != 'complete':
        raise RuntimeError('case already started; no automatic model retry')
    if receipt['files'] != file_hashes(directory):
        raise ValueError('case evidence digest mismatch')
    return receipt


def read_checkpoint(path, plan, context):
    payload = read_gzip(path)
    signature = payload.pop('digest')
    if digest(payload) != signature:
        raise ValueError('checkpoint digest mismatch')
    if payload['context'] != context:
        raise ValueError('checkpoint context mismatch')
    assessment = decode(LocalAssessment, payload['assessment'])
    if assessment.sample != plan:
        raise ValueError('checkpoint sample mismatch')
    if assessment.status != 'completed':
        raise RuntimeError('failed assessment cannot be automatically refitted')
    return assessment, payload


def process_peak_bytes():
    """Child process lifetime OS high-water mark, including imports."""
    if os.name != 'nt':
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == 'darwin' else peak*1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
            (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError('could not read worker OS peak memory')
    return int(counters.PeakWorkingSetSize)


def one_assessment(data, depth, target, candidates, plan, config, path, context):
    path = Path(path)
    if path.exists():
        read_checkpoint(path, plan, context)
        return 'reused'
    start_path = path.with_suffix('.started.json')
    if start_path.exists():
        raise RuntimeError('assessment already started; unknown cost, no automatic retry')
    started = time.monotonic()
    attempt = dict(context=context, status='STARTED', depth=depth, target=target,
                   run_index=plan.run_index, seed=plan.seed, fit_cost_unknown=True,
                   utc=datetime.now(timezone.utc).isoformat())
    write_json(start_path, attempt, exclusive=True)
    risks = []
    native_analysis, native_grouping = analysis.evaluate_risk, grouping.evaluate_risk
    def recorder(native, origin):
        def recorded(*args, **kw):
            result = native(*args, **kw)
            risks.append(dict(origin=origin, risk=portable(result)))
            return result
        return recorded
    analysis.evaluate_risk = recorder(native_analysis, 'analysis')
    grouping.evaluate_risk = recorder(native_grouping, 'grouping')
    try:
        assessment = NATIVE_ASSESS(data, depth=depth, target=target, candidates=candidates,
            plans=(replace(plan, run_index=0),), config=config)[0]
        assessment = replace(assessment, sample=plan)
        payload = dict(context=context, assessment=portable(assessment), risks=risks,
                       seconds=time.monotonic()-started, worker_os_peak_bytes=process_peak_bytes(),
                       memory_scope='worker process lifetime, includes imports; not whole-job peak')
        payload['digest'] = digest(payload)
        write_gzip(path, payload)
        complete = assessment.status == 'completed'
        attempt.update(status=assessment.status, seconds=time.monotonic()-started,
            fit_cost_unknown=not complete,
            outer_fits=assessment.local.outer_model_fit_count if complete else None,
            tuning_fits=assessment.local.tuning_model_fit_count if complete else None)
        if not complete:
            raise RuntimeError(f'assessment {assessment.status}: {assessment.failure_message}')
        return 'computed'
    except BaseException as exc:
        attempt.update(status='FAILED', failure=type(exc).__name__+': '+str(exc)[:2000],
                       seconds=time.monotonic()-started)
        raise
    finally:
        analysis.evaluate_risk, grouping.evaluate_risk = native_analysis, native_grouping
        write_json(path.with_suffix('.finished.json'), attempt)


class CheckpointAssessor:
    def __init__(self, directory, context, deadline, workers=4):
        self.directory, self.context = Path(directory), context
        self.deadline, self.workers = deadline, workers

    def __call__(self, data, *, depth, target, candidates, plans, config):
        context = digest(dict(case=self.context, depth=depth, target=target,
                              candidates=candidates, config=config))
        folder = self.directory/f'd{depth}-{target}'
        results = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(plans), self.workers):
                if time.monotonic() >= self.deadline:
                    raise TimeoutError('case soft deadline; no new assessment batches')
                batch = plans[start:start+self.workers]
                futures = [pool.submit(one_assessment, data, depth, target, candidates, plan,
                            config, folder/f'{plan.run_index:03}.json.gz', context) for plan in batch]
                for future, plan in zip(futures, batch):
                    future.result()
                    results.append(read_checkpoint(folder/f'{plan.run_index:03}.json.gz', plan, context)[0])
                print(f'{target} depth {depth}: {len(results)}/{len(plans)}', flush=True)
        return tuple(results)


def fit_recorded(data, config, directory, context, deadline, workers=4):
    native = engine.assess_layer_target
    engine.assess_layer_target = CheckpointAssessor(directory, context, deadline, workers)
    try:
        return engine.fit_layer_consensus(data, config.analysis.root, tuple(data), config)
    finally:
        engine.assess_layer_target = native


def case_context(spec, data, config, manifest):
    return digest(dict(spec=spec, data=data.to_dict('list'), config=config,
                       manifest=digest(manifest)))


def run_case(spec, directory, manifest, *, data=None, config=None, seconds=900):
    data = generate(spec) if data is None else data
    config = settings(spec) if config is None else config
    directory = Path(directory)
    context = case_context(spec, data, config, manifest)
    if (directory/'receipt.json').exists():
        return verified_receipt(directory, context)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = dict(spec=spec, config=portable(config), context=context, manifest=digest(manifest),
        status='STARTED', github_sha=os.environ.get('GITHUB_SHA'),
        github_run=os.environ.get('GITHUB_RUN_ID'), github_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
        python=platform.python_version(), module=package.__file__,
        runtime={n:importlib.metadata.version(n) for n in manifest['runtime']},
        utc=datetime.now(timezone.utc).isoformat(), workers=4, fit_cost_unknown=True)
    write_json(directory/'receipt.json', receipt, exclusive=True)
    data.to_csv(directory/'data.csv', index=False, lineterminator='\n')
    write_json(directory/'config.json', portable(config))
    start = time.monotonic()
    try:
        network = fit_recorded(data, config, directory/'checkpoints', context, start+seconds)
        metadata = validate_metadata(tuple(data),
            [dict(name=n, type='continuous') for n in data], config.analysis.root)
        write_gzip(directory/'network.json.gz', build_payloads(network, metadata))
        if network.status != 'COMPLETE':
            raise RuntimeError(network.status)
        if time.monotonic()-start > seconds:
            raise TimeoutError('completed after case computation allocation')
        receipt.update(status='complete', fit_cost_unknown=False)
    except BaseException as exc:
        receipt.update(status='incomplete', failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        receipt.update(seconds=time.monotonic()-start, files=file_hashes(directory))
        write_json(directory/'receipt.json', receipt)
    return receipt
