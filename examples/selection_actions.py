"""Transfer the frozen selection study to independent GitHub Actions jobs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_selection_reliability as study
from open.config import parse_config
from open.layer_artifacts import portable

NAME = '2026-10-10-selection-reliability'
OUTPUT = study.ROOT/'examples/results'/NAME
ARCHIVE = study.ROOT/'examples/study_transfers'/f'{NAME}.zip'


def specs():
    return [dict(id=f'{scenario}-n{n}-r{rep}', scenario=scenario, n=n, repetition=rep,
                 seed=171001+si*10000+ni*100+rep)
            for si, scenario in enumerate(study.SCENARIOS)
            for ni, n in enumerate(study.SIZES) for rep in range(4)]


def validate_manifest(manifest):
    if manifest['scope'] != 'development_root_selection' or parse_config(manifest['config']) != study.config():
        raise ValueError('frozen research settings changed')
    known = {s['id']: s for s in specs()}
    ids = [s['id'] for s in manifest['specs']]
    if len(set(ids)) != len(ids) or any(known.get(s['id']) != s for s in manifest['specs']):
        raise ValueError('frozen dataset identities changed')


def verify_sources(manifest, root=study.ROOT, output=OUTPUT):
    for source, expected in manifest['source_hashes'].items():
        actual = hashlib.sha256((root/source).read_bytes()).hexdigest()
        if source == 'examples/run_selection_reliability.py' and manifest.get('execution_sources'):
            archive = output/'runner-at-initial-dispatch.py'
            if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
                raise ValueError('original runner archive changed')
            expected = manifest['execution_sources'][-1]['runner_hash']
        if actual != expected:
            raise ValueError(f'frozen source changed: {source}')


def pack():
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    validate_manifest(manifest)
    verify_sources(manifest)
    checkpoints = study.ROOT/'.superpowers/selection-checkpoints'/NAME
    files = sorted(checkpoints.rglob('*.json.gz'))
    transfer = dict(checkpoints=len(files), completed_cases=len(list(OUTPUT.glob('*/evidence.json.gz'))),
                    helper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note='Synthetic study only; frozen data, settings and source bytes retained. No partial result is accepted as a completed dataset.')
    (OUTPUT/'transfer.json').write_text(json.dumps(transfer, indent=2)+'\n')
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():
                archive.write(p, p.relative_to(study.ROOT).as_posix())
        for p in files:
            archive.write(p, p.relative_to(study.ROOT).as_posix())
        for source in [*manifest['source_hashes'], 'examples/selection_reliability_protocol.md']:
            archive.write(study.ROOT/source, 'frozen-source/'+source)
    print(json.dumps(transfer))


def restore():
    with zipfile.ZipFile(ARCHIVE) as archive:
        transfer = json.loads(archive.read(f'examples/results/{NAME}/transfer.json'))
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != transfer['helper_hash']:
            raise ValueError('transfer orchestration source changed')
        for name in archive.namelist():
            relative = Path(name.removeprefix('frozen-source/'))
            target = (study.ROOT/relative).resolve()
            if not target.is_relative_to(study.ROOT.resolve()):
                raise ValueError('archive path escapes workspace')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    validate_manifest(manifest)
    verify_sources(manifest)
    if manifest['specs'] != specs():
        raise ValueError('transfer must retain all 48 planned datasets')
    if hashlib.sha256((OUTPUT/'protocol.md').read_bytes()).hexdigest() != manifest['protocol_hash']:
        raise ValueError('frozen protocol changed')
    return manifest


def precompute_one(spec, settings, index, output):
    train = study.generate(spec['scenario'], spec['n'], spec['seed'])
    context = study.fingerprint(dict(spec=spec, config=portable(settings), train=train.to_dict('list'),
        sources={p.relative_to(study.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((study.ROOT/'src/open').rglob('*.py'))}))
    plan = study.make_layer_sample_plans(train, settings)[index]
    path = study.ROOT/'.superpowers/selection-checkpoints'/Path(output).name/spec['id']/f'{index:03d}.json.gz'
    if path.exists():
        study.load_checkpoint(path, plan, context)
        return 'reused'
    assessment = study.assess_layer_target(train, depth=0, target='y', candidates=('a', 'b'),
        plans=(replace(plan, run_index=0),), config=settings)[0]
    study.save_checkpoint(path, replace(assessment, sample=plan), context)
    return 'computed'


def run_parallel_case(spec, settings, output=OUTPUT, workers=4):
    started = time.perf_counter()
    counts = dict(reused=0, computed=0)
    if not (output/spec['id']/'evidence.json.gz').exists():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(precompute_one, spec, settings, i, str(output))
                       for i in range(settings.stability.n_resamples)]
            for future in as_completed(pending):
                counts[future.result()] += 1
                print(f"{spec['id']}: {sum(counts.values())}/{settings.stability.n_resamples} checkpoints", flush=True)
    result = study.run_case(spec, str(output), settings, resume=True)
    execution = dict(result=result, checkpoints=counts, elapsed_seconds=time.perf_counter()-started,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            dependencies={name:importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn')}),
        helper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        github_run=os.environ.get('GITHUB_RUN_ID'), github_sha=os.environ.get('GITHUB_SHA'), workers=workers)
    (output/spec['id']/'actions-execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    from audit_selection_reliability import audit_case
    audit_case(output, spec, settings)
    if result['status'] != 'complete':
        raise RuntimeError(f"preserved incomplete case: {spec['id']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('pack', 'restore', 'matrix', 'case', 'finalize'))
    parser.add_argument('--case')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1..4')
    if args.command == 'pack':
        pack()
        return
    if args.command == 'restore':
        restore()
        return
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    validate_manifest(manifest)
    verify_sources(manifest)
    if args.command == 'matrix':
        print(json.dumps([s['id'] for s in manifest['specs']], separators=(',', ':')))
    elif args.command == 'case':
        matches = [s for s in manifest['specs'] if s['id'] == args.case]
        if len(matches) != 1:
            parser.error('case must match one frozen dataset')
        run_parallel_case(matches[0], parse_config(manifest['config']), workers=args.workers)
    else:
        executions = [json.loads((OUTPUT/s['id']/'actions-execution.json').read_text()) for s in manifest['specs']]
        record = dict(planned=48, completed=len(executions),
            failed=sum(e['result']['status'] != 'complete' for e in executions),
            provider='GitHub Actions', github_run=os.environ.get('GITHUB_RUN_ID'),
            elapsed_seconds=None, note='Mixed local checkpoint and distributed execution; no pristine runtime benchmark.',
            case_execution_records=[s['id']+'/actions-execution.json' for s in manifest['specs']])
        (OUTPUT/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
        manifest['distributed_execution'] = record
        (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
