"""Manual, source-frozen orchestration for the approved recursive reference study."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # examples only; never add src

import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
import zipfile

import pandas as pd
import open as package
from open.layer_artifacts import portable, reconstruct_payloads

import recursive_preview_design as design
from recursive_preview_runner import run_case, write_json, read_gzip, file_hashes
from recursive_preview_audit import Store, audit_case, aggregate, validate_noise_gate

ROOT = design.ROOT
MANIFEST = ROOT/'examples/study_manifests/recursive-preview-validation.json'
SEED_AUDIT = ROOT/'examples/study_manifests/recursive-preview-seed-audit.json'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_manifest():
    value = load_json(MANIFEST)
    design.validate_manifest(value)
    return value


def runtime_check(manifest):
    if platform.python_version() != '3.11.9' or 'site-packages' not in Path(package.__file__).parts:
        raise ValueError('requires pinned installed Python3.11.9 package, not source import')
    actual = {n:importlib.metadata.version(n) for n in manifest['runtime']}
    if actual != manifest['runtime']:
        raise ValueError('runtime differs from frozen manifest')
    directory = Path(package.__file__).parent
    installed = {f'src/open/{p.relative_to(directory).as_posix()}':design.text_hash(p)
                 for p in directory.rglob('*.py')}
    expected = {n:h for n,h in manifest['sources'].items() if n.startswith('src/open/')}
    if installed != expected:
        raise ValueError('installed source differs from frozen package')


def freeze():
    from check_recursive_preview_design import seed_values
    proposed = {v for s in design.specs() for v in
                [s['generator_seed'], *range(s['analysis_seed'],s['analysis_seed']+100)]}
    paths = [p for p in sorted((ROOT/'examples/study_manifests').glob('*.json'))
             if p not in (MANIFEST, SEED_AUDIT) and not p.name.startswith('recursive-preview-noise-')]
    paths += sorted((ROOT/'examples/results').rglob('receipt.json'))
    inventory, historical = [], set()
    for path in paths:
        values = {v for _,v in seed_values(load_json(path))}
        historical.update(values)
        inventory.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=design.text_hash(path),
                              explicit_seed_count=len(values)))
    formulas = []
    for path in sorted((ROOT/'examples').glob('*.py')):
        if path.name.startswith(('recursive_preview_', 'check_recursive_preview_')):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        literals = [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                    and type(n.value) is int and n.value >= 1_000_000]
        if set(literals)&proposed:
            raise ValueError(f'historical namespace literal collision in {path.name}')
        formulas.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=design.text_hash(path),
                             large_integer_literals=sorted(set(literals))))
    collision = sorted(proposed&historical)
    if collision:
        raise ValueError('historical explicit seed collision')
    write_json(SEED_AUDIT, dict(reserved=len(proposed), collisions=collision,
        historical_seed_count=len(historical), manifests_and_loose_receipts=inventory,
        historical_source_namespace_inventory=formulas,
        scope='Historical manifests/loose receipts and source numeric namespace inventory; '
              'archive-only receipts use their original manifest identities. No new data or fits.'))
    value = design.manifest()
    write_json(MANIFEST, value)
    return value


def archive_cases(cases, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED) as archive:
        for directory in cases:
            for path in sorted(Path(directory).rglob('*')):
                if path.is_file():
                    archive.write(path, f'cases/{Path(directory).name}/{path.relative_to(directory).as_posix()}')


def indices(stage, shard):
    groups = design.shards(stage)
    if type(shard) is not int or not 0 <= shard < len(groups):
        raise ValueError('shard outside frozen stage')
    return groups[shard]


def validate_preflight(report, manifest):
    try:
        signed = {k:v for k,v in report.items() if k != 'digest'}
        if (report['digest'] != design.digest(signed) or report['status'] != 'PASS'
            or report['manifest'] != design.digest(manifest) or report['equivalence'] is not True
            or set(report['planned_cases']) != {'160','240'}
            or not report['storage_feasible'] or not report['audit_feasible']):
            raise ValueError()
        for key,case in report['planned_cases'].items():
            if case['index'] != int(key) or case['seconds'] > 900 or case['status'] != 'complete':
                raise ValueError()
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('preflight incomplete, edited or source/timing mismatch') from exc
    return True


def preflight(output):
    manifest = checked_manifest()
    runtime_check(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='STARTED', manifest=design.digest(manifest),
                   github_run=os.environ.get('GITHUB_RUN_ID'),
                   github_sha=os.environ.get('GITHUB_SHA'), planned_cases={})
    write_json(output/'preflight.json', receipt)
    try:
        original = ROOT/'examples/results/2026-10-14-default-preview-pilot/default-pilot-chain/.superpowers/default-pilot/chain/bundle'
        payload = {n:load_json(original/n) for n in (
            'analysis_metadata.json','layer_evidence.json','assessment_ledger.json')}
        baseline, _ = reconstruct_payloads(*(payload[n] for n in payload))
        data = pd.read_csv(ROOT/'examples/results/2026-10-08-layer-fixtures/chain/sample.csv')
        spec = dict(id='fixture-chain', fixture='chain')
        fixture = output/'cases'/spec['id']
        run_case(spec, fixture, manifest, data=data, config=baseline.config, seconds=3600)
        recreated = read_gzip(fixture/'network.json.gz')
        network, _ = reconstruct_payloads(*(recreated[n] for n in payload))
        if (design.digest(network.layers) != design.digest(baseline.layers)
            or network.plans != baseline.plans or network.assigned_depth != baseline.assigned_depth):
            raise ValueError('parallel fixture differs from saved sequential default analysis')
        store = Store(fixture)
        try:
            receipt['fixture_audit'] = audit_case(spec, store, manifest, data=data, config=baseline.config)
        finally:
            store.close()
        receipt['equivalence'] = True
        archive_cases([fixture], output/'raw/preflight-fixture.zip')
        timings, sizes = [], []
        for index in manifest['preflight_indices']:
            spec = design.specs()[index]
            directory = output/'cases'/spec['id']
            result = run_case(spec, directory, manifest)
            store = Store(directory)
            start = time.monotonic()
            try:
                audit = audit_case(spec, store, manifest)
            finally:
                store.close()
            timings.append(time.monotonic()-start)
            sizes.append(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()))
            receipt['planned_cases'][str(index)] = dict(index=index,status=audit['status'],
                seconds=result['seconds'], audit_seconds=timings[-1],
                receipt_sha256=audit['receipt_sha256'], assessments=audit['assessments'])
            write_json(output/'preflight.json', receipt)
        archive_cases([output/'cases'/design.specs()[i]['id'] for i in manifest['preflight_indices']],
                      output/'raw/preflight-cases.zip')
        receipt.update(storage_feasible=max(sizes)*320*1.5 < 8_000_000_000,
            audit_feasible=max(timings)*320/4 < 3000,
            projected_largest_signal_stage_bytes=max(sizes)*320*1.5,
            projected_four_process_signal_audit_seconds=max(timings)*320/4,
            raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/'raw').glob('*.zip')})
        if not receipt['storage_feasible'] or not receipt['audit_feasible']:
            raise RuntimeError('evidence size/audit allocation infeasible')
        receipt['status'] = 'PASS'
        receipt['digest'] = design.digest(receipt)
        validate_preflight(receipt, manifest)
    except BaseException as exc:
        receipt.update(status='FAILED',failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        write_json(output/'preflight.json', receipt)
        # Always preserve unfinished case starts/checkpoints if the Python process survives.
        if (output/'cases').exists():
            partial = output/'raw/preflight-all-attempts.zip'
            if not partial.exists():
                archive_cases(list((output/'cases').iterdir()), partial)


def find_one(directory, name):
    matches = list(Path(directory).rglob(name))
    if len(matches) != 1:
        raise ValueError(f'expected exactly one {name}, found {len(matches)}')
    return matches[0]


def prepare(stage, evidence, preflight_run, noise_run=None):
    manifest = checked_manifest()
    runtime_check(manifest)
    report = load_json(find_one(evidence, 'preflight.json'))
    validate_preflight(report, manifest)
    if str(report['github_run']) != str(preflight_run):
        raise ValueError('wrong preflight run receipt')
    if stage == 'signal':
        selection_path = ROOT/'examples/study_manifests/recursive-preview-noise-selection.json'
        selection = load_json(selection_path)
        audit_path = find_one(evidence, 'audit.json')
        audit = load_json(audit_path)
        if (selection['manifest'] != design.digest(manifest) or str(selection['noise_run']) != str(noise_run)
            or selection['audit_sha256'] != hashlib.sha256(audit_path.read_bytes()).hexdigest()):
            raise ValueError('signal stage requires committed exact noise audit/selection')
        validate_noise_gate(audit, design.digest(manifest))
    return list(range(len(design.shards(stage))))


def restore_preflight(archive, directory, spec, manifest):
    store = Store(archive, spec['id'])
    try:
        audit_case(spec, store, manifest)
        for name in store.names:
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('unsafe preflight evidence path')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(store.read(name))
    finally:
        store.close()


def shard(stage, index, output, evidence=None):
    manifest = checked_manifest()
    runtime_check(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    failures = []
    try:
        for case_index in indices(stage, index):
            spec = design.specs()[case_index]
            directory = output/'cases'/spec['id']
            cases.append(directory)
            try:
                if case_index in manifest['preflight_indices']:
                    if evidence is None:
                        raise ValueError('planned preflight dataset must reuse its existing fits')
                    restore_preflight(find_one(evidence, 'preflight-cases.zip'), directory, spec, manifest)
                run_case(spec, directory, manifest)
            except Exception as exc:
                failures.append(dict(index=case_index, failure=type(exc).__name__+': '+str(exc)[:2000]))
    finally:
        archive_cases([p for p in cases if p.exists()], output/'raw'/f'{stage}-{index:03}.zip')
        write_json(output/'shard.json', dict(stage=stage,index=index,failures=failures,
            manifest=design.digest(manifest), github_run=os.environ.get('GITHUB_RUN_ID')))
    if failures:
        raise RuntimeError(f'{len(failures)} incomplete cases; evidence retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze','preflight','prepare','shard','aggregate','pack'))
    parser.add_argument('--stage', choices=('noise','signal'))
    parser.add_argument('--index', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--preflight-run')
    parser.add_argument('--noise-run')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'preflight':
        preflight(args.out)
    elif args.command == 'prepare':
        print(json.dumps(prepare(args.stage,args.evidence,args.preflight_run,args.noise_run)))
    elif args.command == 'shard':
        shard(args.stage,args.index,args.out,args.evidence)
    elif args.command == 'pack':
        raw = args.out/'raw'
        expected = raw/('preflight-all-attempts.zip' if args.stage is None else f'{args.stage}-{args.index:03}.zip')
        cases = args.out/'cases'
        if cases.exists() and not expected.exists():
            archive_cases(list(cases.iterdir()), expected)
    else:
        manifest = checked_manifest()
        runtime_check(manifest)
        aggregate(args.evidence,manifest,args.stage,args.out)


if __name__ == '__main__':
    main()
