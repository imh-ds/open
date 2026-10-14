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
from datetime import datetime
import urllib.request
import zipfile

import pandas as pd
import open as package
from open.layer_artifacts import portable, reconstruct_payloads

import recursive_preview_design as design
from recursive_preview_runner import run_case, write_json, read_gzip, file_hashes
from recursive_preview_audit import Store, audit_case, aggregate, validate_noise_gate
from recursive_preview_equivalence import compare_networks

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
            or not report['storage_feasible'] or not report['audit_feasible']
            or report['archive_crc_checked'] is not True
            or report['fixture_reused'] is not True or report['fixture_new_fits'] != 0
            or report['fixture_producer_run'] != manifest['continuation']['producer_run_id']
            or report['fixture_producer_manifest'] != manifest['continuation']['producer_manifest_digest']
            or report['raw_sha256']['preflight-fixture.zip'] != manifest['continuation']['archive_sha256']):
            raise ValueError()
        for key,case in report['planned_cases'].items():
            if case['index'] != int(key) or case['seconds'] > 900 or case['status'] != 'complete':
                raise ValueError()
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('preflight incomplete, edited or source/timing mismatch') from exc
    return True


def reuse_original_fixture(evidence, manifest, baseline, data, output):
    """Verify original producer bytes and audit them; this path never fits a model."""
    proof = manifest['continuation']
    archive = find_one(evidence,'preflight-all-attempts.zip')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != proof['archive_sha256']:
        raise ValueError('original fixture archive fingerprint differs')
    original = load_json(design.PRODUCER/'raw/producer-manifest.json')
    control = load_json(design.PRODUCER/'raw/preflight.json')
    if (design.digest(original) != proof['producer_manifest_digest']
        or control['manifest'] != proof['producer_manifest_digest']
        or str(control['github_run']) != proof['producer_run_id']
        or control['github_sha'] != proof['producer_source']
        or control['status'] != 'FAILED' or control['planned_cases'] != {}):
        raise ValueError('original producer identity/computation boundary differs')
    old_source = {n:h for n,h in original['sources'].items() if n.startswith('src/open/')}
    new_source = {n:h for n,h in manifest['sources'].items() if n.startswith('src/open/')}
    if old_source != new_source:
        raise ValueError('production package source changed since fixture producer')
    if any(original[k] != manifest[k] for k in ('cases','default_config','runtime','preflight_indices')):
        raise ValueError('scientific settings/identities changed since fixture producer')
    if portable(baseline.config) != original['default_config']:
        raise ValueError('saved sequential fixture does not use the original defaults')
    store = Store(archive,'fixture-chain')
    try:
        if hashlib.sha256(store.read('receipt.json')).hexdigest() != proof['fixture_receipt_sha256']:
            raise ValueError('original fixture receipt fingerprint differs')
        old_receipt = store.json('receipt.json')
        if (old_receipt['github_run'] != proof['producer_run_id']
            or old_receipt['github_sha'] != proof['producer_source'] or old_receipt['github_attempt'] != '1'):
            raise ValueError('original fixture run/source/attempt identity differs')
        saved = store.compressed('network.json.gz')
        names = ('analysis_metadata.json','layer_evidence.json','assessment_ledger.json')
        network,_ = reconstruct_payloads(*(saved[n] for n in names))
        comparison = compare_networks(network,baseline)
        if not comparison['equivalent']:
            raise ValueError('reused fixture differs beyond corrected equivalence policy')
        audit = audit_case(dict(id='fixture-chain',fixture='chain'),store,original,
                           data=data,config=baseline.config)
        if audit['assessments'] != 200 or audit['status'] != 'complete':
            raise ValueError('original fixture is not fully complete')
    finally:
        store.close()
    target = Path(output)/'raw/preflight-fixture.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('rb') as source, target.open('xb') as destination:
        shutil.copyfileobj(source,destination)
    return dict(new_fixture_fits=0, audit=audit, comparison=comparison,
                producer_run_id=proof['producer_run_id'],
                producer_manifest_digest=proof['producer_manifest_digest'])


def github_api(path):
    repository = os.environ.get('GITHUB_REPOSITORY', 'imh-ds/open')
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError('GitHub token required to verify original preflight transport')
    request = urllib.request.Request(f'https://api.github.com/repos/{repository}/{path}',
        headers={'Authorization':f'Bearer {token}', 'Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def verify_preflight_transport(report, api_get=github_api):
    run_id = str(report['github_run'])
    if not run_id.isdigit():
        raise ValueError('invalid original preflight run')
    run = api_get(f'actions/runs/{run_id}')
    if (str(run['id']) != run_id or run['status'] != 'completed' or run['conclusion'] != 'success'
        or run['head_sha'] != report['github_sha'] or run['run_attempt'] != 1):
        raise ValueError('original preflight run not completed successfully at original source/attempt')
    jobs = api_get(f'actions/runs/{run_id}/jobs?per_page=100')['jobs']
    found = [j for j in jobs if j['name'] == 'preflight']
    if len(found) != 1 or found[0]['conclusion'] != 'success':
        raise ValueError('preflight job did not preserve original evidence')
    uploads = [s for s in found[0]['steps'] if 'actions/upload-artifact@' in s['name']]
    if len(uploads) != 2 or any(s['conclusion'] != 'success' for s in uploads):
        raise ValueError('preflight uploads did not both complete successfully')
    def timestamp(text):
        return datetime.fromisoformat(text.replace('Z','+00:00'))
    seconds = [(timestamp(s['completed_at'])-timestamp(s['started_at'])).total_seconds() for s in uploads]
    artifacts = api_get(f'actions/runs/{run_id}/artifacts')['artifacts']
    by_name = {a['name']:a for a in artifacts if not a['expired']}
    required = {'recursive-preview-preflight-control','recursive-preview-preflight-raw'}
    if not required <= set(by_name) or any(by_name[n]['size_in_bytes'] <= 0 for n in required):
        raise ValueError('preflight upload artifact missing, expired or empty')
    return dict(run_id=run_id, source=run['head_sha'], upload_seconds=seconds,
                raw_uploaded_bytes=by_name['recursive-preview-preflight-raw']['size_in_bytes'])


def preflight(output, evidence):
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
        reused = reuse_original_fixture(evidence,manifest,baseline,data,output)
        write_json(output/'equivalence.json',reused['comparison'])
        receipt.update(fixture_audit=reused['audit'],equivalence=True,fixture_reused=True,
            fixture_new_fits=0,fixture_producer_run=reused['producer_run_id'],
            fixture_producer_manifest=reused['producer_manifest_digest'])
        write_json(output/'preflight.json',receipt)
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
        archive_start = time.monotonic()
        archive_cases([output/'cases'/design.specs()[i]['id'] for i in manifest['preflight_indices']],
                      output/'raw/preflight-cases.zip')
        for path in (output/'raw').glob('*.zip'):
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('preflight archive CRC check failed')
        receipt.update(storage_feasible=max(sizes)*320*1.5 < 8_000_000_000,
            audit_feasible=max(timings)*320/4 < 3000,
            projected_largest_signal_stage_bytes=max(sizes)*320*1.5,
            projected_four_process_signal_audit_seconds=max(timings)*320/4,
            raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/'raw').glob('*.zip')},
            archive_crc_checked=True, archive_seconds=time.monotonic()-archive_start,
            preflight_free_disk_bytes=shutil.disk_usage(output).free)
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
    write_json(Path(evidence)/'preflight-transport-verification.json', verify_preflight_transport(report))
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


def restore_preflight(archive, directory, spec, manifest, report):
    validate_preflight(report, manifest)
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != report['raw_sha256']['preflight-cases.zip']:
        raise ValueError('preflight archive differs from original receipt fingerprint')
    store = Store(archive, spec['id'])
    try:
        expected = report['planned_cases'][str(spec['index'])]['receipt_sha256']
        if hashlib.sha256(store.read('receipt.json')).hexdigest() != expected:
            raise ValueError('preflight case differs from original receipt fingerprint')
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
                    control = load_json(find_one(evidence, 'preflight.json'))
                    restore_preflight(find_one(evidence, 'preflight-cases.zip'), directory, spec, manifest, control)
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
        preflight(args.out,args.evidence)
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
