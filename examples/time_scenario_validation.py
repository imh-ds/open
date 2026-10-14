"""Four original timing identities; no scientific qualification or automatic retries."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import os
import time
import zipfile

import recursive_preview_design as design
from recursive_preview_actions import load_json, runtime_check, archive_cases
from recursive_preview_runner import run_case, write_json
from recursive_preview_audit import Store, audit_case

ROOT = design.ROOT
MANIFEST = ROOT/'examples/study_manifests/scenario-validation-timing.json'
CHAIN = ROOT/'examples/results/2026-10-14-recursive-preview-validation/timing-completed-38629900589'
INDICES = (0,80,320,400)


def reference():
    return load_json(ROOT/'examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38621370295/raw/producer-manifest.json')


def manifest():
    old = reference()
    proof = load_json(CHAIN/'audit/verification.json')
    hashes = load_json(CHAIN/'audit/raw-sha256.json')
    paths = set(old['sources']) | {
        'examples/time_scenario_validation.py', 'examples/validation_execution_revision.md',
        '.github/workflows/scenario-validation-timing.yml',
        'examples/study_manifests/recursive-preview-timing.json',
        'examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38621370295/raw/producer-manifest.json',
        (CHAIN/'audit/verification.json').relative_to(ROOT).as_posix(),
        (CHAIN/'audit/raw-sha256.json').relative_to(ROOT).as_posix()}
    return dict(schema='open.scenario_timing.v1', case_indices=list(INDICES),
        reference_manifest=design.digest(old), workers=4, runtime=old['runtime'],
        case_soft_seconds=2880, job_minutes=60, compute_audit_minutes=50,
        aggregate_minutes=15, registration_minutes=1, allocated_runner_minutes=256,
        historical_allocations=250, future_controls=150, remaining_study_allocation=9344,
        total_ceiling=10000, scientific_gate_accepted=False,
        sources={p:design.text_hash(ROOT/p) for p in sorted(paths)},
        chains=dict(run_id=38629900589, source=proof['source'],
            timing_manifest=proof['timing_manifest'], indices=[160,240],
            archive_sha256=hashes['timing-cases.zip'],
            receipt_sha256={k:v['receipt_sha256'] for k,v in proof['cases'].items()}))


def checked_manifest():
    value = load_json(MANIFEST)
    if value != manifest():
        raise ValueError('scenario timing manifest/source/settings changed')
    old = reference()
    if any(design.text_hash(ROOT/p) != h for p,h in old['sources'].items()):
        raise ValueError('original frozen source/settings changed')
    if design.digest(load_json(ROOT/'examples/study_manifests/recursive-preview-timing.json')) != value['chains']['timing_manifest']:
        raise ValueError('saved chain timing manifest differs')
    runtime_check(old)
    return value


def audit_location(location, spec, old):
    store = Store(location, spec['id'])
    try:
        return audit_case(spec, store, old)
    finally:
        store.close()


def audit_saved_chains(value):
    archive = CHAIN/'raw/timing-cases.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != value['chains']['archive_sha256']:
        raise ValueError('saved chain archive fingerprint differs')
    old = reference()
    rows = {}
    for index in (160,240):
        spec = old['cases'][index]
        store = Store(archive, spec['id'])
        try:
            receipt = store.json('receipt.json')
            if (hashlib.sha256(store.read('receipt.json')).hexdigest() != value['chains']['receipt_sha256'][str(index)]
                or receipt['github_run'] != str(value['chains']['run_id'])
                or receipt['github_sha'] != value['chains']['source'] or receipt['github_attempt'] != '1'):
                raise ValueError('saved chain receipt identity differs')
            row = audit_case(spec, store, old)
            if row['status'] != 'complete':
                raise ValueError('saved chain incomplete')
            rows[str(index)] = row
        finally:
            store.close()
    return rows


def pack(output):
    output = Path(output)
    if not (output/'cases').exists():
        return
    files = {p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (output/'cases').rglob('*') if p.is_file()}
    def matches(path):
        try:
            with zipfile.ZipFile(path) as z:
                return (z.testzip() is None and len(z.namelist()) == len(files)
                    and {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} == files)
        except (OSError,zipfile.BadZipFile):
            return False
    archive = output/'raw/case.zip'
    if matches(archive):
        return
    temporary = output/'raw/case.pending.zip'
    temporary.unlink(missing_ok=True)
    archive_cases(list((output/'cases').iterdir()),temporary)
    if not matches(temporary):
        raise ValueError('packed archive differs from saved case files')
    temporary.replace(archive)


def run_case_timing(index, output):
    value = checked_manifest()
    if type(index) is not int or index not in INDICES:
        raise ValueError('case identity outside four original timing conditions')
    if (os.environ.get('GITHUB_RUN_ATTEMPT') != '1' or not os.environ.get('GITHUB_RUN_ID')
        or not os.environ.get('GITHUB_SHA')):
        raise ValueError('requires a first Actions attempt with run/source identity; no automatic retry')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(index=index, status='STARTED', timing_manifest=design.digest(value),
        scientific_gate_accepted=False, timing_only=True, original_preflight_reclassified=False,
        github_run=os.environ.get('GITHUB_RUN_ID'), github_sha=os.environ.get('GITHUB_SHA'),
        github_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'), audit_model_fits=0)
    write_json(output/'control.json', report)
    write_json(output/'manifest.json', value)
    old = reference()
    start = time.monotonic()
    try:
        spec = old['cases'][index]
        directory = output/'cases'/spec['id']
        receipt = run_case(spec, directory, old, seconds=value['case_soft_seconds'])
        audit = audit_location(directory, spec, old)
        if audit['status'] != 'complete':
            raise ValueError('case did not complete its independent audit')
        report.update(status='COMPLETE_TIMING_ONLY', audit=audit, case_seconds=receipt['seconds'])
    except BaseException as exc:
        report.update(status='FAILED', failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        report['compute_audit_seconds'] = time.monotonic()-start
        write_json(output/'control.json', report)
        pack(output)
    return report


def checked_control(control, index, value):
    if (control.get('index') != index or control.get('timing_manifest') != design.digest(value)
        or control.get('github_run') != os.environ.get('GITHUB_RUN_ID')
        or control.get('github_sha') != os.environ.get('GITHUB_SHA')
        or control.get('github_attempt') != '1'
        or control.get('scientific_gate_accepted') is not False):
        raise ValueError('case control source/run/identity differs')


def aggregate_timing(inputs, output):
    value = checked_manifest()
    old = reference()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', timing_manifest=design.digest(value), timing_only=True,
        scientific_gate_accepted=False, original_preflight_reclassified=False,
        audit_model_fits=0, cases=[dict(index=i, spec=old['cases'][i], status='missing',
            fit_cost_unknown=True) for i in INDICES], reused_chains={})
    write_json(output/'aggregate.json', report)
    try:
        controls = {}
        for path in sorted(Path(inputs).rglob('control.json')):
            control = load_json(path)
            index = control.get('index')
            if type(index) is not int or index not in INDICES:
                raise ValueError('foreign case control')
            if index in controls:
                raise ValueError('duplicate case control')
            checked_control(control,index,value)
            controls[index] = path
        for row in report['cases']:
            index = row['index']
            write_json(output/'aggregate.json', report)
            if index not in controls:
                continue
            path = controls[index]
            control = load_json(path)
            row['control'] = control
            archive = path.parent/'raw/case.zip'
            if not archive.exists():
                row['status'] = 'missing_archive'
                continue
            try:
                row['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
                with zipfile.ZipFile(archive) as z:
                    if z.testzip() is not None or {n.split('/')[1] for n in z.namelist()} != {row['spec']['id']}:
                        raise ValueError('archive identity/CRC differs')
                store = Store(archive,row['spec']['id'])
                try:
                    row['started_assessments'] = sum(n.endswith('.started.json') for n in store.names)
                    row['finished_assessments'] = sum(n.endswith('.finished.json') for n in store.names)
                    receipt = store.json('receipt.json')
                    if any(receipt.get(k) != control.get(k) for k in ('github_run','github_sha','github_attempt')):
                        raise ValueError('receipt source/run differs from control')
                    audited = audit_case(row['spec'],store,old)
                finally:
                    store.close()
                if audited['status'] == 'complete':
                    if control.get('status') != 'COMPLETE_TIMING_ONLY' or control.get('audit') != audited:
                        raise ValueError('case control and independent audit differ')
                row.update(status=audited['status'], audit=audited,
                    fit_cost_unknown=audited['status'] != 'complete')
            except Exception as exc:
                row.update(status='audit_failed',fit_cost_unknown=True,
                    failure=type(exc).__name__+': '+str(exc)[:2000])
        report['reused_chains'] = audit_saved_chains(value)
        report['status'] = ('AUDIT_FAILED' if any(r['status'] == 'audit_failed' for r in report['cases']) else
            'COMPLETE_TIMING_ONLY' if all(r['status'] == 'complete' for r in report['cases']) else
            'INCOMPLETE_TIMING_ONLY')
    except BaseException as exc:
        report.update(status='AUDIT_FAILED', failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        write_json(output/'aggregate.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','check','run','pack','aggregate'))
    parser.add_argument('--index',type=int)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--inputs',type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        write_json(MANIFEST,manifest())
    elif args.command == 'check':
        print('PASS',design.digest(checked_manifest()),'frozen scenario timing inputs/runtime')
    elif args.command == 'run':
        run_case_timing(args.index,args.out)
    elif args.command == 'pack':
        pack(args.out)
    else:
        result = aggregate_timing(args.inputs,args.out)
        if result['status'] != 'COMPLETE_TIMING_ONLY':
            raise SystemExit('Incomplete timing; all identities retained, no scientific qualification.')
