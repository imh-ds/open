"""One approved, checkpoint-preserving continuation of 38 original null cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
import time
import zipfile

from open.layer_artifacts import build_payloads, decode, portable
from open.layer_types import LocalAssessment
from open.layer_stability import make_layer_sample_plans
from open.metadata import validate_metadata
import qualify_recursive_validation as q
import recursive_preview_design as design
from recursive_preview_actions import load_json, github_api
from recursive_preview_runner import write_json, write_gzip, fit_recorded, file_hashes
from recursive_preview_audit import Store, audit_case, audit_checkpoint
from time_scenario_validation import pack

ROOT = design.ROOT
PRODUCER_RUN = '38655036215'
PRODUCER_SHA = '157c526f6fe6c5198ff3c2bfa97f15df4b0d136e'
PROOF = 'examples/results/2026-10-14-recursive-qualification/continuation-plan/producer-audit.json'
WORKFLOW = '.github/workflows/qualification-continuation.yml'
MANIFEST = ROOT/'examples/study_manifests/qualification-continuation.json'


def producer():
    return load_json(ROOT/PROOF)


def original():
    return load_json(q.MANIFEST)


def partition():
    report = producer()
    rows = report['cases']
    planned = q.stage_specs('noise')
    if (len(rows) != 110 or len({r['spec']['index'] for r in rows}) != 110
        or sorted((r['spec'] for r in rows), key=lambda s:s['index']) != planned
        or report['github_run'] != PRODUCER_RUN or report['github_sha'] != PRODUCER_SHA
        or report['github_attempt'] != '1' or report['manifest'] != design.digest(original())
        or sum(r['status'] == 'complete' for r in rows) != 72
        or sum(r['status'] == 'incomplete' and r['spec']['n'] == 80 for r in rows) != 38):
        raise ValueError('producer partition/provenance differs')
    return rows


def pending():
    return [r['spec'] for r in partition() if r['status'] == 'incomplete']


def completed():
    return [r['spec'] for r in partition() if r['status'] == 'complete']


def manifest():
    paths = set(original()['sources']) | {
        'examples/study_manifests/recursive-qualification.json', PROOF, WORKFLOW,
        'examples/continue_qualification.py', 'examples/qualification_continuation_protocol.md'}
    return dict(schema='open.qualification_continuation.v1', original_manifest=design.digest(original()),
        producer_run=PRODUCER_RUN, producer_sha=PRODUCER_SHA, pending=pending(), completed=completed(),
        sources={p:design.text_hash(ROOT/p) for p in sorted(paths)},
        job_minutes=60, compute_minutes=48, soft_seconds=2340, max_parallel=20, workers=4,
        prior_reserved=2507, continuation_minutes=2356, reserved_total=4863,
        remaining_ceiling=5137, ceiling=10000, signal_funded=False)


def checked_manifest():
    q.checked_manifest()
    m = load_json(MANIFEST)
    if m != manifest():
        raise ValueError('continuation source/identity/budget manifest changed')
    return m


def verify_once(current_run, *, api=None):
    api = github_api if api is None else api
    page = 1
    while True:
        runs = api(f'actions/workflows/qualification-continuation.yml/runs?event=workflow_dispatch&per_page=100&page={page}')['workflow_runs']
        if any(str(r['id']) != str(current_run) for r in runs):
            raise ValueError('previous continuation dispatch exists; no automatic retry')
        if len(runs) < 100:
            return
        page += 1


def transport(run_id, source, specs, *, original_run=False, api=None):
    api = github_api if api is None else api
    run = api(f'actions/runs/{run_id}')
    path = '.github/workflows/recursive-qualification.yml' if original_run else WORKFLOW
    if (str(run['id']) != str(run_id) or run['head_sha'] != source or run['run_attempt'] != 1
        or run['event'] != 'workflow_dispatch' or run['path'] != path
        or original_run and (run['status'] != 'completed' or run['conclusion'] != 'failure')):
        raise ValueError('run transport provenance differs')
    jobs, page = [], 1
    while True:
        batch = api(f'actions/runs/{run_id}/jobs?per_page=100&page={page}')['jobs']
        jobs += batch
        if len(batch) < 100:
            break
        page += 1
    expected = {f"case ({s['id']})" for s in specs}
    found = [j for j in jobs if j['name'] in expected]
    if len(found) != len(expected) or {j['name'] for j in found} != expected:
        raise ValueError('required completed case transport missing/duplicate')
    for job in found:
        uploads = [s for s in job['steps'] if 'actions/upload-artifact@' in s['name']]
        if job['conclusion'] != 'success' or len(uploads) != 1 or uploads[0]['conclusion'] != 'success':
            raise ValueError('required case/upload transport failed')
    if not original_run and len([j for j in jobs if j['name'].startswith('case (')]) != 38:
        raise ValueError('unexpected continuation case jobs')
    return dict(run_id=str(run_id), source=source, verified=True, case_jobs=len(found))


def validate_archive(spec, archive, *, replay):
    row = next(r for r in partition() if r['spec'] == spec)
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != row['archive_sha256']:
        raise ValueError('producer archive fingerprint differs')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any(not n.startswith(f"cases/{spec['id']}/") for n in z.namelist()):
            raise ValueError('producer archive CRC/identity differs')
    store = Store(archive, spec['id'])
    try:
        receipt = store.json('receipt.json')
        if (hashlib.sha256(store.read('receipt.json')).hexdigest() != row['receipt_sha256']
            or receipt['github_run'] != PRODUCER_RUN or receipt['github_sha'] != PRODUCER_SHA
            or receipt['github_attempt'] != '1'):
            raise ValueError('producer receipt provenance differs')
        checked = audit_case(spec, store, original())
        if checked['status'] != row['status']:
            raise ValueError('producer status differs')
        if row['status'] == 'complete':
            # audit_case above independently replays the whole completed network.
            return dict(checked, original_status='complete')
        if not replay:
            return dict(checked, original_status='incomplete')
        data, config = design.generate(spec), design.settings(spec)
        plans = make_layer_sample_plans(data, config)
        checks, seen = [], set()
        for name in store.names:
            if not name.startswith('checkpoints/') or not name.endswith('.json.gz'):
                continue
            payload = store.compressed(name)
            assessment = decode(LocalAssessment, payload['assessment'])
            if assessment.sample != plans[assessment.sample.run_index]:
                raise ValueError('producer sampled/split identity differs')
            expected_path = f'checkpoints/d{assessment.depth}-{assessment.target}/{assessment.sample.run_index:03}.json.gz'
            if name != expected_path or name in seen:
                raise ValueError('producer checkpoint identity differs')
            seen.add(name)
            context = design.digest(dict(case=receipt['context'], depth=assessment.depth,
                target=assessment.target, candidates=assessment.candidates, config=config))
            checks.append(audit_checkpoint(payload, data.iloc[list(assessment.sample.sampled_positions)],
                                           assessment, context, config))
            for suffix, status in (('.started.json','STARTED'), ('.finished.json','completed')):
                attempt = store.json(name[:-3]+suffix)
                if ((attempt['context'], attempt['depth'], attempt['target'], attempt['run_index'], attempt['seed'])
                    != (context, assessment.depth, assessment.target, assessment.sample.run_index, assessment.sample.seed)
                    or attempt['status'] != status or status == 'completed' and attempt['fit_cost_unknown']):
                    raise ValueError('producer attempt failed/unknown/changed')
        if (len(checks) != checked['started_assessments'] or len(checks) != checked['finished_assessments']
            or checked['unknown_attempts'] != 0):
            raise ValueError('unmatched producer assessment attempts; cannot refit')
        return dict(checked, original_status='incomplete', assessments=len(checks),
                    observation_checks=sum(c['observation_checks'] for c in checks),
                    outer_fits=sum(c['outer_fits'] for c in checks),
                    tuning_fits=sum(c['tuning_fits'] for c in checks))
    finally:
        store.close()


def restore(spec, archive, directory):
    checked = validate_archive(spec, archive, replay=True)
    store = Store(archive, spec['id'])
    try:
        directory.mkdir(parents=True, exist_ok=False)
        for name in store.names:
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('unsafe producer path')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(store.read(name))
        return store.json('receipt.json'), checked
    finally:
        store.close()


def resume(spec, directory, prior, reused, value, output):
    write_json(output/'provenance/original-receipt.json', prior, exclusive=True)
    receipt = dict(prior, status='STARTED', fit_cost_unknown=True, producer_run=PRODUCER_RUN,
        continuation_manifest=design.digest(value), github_run=os.environ['GITHUB_RUN_ID'],
        github_sha=os.environ['GITHUB_SHA'], github_attempt='1', reused_assessments=reused['assessments'],
        prior_seconds=prior['seconds'])
    receipt.pop('failure', None)
    write_json(directory/'receipt.json', receipt)
    data, config = design.generate(spec), design.settings(spec)
    start = time.monotonic()
    try:
        if (directory/'network.json.gz').exists():
            # A late producer network can already be scientifically complete.
            # Audit it in place; reconstruction would change runtime provenance.
            receipt.update(status='complete', fit_cost_unknown=False)
            write_json(directory/'receipt.json', receipt)
            store = Store(directory)
            try:
                checked = audit_case(spec, store, original())
                if checked['status'] != 'complete':
                    raise ValueError('saved late network is incomplete')
            finally:
                store.close()
            receipt['late_network_reused'] = True
        else:
            network = fit_recorded(data, config, directory/'checkpoints', prior['context'],
                                   start+value['soft_seconds'], workers=4)
            metadata = validate_metadata(tuple(data), [dict(name=n,type='continuous') for n in data], config.analysis.root)
            write_gzip(directory/'network.json.gz', build_payloads(network, metadata))
            if network.status != 'COMPLETE':
                raise RuntimeError(network.status)
        if time.monotonic()-start > value['soft_seconds']:
            raise TimeoutError('completed after continuation computation allocation')
        receipt.update(status='complete', fit_cost_unknown=False)
    except BaseException as exc:
        receipt.update(status='incomplete', failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        unchanged = all(hashlib.sha256((directory/n).read_bytes()).hexdigest() == h
                        for n,h in prior['files'].items())
        receipt.update(continuation_seconds=time.monotonic()-start, prior_files_unchanged=unchanged)
        receipt['seconds'] = prior['seconds']+receipt['continuation_seconds']
        receipt['files'] = file_hashes(directory)
        if not unchanged:
            receipt.update(status='incomplete', fit_cost_unknown=True, failure='reused producer evidence changed')
        write_json(directory/'receipt.json', receipt)
        if not unchanged:
            raise ValueError('reused producer evidence changed')
    return receipt


def prepare(output):
    value = checked_manifest()
    q.first_attempt()
    verify_once(os.environ['GITHUB_RUN_ID'])
    old_transport = transport(PRODUCER_RUN, PRODUCER_SHA, completed(), original_run=True)
    output.mkdir(parents=True, exist_ok=False)
    auth = dict(schema='open.qualification_continuation.authorization.v1',
        manifest=design.digest(value), github_run=os.environ['GITHUB_RUN_ID'],
        github_sha=os.environ['GITHUB_SHA'], github_attempt='1', indices=[s['index'] for s in pending()],
        producer_transport=old_transport)
    write_json(output/'authorization.json', auth)
    matrix = dict(include=[dict(index=s['index'],id=s['id']) for s in pending()])
    write_json(output/'matrix.json', matrix)
    write_json(output/'manifest.json', value)
    return matrix


def authorization(path, value):
    auth = load_json(path)
    if (auth.get('schema') != 'open.qualification_continuation.authorization.v1'
        or auth.get('manifest') != design.digest(value)
        or auth.get('github_run') != os.environ['GITHUB_RUN_ID']
        or auth.get('github_sha') != os.environ['GITHUB_SHA'] or auth.get('github_attempt') != '1'
        or auth.get('indices') != [s['index'] for s in pending()]):
        raise ValueError('continuation authorization differs')


def run_one(index, archive, output, auth):
    selected = [s for s in pending() if type(index) is int and s['index'] == index]
    if len(selected) != 1:
        raise ValueError('identity outside the 38 approved incomplete cases')
    value = checked_manifest()
    q.first_attempt()
    authorization(auth, value)
    spec = selected[0]
    output.mkdir(parents=True, exist_ok=False)
    control = dict(index=index, spec=spec, manifest=design.digest(value), status='STARTED',
        github_run=os.environ['GITHUB_RUN_ID'], github_sha=os.environ['GITHUB_SHA'], github_attempt='1')
    write_json(output/'control.json', control)
    try:
        directory = output/'cases'/spec['id']
        prior, reused = restore(spec, archive, directory)
        write_json(output/'reuse-audit.json', reused)
        # Replay is deliberately before the computation clock, under its separate step budget.
        resume(spec, directory, prior, reused, value, output)
        control['status'] = 'complete'
    except BaseException as exc:
        control.update(status='FAILED', failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        write_json(output/'control.json', control)
        pack(output)


def combine(rows):
    seen = set()
    final = {r['spec']['index']:r for r in partition() if r['status'] == 'complete'}
    for row in rows:
        index = row['spec']['index']
        if index in seen:
            raise ValueError('duplicate aggregate case')
        seen.add(index)
        if row['spec'] not in q.stage_specs('noise'):
            raise ValueError('foreign aggregate case')
        if row['spec'] in pending() or all(row.get(k) == v for k,v in final.get(index, {}).items()):
            final[index] = row
        else:
            raise ValueError('completed producer case cannot be replaced')
    return q.summarize(list(final.values()), 'noise', original())


def audit_continued(args):
    spec, control_path, value = args
    store = None
    try:
        control = load_json(control_path)
        archive = control_path.parent/'raw/case.zip'
        if (control.get('manifest') != design.digest(value) or control.get('spec') != spec
            or control.get('github_run') != os.environ['GITHUB_RUN_ID']
            or control.get('github_sha') != os.environ['GITHUB_SHA'] or control.get('github_attempt') != '1'):
            raise ValueError('continuation control provenance differs')
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or any(not n.startswith(f"cases/{spec['id']}/") for n in z.namelist()):
                raise ValueError('continuation archive CRC/identity differs')
        store = Store(archive, spec['id'])
        receipt = store.json('receipt.json')
        prior = load_json(control_path.parent/'provenance/original-receipt.json')
        original_row = next(r for r in partition() if r['spec'] == spec)
        # write_json is canonical with a trailing newline, exactly as the producer receipt.
        prior_bytes = (design.json.dumps(portable(prior), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if hashlib.sha256(prior_bytes).hexdigest() != original_row['receipt_sha256']:
            raise ValueError('original continuation receipt fingerprint differs')
        if (receipt.get('continuation_manifest') != design.digest(value) or receipt.get('producer_run') != PRODUCER_RUN
            or any(receipt.get(k) != control[k] for k in ('github_run','github_sha','github_attempt'))
            or not receipt.get('prior_files_unchanged')
            or any(hashlib.sha256(store.read(n)).hexdigest() != h for n,h in prior['files'].items())):
            raise ValueError('continuation reuse/provenance differs')
        row = audit_case(spec, store, original())
        if row['status'] == 'complete' and control['status'] != 'complete':
            raise ValueError('failed continuation control')
        return dict(row, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    producer_run=PRODUCER_RUN, continuation_manifest=design.digest(value))
    except Exception as exc:
        return dict(spec=spec, status='audit_failed', structure=None, fit_cost_unknown=True,
                    failure=type(exc).__name__+': '+str(exc)[:2000])
    finally:
        if store:
            store.close()


def aggregate(inputs, producer_inputs, output):
    value = checked_manifest()
    q.first_attempt()
    output.mkdir(parents=True, exist_ok=False)
    report = combine([])
    report.update(producer_run=PRODUCER_RUN, original_run_still_failed=True,
                  continuation_manifest=design.digest(value), transport={}, integrity_errors=[])
    write_json(output/'audit.json', report)
    rows = []
    try:
        # Re-audit all 72 originals, never refit or substitute their networks.
        with ProcessPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(audit_original_task, [
                (s, str(next(producer_inputs.rglob(f'qualification-noise-{s["index"]}/**/case.zip'))))
                for s in completed()]))
        controls = {}
        for path in inputs.rglob('control.json'):
            index = load_json(path).get('index')
            if type(index) is not int or index not in {s['index'] for s in pending()} or index in controls:
                raise ValueError('duplicate/foreign continuation control')
            controls[index] = path
        with ProcessPoolExecutor(max_workers=4) as pool:
            rows += list(pool.map(audit_continued, [(s,controls[s['index']],value)
                         for s in pending() if s['index'] in controls]))
        report.update(combine(rows))
        report['transport'] = dict(
            producer=transport(PRODUCER_RUN, PRODUCER_SHA, completed(), original_run=True),
            continuation=transport(os.environ['GITHUB_RUN_ID'], os.environ['GITHUB_SHA'], pending()))
    except Exception as exc:
        report['integrity_errors'].append(type(exc).__name__+': '+str(exc)[:2000])
        report['qualified'] = False
    write_json(output/'audit.json', report)
    q.write_report(report, output)
    return report


def audit_original_task(args):
    spec, archive = args
    return dict(validate_archive(spec, archive, replay=True),
                archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze','check','prepare','case','aggregate','pack'))
    p.add_argument('--out', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--producer-inputs', type=Path)
    p.add_argument('--archive', type=Path)
    p.add_argument('--authorization', type=Path)
    p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.command == 'freeze':
        write_json(MANIFEST, manifest())
    elif a.command == 'check':
        print('PASS', design.digest(checked_manifest()))
    elif a.command == 'prepare':
        print(design.json.dumps(prepare(a.out), separators=(',', ':')))
    elif a.command == 'case':
        run_one(a.index, a.archive, a.out, a.authorization)
    elif a.command == 'aggregate':
        report = aggregate(a.inputs, a.producer_inputs, a.out)
        if not report['qualified']:
            raise SystemExit('Continuation did not qualify; evidence saved, no signal follow-on.')
    else:
        pack(a.out)
