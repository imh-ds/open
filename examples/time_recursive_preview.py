"""Timing-only continuation: no per-case deadline, no refitting saved assessments."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import argparse
import hashlib
import os
import time
import zipfile

from open.layer_artifacts import build_payloads,decode,portable
from open.layer_types import LocalAssessment
from open.layer_stability import make_layer_sample_plans
from open.metadata import validate_metadata
import recursive_preview_design as design
from recursive_preview_actions import load_json,runtime_check,archive_cases,find_one
from recursive_preview_runner import write_json,write_gzip,fit_recorded,run_case,file_hashes
from recursive_preview_audit import Store,audit_case,audit_checkpoint

ROOT = design.ROOT
PRODUCER = ROOT/'examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38621370295'
MANIFEST = ROOT/'examples/study_manifests/recursive-preview-timing.json'


def original_manifest():
    return load_json(PRODUCER/'raw/producer-manifest.json')


def manifest():
    original = original_manifest()
    diagnosis = load_json(PRODUCER/'audit/diagnosis.json')
    hashes = load_json(PRODUCER/'audit/raw-sha256.json')
    paths = [ROOT/n for n in original['sources']]
    paths += [Path(__file__),ROOT/'.github/workflows/recursive-preview-timing.yml',
        ROOT/'examples/recursive_timing_protocol.md',PRODUCER/'raw/producer-manifest.json',
        PRODUCER/'audit/diagnosis.json',PRODUCER/'audit/raw-sha256.json']
    return dict(schema='open.recursive_timing.v1',case_indices=[160,240],case_seconds=None,
        workers=4,new_fixture_fits=0,job_minutes=82,registration_minutes=1,
        prior_allocated_runner_minutes=9917,total_allocated_runner_minutes=10000,
        runtime=original['runtime'],sources={p.relative_to(ROOT).as_posix():design.text_hash(p) for p in paths},
        producer=dict(run_id='38621370295',source=diagnosis['source'],manifest=design.digest(original),
            archive_sha256=hashes['recursive-preview-preflight-raw/preflight-all-attempts.zip'],
            receipt_sha256=diagnosis['partial_case']['receipt_sha256'],assessments=144),
        purpose='Timing only; original failed preflight and scientific denominators unchanged.')


def checked_manifest():
    value = load_json(MANIFEST)
    if value != manifest():
        raise ValueError('timing source/settings/producer manifest changed')
    original = original_manifest()
    if any(design.text_hash(ROOT/n) != h for n,h in original['sources'].items()):
        raise ValueError('original source/settings changed before timing probe')
    runtime_check(original)
    return value


def restore_partial(archive,directory,timing):
    proof = timing['producer']
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != proof['archive_sha256']:
        raise ValueError('original archive fingerprint differs')
    original = original_manifest()
    if design.digest(original) != proof['manifest']:
        raise ValueError('original manifest fingerprint differs')
    spec = original['cases'][160]
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or {n.split('/')[1] for n in z.namelist()} != {spec['id']}:
            raise ValueError('original archive CRC/case identity differs')
    store = Store(archive,spec['id'])
    try:
        if hashlib.sha256(store.read('receipt.json')).hexdigest() != proof['receipt_sha256']:
            raise ValueError('original receipt fingerprint differs')
        row = audit_case(spec,store,original)
        receipt = store.json('receipt.json')
        if (row['status'] != 'incomplete' or row['unknown_attempts'] != 0
            or receipt['failure'] != 'TimeoutError: case soft deadline; no new assessment batches'
            or receipt['github_run'] != proof['run_id'] or receipt['github_sha'] != proof['source']
            or receipt['github_attempt'] != '1'):
            raise ValueError('producer is not the exact clean checkpoint-boundary timeout')
        data,config = design.generate(spec),design.settings(spec)
        plans = make_layer_sample_plans(data,config)
        checks = []
        for name in store.names:
            if not name.endswith('.json.gz'):
                continue
            payload = store.compressed(name)
            assessment = decode(LocalAssessment,payload['assessment'])
            if assessment.sample != plans[assessment.sample.run_index]:
                raise ValueError('saved sampled/split plan changed')
            context = design.digest(dict(case=receipt['context'],depth=assessment.depth,
                target=assessment.target,candidates=assessment.candidates,config=config))
            checks.append(audit_checkpoint(payload,data.iloc[list(assessment.sample.sampled_positions)],
                                           assessment,context,config))
            for suffix,status in (('.started.json','STARTED'),('.finished.json','completed')):
                attempt = store.json(name[:-3]+suffix)
                if (attempt['context'],attempt['depth'],attempt['target'],attempt['run_index'],attempt['seed']) != (
                    context,assessment.depth,assessment.target,assessment.sample.run_index,assessment.sample.seed):
                    raise ValueError('attempt identity changed')
                if attempt['status'] != status or (status == 'completed' and attempt['fit_cost_unknown']):
                    raise ValueError('incomplete/unknown original assessment cannot be retried')
        if len(checks) != proof['assessments'] or row['started_assessments'] != len(checks) or row['finished_assessments'] != len(checks):
            raise ValueError('original assessment coverage changed')
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        for name in store.names:
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('unsafe original evidence path')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                stream.write(store.read(name))
        return receipt,dict(assessments=len(checks),observation_checks=sum(c['observation_checks'] for c in checks),
            outer_fits=sum(c['outer_fits'] for c in checks),tuning_fits=sum(c['tuning_fits'] for c in checks))
    finally:
        store.close()


def resume_case(directory,prior,timing,output):
    original = original_manifest()
    spec = original['cases'][160]
    data,config = design.generate(spec),design.settings(spec)
    write_json(output/'provenance/original-case160-receipt.json',prior,exclusive=True)
    receipt = dict(prior,status='STARTED',fit_cost_unknown=True,github_run=os.environ.get('GITHUB_RUN_ID'),
        github_sha=os.environ.get('GITHUB_SHA'),github_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
        timing_only=True,timing_manifest=design.digest(timing),producer_run=timing['producer']['run_id'],
        prior_seconds=prior['seconds'],reused_assessments=144)
    receipt.pop('failure',None)
    write_json(directory/'receipt.json',receipt)
    start = time.monotonic()
    try:
        network = fit_recorded(data,config,directory/'checkpoints',prior['context'],float('inf'))
        metadata = validate_metadata(tuple(data),[dict(name=n,type='continuous') for n in data],config.analysis.root)
        write_gzip(directory/'network.json.gz',build_payloads(network,metadata))
        if network.status != 'COMPLETE':
            raise RuntimeError(network.status)
        for name,expected in prior['files'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError('reused original evidence changed')
        receipt.update(status='complete',fit_cost_unknown=False)
    except BaseException as exc:
        receipt.update(status='incomplete',failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        receipt.update(continuation_seconds=time.monotonic()-start)
        receipt['seconds'] = prior['seconds']+receipt['continuation_seconds']
        receipt['files'] = file_hashes(directory)
        receipt['new_assessments'] = len(list((directory/'checkpoints').rglob('*.json.gz')))-144
        write_json(directory/'receipt.json',receipt)
    return receipt


def pack(output):
    archive = output/'raw/timing-cases.zip'
    if (output/'cases').exists() and not archive.exists():
        archive_cases(list((output/'cases').iterdir()),archive)


def split_fit_counts(total,reused):
    prior = {n:reused.get(n,0) for n in ('outer_fits','tuning_fits')}
    new = {n:total[n]-prior[n] for n in prior}
    if any(n < 0 for n in (*prior.values(),*new.values())):
        raise ValueError('fit count totals cannot be smaller than reused work')
    return dict(reused=prior,new=new)


def run(output,evidence):
    timing = checked_manifest()
    original = original_manifest()
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='STARTED',timing_manifest=design.digest(timing),timing_only=True,
        original_preflight_reclassified=False,scientific_gate_accepted=False,
        github_run=os.environ.get('GITHUB_RUN_ID'),github_sha=os.environ.get('GITHUB_SHA'),cases={})
    write_json(output/'timing.json',report)
    try:
        directory = output/'cases'/original['cases'][160]['id']
        prior,audit = restore_partial(find_one(evidence,'preflight-all-attempts.zip'),directory,timing)
        report['reuse_audit'] = audit
        write_json(output/'timing.json',report)
        for index in (160,240):
            spec = original['cases'][index]
            directory = output/'cases'/spec['id']
            if index == 160:
                receipt = resume_case(directory,prior,timing,output)
            else:
                receipt = run_case(spec,directory,original,seconds=float('inf'))
            store = Store(directory)
            try:
                checked = audit_case(spec,store,original)
            finally:
                store.close()
            report['cases'][str(index)] = dict(audit=checked,measured_cumulative_case_seconds=receipt['seconds'],
                prior_seconds=receipt.get('prior_seconds',0),
                continuation_seconds=receipt.get('continuation_seconds',receipt['seconds']),
                reused_assessments=receipt.get('reused_assessments',0),
                new_assessments=checked['assessments']-receipt.get('reused_assessments',0),
                fit_counts=split_fit_counts(checked,audit if index == 160 else {}))
            write_json(output/'timing.json',report)
        report['status'] = 'COMPLETE_TIMING_ONLY'
    except BaseException as exc:
        report.update(status='FAILED',failure=type(exc).__name__+': '+str(exc)[:2000])
        raise
    finally:
        write_json(output/'timing.json',report)
        pack(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run','pack','check'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--evidence',type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        write_json(MANIFEST,manifest())
    elif args.command == 'check':
        value = checked_manifest()
        print('PASS',design.digest(value),len(value['sources']),'frozen sources')
    elif args.command == 'pack':
        pack(args.out)
    else:
        run(args.out,args.evidence)
