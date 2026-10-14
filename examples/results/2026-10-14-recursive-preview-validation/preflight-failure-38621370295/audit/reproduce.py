"""No-fit diagnosis of the saved timeout; run from the repository root."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.path.insert(0,str(Path.cwd()/'examples'))
from open.layer_artifacts import decode, portable
from open.layer_types import LocalAssessment
from open.layer_consensus import aggregate_target_assessments
from open.layer_stability import make_layer_sample_plans
import recursive_preview_actions as actions
import recursive_preview_design as design
from recursive_preview_audit import Store, audit_case, audit_checkpoint

BASE = Path(__file__).resolve().parents[1]
RAW = BASE/'raw'

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    manifest = load(RAW/'producer-manifest.json')
    actions.runtime_check(manifest)
    control = load(next(RAW.rglob('preflight.json')))
    run = load(RAW/'run.json')
    jobs = load(RAW/'jobs.json')['jobs']
    assert run['conclusion'] == 'failure' and run['run_attempt'] == 1
    assert control['github_sha'] == run['head_sha']
    assert control['manifest'] == design.digest(manifest)
    assert control['status'] == 'FAILED' and control['planned_cases'] == {}
    assert control['fixture_new_fits'] == 0 and control['equivalence'] is True
    fixture = next(RAW.rglob('preflight-fixture.zip'))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest['continuation']['archive_sha256']
    archive = next(RAW.rglob('preflight-all-attempts.zip'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        identities = sorted({n.split('/')[1] for n in z.namelist()})
    spec = manifest['cases'][160]
    assert identities == [spec['id']]
    store = Store(archive,spec['id'])
    config, data = design.settings(spec), design.generate(spec)
    try:
        partial = audit_case(spec,store,manifest)
        receipt = store.json('receipt.json')
        assert receipt['status'] == 'incomplete' and 'TimeoutError' in receipt['failure']
        assert receipt['github_sha'] == run['head_sha'] and receipt['github_run'] == str(run['id'])
        plans = make_layer_sample_plans(data,config)
        details, assessments = [], {}
        for name in store.names:
            if not name.endswith('.json.gz'):
                continue
            payload = store.compressed(name)
            assessment = decode(LocalAssessment,payload['assessment'])
            assert assessment.sample == plans[assessment.sample.run_index]
            context = design.digest(dict(case=receipt['context'],depth=assessment.depth,
                target=assessment.target,candidates=assessment.candidates,config=config))
            seconds = payload['seconds']
            checked = audit_checkpoint(payload,data.iloc[list(assessment.sample.sampled_positions)],
                                       assessment,context,config)
            for suffix,status in (('.started.json','STARTED'),('.finished.json','completed')):
                attempt = store.json(name[:-3]+suffix)
                assert (attempt['context'],attempt['depth'],attempt['target'],attempt['run_index'],attempt['seed']) == (
                    context,assessment.depth,assessment.target,assessment.sample.run_index,assessment.sample.seed)
                assert attempt['status'] == status
                if status == 'completed':
                    assert attempt['fit_cost_unknown'] is False
                    assert attempt['outer_fits'] == checked['outer_fits']
                    assert attempt['tuning_fits'] == checked['tuning_fits']
            details.append(dict(depth=assessment.depth,target=assessment.target,
                index=assessment.sample.run_index,seconds=seconds,candidates=len(assessment.candidates),
                warnings=len(assessment.warnings),**checked))
            assessments.setdefault(assessment.target,[]).append(assessment)
        assert len(details) == partial['started_assessments'] == partial['finished_assessments'] == 144
        root = aggregate_target_assessments(tuple(assessments['Y']),planned_count=100,
                                            pi_min=config.thresholds.stability)
        layers = {}
        for target, values in assessments.items():
            rows = [d for d in details if d['target'] == target]
            median = statistics.median(d['seconds'] for d in rows)
            layers[target] = dict(depth=values[0].depth,completed=len(rows),planned=100,
                candidate_count=len(values[0].candidates),median_assessment_seconds=median,
                approximate_100_assessment_seconds_at_four_workers=median*25,
                outer_fits=sum(d['outer_fits'] for d in rows),
                tuning_fits=sum(d['tuning_fits'] for d in rows))
        preflight = next(j for j in jobs if j['name'] == 'preflight')
        uploads = [s for s in preflight['steps'] if 'actions/upload-artifact@' in s['name']]
        assert len(uploads) == 2 and all(s['conclusion'] == 'success' for s in uploads)
        result = dict(run_id=run['id'],source=run['head_sha'],manifest=design.digest(manifest),
            status='FAILED',failure=receipt['failure'],case_index=160,n=80,
            case_seconds=receipt['seconds'],case_limit_seconds=900,
            fixture_reused=True,fixture_new_fits=0,fixture_equivalence=True,
            fixture_completed_assessments=control['fixture_audit']['assessments'],
            fixture_archive_original_hash_verified=True,partial_case=partial,layers=layers,
            completed_assessment_audit=dict(assessments=len(details),
                observation_checks=sum(d['observation_checks'] for d in details),
                outer_fits=sum(d['outer_fits'] for d in details),
                tuning_fits=sum(d['tuning_fits'] for d in details),
                warnings=sum(d['warnings'] for d in details),unknown_started_attempts=0),
            root_relations=portable(root.relations),
            root_promoted=sorted({v for r in root.relations if r.qualifies for v in (r.members or (r.source,))}),
            completed_networks=0,case_240_started=False,both_uploads_success=True,
            audit_model_fits=0,
            timing_scope='Partial-case per-assessment medians; projected full layer times assume unchanged throughput, not measured completion.',
            scientific_gate_accepted=False)
        (BASE/'audit/diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (BASE/'audit/assessment-timing.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(details[0]))
            writer.writeheader(); writer.writerows(details)
    finally:
        store.close()
    hashes = {p.relative_to(RAW).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(RAW.rglob('*')) if p.is_file()}
    (BASE/'audit/raw-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('root_relations','partial_case')},indent=2))

if __name__ == '__main__':
    main()
