"""Native full-shard timing, lossless transport and zero-fit-resume evidence."""
import sys,time,json,subprocess,datetime,shutil
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'examples/realistic_interaction_study.py').is_file())
sys.path.insert(0,str(ROOT/'examples'))
import realistic_interaction_study as s
import realistic_interaction_report as r
import prospective_interaction_archive as archive


def main():
    out=ROOT/'.superpowers/realistic-preflight-reproduction';out.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'.superpowers/realistic-native-044';package=out/'raw-044';combined=ROOT/'.superpowers/realistic-preflight-combined'
    s.freeze();tasks=s.shard_tasks('calibration','044')
    if raw.exists():raise ValueError('preflight path already exists; no automatic retry')
    before=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
    s.run_shard('calibration','044',raw);elapsed=time.monotonic()-tick
    costs=[s.attempt_cost(c,raw/c['id']) for c in tasks]
    assert all(c['fits']==692 and c['verification_refits']==20 and not c['unknown_fit_cost'] and not c['unknown_verification_cost'] for c in costs)
    stamp_before={c['id']:archive.digest(raw/c['id']/'verified.json') for c in tasks}
    tick=time.monotonic();s.run_shard('calibration','044',raw);resume=time.monotonic()-tick
    assert costs==[s.attempt_cost(c,raw/c['id']) for c in tasks]
    assert stamp_before=={c['id']:archive.digest(raw/c['id']/'verified.json') for c in tasks}
    tick=time.monotonic();receipt=archive.pack(raw,package);packing=time.monotonic()-tick
    tick=time.monotonic();audit=r.aggregate('calibration',[package],combined);aggregation=time.monotonic()-tick
    assert audit['complete']==80 and audit['missing']==3520 and not audit['audit_errors'] and audit['standalone']['checked']==80
    assert audit['known_attempt_cost']['fits']==55360 and audit['known_attempt_cost']['verification_refits']==1600
    assert audit['warning_count']==0
    disk_calibration=2*45*receipt['bytes']+2*receipt['raw_bytes'];disk_development=2*135*receipt['bytes']+2*receipt['raw_bytes']
    proof=dict(stage='calibration',shard='044',started_at=before,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        manifest=s.identity(),environment=s.io.environment(),planned_reused_ids=[c['id'] for c in tasks],
        tasks=80,workers=2,seconds=elapsed,computation_cap_seconds=720,resume_seconds=resume,resume_fits=0,resume_refits=0,
        native_fits=55360,native_saved_setting_refits=1600,
        native_scope='extra native computation on planned IDs; never additional independent replicates',
        test_scope='separate pytest computation; historical test model calls uninstrumented, not included in Actions cap',
        packing_seconds=packing,raw_files=receipt['files'],raw_bytes=receipt['raw_bytes'],compressed_bytes=receipt['bytes'],raw_sha256=receipt['sha256'],
        streamed_audit_seconds=aggregation,stage_calibration_estimated_audit_seconds=45*aggregation,
        stage_development_estimated_audit_seconds=135*aggregation,
        temporary_extraction_bytes=receipt['raw_bytes'],estimated_calibration_peak_disk_bytes=disk_calibration,
        estimated_development_peak_disk_bytes=disk_development,
        estimates_scope='linear conservative extrapolation from N600 calibration shard; signal timing/size rechecked after calibration',
        standalone=audit['standalone'],audit_errors=audit['audit_errors'],
        gate_pass=bool(elapsed<=720 and 135*aggregation<=3600 and disk_development<12*1024**3))
    s.io.write(out/'preflight.json',proof)
    for name in ('audit.json','dataset_metrics.csv','report.md'):shutil.copy2(combined/name,out/name)
    shutil.copy2(__file__,out/'reproduce_preflight.py')
    print(json.dumps(proof,indent=2))
    if not proof['gate_pass']:raise ValueError('approved preflight gate failed; revision required')


if __name__=='__main__':main()
