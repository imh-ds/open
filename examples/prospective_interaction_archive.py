"""Keep raw shards compressed and audit at most one extracted shard at once."""
from pathlib import Path
import tempfile,zipfile,hashlib,json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def pack(raw,output):
    raw=Path(raw);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'raw-shard.zip';inventory=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(raw.rglob('*')):
            if not p.is_file():continue
            name=p.relative_to(raw).as_posix();z.write(p,name)
            inventory.append(dict(path=name,bytes=p.stat().st_size,sha256=digest(p)))
    receipt=dict(schema='open.prospective_raw_shard.v1',files=len(inventory),raw_bytes=sum(p['bytes'] for p in inventory),
        bytes=target.stat().st_size,sha256=digest(target),inventory=inventory)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt


def archives(inputs):return sorted(p for root in inputs for p in Path(root).rglob('raw-shard.zip'))


def unique_archives(paths):
    hashes={};unique=[];duplicates=0;receipts=[]
    for p in paths:
        r=json.loads((p.parent/'receipt.json').read_text(encoding='utf-8'))
        if r.get('schema')!='open.prospective_raw_shard.v1' or r['bytes']!=p.stat().st_size or r['sha256']!=digest(p):
            raise ValueError('raw shard receipt differs')
        if r['sha256'] in hashes:
            if hashes[r['sha256']]!=r:raise ValueError('duplicate archive inventory differs')
            duplicates+=1
        else:hashes[r['sha256']]=r;unique.append(p)
        receipts.append(dict(path=str(p),sha256=r['sha256'],bytes=r['bytes']))
    return unique,dict(identical_archive_duplicates=duplicates,archives=receipts)


def extract_checked(path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate archive paths')
        for name in z.namelist():
            target=(output/name).resolve()
            if not target.is_relative_to(output.resolve()):raise ValueError('escaping archive path')
            if name.endswith('/'):continue
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    receipt=Path(path).parent/'receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text(encoding='utf-8'))
        inventory=r['inventory'];files=[p for p in output.rglob('*') if p.is_file()]
        if len(files)!=r['files'] or {p.relative_to(output).as_posix() for p in files}!={v['path'] for v in inventory}:
            raise ValueError('archive inventory coverage differs')
        for item in inventory:
            p=output/item['path']
            if p.stat().st_size!=item['bytes'] or digest(p)!=item['sha256']:raise ValueError('raw file digest differs')


def aggregate_archives(stage,inputs,output,manifest,selection=None):
    import prospective_interaction_report as report
    study=report.study;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths,merge=unique_archives(archives(inputs));rows=[];pairs=[];children=[];seen=set()
    for p in paths:
        # tempfile owns this narrowly bounded workspace; original compressed evidence stays untouched.
        with tempfile.TemporaryDirectory(prefix='prospective-shard-',dir=output) as scratch:
            scratch=Path(scratch);extract_checked(p,scratch)
            stamps=list(scratch.glob('shard-*.json'))
            if len(stamps)!=1:raise ValueError('one frozen shard stamp required')
            stamp=study.io.read(stamps[0]);subset=study.shard_tasks(stage,stamp['id'])
            if seen&{c['id'] for c in subset}:raise ValueError('conflicting duplicate shard identity')
            seen.update(c['id'] for c in subset)
            a=report.aggregate_leaf(stage,[scratch],scratch,manifest,selection,subset=subset);children.append(a)
            rows.extend(study.io.read(scratch/'dataset-metrics.json'))
            if a['complete']:pairs.extend(study.pd.read_csv(scratch/'case_metrics.csv').to_dict('records'))
    planned=study.d.stage_tasks(stage,selection);complete={r['id'] for r in rows}
    if len(complete)!=len(rows):raise ValueError('duplicate dataset metrics')
    errors=[r for a in children for r in a['audit_errors']];strata,passed=report.strata_summary(stage,rows,planned)
    if errors:
        passed=[]
        for r in strata:
            if stage=='null':r.update(status='audit_error',explanation='unknown_audit_error',**{'pass':False})
            else:
                for key in ('detection','control','shape'):r[key].update(status='audit_error',explanation='unknown_audit_error',**{'pass':False})
    conditions=[]
    if stage=='signal':
        nulls=study.io.read(study.STAGE_A)['strata']
        for r in strata:
            gates=[next(z for z in nulls if z['cell']==r['cell'] and z['benchmark']==s) for s in ('noise','additive')]
            conditions.append(dict(cell=r['cell'],benchmark=r['benchmark'],**report.rules.qualify_condition(gates,r)))
    skipped=[] if stage=='null' else selection['skipped_ids'];missing=[c['id'] for c in planned if c['id'] not in complete]
    total=lambda key:sum(a[key] for a in children)
    audit=dict(stage=stage,planned=len(planned),complete=len(complete),missing=len(missing),missing_ids=missing,
        skipped_ids=skipped,universe_signal_planned=10800,audit_errors=errors,
        failed_tasks=[r for a in children for r in a['failed_tasks']],historical_failed_tasks=[r for a in children for r in a['historical_failed_tasks']],
        unknown_attempt_cost_ids=sorted({r for a in children for r in a['unknown_attempt_cost_ids']}),
        known_attempt_cost={key:sum(a['known_attempt_cost'][key] for a in children) for key in ('fits','verification_refits','seconds')},
        checkpoint_fits=total('checkpoint_fits'),retained_fitting_seconds=total('retained_fitting_seconds'),
        fits=total('fits'),saved_setting_refits=total('saved_setting_refits'),warning_count=total('warning_count'),
        manifest=study.identity(),thresholds_sha256=study.old.threshold_digest(),selection=study.io.fingerprint(selection) if selection else None,
        merge=merge,strata=strata,conditions=conditions,passed_cells=passed,production_accepted=False,
        cost_scope='Attempt/checkpoint/effective totals overlap. Interrupted costs remain unknown even after completion.',
        **report.global_flags(conditions,len(missing),skipped,errors))
    study.io.write(output/'audit.json',audit);study.io.write(output/'dataset-metrics.json',rows)
    if stage=='null' and not missing and not errors:study.io.write(output/'proposed-selection.json',study.d.make_selection(dict(identity=study.identity()),audit))
    study.pd.DataFrame(pairs).to_csv(output/'case_metrics.csv',index=False)
    study.pd.json_normalize(strata).to_csv(output/'stratum_metrics.csv',index=False)
    study.pd.DataFrame(conditions).to_csv(output/'condition_metrics.csv',index=False)
    report.write_report(audit,strata,output);return audit
