"""Audited counts and scoped interpretation; never pool away failed conditions."""
from pathlib import Path
import shutil
import prospective_interaction_study as study
import prospective_interaction_rules as rules


def merge_inputs(inputs,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);duplicates=0;files=0
    for root in inputs:
        root=Path(root)
        if root.resolve()==output.resolve():continue
        for source in sorted(root.rglob('*')):
            if not source.is_file():continue
            relative=source.relative_to(root);target=output/relative
            if target.exists():
                if study.old.digest(target)!=study.old.digest(source):raise ValueError('conflicting evidence: '+str(relative))
                duplicates+=1
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);files+=1
    return dict(copied_files=files,identical_duplicates=duplicates)


def strata_summary(stage,rows,planned):
    groups={}
    for c in planned:groups.setdefault((c['cell'],c['benchmark']),[]).append(c)
    observed={}
    for r in rows:observed.setdefault((r['cell'],r['benchmark']),[]).append(r)
    result=[]
    for (cell,benchmark),tasks in groups.items():
        values=observed.get((cell,benchmark),[])
        if stage=='null':item=rules.summarize_stratum(values,tasks,'false')
        else:item={name:rules.summarize_stratum(values,tasks,endpoint) for name,endpoint in
                   (('control','false'),('detection','detection'),('shape','shape'))}
        result.append(dict(cell=cell,benchmark=benchmark,**item))
    passed=[]
    if stage=='null':
        for cell in study.d.cell_ids():
            values=[r for r in result if r['cell']==cell]
            if len(values)==2 and all(r['pass'] for r in values):passed.append(cell)
    return result,passed


def global_flags(conditions,missing,skipped,errors):
    eligible=not missing and not skipped and not errors and len(conditions)==24
    prediction=bool(eligible and all(c['predictive_support'] for c in conditions))
    pattern=bool(prediction and all(c['curve_pattern_support'] for c in conditions))
    return dict(global_predictive_support=prediction,global_curve_pattern_support=pattern,global_support=pattern)


def aggregate(stage,input_paths,output,manifest,selection=None):
    import prospective_interaction_archive as archive
    if archive.archives(input_paths):return archive.aggregate_archives(stage,input_paths,output,manifest,selection)
    return aggregate_leaf(stage,input_paths,output,manifest,selection)


def aggregate_leaf(stage,input_paths,output,manifest,selection=None,subset=None):
    output=Path(output);merge=merge_inputs(input_paths,output)
    if study.io.fingerprint(manifest)!=study.identity():raise ValueError('aggregate manifest differs')
    if stage=='signal' and selection!=study.selection():raise ValueError('aggregate selection differs')
    universe=study.d.stage_tasks(stage,selection)
    planned=universe if subset is None else subset
    lookup={c['id']:c for c in universe}
    if len({c['id'] for c in planned})!=len(planned) or any(lookup.get(c['id'])!=c for c in planned):raise ValueError('foreign audit subset')
    ids={c['id'] for c in planned}
    rows=[];pair_rows=[];complete=[];errors=[];failures=[];unknown=[];cost=0;seconds=0.;warning_count=0
    attempts=dict(fits=0,verification_refits=0,seconds=0.)
    for c in planned:
        folder=output/c['id']
        if not folder.exists():continue
        try:
            a=study.attempt_cost(c,folder)
            for key in attempts:attempts[key]+=a[key]
            if a['unknown_fit_cost'] or a['unknown_verification_cost']:unknown.append(c['id'])
        except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if (folder/'receipt.json').exists():
            try:
                expected=dict(task=c,context=study.context(c),environment=study.io.environment())
                if study.io.read(folder/'receipt.json')!=expected:raise ValueError('partial receipt context differs')
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if (folder/'error.json').exists():
            try:
                f=study.io.read(folder/'error.json')
                if f['task']!=c or f['context']!=study.context(c) or any(type(a) is not str for a in f['attempts']):
                    raise ValueError('failure context/history differs')
                failures.append(dict(id=c['id'],failure=f))
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if (folder/'result.json').exists():
            try:
                v=study.audit_task(c,folder)
                if study.io.read(folder/'verified.json')!=study.verification(c,folder):raise ValueError('verification stamp differs')
                metrics=study.result_rows(v);pair_rows.extend(metrics);complete.append(c['id'])
                true,control=metrics;displayed=bool(true['above_threshold'] and true['pattern_status']=='assessed' and true['pattern_consistent'])
                rows.append(dict(id=c['id'],cell=c['cell'],benchmark=c['benchmark'],
                    control=bool(any(r['above_threshold'] for r in metrics) if stage=='null' else control['above_threshold']),
                    detected=bool(true['above_threshold']),displayed=displayed,
                    shape_correct=bool(displayed and true['truth_cosine'] is not None and true['truth_cosine']>=.8)))
                cost+=692;warning_count+=len(v['warnings']);seconds+=study.io.read(folder/'timing.json')['retained_fitting_seconds']
            except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
        if c['id'] not in complete:
            if (folder/'receipt.json').exists() or (folder/'error.json').exists() or any(folder.glob('comparison-*.json')):unknown.append(c['id'])
            data=study.old.comparisons(c)
            for i in range(4):
                p=folder/f'comparison-{i}.json'
                if p.exists():
                    try:
                        s=study.io.read(p);study.check_checkpoint(c,s,data[i]);cost+=173;seconds+=s['fitting_seconds']
                    except Exception as e:errors.append(dict(id=c['id'],checkpoint=p.name,error=repr(e)))
    for p in output.glob('prospective--*'):
        if p.is_dir() and p.name not in ids:errors.append(dict(id=p.name,error='foreign task identity'))
    for p in output.glob('shard-*.json'):
        try:
            v=study.io.read(p);i=v['id']
            expected=dict(id=i,stage=stage,tasks=study.shard_tasks(stage,i),manifest=study.identity(),
                selection=study.io.fingerprint(selection) if stage=='signal' else None)
            if p.name!=f'shard-{stage}-{i}.json' or v!=expected:raise ValueError('foreign shard stamp')
        except Exception as e:errors.append(dict(id=p.name,error=repr(e)))
    strata,passed=strata_summary(stage,rows,planned)
    if errors:
        passed=[]
        for r in strata:
            if stage=='null':r.update(status='audit_error',**{'pass':False})
            else:
                for key in ('detection','control','shape'):r[key].update(status='audit_error',**{'pass':False})
    conditions=[]
    if stage=='signal':
        nulls=study.io.read(study.STAGE_A)['strata']
        for r in strata:
            gates=[next(z for z in nulls if z['cell']==r['cell'] and z['benchmark']==s) for s in ('noise','additive')]
            conditions.append(dict(cell=r['cell'],benchmark=r['benchmark'],**rules.qualify_condition(gates,r)))
    skipped=[] if stage=='null' else selection['skipped_ids']
    audit=dict(stage=stage,planned=len(planned),complete=len(complete),missing=len(planned)-len(complete),
        missing_ids=[c['id'] for c in planned if c['id'] not in complete],skipped_ids=skipped,universe_signal_planned=10800,
        audit_errors=errors,failed_tasks=[f for f in failures if f['id'] not in complete],
        historical_failed_tasks=[f for f in failures if f['id'] in complete],unknown_attempt_cost_ids=sorted(set(unknown)),
        known_attempt_cost=attempts,
        checkpoint_fits=cost,retained_fitting_seconds=seconds,fits=692*len(complete),saved_setting_refits=20*len(complete),
        cost_scope='Complete totals overlap checkpoint totals; failed/interrupted attempt work may be additional unknown cost.',
        warning_count=warning_count,manifest=study.identity(),thresholds_sha256=study.old.threshold_digest(),
        selection=study.io.fingerprint(selection) if selection else None,merge=merge,strata=strata,conditions=conditions,
        passed_cells=passed,**global_flags(conditions,len(planned)-len(complete),skipped,errors),production_accepted=False)
    study.io.write(output/'audit.json',audit)
    if stage=='null' and subset is None and not audit['missing'] and not errors:
        study.io.write(output/'proposed-selection.json',study.d.make_selection(dict(identity=study.identity()),audit))
    study.io.write(output/'dataset-metrics.json',rows)
    study.pd.DataFrame(pair_rows).to_csv(output/'case_metrics.csv',index=False)
    study.pd.json_normalize(strata).to_csv(output/'stratum_metrics.csv',index=False)
    study.pd.DataFrame(conditions).to_csv(output/'condition_metrics.csv',index=False)
    write_report(audit,strata,output);return audit


def write_report(audit,strata,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    text=['# Prospective interaction reliability', '',
        f"Stage {audit['stage']}: {audit['complete']}/{audit['planned']} planned datasets completed; {audit['missing']} missing; {len(audit['audit_errors'])} audit errors.",
        'Missing and errored results are unknown. No incomplete stratum receives a support label.', '',
        'This tests two fixed pairs with one frozen Gaussian spline method. It does not certify other engines, causal moderation or the full recursive network.',
        'Curve-pattern support checks relative direction/pattern at common supported locations, not magnitude, pointwise accuracy or the full surface.', '']
    if audit['stage']=='null':
        text += [f"Conditions passing both null-class gates: {len(audit['passed_cells'])}/6.",
            'Meaning: only these conditions may proceed to testing reliable interaction detection. A failing null gate is insufficient assurance about false alarms, not proof that interactions do not exist.', '',
            '| Condition | Null type | False alarms / completed | Upper bound | Status |', '|---|---|---:|---:|---|']
        for r in strata:
            b='unknown' if r['bound'] is None else f"{r['bound']:.4f}"
            text.append(f"| {r['cell']} | {r['benchmark']} | {r['k']}/{r['complete']} | {b} | {r['explanation']} |")
        text+=['', 'Observed-rate miss means the measured rate misses the requirement; it is descriptive. Uncertain despite an observed rate meeting the target means the bound is still too imprecise to qualify. Neither is a certificate of the underlying population rate.']
    else:
        text += [f"Skipped alternatives: {len(audit.get('skipped_ids',[]))}; these are unassessed, not failed detections.",
            'Predictive support requires low false alarms in both null classes and the alternative control pair, plus reliable true-pair detection.',
            'Curve support adds evidence that displayed patterns match the known relative pattern; report display frequency separately.', '',
            '| Condition | Benchmark | Detection | Displayed / planned | Predictive support | Pattern support |', '|---|---|---:|---:|---|---|']
        for r,c in zip(strata,audit.get('conditions',[])):
            text.append(f"| {r['cell']} | {r['benchmark']} | {r['detection']['k']}/{r['detection']['n']} ({r['detection']['explanation']}) | {r['shape']['n']}/{r['detection']['planned']} ({r['shape']['explanation']}) | {c['predictive_support']} | {c['curve_pattern_support']} |")
        text += ['', 'Lower/upper bounds crossing their targets mean uncertainty, not evidence that the effect is absent. Quantitative curve magnitude is not validated by this pattern criterion. No universal minimum N follows from these fixed conditions.']
    if strata:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,max(4,len(strata)*.3)))
        values=[r.get('bound') if audit['stage']=='null' else r['detection']['bound'] for r in strata]
        labels=[r['cell']+' / '+r['benchmark'] for r in strata]
        for i,v in enumerate(values):
            if v is not None:ax.plot(v,i,'o',color='#285f8f')
            else:ax.text(0,i,'unknown',va='center',fontsize=8)
        target=.05 if audit['stage']=='null' else .8
        ax.axvline(target,color='#a53e3e',linestyle='--');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
        ax.set_xlabel('Simultaneous exact upper false-alarm bound' if audit['stage']=='null' else 'Simultaneous exact lower detection bound')
        ax.set_xlim(0,1);fig.tight_layout();fig.savefig(output/'SUMMARY.png',dpi=150);plt.close(fig)
        text+=['','![Condition-specific bounds](SUMMARY.png)']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
