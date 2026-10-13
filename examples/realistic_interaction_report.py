"""Streamed raw audit, fresh calibration and descriptive D-043 reporting."""
from pathlib import Path
import tempfile,time
import numpy as np
from scipy.stats import beta
import realistic_interaction_study as study
import realistic_interaction_verify as verify
import prospective_interaction_archive as archive


def pair_decisions(v):
    c=v['task'];threshold=study.thresholds()['thresholds'][c['cell']] if c['stage']=='development' else 0.
    rows=[]
    for pos in (0,1):
        r=study.previous.diagnostic.decision(study.old.models.fold_records(v['comparisons'][:3],pos),threshold)
        value={k:r[k] for k in ('eligible','score','overall','matched','positive_folds')}
        if c['stage']=='development':value.update(selected=r['above_threshold'],threshold=threshold)
        rows.append(value)
    return rows


def learn_thresholds(rows):
    expected={c['id']:c for c in study.d.stage_tasks('calibration')}
    if len(rows)!=3600 or {r['id'] for r in rows}!=set(expected):raise ValueError('complete unique calibration identities required')
    result={}
    for row in rows:
        c=expected[row['id']]
        if row['cell']!=c['cell'] or row['benchmark']!=c['benchmark'] or len(row['pairs'])!=2:raise ValueError('foreign calibration row')
        for p in row['pairs']:
            if type(p['eligible']) is not bool or type(p['score']) not in (float,int) or not np.isfinite(p['score']) or p['score']<0 or (not p['eligible'] and p['score']!=0):raise ValueError('invalid score')
        result[c['cell']]=max(result.get(c['cell'],0.),*(p['score'] for p in row['pairs']))
    return dict(sorted(result.items()))


def interval(k,n):
    if type(k) is not int or type(n) is not int or not 0<=k<=n:raise ValueError('invalid counts')
    return dict(k=k,n=n,rate=k/n if n else None,
        low=(0. if k==0 else float(beta.ppf(.025,k,n-k+1))) if n else None,
        high=(1. if k==n else float(beta.ppf(.975,k+1,n-k))) if n else None,
        scope='two-sided 95% exact descriptive interval; unadjusted')


def curve_metrics(curves,surfaces,selected):
    pattern=study.old.rules.pattern_summary(curves)
    a=np.asarray([z['contrast'] for z in curves]);mask=np.asarray([z['support'] for z in curves]).all(axis=0)
    mask[2,:]=False;mask[:,2]=False
    target=np.asarray([study.old.models.anchored(z)/c['training_sd'] for z,c in zip(surfaces,curves)])
    result=dict(displayed=bool(selected and pattern['consistent']),shape_correct=None,common_supported=pattern['common_supported'],
        fold_cosine=pattern['min_cosine'],truth_cosine=None,standardized_rmse=None,relative_magnitude_error=None)
    if mask.sum()<4:return result
    x=a[:,mask];y=target[:,mask];nx=np.linalg.norm(x,axis=1);ny=np.linalg.norm(y,axis=1)
    result['standardized_rmse']=float(np.sqrt(np.square(x-y).mean()))
    if (ny>=1e-8).all():result['relative_magnitude_error']=float(np.mean(np.abs(nx-ny)/ny))
    if (nx>=1e-8).all() and (ny>=1e-8).all():
        cos=np.clip(np.sum(x*y,axis=1)/(nx*ny),-1.,1.)
        result.update(truth_cosine=float(cos.min()),shape_correct=bool(cos.min()>=.8))
    return result


def dataset_metrics(v):
    c=v['task'];pairs=pair_decisions(v)
    r=dict(id=c['id'],cell=c['cell'],benchmark=c['benchmark'],pairs=pairs)
    if c['stage']=='development':
        curves=[z['pairs'][0]['curve'] for z in v['comparisons'][:3]]
        surfaces=[z['surfaces'][0] for z in v['population'][:3]]
        r.update(detected=pairs[0]['selected'],control=pairs[1]['selected'],any_selected=any(z['selected'] for z in pairs),
            **curve_metrics(curves,surfaces,pairs[0]['selected']),
            teacher_independent_mse=v['population'][3]['mse'],
            fitted_independent_additive_mse=v['comparisons'][3]['baseline_mse'],
            fitted_independent_joint_mse=v['comparisons'][3]['pairs'][0]['joint_mse'],
            ideal_additive='unassessed')
    return r


def summarize(rows,planned):
    expected={c['id'] for c in planned};ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or not set(ids)<=expected:raise ValueError('duplicate or foreign metrics identity')
    displayed=[r for r in rows if r['displayed']];assessed=[r for r in displayed if r['shape_correct'] is not None]
    r=dict(planned=len(planned),complete=len(rows),missing=len(planned)-len(rows),
        status='complete' if len(rows)==len(planned) else 'incomplete',
        detection=interval(sum(z['detected'] for z in rows),len(rows)),
        control=interval(sum(z['control'] for z in rows),len(rows)),
        false_finding=interval(sum(z['any_selected'] for z in rows),len(rows)),
        display=interval(len(displayed),len(rows)),
        pattern=interval(sum(z['shape_correct'] is True for z in assessed),len(displayed)),
        pattern_assessed=interval(sum(z['shape_correct'] is True for z in assessed),len(assessed)),
        displayed=len(displayed),pattern_unassessed=len(displayed)-len(assessed))
    r['pattern']['unknown']=r['pattern_unassessed']
    if r['pattern_unassessed']:
        r['pattern'].update(rate=None,low=None,high=None,status='unresolved_pattern_outcomes')
    return r


def stage_summary(stage,rows):
    if stage=='calibration':return []
    tasks=study.d.stage_tasks(stage);groups=[]
    for cell,benchmark in dict.fromkeys((c['cell'],c['benchmark']) for c in tasks):
        planned=[c for c in tasks if c['cell']==cell and c['benchmark']==benchmark];c=planned[0]
        group=[r for r in rows if r['cell']==cell and r['benchmark']==benchmark]
        groups.append(dict(cell=cell,benchmark=benchmark,n=c['n'],gamma=c['gamma'],observed_rho=c['rho'],reliability=c['reliability'],
            interpretation='measurement-induced predictive structure' if benchmark=='latent-additive' else 'observed predictive null' if benchmark in ('noise','observed-additive') else 'mixed signal',
            **summarize(group,planned)))
    return groups


def aggregate(stage,inputs,output):
    study.freeze();output=Path(output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    paths,merge=archive.unique_archives(archive.archives(inputs));seen=set();rows=[];errors=[];failures=[];costs=[];warnings=0;difference=0.
    inventories=[];checkpoint_fits=0
    for path in paths:
        with tempfile.TemporaryDirectory(prefix='realistic-shard-',dir=output) as scratch:
            scratch=Path(scratch);archive.extract_checked(path,scratch)
            stamps=list(scratch.glob('shard-*.json'))
            if len(stamps)!=1:raise ValueError('one shard stamp required')
            stamp=study.io.read(stamps[0]);planned=study.shard_tasks(stage,stamp['id'])
            expected=dict(id=stamp['id'],stage=stage,tasks=planned,manifest=study.identity(),
                thresholds=study.io.fingerprint(study.thresholds()) if stage=='development' else None)
            if stamp!=expected:raise ValueError('frozen shard stamp differs')
            ids={c['id'] for c in planned}
            if seen&ids:raise ValueError('duplicate shard identities')
            seen.update(ids)
            if {p.name for p in scratch.iterdir() if p.is_dir()}-ids:raise ValueError('foreign dataset folder')
            inventories.append(dict(shard=stamp['id'],archive_sha256=archive.digest(path),bytes=path.stat().st_size))
            for c in planned:
                folder=scratch/c['id']
                if not folder.exists():continue
                try:
                    cost=study.attempt_cost(c,folder);costs.append(dict(id=c['id'],**cost))
                    checkpoint_fits+=173*len(list(folder.glob('comparison-*.json')))
                    if (folder/'error.json').exists():failures.append(study.io.read(folder/'error.json'))
                    if not (folder/'verified.json').exists():continue
                    v=study.audit_task(c,folder)
                    if study.io.read(folder/'verified.json')!=study.verification(c,folder):raise ValueError('verification stamp differs')
                    r=dataset_metrics(v);independent=verify.vector_scores(v['comparisons'][:3])
                    for own,other in zip(r['pairs'],independent):
                        delta=abs(own['score']-other['score']);difference=max(difference,delta)
                        if delta>1e-10 or own['eligible']!=other['eligible']:raise ValueError('standalone vector score differs')
                    r['result_sha256']=archive.digest(folder/'result.json');rows.append(r);warnings+=len(v['warnings'])
                except Exception as e:errors.append(dict(id=c['id'],error=repr(e)))
    planned=study.d.stage_tasks(stage);complete={r['id'] for r in rows};missing=[c['id'] for c in planned if c['id'] not in complete]
    audit=dict(stage=stage,manifest=study.identity(),planned=len(planned),complete=len(rows),missing=len(missing),missing_ids=missing,
        audit_errors=errors,rows=rows,standalone=dict(status='complete' if not missing and not errors else 'incomplete',checked=len(rows),max_score_difference=difference),
        merge=merge,inventories=inventories,warning_count=warnings,failed_tasks=failures,
        attempt_costs=costs,known_attempt_cost={k:sum(c[k] for c in costs) for k in ('fits','verification_refits','seconds')},
        unknown_attempt_cost_ids=sorted({c['id'] for c in planned}-{c['id'] for c in costs} |
            {c['id'] for c in costs if c['unknown_fit_cost'] or c['unknown_verification_cost']}),
        checkpoint_fits=checkpoint_fits,fits=692*len(rows),saved_setting_refits=20*len(rows),
        cost_scope='effective, checkpoint and attempt counts overlap; missing/interrupted costs remain unknown',
        aggregation_seconds=time.monotonic()-start,production_accepted=False)
    study.io.write(output/'audit.json',audit)
    if stage=='calibration' and not missing and not errors:
        cutoffs=verify.verify_calibration(audit,study.identity())
        if cutoffs['thresholds']!=learn_thresholds(rows):raise ValueError('independent cutoffs differ')
        study.io.write(output/'proposed-thresholds.json',cutoffs)
    groups=stage_summary(stage,rows);study.io.write(output/'strata.json',groups)
    study.pd.json_normalize(rows).to_csv(output/'dataset_metrics.csv',index=False)
    study.pd.json_normalize(groups).to_csv(output/'stratum_metrics.csv',index=False)
    write_report(audit,groups,output)
    return audit


def write_report(audit,groups,output):
    output=Path(output)
    text=['# Realistic interaction development study',
        f"Completed {audit['complete']}/{audit['planned']} planned datasets; {audit['missing']} unknown; {len(audit['audit_errors'])} audit errors.",
        'This is a development map. It does not certify individual findings or establish a universal minimum sample size.',
        'Each calibration cutoff is the largest eligible score across both candidate pairs and both observed-input null classes. Development uses new datasets.',
        'Displayed/80 measures how often a usable plot appeared. Correct/displayed measures its pattern agreement when assessed. Zero plots leave pattern accuracy unresolved.',
        'Intervals are exact two-sided 95% descriptive intervals, without correction for the many comparisons. Incomplete groups cannot support recommendations.',
        'Noisy latent-additive cases represent measurement-induced predictive structure, not false latent-moderation discoveries.',
        'Magnitude errors and standardized contrast RMSE are descriptive; pattern agreement alone does not prove accurate effect sizes.',
        'N150 uses outer training100 / inner66–67; N300 uses200 /133–134; N600 uses400 /266–267. Independent assessment adds no training participants.']
    if groups:
        text+=['','|Context (latent gamma / observed rho / reliability)|Benchmark|Detected|Unrelated pair|Displayed|Correct pattern / displayed (unknown)|',
               '|---|---|---:|---:|---:|---:|']
        for r in groups:
            fmt=lambda z:f"{z['k']}/{z['n']}" if z['n'] else 'unresolved'
            text.append(f"|N{r['n']} / {r['gamma']} / {r['observed_rho']:.2f} / {r['reliability']}|{r['benchmark']}|{fmt(r['detection'])}|{fmt(r['control'])}|{fmt(r['display'])}|{fmt(r['pattern'])} ({r['pattern_unassessed']} unknown)|")
        plots(groups,output)
        text+=['','![Detection across sample sizes](detection.png)','![Display coverage and conditional pattern agreement](display-pattern.png)',
               'Read each panel separately: rising detection with N suggests a sample-size limitation in that condition. Compare reliability panels to see the additional loss from measurement error. A strong reference succeeding while weaker signals fail suggests limited sensitivity, rather than complete inability to model interactions.']
    (output/'report.md').write_text('\n\n'.join(text[:9])+'\n'+'\n'.join(text[9:])+'\n',encoding='utf-8')


def plots(groups,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind,name in [('detection','detection.png'),('display','display-pattern.png')]:
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
        for i,r in enumerate((1.,.8)):
            for j,g in enumerate((.1,.3,.5)):
                ax=axes[i,j]
                for bench in ('product-b0.2','product-b0.4','tanh-b0.2','tanh-b0.4','product-b0.8'):
                    values=sorted([x for x in groups if x['reliability']==r and x['gamma']==g and x['benchmark']==bench],key=lambda x:x['n'])
                    values=[x for x in values if x['status']=='complete']
                    if not values:continue
                    line,=ax.plot([x['n'] for x in values],[x[kind]['rate'] for x in values],'o-',label=bench)
                    if kind=='display':ax.plot([x['n'] for x in values],[x['pattern']['rate'] if x['pattern']['rate'] is not None else np.nan for x in values],':',alpha=.65,color=line.get_color())
                ax.set(title=f'Latent correlation {g}; observed {r*g:.2f}; reliability {r}',ylim=(0,1.05),xticks=(150,300,600))
                ax.grid(alpha=.2);ax.set_xlabel('People per dataset');ax.set_ylabel('Detection rate' if kind=='detection' else 'Solid: display rate; dotted: correct / displayed')
        axes[0,0].legend(fontsize=8);fig.suptitle('Development estimates; interpret each condition separately');fig.tight_layout()
        fig.savefig(output/name,dpi=160);plt.close(fig)
