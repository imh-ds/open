"""Method-specific survey demonstration; preserves historical outputs and APIs."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import warnings

from open.config import (AnalysisConfig, OpenConfig, PerformanceCVConfig, StabilityConfig,
                         TuningCVConfig, dump_config)
from open.io import write_open_outputs
from open.layer_sensitivity import fit_layer_sensitivity
from open.layer_stability import fit_layer_consensus
from open.layer_io import load_layer_bundle
from open.layer_visualization import CAPTION, evidence_lines
from open.metadata import validate_metadata
from open.visualization import plot_open_network


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')


def save_fit(result, directory, metadata):
    from examples.run_synthetic_demo import _save_svg
    directory.mkdir(parents=True,exist_ok=True)
    write_open_outputs(result,directory/'bundle',metadata=metadata)
    bundle=load_layer_bundle(directory/'bundle')
    fig=plot_open_network(bundle,output_path=directory/'network.png')
    _save_svg(fig,directory/'network.svg')
    records = [a for layer in result.layers for target in layer.evidence for a in target.assessments]
    summary={'method':'layer_consensus','threshold':result.config.thresholds.stability,
        'status':result.status,'planned_per_target':result.effective_resamples,
        'assigned_depth':dict(result.assigned_depth),'parents':dict(result.parents),
        'stop_reasons':dict(result.stop_reasons),'assessment_count':len(records),
        'completed_count':sum(a.status=='completed' for a in records),
        'failed_count':sum(a.status=='failed' for a in records),
        'infeasible_count':sum(a.status=='infeasible' for a in records),
        'warning_count':sum(len(a.warnings) for a in records),
        'outer_model_fit_count':sum(a.local.outer_model_fit_count for a in records if a.local),
        'tuning_model_fit_count':sum(a.local.tuning_model_fit_count for a in records if a.local),
        'elapsed_seconds':result.provenance.elapsed_seconds,
        'peak_traced_memory_bytes':result.provenance.peak_traced_memory_bytes,
        'engine_revision':result.provenance.git_revision,'path_frequency_available':False}
    write_json(directory/'summary.json',summary)
    return bundle,summary


def run_layer_demo(output, *, n_resamples=20, max_depth=5, sensitivity=False):
    from examples.run_survey_demo import LABELS, survey_fixture
    if isinstance(n_resamples,bool) or not isinstance(n_resamples,int) or n_resamples<1:
        raise ValueError('n_resamples must be positive integer')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    frame=survey_fixture()
    sample=frame.to_csv(index=False,lineterminator='\n').encode('utf-8')
    (output/'sample.csv').write_bytes(sample)
    config=OpenConfig(AnalysisConfig('wellbeing',613,max_depth=max_depth),
        performance_cv=PerformanceCVConfig(2,1),tuning_cv=TuningCVConfig(2),
        stability=StabilityConfig(fraction=.8,n_resamples=n_resamples,
            final_resamples=n_resamples,scope='layer_consensus'))
    dump_config(config,output/'config.yaml')
    metadata=validate_metadata(frame.columns,[{'name':name,'display_label':LABELS[name],
        'type':'continuous','role':'root' if name=='wellbeing' else 'candidate'} for name in frame],'wellbeing')
    print(f'Starting layer consensus: 180 rows, {n_resamples} assessments per target; sensitivity={sensitivity}',flush=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if sensitivity:
            results=fit_layer_sensitivity(frame,'wellbeing',tuple(frame),config)
        else:
            results=(fit_layer_consensus(frame,'wellbeing',tuple(frame),config),)
    summaries=[]
    reference=None
    for result in results:
        p=result.config.thresholds.stability
        directory=output if p==config.thresholds.stability else output/'sensitivity'/f'{p:.2f}'
        bundle,summary=save_fit(result,directory,metadata)
        summary['directory']='.' if directory==output else directory.relative_to(output).as_posix()
        summaries.append(summary)
        if directory==output: reference=bundle
    warnings_summary=[{'category':c,'message':m,'count':n} for (c,m),n in sorted(Counter(
        (w.category.__name__,str(w.message)) for w in captured).items())]
    write_json(output/'warnings.json',warnings_summary)
    root_summary=next(s for s in summaries if s['directory']=='.')
    root_summary.update(sample_sha256=hashlib.sha256(sample).hexdigest(),
        sensitivity=summaries if sensitivity else [])
    # Avoid a self-reference in JSON.
    root_summary['sensitivity']=[dict(s, sensitivity=[]) for s in summaries] if sensitivity else []
    write_json(output/'summary.json',root_summary)
    rows=['| Threshold | Status | Assigned depths | Assessments | Graph |',
          '| --- | --- | --- | --- | --- |']
    for s in summaries:
        link='network.png' if s['directory']=='.' else s['directory']+'/network.png'
        rows.append(f"| {s['threshold']:.2f} | {s['status']} | {s['assigned_depth']} | "
                    f"{s['completed_count']}/{s['assessment_count']} completed | [graph]({link}) |")
    report=['# Layer-consensus survey demonstration','',
        '**Synthetic data only.** 180 rows; Well-being is the outcome. Fixed seed 613.',
        '',f"Reference status: **{root_summary['status']}**. Each scheduled target has {n_resamples} planned assessments.",
        '', '![Reference network](network.png)','',CAPTION,'',
        '## What the layers mean','',
        'First finalize predictors of Well-being. Only retained sources become next-layer targets. '
        'All targets at the same depth share the remaining candidate set. Finalize that entire layer before continuing.',
        '', '```text',*evidence_lines(reference),*evidence_lines(reference,finalized=False),'```','',
        '## Full-construction threshold sensitivity','',*rows,'',
        ('Each threshold is fitted independently on identical seeded samples. Differences upstream can change '
         'later candidate sets. These graphs are separate conditional reference networks, not image filters.'
         if sensitivity else '**Not assessed.** Only the reference threshold was fitted; no threshold comparison was run.'),
        '', '## Audit and limitations','',
        f"Input SHA256: `{root_summary['sample_sha256']}`. Engine revision: `{root_summary['engine_revision']}`.",
        f"Reference fits: {root_summary['outer_model_fit_count']} outer, {root_summary['tuning_model_fit_count']} tuning; "
        f"runtime {root_summary['elapsed_seconds']:.2f}s; {root_summary['warning_count']} captured warnings.",
        'Local nested CV evaluates prediction for the current candidates; it does not independently validate '
        'discovery of the entire network. Frequencies are conditional stability descriptions, not confidence '
        'or calibrated error rates. Arrows do not establish causality. No empirical whole-path probability '
        'is available. Thresholds are development defaults, not scientifically calibrated.',
        '', '[Summary](summary.json), [warnings](warnings.json), [sample](sample.csv), [configuration](config.yaml),',
        '[metadata](bundle/analysis_metadata.json), [all layer evidence](bundle/layer_evidence.json),',
        '[assessment ledger](bundle/assessment_ledger.json), [local targets](bundle/local_targets.csv).','']
    (output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps({k:root_summary[k] for k in ('status','assigned_depth','assessment_count','warning_count')}) ,flush=True)
    return root_summary
