"""Run fixed chain/missing-outcome/redundant fixtures with layer consensus."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from examples.run_synthetic_demo import sample_fixtures
from examples.run_layer_demo import save_fit, write_json
from examples.audit_layer_demo import audit_demo
from open.config import AnalysisConfig, OpenConfig, PerformanceCVConfig, StabilityConfig, ThresholdConfig, TuningCVConfig, dump_config
from open.metadata import validate_metadata
from open.layer_stability import fit_layer_consensus

def run_fixtures(output, n_resamples=20):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for name,frame,root,seed in sample_fixtures():
        print(f'Starting {name}: {n_resamples} layer assessments per target',flush=True)
        directory=output/name;directory.mkdir()
        sample=frame.to_csv(index=False,lineterminator='\n').encode('utf-8')
        (directory/'sample.csv').write_bytes(sample)
        config=OpenConfig(AnalysisConfig(root,27,max_depth=3),
            thresholds=ThresholdConfig(unique_relative=.15),
            performance_cv=PerformanceCVConfig(2,1),tuning_cv=TuningCVConfig(2),
            stability=StabilityConfig(fraction=.8,n_resamples=n_resamples,scope='layer_consensus'))
        dump_config(config,directory/'config.yaml')
        metadata=validate_metadata(frame.columns,[{'name':n,'display_label':n.replace('_',' '),
            'type':'continuous','role':'root' if n==root else 'candidate'} for n in frame],root)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fitted=fit_layer_consensus(frame,root,tuple(frame),config)
        bundle,summary=save_fit(fitted,directory,metadata)
        summary.update(sample_sha256=hashlib.sha256(sample).hexdigest(),data_seed=seed,fixture=name)
        write_json(directory/'summary.json',summary)
        write_json(directory/'warnings.json',[{'category':c,'message':m,'count':n} for (c,m),n in
            Counter((w.category.__name__,str(w.message)) for w in captured).items()])
        audit_demo(directory)
        summaries.append(summary)
    write_json(output/'summary.json',summaries)
    write_fixture_report(output,summaries)
    return summaries


def write_fixture_report(output,summaries):
    lines=['# Layer-consensus regression demonstrations','',
        'These fixed synthetic fixtures illustrate software behavior; they do not calibrate the method.',
        '', 'All cases use 20 assessments per target, 80% subsampling, seed 27, two outer/tuning folds,',
        'depth cap 3, unique relevance cutoff .15, and stability cutoff .70. Settings match the historical fixtures.',
        '', '| Fixture | Result | Completed assessments | Warnings |',
        '| --- | --- | --- | --- |']
    for s in summaries:
        lines.append(f"| {s['fixture']} | {s['assigned_depth']} | {s['completed_count']}/{s['assessment_count']} | {s['warning_count']} |")
    lines += ['', 'The chain expands from Anxiety to Depression, then to Stress. Each retained link is 20/20.',
        'The identical chain with one missing Depression outcome still completes; sampled target coverage is',
        '63 observed + 1 missing or 64 observed + 0 missing. Missing target values are excluded, never imputed.',
        'The near-copy scales are retained jointly in 20/20 assessments. Both enter depth 1; no unique',
        'member arrow is invented. The single joint loss contrast belongs to the pair.', '']
    for s in summaries:
        case=s['fixture']
        lines += [f'## {case}','', f'![{case}]({case}/network.png)','',
            f'[Summary]({case}/summary.json), [offline audit]({case}/audit.json),',
            f'[configuration]({case}/config.yaml), [sample]({case}/sample.csv),',
            f'[all assessments]({case}/bundle/assessment_ledger.json).','']
    lines += ['Frequencies describe stability conditional on earlier finalized layers, not confidence,',
        'causal direction or empirical whole-path probability. Actual counts and precision are in the saved bundles.',
        'Fit revision is recorded per case; later audit validation and display formatting do not change fitted scores.','']
    (Path(output)/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resamples',type=int,default=20)
    args=parser.parse_args();run_fixtures(args.output,args.resamples)
