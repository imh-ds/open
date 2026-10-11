"""Bounded pair explanations on analytic functions and a synthetic RF survey.

Run: python examples/run_pair_interactions.py --output NEW_DIRECTORY
Requires the visualization extra. This is a smoke demonstration, not calibration.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from open.config import AnalysisConfig,OpenConfig,PerformanceCVConfig,TuningCVConfig
from open.effects import compare_target_engines,save_engine_comparison
from open.interactions import PairScreenConfig,screen_pair_interactions,aggregate_pair_screens
from open.interaction_visualization import save_interaction_report
from open.splits import make_split_plan


class FunctionModel:
    def __init__(self,function): self.function=function
    def predict(self,X): return self.function(X)


def run(output):
    if output.exists():
        raise FileExistsError('Use a new directory to preserve existing evidence')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    rng=np.random.default_rng(32001)
    frame=pd.DataFrame(rng.normal(size=(600,5)),columns=[
        'social_support','sleep_quality','perceived_stress','physical_activity','work_demands'])
    settings=PairScreenConfig(max_details=2)
    analytic={}
    functions={
        'additive':lambda X: X.social_support**2+np.sin(X.perceived_stress),
        'pure_interaction':lambda X:X.social_support*X.perceived_stress,
        'mixed':lambda X:.5*X.social_support-.5*X.perceived_stress+X.social_support*X.perceived_stress,
    }
    for name,function in functions.items():
        score=screen_pair_interactions(FunctionModel(function),frame,tuple(frame),
            outcome_sd=float(np.std(function(frame))),config=settings)
        score.update(repeat=0,fold=0)
        summary=aggregate_pair_screens([score])
        summary['fold_screens']=[score]
        value=dict(target='wellbeing',engines=[dict(engine='known_function',interactions=summary)])
        folder=output/name
        folder.mkdir()
        (folder/'comparison.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        save_interaction_report(value,folder,plots=True)
        analytic[name]=score
    assert not any(p['selected_for_detail'] for p in analytic['additive']['pairs'])
    for name in ('pure_interaction','mixed'):
        selected=[p for p in analytic[name]['pairs'] if p['selected_for_detail']]
        assert len(selected)==1
        assert selected[0]['members']==['perceived_stress','social_support']
    frame['wellbeing']=(.9*frame.sleep_quality+.6*frame.physical_activity+
        .4*frame.social_support-.5*frame.perceived_stress+
        .8*frame.social_support*frame.perceived_stress+rng.normal(0,.6,len(frame)))
    frame.to_csv(output/'sample.csv',index=False)
    candidates=tuple(n for n in frame if n!='wellbeing')
    config=OpenConfig(AnalysisConfig('wellbeing',32002),performance_cv=PerformanceCVConfig(3,1),
                      tuning_cv=TuningCVConfig(2))
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        comparison=compare_target_engines(frame,'wellbeing',candidates,config,make_split_plan(frame,config),
            engines=('random_forest',),bins=3,interaction_screen=settings)
    save_engine_comparison(comparison,output/'rf')
    save_interaction_report(comparison,output/'rf',plots=True)
    engine=comparison['engines'][0]
    summary=engine['interactions']
    assert summary['additional_model_fits']==0
    assert summary['planned_folds']==3
    assert len(summary['pairs'])==10
    assert summary['prediction_rows'] <= 3*settings.prediction_rows
    ranked=sorted([p for p in summary['pairs'] if p['mean_screen_rms'] is not None],
                  key=lambda p:(-p['mean_screen_rms'],p['members']))
    receipt=dict(seed=32001,rows=600,candidates=list(candidates),pair_config=asdict(settings),
        additional_screening_model_fits=0,interaction_prediction_rows=summary['prediction_rows'],
        scored_outer_fit_count=engine['scored_outer_fit_count'],
        tuning_model_fit_count=engine['tuning_model_fit_count'],
        explanatory_refit_count=engine['explanatory_refit_count'],
        comparison_elapsed_seconds=time.perf_counter()-started,cv_r2=engine['cv_r2'],
        ranking=[dict(members=p['members'],mean_screen_rms=p['mean_screen_rms'],
                      supported_detail_folds=p['supported_detail_folds']) for p in ranked],
        warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in captured],
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in list((ROOT/'src/open').rglob('*.py'))+[Path(__file__)]})
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    rank=next((i+1 for i,p in enumerate(ranked) if p['members']==['perceived_stress','social_support']),None)
    lines=['# Pair interaction smoke demonstration','',
        '600 synthetic observations, five predictors and Well-being as the outcome. '
        'Sleep Quality, Physical Activity, Social Support and Perceived Stress have additive contributions; '
        'Support and Stress also have a product interaction. Work Demands is independent noise. '
        'These are illustrative standard-normal scores, not actual survey measurements.','',
        'Analytic checks: nonlinear separate effects produce no candidate junction; pure and mixed '
        'functions identify the known Support–Stress pair without marginal importance filtering. '
        'This checks calculation, not statistical detection performance.','',
        f"Actual fitted RF: CV R² {engine['cv_r2']:.3f}; known interacting pair ranks {rank} of ten pairs. "
        f"Screening used {summary['prediction_rows']:,} evaluated prediction rows across three folds, "
        'with zero additional model fits or tuning searches for the pair supplement. '
        'The larger effect comparison still performs its usual full, ablation, tuning and explanatory fits.','',
        f"Whole comparison/export took {receipt['comparison_elapsed_seconds']:.1f} seconds. This is a single "
        'machine/synthetic-case smoke timing, not a general runtime guarantee.','',
        '[RF candidate graph and supported local-contrast plots](rf/INTERACTIONS.md), '
        '[pure known interaction](pure_interaction/INTERACTIONS.md), '
        '[known additive function](additive/INTERACTIONS.md), '
        '[mixed effects](mixed/INTERACTIONS.md).','',
        'RF can learn incidental nonadditivity involving other pairs. A shortlist entry does not establish '
        'a true population interaction; this demo does not calibrate false positives, stable recovery or '
        'minimum sample sizes. The recursive graph is not modified. Pure/mixed interactions are visually '
        'distinguished using a separate exploratory junction view, not automatic relabelling of ordinary arrows.','',
        f"In this RF fit, {sum(p['supported_detail_folds']>0 and p['members']!=['perceived_stress','social_support'] for p in ranked)} "
        'additional pairs received supported detailed examinations across folds despite not being interaction '
        'terms in the generating function. This is evidence of incidental fitted-model patterns and a reason '
        'to keep the candidate label; the screen is not a validated population interaction detector.','',
        'Raw model settings, fold scores/support, budgets and identities are in rf/comparison.json and CSVs. '
        'receipt.json records runtime, warnings and source hashes; sample.csv preserves the synthetic data.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('cv_r2','comparison_elapsed_seconds',
        'interaction_prediction_rows','additional_screening_model_fits','ranking')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
