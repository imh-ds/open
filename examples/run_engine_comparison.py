"""Reproduce four-engine matched effects on survey and nonlinear fixtures.

From repository root: python examples/run_engine_comparison.py --output NEW_DIR
Requires the visualization and LightGBM extras for the default four engines.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from open.config import load_config, PerformanceCVConfig
from open.effects import compare_target_engines,save_engine_comparison
from open.splits import make_split_plan


def run(output, engines, bins=10):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory; existing evidence is preserved')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ROOT/'examples/results/2026-10-08-survey-layer-consensus'
    survey=pd.read_csv(source/'sample.csv')
    config=replace(load_config(source/'config.yaml'),performance_cv=PerformanceCVConfig(3,1))
    candidates=tuple(n for n in survey if n!='wellbeing')
    output.mkdir(parents=True)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        survey_result=compare_target_engines(survey,'wellbeing',candidates,config,
            make_split_plan(survey,config),engines=engines,
            joint_groups=(('social_support','sleep_quality'),),bins=bins,adaptive_bins=True)
        save_engine_comparison(survey_result,output/'survey')
        x=np.linspace(-2,2,120)
        nonlinear=pd.DataFrame({'x':x,'y':x*x})
        nonlinear_config=replace(config,analysis=replace(config.analysis,root='y',random_seed=13))
        nonlinear_result=compare_target_engines(nonlinear,'y',('x',),nonlinear_config,
            make_split_plan(nonlinear,nonlinear_config),engines=engines,bins=bins,adaptive_bins=True)
        save_engine_comparison(nonlinear_result,output/'nonlinear')
        nonlinear.to_csv(output/'nonlinear/sample.csv',index=False)
        warning_rows=[dict(category=w.category.__name__,message=str(w.message)) for w in captured]
    results={'survey':survey_result,'nonlinear':nonlinear_result}
    # Offline checks of matched settings, scoring formulas, curve/contrast identity
    # and JSON serialization. These do not pick an engine or alter any threshold.
    for result in results.values():
        reference_keys=result['engines'][0]['heldout_keys']
        for engine in result['engines']:
            assert engine['heldout_keys']==reference_keys
            assert engine['cv_r2']==1-engine['full_mse']/engine['heldout_variance']
            assert not engine['ablations_retuned']
            for effect in engine['effects']:
                if effect['status']=='available':
                    difference=np.interp(effect['high'],effect['edges'],effect['standardized_curve'])-np.interp(effect['low'],effect['edges'],effect['standardized_curve'])
                    np.testing.assert_allclose(difference,effect['standardized_contrast'],atol=1e-12)
                if engine['engine']!='elastic_net': assert effect['beta'] is None
                if effect['magnitude_status']=='available':
                    frame=survey if result is survey_result else nonlinear
                    values=frame[effect['source']].dropna().to_numpy()
                    expected=np.std(np.interp(values,effect['edges'],effect['standardized_curve']),ddof=0)
                    np.testing.assert_allclose(expected,effect['standardized_curve_sd'],atol=1e-12)
                for interval in effect['interval_contrasts']:
                    if interval['status']=='available' and effect['magnitude_status']=='available':
                        expected=np.interp(interval['high'],effect['edges'],effect['standardized_curve'])-np.interp(interval['low'],effect['edges'],effect['standardized_curve'])
                        np.testing.assert_allclose(expected,interval['standardized_contrast'],atol=1e-12)
    colors=dict(zip(engines,['#1b6b93','#e08c25','#4c956c','#a34d91']))
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    labels={name:name.replace('_',' ').title() for name in candidates}
    labels['wellbeing']='Well-being'
    survey_levels=[value for engine in survey_result['engines'] for effect in engine['effects']
                   for value in effect['standardized_curve'] if value is not None]
    padding=.05*(max(survey_levels)-min(survey_levels)) if survey_levels else .1
    for ax,source_name in zip(axes.flat,candidates):
        for engine in survey_result['engines']:
            effect=next(e for e in engine['effects'] if e['source']==source_name)
            ax.plot(effect['edges'],effect['standardized_curve'],marker='o',markersize=3,
                    label=engine['engine'],color=colors[engine['engine']])
        ax.set_title(labels[source_name])
        ax.set_xlabel('Predictor score')
        ax.set_ylabel('ALE / Well-being SD')
        ax.axhline(0,color='gray',linewidth=.6)
        if survey_levels: ax.set_ylim(min(survey_levels)-padding,max(survey_levels)+padding)
    ax=axes.flat[-1]
    for engine in nonlinear_result['engines']:
        effect=engine['effects'][0]
        ax.plot(effect['edges'],effect['standardized_curve'],marker='o',markersize=3,
                label=engine['engine'],color=colors[engine['engine']])
    ax.set_title('Nonlinear check: Y = X²')
    ax.set_xlabel('X')
    ax.set_ylabel('ALE / Y SD')
    ax.legend(fontsize=8)
    fig.suptitle('Matched engines: standardized predictive relationships\nSame reference endpoints and outcome SD; curves are descriptive, not causal',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(output/'ale_curves.png',dpi=160)
    plt.close(fig)
    for kind in ('percentile_curves','interval_contrasts'):
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,(result,source_name) in zip(axes.flat,
                [(survey_result,n) for n in candidates]+[(nonlinear_result,'x')]):
            for engine in result['engines']:
                effect=next(e for e in engine['effects'] if e['source']==source_name)
                if kind=='percentile_curves':
                    points=effect['percentile_points']
                    positions=[p['percentile'] for p in points]
                    levels=[p['standardized_ale'] if p['standardized_ale'] is not None else np.nan for p in points]
                else:
                    intervals=effect['interval_contrasts']
                    positions=[(r['low_percentile']+r['high_percentile'])/2 for r in intervals]
                    levels=[r['standardized_contrast'] if r['standardized_contrast'] is not None else np.nan for r in intervals]
                ax.plot(positions,levels,marker='o',markersize=3,
                    label=engine['engine'],color=colors[engine['engine']])
            ax.set_title(labels[source_name] if source_name in labels else 'Nonlinear check: Y = X²')
            ax.set_xlabel('Predictor percentile' if kind=='percentile_curves' else '10-percentile interval midpoint')
            ax.set_ylabel('ALE / outcome SD' if kind=='percentile_curves' else 'Signed interval change / outcome SD')
            ax.set_xticks(range(0,101,20))
            ax.axhline(0,color='gray',linewidth=.6)
            if kind=='percentile_curves' and result is survey_result and survey_levels:
                ax.set_ylim(min(survey_levels)-padding,max(survey_levels)+padding)
        axes.flat[-1].legend(fontsize=8)
        fig.suptitle(('Curve levels at P0, P10 … P100' if kind=='percentile_curves' else 'Adjacent percentile contrasts: P0→P10 … P90→P100')+
            '\nDescriptive ALE; interpolated within supported estimation bins; gaps remain unavailable',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.93))
        fig.savefig(output/(kind+'.png'),dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(name,result) in zip(axes,results.items()):
        names=[r['engine'] for r in result['engines']]
        values=[r['cv_r2'] for r in result['engines']]
        ax.bar(names,values,color=[colors[name] for name in names])
        ax.axhline(0,color='gray',linewidth=.6)
        ax.set_title('Synthetic '+name)
        ax.set_ylabel('Cross-validated R²')
        ax.tick_params(axis='x',rotation=20)
        for i,value in enumerate(values): ax.text(i,value+.02,f'{value:.3f}',ha='center')
    fig.tight_layout()
    fig.savefig(output/'prediction_performance.png',dpi=160)
    plt.close(fig)
    lines=['# Matched engine comparison: synthetic demonstrations','',
        'New fixed-target nested-CV comparisons; these are not the earlier 20-resample network analysis. No engine is chosen as a winner and no consensus network is pooled across engines.',
        '', f"Survey: the same 180 saved rows and all five Well-being candidates. Shared three outer folds, two inner folds, seed 613. Requested ALE resolution: {bins} quantile bins, with Q25/Q75 boundaries also included. Held-out covariate support selected {survey_result['estimation_bins']} nominal bins for the survey and {nonlinear_result['estimation_bins']} for the nonlinear fixture, before fitting any engine. Nonlinear check: 120 deterministic rows, Y = X², seed 13, same CV dimensions. Both use held-out reference rows and common reference outcome SD.",
        '', '## Prediction performance','', '| Engine | Survey CV R² | Nonlinear CV R² |','| --- | ---: | ---: |']
    for a,b in zip(survey_result['engines'],nonlinear_result['engines']):
        lines.append(f"| {a['engine']} | {a['cv_r2']:.4f} | {b['cv_r2']:.4f} |")
    lines+=['', '![Performance](prediction_performance.png)', '', '## Comparable effects on Well-being','',
        'Signed standardized ALE contrast: change along the estimated prediction function from predictor Q25 to Q75. Overall curve magnitude is the empirical SD of the combined curve evaluated at observed predictor values. Both use Well-being SD units, but magnitude is unsigned and captures a U shape without endpoint cancellation. ΔCV R² is the unique loss in predictive R² after removing that predictor; it is a different quantity. These are descriptive predictive summaries, not causal effects or coefficients.',
        '', '| Predictor | Engine | Q25→Q75 contrast (Y SD) | Curve magnitude (Y SD) | Unique ΔCV R² | Contrast / magnitude status |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    def number(value): return 'Unavailable' if value is None else f'{value:+.4f}'
    for source_name in candidates:
        for engine in survey_result['engines']:
            effect=next(e for e in engine['effects'] if e['source']==source_name)
            lines.append(f"| {labels[source_name]} | {engine['engine']} | {number(effect['standardized_contrast'])} | {number(effect['standardized_curve_sd'])} | {number(effect['delta_cv_r2'])} | {effect['status']} / {effect['magnitude_status']} |")
    lines+=['', '## Percentile curves and adjacent changes','',
        'The first graph shows centered ALE levels at P0, P10 … P100. The second shows signed P0→P10, P10→P20 … P90→P100 changes, plotted at interval midpoints. Positive means the fitted prediction rises; negative means it falls; zero means a supported flat segment. These are model-effect points, not individual-person scatterplots or absolute expected outcome scores. Actual predictor values accompany each point and interval in the CSV tables.',
        '', 'Percentile points interpolate the supported piecewise-linear ALE curve. Coarser estimation bins do not create ten independently estimated segments. Adjacent contrasts integrate supported local bin increments; an empty bin withholds any interval crossing it, while an interval outside the gap can remain available. Tied endpoints have zero width and remain unavailable. Aggregate estimates require support in every contributing fold and use the same nonmissing-reference weights as the original contrast.',
        '', 'P0 and P100 are observed extremes and can be sensitive to unusual values. A 10-percentile step can span very different score distances. Nonempty bins are a minimum support check, not a guarantee of precision. These displays do not change model selection, path retention, or the consensus threshold.',
        '', '![Percentile curves](percentile_curves.png)', '', '![Adjacent changes](interval_contrasts.png)',
        '', '[Survey percentile points](survey/percentile_points.csv), [survey adjacent changes](survey/interval_contrasts.csv), [nonlinear percentile points](nonlinear/percentile_points.csv), [nonlinear adjacent changes](nonlinear/interval_contrasts.csv).',
        '', 'Original score-axis curves: ![ALE curves](ale_curves.png)', '', '## Joint removal diagnostic','',
        'Social Support and Sleep Quality are removed together on the same held-out observations. This is the total predictive contribution of the pair, including unique and overlapping information; it does not establish that this pair is redundant or promote a shared group into the network.',
        '', '| Engine | Joint ΔCV R² |','| --- | ---: |']
    for engine in survey_result['engines']:
        lines.append(f"| {engine['engine']} | {number(engine['joint'][0]['delta_cv_r2'])} |")
    lines+=['', '## U-shaped fixture: contrast versus magnitude','',
        '| Engine | Q25→Q75 contrast (Y SD) | Overall curve magnitude (Y SD) |',
        '| --- | ---: | ---: |']
    for engine in nonlinear_result['engines']:
        effect=engine['effects'][0]
        lines.append(f"| {engine['engine']} | {number(effect['standardized_contrast'])} | {number(effect['standardized_curve_sd'])} |")
    lines+=['', '## Interpretation and limits','',
        'In the nonlinear fixture, elastic net can only fit a straight line; the tree engines can learn the U-shaped relationship. The Q25→Q75 contrast can be near zero despite substantial overall curve magnitude. Magnitude measures typical variation of the first-order curve, not unique importance, variance explained, or all interactions. An interaction can average to a flat first-order curve while still helping prediction. Inspect the curve together with predictive R². Magnitude depends on the reference distribution and bin resolution; no universal effect-size cutoffs are assigned.',
        '', 'Every engine shares the data, folds and effect-size reference. Elastic net searches 25 tuning candidates; each tree engine searches four. This is not an equal-budget benchmark, an engine-selection test, or a scientific calibration. No confidence intervals are inferred from folds. Different recursive networks require separately labeled candidate sets.',
        '', '[Survey JSON](survey/comparison.json), [survey effect table](survey/effects.csv), [nonlinear JSON](nonlinear/comparison.json), [nonlinear effect table](nonlinear/effects.csv), [audit](audit.json).',
        '', '[Method and usage](../../../docs/engines-and-effect-sizes.md).']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(survey_source_sha256=hashlib.sha256((ROOT/'examples/results/2026-10-08-survey-layer-consensus/sample.csv').read_bytes()).hexdigest(),
        engines=list(engines),elapsed_seconds=time.perf_counter()-started,warnings=warning_rows,
        matched_heldout_keys_verified=True,r2_formula_verified=True,curve_contrast_identity_verified=True,
        curve_magnitude_formula_verified=True,
        adjacent_contrast_curve_identity_verified=True,
        requested_bins=bins,estimation_bins={name:r['estimation_bins'] for name,r in results.items()},
        nonlinear_tree_signal_verified=all(e['cv_r2']>.8 for e in nonlinear_result['engines'] if e['engine']!='elastic_net'),
        survey_scored_outer_fits=sum(e['scored_outer_fit_count'] for e in survey_result['engines']),
        survey_tuning_fits=sum(e['tuning_model_fit_count'] for e in survey_result['engines']))
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    print('Report:',output/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--engines',nargs='+',default=['elastic_net','random_forest','gbm','lightgbm'])
    parser.add_argument('--bins',type=int,default=10,help='Maximum requested ALE resolution; reduced if held-out bins lack support')
    args=parser.parse_args()
    run(args.output,tuple(args.engines),args.bins)
