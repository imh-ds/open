"""Matched engine comparisons and standardized predictive effects.

These are descriptive prediction-function summaries, not causal estimates or
substitutes for the recursive engine's selection/stability evidence.
"""
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import csv
import hashlib
import json
import math
from numbers import Real

import numpy as np
import pandas as pd

from open.analysis import _validate_inputs
from open.config import LearnerConfig
from open.learners import make_predictor
from open.risk import evaluate_risk
from open.splits import observed_target_splits


def curve_magnitude(edges, curve, reference_values):
    """Empirical population SD of a supported piecewise-linear ALE curve.

    Evaluate at observed predictor values, rather than giving every boundary
    equal weight. A constant level offset is irrelevant. Unsupported levels
    remain undefined; interpolation never crosses an unsupported bin.
    """
    bounds=np.asarray(edges,dtype=float)
    if len(bounds)!=len(curve) or len(bounds)<2 or not np.isfinite(bounds).all() or (np.diff(bounds)<=0).any():
        raise ValueError("magnitude requires matching increasing boundaries and curve levels")
    if any(value is None for value in curve):
        return None
    levels=np.asarray(curve,dtype=float)
    if not np.isfinite(levels).all():
        raise ValueError("curve levels must be finite or explicitly unavailable")
    values=np.asarray(reference_values,dtype=float)
    if np.isinf(values).any():
        raise ValueError("reference values must not contain infinity")
    values=values[np.isfinite(values)]
    if not len(values): return None
    if values.min()<bounds[0] or values.max()>bounds[-1]:
        raise ValueError("reference values must lie within supported curve boundaries")
    return float(np.std(np.interp(values,bounds,levels),ddof=0))


def accumulated_local_effects(model, reference, source, *, edges, outcome_sd, contrast=None):
    """First-order binned ALE; undefined support is never converted to zero.

    Curves are evaluated at bin boundaries. Complete curves are centered by
    the reference-weighted mean of bin midpoint effects. An empty bin makes
    cumulative/centered levels undefined, but supported local increments and
    a contrast entirely outside that gap can still be reported.
    """
    if (isinstance(outcome_sd, bool) or not isinstance(outcome_sd, Real)
            or not math.isfinite(outcome_sd) or outcome_sd < 0):
        raise ValueError("outcome_sd must be finite and nonnegative")
    if not isinstance(reference, pd.DataFrame) or source not in reference:
        raise ValueError("reference must contain the source predictor")
    bounds = np.asarray(edges, dtype=float)
    if (bounds.ndim != 1 or not len(bounds) or not np.isfinite(bounds).all()
            or (np.diff(bounds) <= 0).any()):
        raise ValueError("edges must be finite, distinct, increasing boundaries")
    x = reference[source].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(x).any():
        raise ValueError("reference predictors must not contain infinity")
    valid = np.isfinite(x)
    values = x[valid]
    result = dict(edges=bounds.tolist(), bin_counts=[], bin_differences=[],
                  raw_curve=[], standardized_curve=[], raw_contrast=None,
                  standardized_contrast=None, low=None, high=None,
                  outcome_sd=float(outcome_sd), reference_count=int(valid.sum()),
                  missing_reference_count=int((~valid).sum()), status="available",
                  curve_sd=None,standardized_curve_sd=None,magnitude_status="unavailable")
    if not len(values):
        result['status'] = 'no_reference_values'
        result['magnitude_status']='no_reference_values'
        return result
    low, high = np.quantile(values,[.25,.75]) if contrast is None else contrast
    if not np.isfinite([low,high]).all() or low > high or low < bounds[0] or high > bounds[-1]:
        raise ValueError("contrast endpoints must be finite, ordered and within edges")
    result.update(low=float(low), high=float(high))
    if len(bounds) == 1 or np.ptp(values)==0:
        result['status'] = 'constant_predictor'
        result['magnitude_status']='constant_predictor'
        return result
    if values.min() < bounds[0] or values.max() > bounds[-1]:
        raise ValueError("edges must cover all observed reference values")
    indices = np.clip(np.searchsorted(bounds, values, side='right')-1,0,len(bounds)-2)
    clean = reference.loc[valid].copy()
    differences, counts = [], []
    for i in range(len(bounds)-1):
        rows = clean.iloc[np.flatnonzero(indices==i)]
        counts.append(len(rows))
        if not len(rows):
            differences.append(None)
            continue
        left, right = rows.copy(), rows.copy()
        left[source], right[source] = bounds[i], bounds[i+1]
        a, b = np.asarray(model.predict(left),float), np.asarray(model.predict(right),float)
        if a.shape!=(len(rows),) or b.shape!=a.shape or not np.isfinite([a,b]).all():
            raise ValueError("ALE model returned invalid predictions")
        differences.append(float(np.mean(b-a)))
    result.update(bin_counts=counts,bin_differences=differences)
    if all(value is not None for value in differences):
        curve = np.r_[0.,np.cumsum(differences)]
        center = np.average((curve[:-1]+curve[1:])/2,weights=counts)
        curve -= center
        result['raw_curve'] = curve.tolist()
        result['standardized_curve'] = (curve/outcome_sd).tolist() if outcome_sd>0 else [None]*len(curve)
        result['curve_sd']=curve_magnitude(bounds,curve,values)
        result['standardized_curve_sd']=result['curve_sd']/outcome_sd if outcome_sd>0 else None
        result['magnitude_status']='available' if outcome_sd>0 else 'constant_outcome'
    else:
        result['raw_curve'] = [None]*len(bounds)
        result['standardized_curve'] = [None]*len(bounds)
        result['magnitude_status']='unsupported_bins' if outcome_sd>0 else 'constant_outcome'
    overlap = np.maximum(0,np.minimum(bounds[1:],high)-np.maximum(bounds[:-1],low)) / np.diff(bounds)
    if any(weight>0 and value is None for weight,value in zip(overlap,differences)):
        result['status'] = 'unsupported_bins'
    else:
        raw = sum(weight*value for weight,value in zip(overlap,differences) if weight>0)
        result['raw_contrast'] = float(raw)
        result['standardized_contrast'] = float(raw/outcome_sd) if outcome_sd>0 else None
    if outcome_sd==0:
        result['status']='constant_outcome'
    elif low==high:
        result.update(status='zero_width_contrast',raw_contrast=None,standardized_contrast=None)
    return result


def _observations(risk):
    return {(f.repeat_index,f.fold_index,r.row_position):(r.observed,r.loss)
            for f in risk.folds for r in f.observations}


def _mean_available(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def percentile_effects(ale, percentile_values):
    """Decile levels and signed adjacent contrasts on a supported binned ALE.

    Percentile positions are common descriptive references, not new fitted
    observations. Local increments can remain available outside an empty bin
    even when the globally centered curve is unavailable.
    """
    if ale['magnitude_status']=='no_reference_values' and list(percentile_values)==[None]*11:
        return dict(
            percentile_points=[dict(percentile=p,predictor_value=None,raw_ale=None,
                standardized_ale=None,status='no_reference_values') for p in range(0,101,10)],
            interval_contrasts=[dict(low_percentile=p,high_percentile=p+10,low=None,high=None,
                raw_contrast=None,standardized_contrast=None,status='no_reference_values') for p in range(0,100,10)])
    positions=np.asarray(percentile_values,dtype=float)
    bounds=np.asarray(ale['edges'],dtype=float)
    if (positions.shape!=(11,) or not np.isfinite(positions).all()
            or (np.diff(positions)<0).any()
            or positions[0]<bounds[0] or positions[-1]>bounds[-1]):
        raise ValueError('percentile values must be eleven ordered finite positions within edges')
    sd=ale['outcome_sd']
    degenerate=ale['magnitude_status'] if ale['magnitude_status'] in (
        'no_reference_values','constant_predictor','constant_outcome') else None
    levels=ale['raw_curve']
    supported=len(levels)==len(bounds) and all(v is not None for v in levels)
    points=[]
    for percentile,value in zip(range(0,101,10),positions):
        raw=float(np.interp(value,bounds,levels)) if supported and degenerate is None else None
        points.append(dict(percentile=percentile,predictor_value=float(value),raw_ale=raw,
            standardized_ale=raw/sd if raw is not None and sd>0 else None,
            status=degenerate or ('available' if supported else 'unsupported_bins')))
    intervals=[]
    for i,(low,high) in enumerate(zip(positions[:-1],positions[1:])):
        status=degenerate or ('zero_width_contrast' if low==high else 'available')
        raw=None
        if status=='available':
            overlap=np.maximum(0,np.minimum(bounds[1:],high)-np.maximum(bounds[:-1],low))/np.diff(bounds)
            differences=ale['bin_differences']
            if len(differences)!=len(overlap) or any(w>0 and v is None for w,v in zip(overlap,differences)):
                status='unsupported_bins'
            else:
                raw=float(sum(w*v for w,v in zip(overlap,differences) if w>0))
        intervals.append(dict(low_percentile=i*10,high_percentile=(i+1)*10,
            low=float(low),high=float(high),raw_contrast=raw,
            standardized_contrast=raw/sd if raw is not None and sd>0 else None,status=status))
    return dict(percentile_points=points,interval_contrasts=intervals)


def _reference_edges(values,bins):
    quantiles=np.unique(np.r_[np.linspace(0,1,bins+1),.25,.75])
    return np.unique(np.quantile(values,quantiles)) if len(values) else np.array([0.])


def _supported_resolution(observed,data,candidates,splits,bins):
    """Choose shared resolution from held-out covariate support, before fitting."""
    for count in range(bins,1,-1):
        supported=True
        for source in candidates:
            values=observed[source].dropna().to_numpy(float)
            if not len(values) or np.ptp(values)==0: continue
            edges=_reference_edges(values,count)
            for split in splits:
                test=data.iloc[list(split.test_indices)][source].dropna().to_numpy(float)
                indices=np.clip(np.searchsorted(edges,test,side='right')-1,0,len(edges)-2)
                if not len(test) or not np.bincount(indices,minlength=len(edges)-1).all():
                    supported=False
                    break
            if not supported: break
        if supported: return count
    # Even the coarsest attempt may lack support; statuses still disclose this.
    return 2


def compare_target_engines(data, target, candidates, config, split_plan, *,
                           engines=("elastic_net","random_forest","gbm"), joint_groups=(), bins=10,
                           adaptive_bins=False, interaction_screen=None):
    """Nested-CV comparison on one frozen local problem, with paired ablations.

    Reference quantiles/SD are descriptive constants computed from observed
    target rows, shared across engines; they are never used to train or tune.
    ALE uses each outer model on that fold's held-out reference covariates.
    """
    _validate_inputs(data,target,candidates,config,split_plan)
    from open.interactions import PairScreenConfig,screen_pair_interactions,aggregate_pair_screens
    if interaction_screen is not None and not isinstance(interaction_screen,PairScreenConfig):
        raise ValueError('interaction_screen must be PairScreenConfig or None')
    if (not isinstance(engines,tuple) or not engines or len(set(engines))!=len(engines)
            or any(type(e) is not str or e not in ('elastic_net','random_forest','gbm','lightgbm') for e in engines)):
        raise ValueError("engines must be distinct supported engine names")
    if type(bins) is not int or bins<2:
        raise ValueError("bins must be an integer at least two")
    if type(adaptive_bins) is not bool:
        raise ValueError('adaptive_bins must be boolean')
    groups=[]
    for group in joint_groups:
        if (not isinstance(group,tuple) or len(group)<2 or len(set(group))!=len(group)
                or not set(group).issubset(candidates)):
            raise ValueError("joint groups must contain distinct candidate names")
        identity=tuple(sorted(group))
        if identity in groups:
            raise ValueError("joint groups must not be duplicated")
        groups.append(identity)
    observed=data.loc[data[target].notna()]
    outcome_sd=float(np.std(observed[target],ddof=0))
    splits=observed_target_splits(split_plan,data[target].to_numpy(float,na_value=np.nan),
                                  min_training_rows=config.tuning_cv.folds)
    estimation_bins=_supported_resolution(observed,data,candidates,splits,bins) if adaptive_bins else bins
    reference={}
    for source in candidates:
        values=observed[source].dropna().to_numpy(float)
        bounds=_reference_edges(values,estimation_bins)
        anchors=np.quantile(values,[.25,.75]).tolist() if len(values) else [None,None]
        reference[source]=dict(edges=bounds.tolist(),low=anchors[0],high=anchors[1],
                               percentile_values=np.quantile(values,np.linspace(0,1,11)).tolist() if len(values) else [None]*11,
                               predictor_sd=float(np.std(values,ddof=0)) if len(values) else None)
    versions={}
    for package in ('numpy','pandas','scikit-learn','lightgbm'):
        try: versions[package]=version(package)
        except PackageNotFoundError: versions[package]=None
    from open.stability import read_git_revision
    revision,revision_reason=read_git_revision()
    result=dict(schema='open.engine_comparison',schema_version=1,target=target,candidates=list(candidates),
        observed_target_rows=len(observed),missing_target_rows=int(data[target].isna().sum()),
        outcome_sd=outcome_sd,reference=reference,config=asdict(config),
        requested_bins=bins,estimation_bins=estimation_bins,adaptive_bins=adaptive_bins,
        split_plan=split_plan.to_dict(),split_plan_sha256=hashlib.sha256(split_plan.to_json(indent=None).encode()).hexdigest(),
        data_sha256=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        versions=versions,git_revision=revision,git_revision_unavailable_reason=revision_reason,
        inference='descriptive_predictive_not_causal',engines=[])
    # Make the exported in-memory value exactly the portable JSON representation.
    result=json.loads(json.dumps(result,allow_nan=False))
    for engine in engines:
        run_config=replace(config,learner=LearnerConfig(library=(engine,)))
        predictor=make_predictor(run_config)
        common=dict(target=target,candidate_set=candidates,split_plan=split_plan,
                    predictor=predictor,tuning_folds=config.tuning_cv.folds)
        baseline=evaluate_risk(data,predictors=(),**common)
        full=evaluate_risk(data,predictors=candidates,**common)
        full_obs=_observations(full)
        keys=sorted(full_obs)
        variance=float(np.var([full_obs[key][0] for key in keys],ddof=0))
        params={(f.repeat_index,f.fold_index):f.fitted_parameters for f in full.folds}
        unique,joint={},[]
        fits=[baseline,full]
        for group in [*((source,) for source in candidates),*groups]:
            ablation=evaluate_risk(data,predictors=tuple(n for n in candidates if n not in group),
                                   fitted_parameters=params,**common)
            fits.append(ablation)
            obs=_observations(ablation)
            if sorted(obs)!=keys or any(obs[key][0]!=full_obs[key][0] for key in keys):
                raise ValueError("ablation must use identical held-out observations")
            delta=(ablation.aggregate_risk-full.aggregate_risk)/variance if variance>0 else None
            if len(group)==1: unique[group[0]]=delta
            else: joint.append(dict(members=list(group),delta_cv_r2=delta,ablated_mse=ablation.aggregate_risk))
        fold_effects={source:[] for source in candidates}
        coefficients={source:[] for source in candidates}
        pair_screens=[]
        for split in splits:
            train,test=list(split.train_indices),list(split.test_indices)
            model=predictor.clone().fit(data.iloc[train][list(candidates)],data.iloc[train][target],
                                        params[(split.repeat,split.fold)])
            # Verify this explanatory refit describes exactly the scored model.
            scored=next(f for f in full.folds if (f.repeat_index,f.fold_index)==(split.repeat,split.fold))
            np.testing.assert_allclose(model.predict(data.iloc[test][list(candidates)]),
                [row.predicted for row in scored.observations],rtol=1e-12,atol=1e-12)
            if interaction_screen is not None:
                screen=screen_pair_interactions(model,data.iloc[test][list(candidates)],
                    candidates,outcome_sd=outcome_sd,config=interaction_screen)
                screen.update(repeat=split.repeat,fold=split.fold)
                pair_screens.append(screen)
            for i,source in enumerate(candidates):
                ref=reference[source]
                ale=accumulated_local_effects(model,data.iloc[test][list(candidates)],source,
                    edges=ref['edges'],outcome_sd=outcome_sd,
                    contrast=(ref['low'],ref['high']) if ref['low'] is not None else None)
                ale.update(repeat=split.repeat,fold=split.fold)
                # Absent references retain eleven unavailable points and ten
                # unavailable contrasts rather than dropping the predictor.
                ale.update(percentile_effects(ale,ref['percentile_values']))
                fold_effects[source].append(ale)
                if engine=='elastic_net':
                    pipeline=model._pipeline
                    b=float(pipeline['regressor'].coef_[i]/pipeline['preprocessor']['scaler'].scale_[i])
                    coefficients[source].append(b)
        effects=[]
        for source in candidates:
            folds=fold_effects[source]
            weights=[f['reference_count'] for f in folds]
            def combine(field):
                values=[f[field] for f in folds]
                return float(np.average(values,weights=weights)) if all(v is not None for v in values) and sum(weights)>0 else None
            npoints=len(reference[source]['edges'])
            curve=[]
            for i in range(npoints):
                values=[f['standardized_curve'][i] if len(f['standardized_curve'])==npoints else None for f in folds]
                curve.append(float(np.average(values,weights=weights)) if all(v is not None for v in values) and sum(weights)>0 else None)
            b=_mean_available(coefficients[source])
            beta=b*reference[source]['predictor_sd']/outcome_sd if b is not None and outcome_sd>0 and reference[source]['predictor_sd'] is not None else None
            magnitude_status=next((f['magnitude_status'] for f in folds if f['magnitude_status']!='available'),'available')
            magnitude=curve_magnitude(reference[source]['edges'],curve,observed[source].to_numpy(float,na_value=np.nan)) if magnitude_status=='available' else None
            differences=[]
            for i in range(npoints-1):
                values=[f['bin_differences'][i] if len(f['bin_differences'])==npoints-1 else None for f in folds]
                differences.append(float(np.average(values,weights=weights)) if all(v is not None for v in values) and sum(weights)>0 else None)
            combined=dict(edges=reference[source]['edges'],outcome_sd=outcome_sd,
                raw_curve=[v*outcome_sd if v is not None else None for v in curve],
                bin_differences=differences,magnitude_status=magnitude_status)
            percentiles=percentile_effects(combined,reference[source]['percentile_values'])
            effects.append(dict(engine=engine,source=source,target=target,
                low=reference[source]['low'],high=reference[source]['high'],outcome_sd=outcome_sd,
                raw_contrast=combine('raw_contrast'),standardized_contrast=combine('standardized_contrast'),
                status='available' if all(f['status']=='available' for f in folds) else next(f['status'] for f in folds if f['status']!='available'),
                delta_cv_r2=unique[source],b=b,beta=beta,standardized_curve=curve,
                edges=reference[source]['edges'],folds=folds,
                standardized_curve_sd=magnitude,curve_sd=magnitude*outcome_sd if magnitude is not None else None,
                magnitude_status=magnitude_status,**percentiles))
        from open.learners.trees import GRIDS
        from open.learners.elastic_net import ALPHA_GRID,L1_RATIO_GRID
        grid={'alpha':list(ALPHA_GRID),'l1_ratio':list(L1_RATIO_GRID)} if engine=='elastic_net' else GRIDS[engine]
        result['engines'].append(dict(engine=engine,cv_r2=1-full.aggregate_risk/variance if variance>0 else None,
            full_mse=full.aggregate_risk,baseline_mse=baseline.aggregate_risk,heldout_variance=variance,
            heldout_keys=[list(key) for key in keys],full_parameters=[dict(repeat=k[0],fold=k[1],params=v) for k,v in params.items()],
            tuning_grid=grid,tuning_grid_size=predictor.tuning_grid_size,ablations_retuned=False,
            scored_outer_fit_count=sum(r.outer_model_fit_count for r in fits),
            explanatory_refit_count=len(splits),tuning_model_fit_count=full.tuning_model_fit_count,
            effects=effects,joint=joint))
        if interaction_screen is not None:
            summary=aggregate_pair_screens(pair_screens)
            summary.update(config=asdict(interaction_screen),fold_screens=pair_screens)
            result['engines'][-1]['interactions']=summary
    return json.loads(json.dumps(result,allow_nan=False))


def save_engine_comparison(result, directory):
    """Write a versioned supplement without modifying network bundle schemas."""
    if result.get('schema')!='open.engine_comparison' or result.get('schema_version')!=1:
        raise ValueError("unsupported engine comparison schema")
    payload=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'comparison.json').write_text(payload,encoding='utf-8')
    fields=('engine','source','target','low','high','outcome_sd','raw_contrast',
            'standardized_contrast','status','delta_cv_r2','b','beta',
            'curve_sd','standardized_curve_sd','magnitude_status')
    with (directory/'effects.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore')
        writer.writeheader()
        for engine in result['engines']: writer.writerows(engine['effects'])
    for filename,key,fields in (
        ('percentile_points.csv','percentile_points',('percentile','predictor_value','raw_ale','standardized_ale','status')),
        ('interval_contrasts.csv','interval_contrasts',('low_percentile','high_percentile','low','high','raw_contrast','standardized_contrast','status')),
    ):
        with (directory/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,('engine','source','target',*fields))
            writer.writeheader()
            for engine in result['engines']:
                for effect in engine['effects']:
                    writer.writerows(dict(engine=effect['engine'],source=effect['source'],target=effect['target'],**row) for row in effect.get(key,()))
    if any('interactions' in e for e in result['engines']):
        fields=('engine','target','first','second','planned_folds','supported_screen_folds',
                'shortlisted_folds','supported_detail_folds','status','mean_screen_rms')
        with (directory/'interaction_pairs.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fields,extrasaction='ignore')
            writer.writeheader()
            for engine in result['engines']:
                for pair in engine.get('interactions',{}).get('pairs',[]):
                    writer.writerow(dict(engine=engine['engine'],target=result['target'],
                        first=pair['members'][0],second=pair['members'][1],**pair))
        fields=('engine','target','first','second','repeat','fold','stage','status',
                'i','j','count','raw_mean','raw_rms','standardized_mean','standardized_rms')
        with (directory/'interaction_cells.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fields,extrasaction='ignore')
            writer.writeheader()
            for engine in result['engines']:
                for fold in engine.get('interactions',{}).get('fold_screens',[]):
                    for pair in fold['pairs']:
                        for stage in ('screen','detail'):
                            diagnostic=pair[stage]
                            for cell in diagnostic['cells']:
                                writer.writerow(dict(engine=engine['engine'],target=result['target'],
                                    first=pair['members'][0],second=pair['members'][1],
                                    repeat=fold['repeat'],fold=fold['fold'],stage=stage,
                                    status=diagnostic['status'],**cell))
        from open.interaction_visualization import save_interaction_report
        save_interaction_report(result,directory)
    else:
        # These derived files belong to the optional supplement, not the current
        # opt-out payload. Never leave an older pair report beside a new run.
        for filename in ('interaction_pairs.csv','interaction_cells.csv','INTERACTIONS.md'):
            (directory/filename).unlink(missing_ok=True)
