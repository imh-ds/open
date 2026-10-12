"""Bounded exploratory nonadditivity diagnostics for already fitted models.

Four-corner local differences, not a complete second-order ALE estimator,
population interaction test or feature-generation/selection procedure.
"""
from dataclasses import asdict, dataclass
from itertools import combinations
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairScreenConfig:
    max_pairs: int = 250
    screen_rows: int = 64
    screen_bins: int = 2
    max_details: int = 5
    detail_rows: int = 256
    detail_bins: int = 4
    min_cell: int = 3
    prediction_rows: int = 100000
    seed: int = 0
    adaptive_details: bool = False

    def __post_init__(self):
        if type(self.adaptive_details) is not bool:
            raise ValueError('adaptive_details must be boolean')
        for name in ('max_pairs','max_details','prediction_rows','seed'):
            value = getattr(self,name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('screen_rows','detail_rows','min_cell'):
            value = getattr(self,name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('screen_bins','detail_bins'):
            value = getattr(self,name)
            if type(value) is not int or value < 2:
                raise ValueError(f'{name} must be an integer at least two')


def _unavailable(status):
    return dict(status=status,standardized_rms=None,raw_rms=None,
                prediction_rows=0,reference_rows=0,cells=[])


def _moments(values,outcome_sd):
    scale=float(np.max(np.abs(values)))
    normalized=values/scale if scale else values
    mean=float(np.mean(normalized))*scale
    rms=float(np.sqrt(np.mean(np.square(normalized))))*scale
    standardized_mean=mean/outcome_sd
    standardized_rms=rms/outcome_sd
    if not all(np.isfinite(v) for v in (mean,rms,standardized_mean,standardized_rms)):
        raise ValueError('contrast summaries exceed finite numeric range')
    return mean,rms,standardized_mean,standardized_rms


def _contrast(model,reference,pair,rows,bins,config,outcome_sd,remaining,*,partial=False):
    if outcome_sd == 0:
        return _unavailable('constant_outcome')
    valid = np.isfinite(reference[list(pair)].to_numpy(float)).all(axis=1)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return _unavailable('missing_reference')
    rng = np.random.default_rng(config.seed)
    if len(positions) > rows:
        positions = np.sort(rng.choice(positions,size=rows,replace=False))
    X = reference.iloc[positions].copy()
    edges = [np.unique(np.quantile(reference.loc[valid,name],np.linspace(0,1,bins+1)))
             for name in pair]
    if any(len(e) < 2 for e in edges):
        return _unavailable('constant_predictor')
    indices = [np.clip(np.searchsorted(e,X[name].to_numpy(float),side='right')-1,0,len(e)-2)
               for name,e in zip(pair,edges)]
    counts = np.zeros((len(edges[0])-1,len(edges[1])-1),dtype=int)
    np.add.at(counts,tuple(indices),1)
    cells = [dict(i=i,j=j,count=int(counts[i,j]),raw_mean=None,raw_rms=None,
                  standardized_mean=None,standardized_rms=None)
             for i in range(counts.shape[0]) for j in range(counts.shape[1])]
    metadata = dict(edges=[e.tolist() for e in edges],reference_rows=len(X),
        missing_reference_rows=int((~valid).sum()),reference_positions=positions.tolist(),
        cells=cells,requested_bins=bins,realized_bins=list(counts.shape))
    supported=counts>=config.min_cell
    if partial:
        for cell in cells:cell['supported']=bool(supported[cell['i'],cell['j']])
        keep=supported[tuple(indices)]
        metadata.update(coverage='complete' if supported.all() else 'partial',
            supported_reference_rows=int(keep.sum()),supported_cells=int(supported.sum()),
            evaluated_reference_positions=positions[keep].tolist())
    else:
        keep=np.ones(len(X),dtype=bool)
    if (not supported.any()) or (not partial and not supported.all()):
        return {**_unavailable('unsupported_cells'),**metadata}
    X=X.iloc[np.flatnonzero(keep)].copy()
    indices=[index[keep] for index in indices]
    cost = 4*len(X)
    if cost > remaining:
        return {**_unavailable('prediction_budget'),**metadata}
    corners = []
    # Order HH, HL, LH, LL holds all other features at observed reference rows.
    for first,second in ((1,1),(1,0),(0,1),(0,0)):
        corner = X.copy()
        corner[pair[0]] = edges[0][indices[0]+first]
        corner[pair[1]] = edges[1][indices[1]+second]
        corners.append(corner)
    predictions = np.asarray(model.predict(pd.concat(corners,ignore_index=True)),dtype=float)
    if predictions.shape != (cost,) or not np.isfinite(predictions).all():
        raise ValueError('model predictions must be finite with one value per row')
    hh,hl,lh,ll = predictions.reshape(4,len(X))
    differences = hh-hl-lh+ll
    if not np.isfinite(differences).all():
        raise ValueError('mixed prediction contrasts must be finite')
    _,rms,_,standardized_rms=_moments(differences,outcome_sd)
    for cell in cells:
        if partial and not cell['supported']:continue
        subset = differences[(indices[0]==cell['i']) & (indices[1]==cell['j'])]
        mean,magnitude,standardized_mean,standardized_magnitude=_moments(subset,outcome_sd)
        cell.update(raw_mean=mean,raw_rms=magnitude,
            standardized_mean=standardized_mean,standardized_rms=standardized_magnitude)
    return dict(status='available',raw_rms=rms,standardized_rms=standardized_rms,
                prediction_rows=cost,**metadata)


def _adaptive_detail(model,reference,pair,config,outcome_sd,remaining):
    attempts=[]
    for bins in range(config.detail_bins,1,-1):
        value=_contrast(model,reference,pair,config.detail_rows,bins,config,outcome_sd,remaining)
        attempts.append(dict(bins=bins,status=value['status']))
        if value['status']!='unsupported_cells':
            return dict(value,adaptive=True,coverage='complete' if value['status']=='available' else 'unavailable',
                initial_requested_bins=config.detail_bins,attempts=attempts)
    value=_contrast(model,reference,pair,config.detail_rows,2,config,outcome_sd,remaining,partial=True)
    return dict(value,adaptive=True,initial_requested_bins=config.detail_bins,attempts=attempts)


def screen_pair_interactions(model,reference,candidates,*,outcome_sd,
                             config=PairScreenConfig()):
    """Query fitted predictions with hard caps; retain every planned pair/status.

    Supported pairwise differences can include higher-order interaction effects.
    Nonzero fitted scores and shortlist rank are descriptive, not validation.
    """
    if not isinstance(config,PairScreenConfig):
        raise ValueError('config must be PairScreenConfig')
    if (isinstance(outcome_sd,bool) or not isinstance(outcome_sd,Real)
            or not np.isfinite(outcome_sd) or outcome_sd < 0):
        raise ValueError('outcome_sd must be finite and nonnegative')
    if (not isinstance(reference,pd.DataFrame) or not reference.columns.is_unique
            or not isinstance(candidates,tuple) or len(set(candidates)) != len(candidates)
            or any(type(n) is not str or n not in reference for n in candidates)):
        raise ValueError('candidates must be distinct reference column names')
    if set(candidates) != set(reference.columns):
        raise ValueError('reference must contain exactly the model candidate columns')
    # Candidate order is the feature order used to fit the supplied estimator.
    reference=reference.loc[:,list(candidates)]
    planned = list(combinations(sorted(candidates),2))
    used = 0
    pairs = []
    for index,pair in enumerate(planned):
        value = (_unavailable('pair_limit') if index >= config.max_pairs else
            _contrast(model,reference,pair,config.screen_rows,config.screen_bins,
                      config,float(outcome_sd),config.prediction_rows-used))
        used += value['prediction_rows']
        pairs.append(dict(members=list(pair),screen=value,selected_for_detail=False,
                          detail=_unavailable('not_shortlisted')))
    # Numerical cancellation tolerance is not a scientific interaction cutoff.
    supported = [p for p in pairs if p['screen']['status']=='available'
                 and p['screen']['standardized_rms'] > 1e-12]
    supported.sort(key=lambda p:(-p['screen']['standardized_rms'],p['members']))
    for pair in supported[:config.max_details]:
        pair['selected_for_detail'] = True
        value = (_adaptive_detail(model,reference,tuple(pair['members']),config,float(outcome_sd),
            config.prediction_rows-used) if config.adaptive_details else
            _contrast(model,reference,tuple(pair['members']),config.detail_rows,
                config.detail_bins,config,float(outcome_sd),config.prediction_rows-used))
        pair['detail'] = value
        used += value['prediction_rows']
    return dict(schema='open.pair_interaction_screen',schema_version=1,
        inference='exploratory_fitted_model_nonadditivity',config=asdict(config),
        outcome_sd=float(outcome_sd),planned_pairs=len(planned),
        screened_pairs=sum(p['screen']['status']=='available' for p in pairs),
        shortlisted_pairs=sum(p['selected_for_detail'] for p in pairs),
        detailed_pairs=sum(p['detail']['status']=='available' for p in pairs),
        prediction_rows=used,additional_model_fits=0,pairs=pairs)


def aggregate_pair_screens(screens):
    """Descriptive fold coverage with fixed planned denominators, no inference."""
    if not screens:
        return dict(planned_folds=0,pairs=[],prediction_rows=0,additional_model_fits=0)
    identities = [p['members'] for p in screens[0]['pairs']]
    if any([p['members'] for p in s['pairs']] != identities for s in screens):
        raise ValueError('pair identities must match across planned folds')
    result = []
    for i,pair in enumerate(identities):
        folds = [s['pairs'][i] for s in screens]
        values = [f['screen']['standardized_rms'] for f in folds if f['screen']['status']=='available']
        result.append(dict(members=pair,planned_folds=len(screens),
            supported_screen_folds=len(values),
            shortlisted_folds=sum(f['selected_for_detail'] for f in folds),
            supported_detail_folds=sum(f['detail']['status']=='available' for f in folds),
            status='complete_descriptive' if len(values)==len(screens) else 'incomplete_support',
            mean_screen_rms=float(np.mean(values)) if len(values)==len(screens) else None,
            folds=folds))
    return dict(planned_folds=len(screens),pairs=result,
        prediction_rows=sum(s['prediction_rows'] for s in screens),
        additional_model_fits=0,
        inference='descriptive_overlapping_cv_folds_not_stability_probability')
