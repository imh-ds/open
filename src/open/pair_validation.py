"""Experimental held-out additive-versus-joint prediction comparisons.

This diagnostic model family does not extract coefficients from tree engines,
certify moderation or select network edges. Candidate selection must use training
data only. Tensor features are built only in the separate validation model.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


@dataclass(frozen=True)
class PairValidationConfig:
    max_pairs: int = 2
    max_fits: int = 100
    knots: tuple = (3, 5)
    alphas: tuple = (.1, 1., 10., 100.)
    inner_folds: int = 3
    seed: int = 0

    def __post_init__(self):
        for name in ('max_pairs','max_fits','seed'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if type(self.inner_folds) is not int or self.inner_folds<2:
            raise ValueError('inner_folds must be at least two')
        if (type(self.knots) is not tuple or not self.knots or
            any(type(k) is not int or k<2 for k in self.knots) or len(set(self.knots))!=len(self.knots)):
            raise ValueError('knots must be distinct integers at least two')
        if (type(self.alphas) is not tuple or not self.alphas or any(
            isinstance(a,bool) or not isinstance(a,(float,int)) or not np.isfinite(a) or a<=0
            for a in self.alphas) or len(set(self.alphas))!=len(self.alphas)):
            raise ValueError('alphas must be distinct positive finite numbers')


class SplinePairFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,n_knots=3,pair=None):
        self.n_knots=n_knots
        self.pair=pair

    def fit(self,X,y=None):
        self.splines_=SplineTransformer(n_knots=self.n_knots,degree=3,knots='quantile',
            extrapolation='linear',include_bias=False).fit(X)
        self.width_=self.splines_.n_features_out_//X.shape[1]
        base=self.splines_.transform(X)
        self.marginal_scaler_=StandardScaler().fit(base)
        if self.pair is not None:
            a,b=self._pair_blocks(base)
            marginal=np.column_stack([np.ones(len(X)),a,b])
            tensor=(a[:,:,None]*b[:,None,:]).reshape(len(X),-1)
            # Remove directions already spanned by this pair's marginal curves.
            # All fitting/projection uses this training partition only.
            self.projection_=np.linalg.lstsq(marginal,tensor,rcond=None)[0]
        return self

    def _pair_blocks(self,base):
        return tuple(base[:,i*self.width_:(i+1)*self.width_] for i in self.pair)

    def transform(self,X):
        base=self.splines_.transform(X)
        marginal=self.marginal_scaler_.transform(base)
        if self.pair is None:
            return marginal
        a,b=self._pair_blocks(base)
        tensor=(a[:,:,None]*b[:,None,:]).reshape(len(X),-1)
        residual=tensor-np.column_stack([np.ones(len(X)),a,b])@self.projection_
        # Tensor columns stay on their natural spline scale. Dividing each by
        # its own (possibly almost zero) training SD destroys ridge shrinkage
        # for weakly supported directions and can explode held-out predictions.
        return np.column_stack([marginal,residual])


def _estimator(pair=None):
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('curves',SplinePairFeatures(pair=pair)),
        ('ridge',Ridge(solver='lsqr',tol=1e-8))])


def _tune(X,y,pair,config):
    pipeline=_estimator(pair)
    splits=list(KFold(config.inner_folds,shuffle=True,random_state=config.seed).split(X))
    search=GridSearchCV(pipeline,{'curves__n_knots':list(config.knots),
        'ridge__alpha':list(config.alphas)},cv=splits,scoring='neg_mean_squared_error',
        refit=True,error_score='raise',n_jobs=1).fit(X,y)
    return search.best_estimator_,dict(search.best_params_),[len(a) for a,b in splits]


def validate_pairs(X,y,train_indices,assessment_indices,pairs,*,config=PairValidationConfig()):
    """Compare curves with/without each preselected pair on disjoint observations.

    Caller must select supplied pairs using training observations only. Missing
    budgets/results are unavailable, never negative evidence of an interaction.
    """
    if not isinstance(config,PairValidationConfig):
        raise ValueError('config must be PairValidationConfig')
    if not isinstance(X,pd.DataFrame) or not X.columns.is_unique or len(X.columns)<2:
        raise ValueError('X must have at least two distinct numeric columns')
    if any(type(c) is not str for c in X.columns):
        raise ValueError('feature names must be strings')
    values=X.to_numpy(dtype=float)
    y=np.asarray(y,dtype=float)
    if y.shape!=(len(X),) or not np.isfinite(y).all() or np.isinf(values).any():
        raise ValueError('outcome must be finite; predictors cannot contain infinity')
    for positions in (train_indices,assessment_indices):
        if (type(positions) is not tuple or not positions or len(set(positions))!=len(positions)
            or any(type(i) is not int or not 0<=i<len(X) for i in positions)):
            raise ValueError('split indices must be distinct valid integer tuples')
    if set(train_indices)&set(assessment_indices):
        raise ValueError('training and assessment indices must be disjoint')
    if len(train_indices)<2*config.inner_folds:
        raise ValueError('training sample too small for inner folds')
    if (type(pairs) is not tuple or any(
        type(p) is not tuple or len(p)!=2 or p[0]==p[1] or any(n not in X for n in p) for p in pairs)):
        raise ValueError('pairs must be distinct two-feature tuples')
    if len({frozenset(p) for p in pairs})!=len(pairs):
        raise ValueError('pairs must be distinct two-feature tuples')
    train=list(train_indices);assessment=list(assessment_indices)
    cost=len(config.knots)*len(config.alphas)*config.inner_folds+1
    records=[dict(members=list(p),status='pair_limit' if i>=config.max_pairs else 'fit_budget',
        gain=None,predicted=None,baseline_predicted=None) for i,p in enumerate(pairs)]
    result=dict(schema='open.experimental_pair_validation.v2',config=asdict(config),
        feature_scaling='standardized_marginals_natural_joint',
        solver=dict(name='lsqr',tol=1e-8),
        train_indices=train,assessment_indices=assessment,additional_fits=0,
        baseline_parameters=None,pairs=records,
        interpretation='heldout predictive gain; no significance or certified moderation')
    if not pairs or not config.max_pairs or cost>config.max_fits:
        return result
    if np.std(y[train])==0:
        for r in records:
            if r['status']!='pair_limit':r['status']='constant_training_outcome'
        return result
    base,params,sizes=_tune(X.iloc[train],y[train],None,config)
    predicted=base.predict(X.iloc[assessment])
    observed=y[assessment]
    base_mse=float(np.square(observed-predicted).mean())
    reference_mse=float(np.square(observed-y[train].mean()).mean())
    result.update(additional_fits=cost,baseline_parameters=params,
        baseline_mse=base_mse,reference_mse=reference_mse,observed=observed.tolist(),
        inner_training_n=sizes)
    for record,pair in zip(records,pairs):
        if record['status']=='pair_limit':continue
        if result['additional_fits']+cost>config.max_fits:continue
        joint,settings,_=_tune(X.iloc[train],y[train],tuple(X.columns.get_loc(n) for n in pair),config)
        joint_predicted=joint.predict(X.iloc[assessment])
        mse=float(np.square(observed-joint_predicted).mean())
        gain=(base_mse-mse)/reference_mse if reference_mse>0 else None
        record.update(status='available' if gain is not None else 'constant_assessment_baseline',
            gain=gain,joint_mse=mse,baseline_mse=base_mse,reference_mse=reference_mse,
            predicted=joint_predicted.tolist(),baseline_predicted=predicted.tolist(),parameters=settings)
        result['additional_fits']+=cost
    return result
