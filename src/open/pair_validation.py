"""Additive spline ridge model used by exploratory-mode interaction tests.

``_tune`` fits the additive spline model (one curve per predictor) that
``open.explore`` uses as the baseline for interaction gains.
"""
from dataclasses import dataclass

import numpy as np
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
