"""Additive spline ridge model used by discovery-stage interaction tests.

``_tune`` fits the additive spline model (one curve per predictor) that
``open.explore`` uses as the baseline for interaction gains.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.utils import _safe_indexing, indexable


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


def _tune(X, y, pair, config):
    """Search the unchanged grid while reusing each fold's feature preparation.

    Imputation depends on the training split; spline features also depend on
    the knot count. Ridge penalties share those features, never fitted models.
    The selected complete pipeline is then fitted afresh on all supplied rows.
    """
    X, y = indexable(X, y)
    pipeline = _estimator(pair)
    splits = list(KFold(config.inner_folds, shuffle=True, random_state=config.seed).split(X))
    scores = np.empty((len(config.knots) * len(config.alphas), len(splits)))
    scorer = get_scorer("neg_mean_squared_error")
    for fold, (train, test) in enumerate(splits):
        ytrain, ytest = _safe_indexing(y, train), _safe_indexing(y, test)
        imputer = clone(pipeline["imputer"])
        train_imputed = imputer.fit_transform(_safe_indexing(X, train), ytrain)
        test_imputed = imputer.transform(_safe_indexing(X, test))
        for knot_index, knots in enumerate(config.knots):
            curves = clone(pipeline["curves"]).set_params(n_knots=knots)
            Xtrain = curves.fit_transform(train_imputed, ytrain)
            Xtest = curves.transform(test_imputed)
            for alpha_index, alpha in enumerate(config.alphas):
                ridge = clone(pipeline["ridge"]).set_params(alpha=alpha).fit(Xtrain, ytrain)
                index = knot_index * len(config.alphas) + alpha_index
                scores[index, fold] = scorer(ridge, Xtest, ytest)
    # GridSearchCV orders curves__n_knots before ridge__alpha, preserves the
    # configured value order, and chooses the first candidate in a mean tie.
    means = scores.mean(axis=1)
    if not np.isfinite(means).all():
        warnings.warn(f"One or more of the test scores are non-finite: {means}", UserWarning)
    best = int(np.argmax(np.where(np.isnan(means), -np.inf, means)))
    knot_index, alpha_index = divmod(best, len(config.alphas))
    params = {"curves__n_knots": config.knots[knot_index], "ridge__alpha": config.alphas[alpha_index]}
    fitted = pipeline.set_params(**params).fit(X, y)
    return fitted, params, [len(train) for train, _ in splits]
