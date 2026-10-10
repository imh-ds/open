"""Outer evaluation of complete layer-consensus discovery on independent rows.

The configuration must be fixed before calling this workflow. Outer outcomes
are used only for scoring; they never select candidates, settings or engines.
"""
from dataclasses import dataclass, replace
import hashlib
from typing import Mapping
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from open.config import OpenConfig
from open.graph import _freeze
from open.layer_stability import fit_layer_consensus
from open.layer_types import LayeredOPENResult
from open.learners import make_predictor
from open.recurse import _validate_inputs


@dataclass(frozen=True)
class ProcedurePrediction:
    row_position: int
    observed: float
    baseline: float
    reduced: float
    full: float


@dataclass(frozen=True)
class ProcedureFold:
    fold: int
    seed: int
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    status: str
    network: LayeredOPENResult | None
    selected_predictors: tuple[str, ...]
    reduced_parameters: Mapping[str, object] | None
    full_parameters: Mapping[str, object] | None
    predictions: tuple[ProcedurePrediction, ...]
    warnings: tuple[str, ...]
    failure_type: str | None = None
    failure_message: str | None = None

    def __post_init__(self):
        for name in ('reduced_parameters', 'full_parameters'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class ProcedureEvaluation:
    config: OpenConfig
    variables: tuple[str, ...]
    data_hash: str
    n_rows: int
    outer_seed: int
    folds: tuple[ProcedureFold, ...]
    scores: Mapping[str, Mapping[str, float | None]] | None

    def __post_init__(self):
        object.__setattr__(self, 'scores', _freeze(self.scores))

    @property
    def status(self):
        return 'COMPLETE' if all(f.status == 'completed' for f in self.folds) else 'INCOMPLETE'


def data_fingerprint(data, variables):
    """Hash ordered analyzed values/columns; index labels are not row identities."""
    content = data.loc[:, list(variables)].reset_index(drop=True).to_json(
        orient='split', double_precision=15)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _refit_predict(train, test, root, predictors, config):
    if not predictors:
        return np.full(len(test), float(train[root].mean())), None
    model = make_predictor(config)
    inner = tuple(KFold(config.tuning_cv.folds, shuffle=True,
                        random_state=config.analysis.random_seed).split(train))
    X, y = train.loc[:, list(predictors)], train[root]
    params = dict(model.tune(X, y, inner, config.outcome.loss))
    model.fit(X, y, params)
    predicted = np.asarray(model.predict(test.loc[:, list(predictors)]), dtype=float)
    if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
        raise ValueError('prediction must provide one finite value per held-out participant')
    return predicted, params


def _scores(folds):
    if any(f.status != 'completed' for f in folds):
        return None
    predictions = tuple(p for f in folds for p in f.predictions)
    y = np.array([p.observed for p in predictions])
    variance = float(np.sum((y - y.mean())**2))
    baseline_error = float(sum((p.observed-p.baseline)**2 for p in predictions))
    result = {}
    for name in ('baseline', 'reduced', 'full'):
        error = float(sum((p.observed-getattr(p, name))**2 for p in predictions))
        result[name] = dict(mse=error/len(y), rmse=float(np.sqrt(error/len(y))),
                            r_squared=1-error/variance if variance > 0 else None,
                            prediction_skill=1-error/baseline_error if baseline_error > 0 else None)
    return result


def evaluate_procedure(data: pd.DataFrame, root: str, variables: tuple[str, ...],
                       config: OpenConfig, *, outer_folds: int = 5,
                       outer_seed: int | None = None) -> ProcedureEvaluation:
    """Rebuild discovery within every training fold, then score root prediction.

    Includes shared-group members in the reduced model. Models are separately
    tuned on outer training data; these are not LOCO ablation scores. Incomplete
    discovery anywhere in a fold withholds that fold's prediction. All planned
    folds are attempted; aggregate scores require every fold to complete.
    """
    _validate_inputs(data, root, variables, config)
    if config.stability.scope != 'layer_consensus':
        raise ValueError('procedure evaluation requires layer_consensus scope')
    if type(outer_folds) is not int or not 2 <= outer_folds <= len(data):
        raise ValueError('outer_folds must be an integer between two and row count')
    seed = config.analysis.random_seed if outer_seed is None else outer_seed
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('outer_seed must be an integer in [0, 2**32)')
    if config.analysis.random_seed + outer_folds + config.stability.n_resamples - 2 >= 2**32:
        raise ValueError('evaluation discovery seeds exceed [0, 2**32)')
    frame = data.loc[:, list(variables)].reset_index(drop=True).copy()
    candidates = tuple(n for n in variables if n != root)
    folds = []
    for index, (training, testing) in enumerate(KFold(outer_folds, shuffle=True, random_state=seed).split(frame)):
        fold_seed = config.analysis.random_seed + index
        fold_config = replace(config, analysis=replace(config.analysis, random_seed=fold_seed))
        train, test = frame.iloc[training].reset_index(drop=True), frame.iloc[testing].reset_index(drop=True)
        network = None
        selected = predictions = ()
        reduced_params = full_params = failure_type = failure_message = None
        status = 'completed'
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            try:
                network = fit_layer_consensus(train, root, variables, fold_config)
                if network.status != 'COMPLETE':
                    raise ValueError(f'discovery status {network.status}; no complete-procedure prediction')
                selected = tuple(n for n in candidates if root in network.parents.get(n, ()))
                baseline = float(train[root].mean())
                reduced, reduced_params = _refit_predict(train, test, root, selected, fold_config)
                full, full_params = _refit_predict(train, test, root, candidates, fold_config)
                predictions = tuple(ProcedurePrediction(int(pos), float(frame.iloc[pos][root]),
                    baseline, float(reduced[j]), float(full[j])) for j, pos in enumerate(testing))
            except Exception as exc:
                status = 'failed'
                failure_type, failure_message = type(exc).__name__, str(exc)[:2000] or type(exc).__name__
        messages = tuple(f'{w.category.__name__}: {w.message}' for w in captured)
        folds.append(ProcedureFold(index, fold_seed, tuple(map(int, training)), tuple(map(int, testing)),
            status, network, selected, reduced_params, full_params, predictions, messages,
            failure_type, failure_message))
    return ProcedureEvaluation(config, variables, data_fingerprint(frame, variables), len(frame),
                               seed, tuple(folds), _scores(folds))
