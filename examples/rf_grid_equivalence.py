"""Matched RF singleton-grid probe. Does not change production defaults.

Run serially: scoped patches instrument real fits and temporarily supply the
candidate grid to the existing TreePredictor. Never import this into a worker.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from open.learners import trees


def require_parity(original, reduced):
    """Fail closed on numerical or tie-selection differences; no tolerance."""
    assert original['selected'] == reduced['selected'], 'selected parameters differ'
    np.testing.assert_array_equal(original['scores'][:2], original['scores'][2:],
                                  err_msg='original candidates are not duplicates')
    np.testing.assert_array_equal(original['scores'][:2], reduced['scores'],
                                  err_msg='CV scores differ')
    np.testing.assert_array_equal(original['predictions'], reduced['predictions'],
                                  err_msg='held-out predictions differ')


def _run(X, y, heldout, splits, seed, reduced):
    grid = {'min_samples_leaf': [2, 5], 'max_features': [.7] if reduced else [.7, 1.]}
    counts = {'tuning': 0, 'final': 0}
    searches = []
    phase = 'tuning'
    original_fit = RandomForestRegressor.fit
    original_search = trees.GridSearchCV

    def counted_fit(estimator, *args, **kwargs):
        counts[phase] += 1
        return original_fit(estimator, *args, **kwargs)

    def recorded_search(*args, **kwargs):
        search = original_search(*args, **kwargs)
        searches.append(search)
        return search

    with patch.dict(trees.GRIDS, {'random_forest': grid}), \
            patch.object(RandomForestRegressor, 'fit', counted_fit), \
            patch.object(trees, 'GridSearchCV', recorded_search):
        predictor = trees.TreePredictor('random_forest', random_seed=seed)
        start = perf_counter()
        params = predictor.tune(X, y, splits)
        seconds = perf_counter() - start
        reported = predictor.tuning_grid_size * len(splits)
        assert counts['tuning'] == reported, 'reported tuning fit count differs from actual'
        phase = 'final'
        predictor.fit(X, y, params)
        predictions = predictor.predict(heldout)
        search = searches[0]
        scores = search.cv_results_['mean_test_score'].tolist()
        fold_scores = [search.cv_results_[f'split{i}_test_score'].tolist()
                       for i in range(len(splits))]
    return dict(selected=params, scores=scores, fold_scores=fold_scores,
                predictions=predictions.tolist(), tuning_seconds=seconds,
                actual_tuning_fits=counts['tuning'], actual_final_fits=counts['final'],
                reported_tuning_fits=reported)


def compare(X, y, heldout, *, seed, repeats=4):
    """Compare both grids on identical folds, alternating execution order."""
    if np.ndim(X) != 2 or X.shape[1] != 1 or np.ndim(heldout) != 2 or heldout.shape[1] != 1:
        raise ValueError('probe requires exactly one predictor')
    if type(repeats) is not int or not 1 <= repeats <= 6:
        raise ValueError('repeats must be an integer in [1, 6]')
    splits = list(KFold(3, shuffle=True, random_state=seed).split(X))
    pairs = []
    with threadpool_limits(limits=1):
        for repetition in range(repeats):
            pair = {}
            order = [('original', False), ('deduplicated', True)]
            if repetition % 2:
                order.reverse()
            for label, reduced in order:
                pair[label] = _run(X, y, heldout, splits, seed, reduced)
            require_parity(pair['original'], pair['deduplicated'])
            for full, reduced in zip(pair['original']['fold_scores'], pair['deduplicated']['fold_scores']):
                np.testing.assert_array_equal(full[:2], full[2:])
                np.testing.assert_array_equal(full[:2], reduced)
            pair['order'] = [label for label, _ in order]
            pairs.append(pair)
    old = median(p['original']['tuning_seconds'] for p in pairs)
    new = median(p['deduplicated']['tuning_seconds'] for p in pairs)
    return dict(exact_parity=True, seed=seed, training_rows=len(y),
                test_rows=len(heldout), inner_folds=3, pairs=pairs,
                median_original_tuning_seconds=old,
                median_deduplicated_tuning_seconds=new,
                tuning_time_reduction=1-new/old)


def fixture(n, scenario):
    """Explicit probe design, not a reliability-study dataset generator."""
    seed = 615+n
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1))
    y = 2*X[:, 0] + rng.normal(scale=.5, size=n)
    if scenario == 'u_shape':
        y = X[:, 0]**2 + rng.normal(scale=.5, size=n)
    elif scenario == 'missing':
        X[rng.random(n) < .2, 0] = np.nan
    elif scenario == 'constant_predictor':
        X[:] = 1.
    elif scenario == 'all_missing_predictor':
        X[:] = np.nan
    elif scenario == 'constant_target':
        y[:] = 1.
    elif scenario != 'linear':
        raise ValueError('unknown probe scenario')
    cut = int(.8*n)
    return X[:cut], y[:cut], X[cut:], seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for n in (80, 200):
        for scenario in ('linear', 'u_shape', 'missing', 'constant_predictor',
                         'all_missing_predictor', 'constant_target'):
            X, y, heldout, seed = fixture(n, scenario)
            name = f'{scenario}-n{n}'
            np.savez(args.output/f'{name}.npz', X=X, y=y, heldout=heldout)
            result = compare(X, y, heldout, seed=seed, repeats=args.repeats)
            result.update(id=name, scenario=scenario, n=n,
                          dataset_sha256=sha256((args.output/f'{name}.npz').read_bytes()).hexdigest())
            cases.append(result)
            print(name, 'exact parity', f"tuning time reduction {result['tuning_time_reduction']:.1%}", flush=True)
    source_paths = ('src/open/learners/trees.py', 'src/open/preprocessing.py',
                    'src/open/risk.py', 'examples/rf_grid_equivalence.py')
    report = dict(schema='open.rf_grid_equivalence', schema_version=1,
                  production_defaults_changed=False, scientific_qualification=False,
                  python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                  source_sha256={p:sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
                  cases=cases, completed_cases=len(cases),
                  matched_pairs=sum(len(c['pairs']) for c in cases),
                  median_case_tuning_time_reduction=median(c['tuning_time_reduction'] for c in cases))
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
