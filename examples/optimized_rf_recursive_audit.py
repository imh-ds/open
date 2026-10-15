"""Explicit RF singleton-v1 audit; historical four-candidate audits stay intact."""
from contextlib import contextmanager
from threading import Lock

import multiengine_recursive_audit as legacy
from open.layer_artifacts import portable, reconstruct_payloads
from recursive_preview_equivalence import compare_payloads

POLICY = 'rf-singleton-v1'
NOMINAL_GRID = dict(min_samples_leaf=[2, 5], max_features=[.7, 1.])
SINGLETON_GRID = dict(min_samples_leaf=[2, 5], max_features=[.7])
Store = legacy.Store
_lock = Lock()
_original_verify = legacy.verify_tuning


def expected_tuning(risk, config, tuned):
    if config.learner.library != ('random_forest',):
        raise ValueError('singleton-v1 audit requires RF')
    if not tuned:
        return 0
    width = len(risk['predictors'])
    if width < 1:
        raise ValueError('baseline cannot be tuned')
    return len(risk['folds'])*(2 if width == 1 else 4)*config.tuning_cv.folds


def verify_tuning(risk, config, tuned):
    expected = expected_tuning(risk, config, tuned)
    if risk['tuning_model_fit_count'] != expected:
        raise ValueError('saved tuning count differs from explicit singleton-v1 policy')
    if tuned:
        grid = SINGLETON_GRID if len(risk['predictors']) == 1 else NOMINAL_GRID
        for fold in risk['folds']:
            params = fold['fitted_parameters']
            if (not legacy.valid_parameters(params, 'random_forest')
                    or any(params[k] not in grid[k] for k in grid)):
                raise ValueError('tuned parameters differ from effective search grid')
    return expected


@contextmanager
def _policy_scope():
    # One audit per process. Fail closed rather than share a temporary policy
    # across threads or nest it inside another override. No estimator is fitted.
    if not _lock.acquire(blocking=False):
        raise RuntimeError('concurrent audit policy override')
    try:
        if legacy.verify_tuning is not _original_verify:
            raise RuntimeError('historical audit policy already overridden')
        legacy.verify_tuning = verify_tuning
        try:
            yield
        finally:
            legacy.verify_tuning = _original_verify
    finally:
        _lock.release()


def audit_case(spec, store, manifest, **kwargs):
    if manifest.get('grid_policy') != POLICY:
        raise ValueError('explicit grid policy required')
    with _policy_scope():
        return legacy.audit_case(spec, store, manifest, **kwargs)


def audit_checkpoint(*args, **kwargs):
    with _policy_scope():
        return legacy.audit_checkpoint(*args, **kwargs)


def compare_statistics(current, original):
    # Counts are checked independently against their declared policies. Every
    # scientific decision, parameter and observation remains in this comparison.
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k != 'tuning_model_fit_count'}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return compare_payloads(strip(current), strip(original))


def compare_reference(current, original):
    def network(store):
        p = store.compressed('network.json.gz')
        n, _ = reconstruct_payloads(*(p[k] for k in
            ('analysis_metadata.json', 'layer_evidence.json', 'assessment_ledger.json')))
        return portable({k: getattr(n, k) for k in ('layers', 'config', 'variables',
            'plans', 'assigned_depth', 'parents', 'stop_reasons', 'status')})

    report = compare_statistics(network(current), network(original))
    paths = {p for p in current.names if p.startswith('checkpoints/') and p.endswith('.json.gz')}
    old_paths = {p for p in original.names if p.startswith('checkpoints/') and p.endswith('.json.gz')}
    if paths != old_paths:
        raise ValueError('reference checkpoint coverage differs')
    for path in sorted(paths):
        row = compare_statistics(current.compressed(path)['risks'], original.compressed(path)['risks'])
        report['equivalent'] = report['equivalent'] and row['equivalent']
        report['changed_float_fields'] += row['changed_float_fields']
        report['max_absolute_difference'] = max(report['max_absolute_difference'], row['max_absolute_difference'])
        report['mismatches'].extend(dict(path=path+'/'+r['path'], reason=r['reason'])
                                    for r in row['mismatches'][:100])
    report['checkpoints_compared'] = len(paths)
    return report
