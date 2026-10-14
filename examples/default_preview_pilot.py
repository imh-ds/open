"""Bounded shipped-default fixture pilot; no scientific error-rate claim."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

CASES = ('chain', 'chain_missing', 'shared_predictors')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    import pandas as pd
    import open
    from open.config import load_config
    from open.layer_artifacts import portable
    from open.layer_stability import fit_layer_consensus
    from open.layer_io import write_layer_outputs, load_layer_bundle
    from open.metadata import validate_metadata
    root = Path(__file__).resolve().parents[1]
    sample = root / 'examples/results/2026-10-08-layer-fixtures' / a.case / 'sample.csv'
    config_file = root / 'configs/default.yaml'
    c = load_config(config_file)
    target = 'Outcome' if a.case == 'shared_predictors' else 'Anxiety'
    c = replace(c, analysis=replace(c.analysis, root=target))
    assert (c.stability.n_resamples, c.performance_cv.folds,
            c.performance_cv.repeats, c.tuning_cv.folds) == (100, 5, 2, 3)
    assert c.thresholds.unique_relative == .05
    a.out.mkdir(parents=True, exist_ok=False)
    receipt = dict(case=a.case, status='PREFLIGHT' if a.preflight_only else 'STARTED',
                   sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
                   default_config_sha256=hashlib.sha256(config_file.read_bytes()).hexdigest(),
                   override='outcome column name only; no numerical settings changed',
                   configuration=portable(c), installed_module=open.__file__,
                   github_revision=os.environ.get('GITHUB_SHA'),
                   github_run_id=os.environ.get('GITHUB_RUN_ID'),
                   github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
                   scope='fixed-fixture runtime/regression pilot, not independent simulation')
    (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if a.preflight_only:
        return
    assert 'site-packages' in Path(open.__file__).parts
    d = pd.read_csv(sample)
    start = time.perf_counter()
    try:
        result = fit_layer_consensus(d, target, tuple(d), c)
        metadata = validate_metadata(tuple(d),
            [{'name': n, 'type': 'continuous'} for n in d], target)
        write_layer_outputs(result, a.out / 'bundle', metadata=metadata)
        load_layer_bundle(a.out / 'bundle')
        receipt.update(status=result.status, seconds=time.perf_counter()-start,
                       assigned_depth=portable(result.assigned_depth),
                       provenance=portable(result.provenance),
                       planned_resamples_per_target=result.effective_resamples,
                       assessment_count=sum(len(t.assessments) for l in result.layers for t in l.evidence),
                       completed_assessment_count=sum(t.completed_count for l in result.layers for t in l.evidence))
    except Exception as exc:
        receipt.update(status='FAILED', seconds=time.perf_counter()-start,
                       failure_type=type(exc).__name__, failure_message=str(exc))
        raise
    finally:
        (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert receipt['status'] == 'COMPLETE'
    assert receipt['planned_resamples_per_target'] == 100


if __name__ == '__main__':
    main()
