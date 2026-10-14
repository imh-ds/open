"""No-fit integrity and expected-structure audit for a completed default pilot."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

CASES = ('chain', 'chain_missing', 'shared_predictors')
EXPECTED_DEPTHS = {
    'chain': {'Anxiety': 0, 'Depression': 1, 'Stress': 2},
    'chain_missing': {'Anxiety': 0, 'Depression': 1, 'Stress': 2},
    'shared_predictors': {'Outcome': 0, 'Scale_A': 1, 'Scale_B': 1},
}
EXPECTED_EDGES = {
    'chain': {('Depression', 'Anxiety'), ('Stress', 'Depression')},
    'chain_missing': {('Depression', 'Anxiety'), ('Stress', 'Depression')},
    'shared_predictors': set(),
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'src'))
    from open.layer_io import load_layer_bundle

    cases = []
    for name in CASES:
        case_dir = args.evidence / f'default-pilot-{name}'
        output_dir = case_dir / '.superpowers/default-pilot' / name
        receipt = json.loads((output_dir / 'receipt.json').read_text())
        assert receipt['case'] == name and receipt['status'] == 'COMPLETE'
        assert receipt['github_run_id'] == '38525190948'
        assert receipt['github_revision'] == '284b474e0c26b8fb348e6c5ae3306ff53184490b'
        assert receipt['planned_resamples_per_target'] == 100
        assert receipt['assessment_count'] == receipt['completed_assessment_count']
        assert receipt['configuration']['performance_cv']['folds'] == 5
        assert receipt['configuration']['performance_cv']['repeats'] == 2
        assert receipt['configuration']['tuning_cv']['folds'] == 3
        assert receipt['configuration']['stability']['n_resamples'] == 100
        assert receipt['configuration']['thresholds']['unique_relative'] == .05

        sample = args.repo / 'examples/results/2026-10-08-layer-fixtures' / name / 'sample.csv'
        assert hashlib.sha256(sample.read_bytes()).hexdigest() == receipt['sample_sha256']
        config = args.repo / 'configs/default.yaml'
        assert hashlib.sha256(config.read_bytes()).hexdigest() == receipt['default_config_sha256']

        bundle_dir = output_dir / 'bundle'
        bundle = load_layer_bundle(bundle_dir)
        nodes = {row['node']: int(row['depth']) for row in read_csv(bundle_dir / 'nodes.csv')}
        assert nodes == EXPECTED_DEPTHS[name], (name, nodes)
        edge_rows = read_csv(bundle_dir / 'edges.csv')
        edge_set = {(r['source'], r['target']) for r in edge_rows}
        assert edge_set == EXPECTED_EDGES[name], (name, edge_set)
        for edge in edge_rows:
            assert int(edge['selected_count']) == int(edge['planned_count']) == 100
            assert int(edge['completed_count']) == 100
        groups = read_csv(bundle_dir / 'shared_groups.csv')
        if name == 'shared_predictors':
            assert len(groups) == 1
            members = json.loads(groups[0]['members'])
            assert members == ['Scale_A', 'Scale_B']
            assert int(groups[0]['selected_count']) == int(groups[0]['planned_count']) == 100
        else:
            assert not groups

        ledger = json.loads((bundle_dir / 'assessment_ledger.json').read_text())['assessments']
        assert len(ledger) == receipt['assessment_count']
        assert all(row['status'] == 'completed' for row in ledger)
        warning_count = sum(len(row['warnings']) for row in ledger)
        assert warning_count == 0
        row_patterns = Counter((row['target'], row['n_observed_target_rows'],
                                 row['n_missing_target_rows']) for row in ledger)
        missing_counts = {}
        for (target, observed, missing), count in sorted(row_patterns.items()):
            missing_counts.setdefault(target, []).append({
                'observed': observed, 'missing': missing, 'resamples': count})
        cases.append({
            'case': name,
            'status': receipt['status'],
            'seconds': receipt['seconds'],
            'planned_and_completed_assessments': receipt['assessment_count'],
            'warnings': warning_count,
            'depths': nodes,
            'edges': [f"{a} -> {b}" for a, b in sorted(edge_set)],
            'shared_groups': [json.loads(g['members']) for g in groups],
            'observed_and_missing_target_rows_per_resample': missing_counts,
            'strict_bundle_reload': True,
            'source_sample_and_config_hashes_match': True,
        })
    result = {
        'status': 'PASS',
        'run_id': '38525190948',
        'source_revision': '284b474e0c26b8fb348e6c5ae3306ff53184490b',
        'cases': cases,
        'total_assessments': sum(x['planned_and_completed_assessments'] for x in cases),
        'total_recorded_warnings': sum(x['warnings'] for x in cases),
        'model_fits_during_audit': 0,
        'scope': 'fixed historical-fixture regression and runtime check; not an independent false-selection-rate study',
    }
    files = sorted(p for p in args.evidence.rglob('*') if p.is_file()
                   and p.name not in {'independent-audit.json', 'artifact_sha256.json'})
    result['preserved_artifacts'] = len(files)
    result['preserved_artifact_bytes'] = sum(p.stat().st_size for p in files)
    (args.evidence / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    hashes = {p.relative_to(args.evidence).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    (args.evidence / 'artifact_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
