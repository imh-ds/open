"""Recount saved outer evaluation without fitting or using report assertions."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from open.layer_io import load_layer_bundle
from open.evaluation import data_fingerprint


def audit(directory):
    counts = dict(evaluations=0, folds=0, failed_folds=0, predictions=0, verified_networks=0,
                  recomputed_scores=0, warnings=0, input_datasets_verified=0,
                  training_baselines_verified=0)
    for path in sorted(directory.rglob('evaluation.json')):
        evidence = json.loads(path.read_text(encoding='utf-8'))
        # Saved demonstration inputs allow an extra check of observed values and
        # training-only baseline means, beyond recounting the prediction files.
        input_data = None
        for parent in (path.parent, *path.parents):
            if (parent/'protocol.md').exists() and (parent/'survey.csv').exists():
                name = 'survey' if path.parent.name == 'survey' else path.parent.name.rsplit('-', 1)[0]
                input_path = parent/f'{name}.csv'
                if input_path.exists():
                    input_data = pd.read_csv(input_path, float_precision='round_trip')
                    assert data_fingerprint(input_data, tuple(evidence['variables'])) == evidence['data_sha256']
                    counts['input_datasets_verified'] += 1
                break
        counts['evaluations'] += 1
        counts['folds'] += len(evidence['folds'])
        positions = []
        predicted = []
        for fold in evidence['folds']:
            train, test = set(fold['train_positions']), set(fold['test_positions'])
            assert not train & test
            assert train | test == set(range(evidence['n_rows']))
            assert len(train) == len(fold['train_positions']) and len(test) == len(fold['test_positions'])
            positions.extend(test)
            counts['warnings'] += len(fold['warnings'])
            if fold['network_bundle']:
                bundle = load_layer_bundle(path.parent/fold['network_bundle'])
                counts['verified_networks'] += 1
                if fold['status'] == 'completed':
                    assert bundle.metadata['status'] == 'COMPLETE'
                    root = evidence['config']['analysis']['root']
                    parents = bundle.metadata['parents']
                    selected = [n for n in evidence['variables'] if root in parents.get(n, ())]
                    assert selected == fold['selected_predictors']
                assert all(max(p['sampled_positions'], default=-1) < len(train) for p in bundle.metadata['plans'])
            if fold['status'] == 'completed':
                assert {p['row_position'] for p in fold['predictions']} == test
                assert len(fold['predictions']) == len(test)
                predicted.extend(dict(fold=fold['fold'], **p) for p in fold['predictions'])
                if input_data is not None:
                    root = evidence['config']['analysis']['root']
                    mean = float(input_data.iloc[fold['train_positions']][root].mean())
                    for prediction in fold['predictions']:
                        assert prediction['observed'] == float(input_data.iloc[prediction['row_position']][root])
                        assert np.isclose(prediction['baseline'], mean, rtol=1e-14, atol=1e-14)
                    counts['training_baselines_verified'] += 1
            else:
                counts['failed_folds'] += 1
                assert not fold['predictions'] and fold['failure_type']
        assert sorted(positions) == list(range(evidence['n_rows']))
        assert predicted == evidence['predictions']
        counts['predictions'] += len(predicted)
        with (path.parent/'predictions.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(predicted)
        for row, expected in zip(rows, predicted):
            for key, value in expected.items():
                assert float(row[key]) == value
        if evidence['status'] == 'COMPLETE':
            assert len(predicted) == evidence['n_rows']
            y = np.array([p['observed'] for p in predicted])
            variance = np.sum((y-y.mean())**2)
            baseline = sum((p['observed']-p['baseline'])**2 for p in predicted)
            for model in ('baseline', 'reduced', 'full'):
                errors = sum((p['observed']-p[model])**2 for p in predicted)
                expected = dict(mse=errors/len(y), rmse=np.sqrt(errors/len(y)),
                    r_squared=1-errors/variance if variance > 0 else None,
                    prediction_skill=1-errors/baseline if baseline > 0 else None)
                for name, value in expected.items():
                    actual = evidence['scores'][model][name]
                    assert actual is None if value is None else np.isclose(actual, value, rtol=1e-12, atol=1e-12)
                    counts['recomputed_scores'] += 1
        else:
            assert evidence['scores'] is None
        report = (path.parent/'REPORT.md').read_text(encoding='utf-8')
        for link in re.findall(r'\]\(([^)]+)\)', report):
            assert (path.parent/link).exists(), link
        if (path.parent/'reference').exists():
            load_layer_bundle(path.parent/'reference')
            counts['verified_networks'] += 1
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    counts = audit(args.directory)
    print(json.dumps(counts, indent=2))
    (args.directory/'audit.json').write_text(json.dumps(counts, indent=2)+'\n', encoding='utf-8')
