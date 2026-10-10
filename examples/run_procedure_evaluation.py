"""Execute the predefined integration pilot and the existing survey fixture."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from open.config import (AnalysisConfig, LearnerConfig, OpenConfig,
                         PerformanceCVConfig, StabilityConfig, TuningCVConfig)
from open.evaluation import evaluate_procedure
from open.layer_artifacts import portable
from open.metadata import validate_metadata
from open.reporting import ResearcherAnalysis, analyze_researcher, write_researcher_outputs


def config(root, engine, *, budget=3, depth=2):
    return OpenConfig(AnalysisConfig(root, 613, depth), learner=LearnerConfig(library=(engine,)),
        performance_cv=PerformanceCVConfig(2, 1), tuning_cv=TuningCVConfig(2),
        stability=StabilityConfig(n_resamples=budget, scope='layer_consensus'))


def scenarios():
    for index, name in enumerate(('noise', 'direct', 'redundant', 'u_shape', 'interaction')):
        rng = np.random.default_rng(9301+index)
        a, b = rng.normal(size=(2, 80))
        error = rng.normal(scale=.3, size=80)
        if name == 'noise':
            y = rng.normal(size=80)
        elif name == 'direct':
            y = 2*a+error
        elif name == 'redundant':
            b = a+rng.normal(scale=.03, size=80)
            y = 2*a+error
        elif name == 'u_shape':
            y = 2*a**2+error
        else:
            y = 2*a*b+error
        yield name, pd.DataFrame(dict(y=y, a=a, b=b))


def metadata(data, root):
    return validate_metadata(tuple(data), [dict(name=n, type='continuous') for n in data], root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    base = Path(__file__).resolve().parent
    shutil.copyfile(base/'procedure-validation-protocol.md', args.output/'protocol.md')
    started = time.perf_counter()
    rows = []
    for name, data in scenarios():
        data.to_csv(args.output/f'{name}.csv', index=False)
        for engine in ('elastic_net', 'gbm'):
            evaluation = evaluate_procedure(data, 'y', tuple(data), config('y', engine),
                                            outer_folds=3, outer_seed=8401)
            write_researcher_outputs(ResearcherAnalysis(evaluation, None, None),
                args.output/'pilot'/f'{name}-{engine}', metadata=metadata(data, 'y'))
            for model in ('baseline', 'reduced', 'full'):
                rows.append(dict(scenario=name, engine=engine, model=model,
                    completed=sum(f.status == 'completed' for f in evaluation.folds), planned=3,
                    **(portable(evaluation.scores[model]) if evaluation.scores is not None else
                       dict(mse=None, rmse=None, r_squared=None, prediction_skill=None))))
            print(f'{name}/{engine}: {evaluation.status}', flush=True)
    survey = pd.read_csv(base/'results/2026-10-08-survey-layer-consensus/sample.csv')
    survey = survey.rename(columns=dict(wellbeing='Well-being', sleep_quality='Sleep Quality',
        perceived_stress='Perceived Stress', social_support='Social Support',
        physical_activity='Physical Activity', work_demands='Work Demands'))
    survey.to_csv(args.output/'survey.csv', index=False)
    root = 'Well-being'
    analysis = analyze_researcher(survey, root, tuple(survey), config(root, 'elastic_net', budget=10, depth=3),
                                  outer_folds=3, outer_seed=8401)
    write_researcher_outputs(analysis, args.output/'survey', metadata=metadata(survey, root), plots=True)
    with (args.output/'pilot_scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'execution.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
        pilot_conditions=10, planned_pilot_folds=30, survey_status=analysis.evaluation.status,
        survey_scores=portable(analysis.evaluation.scores),
        claim='development_integration_not_scientific_calibration'), indent=2)+'\n', encoding='utf-8')
    print(f'Survey: {analysis.evaluation.status}; evidence saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
