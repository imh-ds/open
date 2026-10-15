"""Independently replay all 38 partial prefixes locally, never fit a model."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'examples'))
import continue_qualification as c
import recursive_preview_runner as runner


def forbidden(*args, **kwargs):
    raise RuntimeError('scientific model fitting is forbidden in this evidence replay')


def check(spec):
    runner.NATIVE_ASSESS = forbidden
    archive = ROOT/f'examples/results/2026-10-14-recursive-qualification/noise-38655036215/raw/case-{spec["index"]}.zip'
    return c.validate_archive(spec, archive, replay=True)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, c.pending()))
    c.write_json(Path(__file__).with_name('partial-replay.json'), dict(
        status='PASS', model_fits=0, producer_run=c.PRODUCER_RUN, cases=rows,
        assessments=sum(r['assessments'] for r in rows),
        observation_checks=sum(r['observation_checks'] for r in rows),
        unknown_attempts=sum(r['unknown_attempts'] for r in rows)))
    print('PASS', len(rows), 'partial cases;', sum(r['assessments'] for r in rows), 'checkpoints; no fits')
