"""Replay the frozen combined audit with complete downloads; never fit models.

Actions' aggregate downloaded 100 of 112 producer artifacts. This local replay
uses all locally preserved inputs and saved, completed GitHub API receipts.
Run under the pinned installed scientific runtime, from the repo root.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'examples'))
import continue_qualification as c
import recursive_preview_runner as runner

BASE = Path(__file__).resolve().parent


def forbidden(*args, **kwargs):
    raise RuntimeError('model fitting is prohibited in this independent evidence replay')


runner.NATIVE_ASSESS = forbidden


def saved_api(path):
    prefix = 'actions/runs/'
    if not path.startswith(prefix):
        raise ValueError('unexpected API request in offline audit')
    suffix = path[len(prefix):]
    if suffix == c.PRODUCER_RUN:
        return c.load_json(BASE/'raw/producer-run.json')
    if suffix == '38676310704':
        return c.load_json(BASE/'raw/run.json')
    if suffix.startswith(c.PRODUCER_RUN+'/jobs?'):
        page = int(path.rsplit('page=', 1)[1])
        pages = c.load_json(BASE/'raw/producer-jobs-pages.json')
        return pages[page-1] if page <= len(pages) else dict(jobs=[])
    if suffix.startswith('38676310704/jobs?'):
        page = int(path.rsplit('page=', 1)[1])
        return c.load_json(BASE/'raw/jobs.json') if page == 1 else dict(jobs=[])
    raise ValueError('unexpected API request in offline audit')


def stage_preserved(directory):
    """Reconstruct download layout from committed bytes, without extracting ZIPs."""
    directory.mkdir(parents=True,exist_ok=False)
    inputs,producer_inputs = directory/'inputs',directory/'producer-inputs'
    def link(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            os.link(source,target)
        except OSError:
            shutil.copyfile(source,target)
    original = ROOT/'examples/results/2026-10-14-recursive-qualification/noise-38655036215/raw'
    for spec in c.completed():
        i=spec['index']
        link(original/f'case-{i}.zip',producer_inputs/f'qualification-noise-{i}/raw/case.zip')
    for spec in c.pending():
        i=spec['index']
        folder=inputs/f'qualification-continuation-{i}'
        for source,target in ((f'continued-case-{i}.zip','raw/case.zip'),
                (f'continued-control-{i}.json','control.json'),
                (f'original-receipt-{i}.json','provenance/original-receipt.json'),
                (f'reuse-audit-{i}.json','reuse-audit.json')):
            link(BASE/'raw'/source,folder/target)
    return inputs,producer_inputs


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=ROOT/'.superpowers/continuation-download-38676310704')
    parser.add_argument('--producer-inputs',type=Path,default=ROOT/'.superpowers/qualification-noise-download-38655036215')
    parser.add_argument('--out',type=Path,default=BASE/'audit',help='Fresh output directory; existing evidence is never overwritten')
    parser.add_argument('--stage-preserved',type=Path,help='Fresh directory to reconstruct inputs from retained raw archives')
    args=parser.parse_args()
    if args.stage_preserved:
        args.inputs,args.producer_inputs=stage_preserved(args.stage_preserved)
    # These bind replay checks to the original case execution, not a new run.
    os.environ.update(GITHUB_RUN_ID='38676310704', GITHUB_RUN_ATTEMPT='1',
                      GITHUB_SHA='bab7b69b367ba577b7305ea633d1fe12b412bd78')
    c.github_api = saved_api
    result = c.aggregate(args.inputs,args.producer_inputs,args.out)
    result.update(audit_execution='local no-fit replay using frozen code and complete downloads',
                  continuation_run_still_failed=True, model_refits=0,
                  local_source_continuation_manifest=c.design.digest(c.checked_manifest()))
    c.write_json(args.out/'audit.json', result)
    print({k:result[k] for k in ('planned','complete','qualified','integrity_errors','cells')})
    if result['integrity_errors'] or result['complete'] != 110:
        raise SystemExit('Combined audit incomplete; preserve all unknowns.')
