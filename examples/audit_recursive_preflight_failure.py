"""Reproduce the failed preflight diagnosis from saved evidence; zero model fits."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import pandas as pd

from open.layer_artifacts import reconstruct_payloads
from recursive_preview_audit import Store, audit_case
from recursive_preview_design import ROOT, digest
from recursive_preview_equivalence import compare_networks


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def audit(evidence, output):
    evidence, output = Path(evidence), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    control = read(evidence/'preflight.json')
    run = read(evidence/'run.json')
    manifest = read(evidence/'producer-manifest.json')
    if (control['manifest'] != digest(manifest) or control['github_sha'] != run['head_sha']
        or str(control['github_run']) != str(run['id']) or run['conclusion'] != 'failure'):
        raise ValueError('original failed-run/producer identity mismatch')
    archive = evidence/'preflight-all-attempts.zip'
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('archive CRC mismatch')
        actual = json.loads(__import__('gzip').decompress(z.read('cases/fixture-chain/network.json.gz')))
    names = ('analysis_metadata.json','layer_evidence.json','assessment_ledger.json')
    baseline_path = ROOT/'examples/results/2026-10-14-default-preview-pilot/default-pilot-chain/.superpowers/default-pilot/chain/bundle'
    original = {n:read(baseline_path/n) for n in names}
    a,_ = reconstruct_payloads(*(actual[n] for n in names))
    b,_ = reconstruct_payloads(*(original[n] for n in names))
    comparison = compare_networks(a,b)
    data = pd.read_csv(ROOT/'examples/results/2026-10-08-layer-fixtures/chain/sample.csv')
    store = Store(archive,'fixture-chain')
    start = time.monotonic()
    try:
        checked = audit_case(dict(id='fixture-chain',fixture='chain'),store,manifest,
                             data=data,config=b.config)
    finally:
        store.close()
    checked['audit_seconds'] = time.monotonic()-start
    result = dict(original_run=run['id'],original_conclusion='failure',
        original_strict_comparison_failed=True, fixture_audit=checked, comparison=comparison,
        completed_fixture_assessments=checked['assessments'],
        new_study_datasets_fitted=len(control['planned_cases']),
        audit_model_fits=0, original_protocol_reclassified=False,
        scientific_qualification='Not assessed; no independent study datasets fitted.',
        scope='All discrete saved decisions, sample/split/config identities and available fitted-parameter fields exact. '
              'Derived float fields compared at rtol1e-12/atol1e-14. Original fixture lacks full/broad prediction objects; '
              'new captured observations are independently replayed without claiming unavailable original predictions.')
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(evidence.iterdir()) if p.is_file()}
    (output/'raw-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.evidence,args.out)
    print(json.dumps({k:v for k,v in result.items() if k != 'fixture_audit'},indent=2))
