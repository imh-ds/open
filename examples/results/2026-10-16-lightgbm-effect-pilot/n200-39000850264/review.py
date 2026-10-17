"""Verify saved N200 pilot counts and describe root prediction; zero model fits."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
from statistics import median
import zipfile


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    report=load(root/'report.json'); manifest=load(root/'manifest.json')
    reference=load(root.parent/'launch/manifest-n200.json')
    assert manifest==reference
    producer=load(root/'run-api.json'); jobs=load(root/'jobs-api.json')['jobs']
    assert producer['id']==39000850264 and producer['run_attempt']==1
    assert producer['head_sha']==report['source']=='8108299ffc777e8ec584bc08fef475ed833f7247'
    assert producer['event']=='workflow_dispatch' and producer['conclusion']=='success'
    assert producer['path']=='.github/workflows/lightgbm-effect-pilot.yml'
    assert report['execution_complete'] and len(report['cases'])==30
    assert len(jobs)==34 and sum(j['conclusion']=='success' for j in jobs)==33
    expected={'case ('+s['id']+')' for s in manifest['cases']}
    assert {j['name'] for j in jobs if j['name'].startswith('case (')}==expected
    publication=load(root/'publication-verification.json')
    metadata=load(root/'release-api.json')
    assert len(metadata['assets'])==10
    for asset in metadata['assets']:
        path=args.bundle if asset['name'].endswith('.zip') else root/asset['name']
        assert asset['digest']=='sha256:'+sha(path.read_bytes())
        assert asset['size']==path.stat().st_size
    for asset in publication['assets']:
        path=args.bundle if asset['name'].endswith('.zip') else root/asset['name']
        assert sha(path.read_bytes())==asset['sha256'],asset['name']
    inventory=load(root/'inventory.json')[0]
    assert sha(args.bundle.read_bytes())==inventory['sha256']
    audited={r['case_id']:r for r in report['cases']}
    details=[]; warnings=Counter()
    with zipfile.ZipFile(args.bundle) as outer:
        assert set(outer.namelist())==set(inventory['members'])
        for name,member in inventory['members'].items():
            raw=outer.read(name); assert sha(raw)==member['sha256']
            case=audited[member['case_id']]; spec=case['spec']
            assert case['status']=='complete' and not case['fit_cost_unknown']
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                prefix='cases/'+spec['id']+'/'
                receipt=json.loads(archive.read(prefix+'receipt.json'))
                assert receipt['spec']==spec and receipt['status']=='complete'
                assert receipt['github_run']=='39000850264' and receipt['github_sha']==report['source']
                assert receipt['github_attempt']=='1' and receipt['manifest']==report['manifest']
                sealed={n[len(prefix):]:sha(archive.read(n)) for n in archive.namelist() if n!=prefix+'receipt.json'}
                assert sealed==receipt['files']
                assert sha(archive.read(prefix+'data.csv'))==manifest['data_sha256'][spec['id']]
                network=json.loads(gzip.decompress(archive.read(prefix+'network.json.gz')))
                assessments=[a for a in network['assessment_ledger.json']['assessments'] if a['depth']==0 and a['target']=='Y']
                assert len(assessments)==100 and all(a['status']=='completed' for a in assessments)
                gains=[]; alone=[]; selected=0; pass_gate=0
                for a in assessments:
                    local=a['local']; relevance=local['relevance']
                    gain=(relevance['baseline_risk']-relevance['full_risk'])/relevance['baseline_risk']
                    assert abs(gain-local['total_gain_relative'])<1e-12
                    assert local['stop']==(gain<.02)
                    gains.append(gain)
                    alone.append(relevance['broad']['X2']['raw']/relevance['baseline_risk'])
                    selected+=local['classes'].get('X2',{}).get('kind')=='PROXIMAL_UNIQUE'
                    pass_gate+=not local['stop']
                    warnings.update(w['category']+': '+w['message'] for w in a['warnings'])
                assert (selected>=70)==(['X2','Y'] in case['edges'])
                details.append(dict(case_id=spec['id'],effect=spec['effect_percent']/100,
                    median_full_gain_relative=median(gains),median_X2_alone_gain_relative=median(alone),
                    root_assessments_passing_gain_gate=pass_gate,
                    root_assessments_full_worse_than_baseline=sum(g<0 for g in gains),
                    direct_proximal_selected=selected))
    cells=[]
    for effect in (.2,.3,.4):
        rows=[r for r in report['cases'] if r['spec']['effect_percent']==round(100*effect)]
        saved=next(c for c in report['cells'] if c['effect']==effect)
        assert len(rows)==10
        assert saved['direct_links']==sum(['X2','Y'] in r['edges'] for r in rows)
        assert saved['indirect_links']==sum(['X1','X2'] in r['edges'] for r in rows)
        assert saved['exact_networks']==sum(r['structure']['exact'] for r in rows)
        assert saved['extra_finding_networks']==sum(r['structure']['incorrect'] for r in rows)
        sub=[d for d in details if d['effect']==effect]
        cells.append(dict(effect=effect,
            median_of_dataset_median_full_gain_relative=median(d['median_full_gain_relative'] for d in sub),
            median_of_dataset_median_X2_alone_gain_relative=median(d['median_X2_alone_gain_relative'] for d in sub),
            full_model_worse_than_baseline_assessments=sum(d['root_assessments_full_worse_than_baseline'] for d in sub),
            root_gain_gate_passes=sum(d['root_assessments_passing_gain_gate'] for d in sub),
            root_assessments=1000,median_direct_selections=median(d['direct_proximal_selected'] for d in sub)))
    output=dict(checks_passed=True,scientific_fits=0,case_count=30,root_assessments=3000,
        source=report['source'],run=report['run'],cells=cells,cases=details,
        remote_asset_digests_verified=10,
        root_warning_counts=dict(warnings),
        scope='No-fit verification and dependent within-dataset prediction descriptions; not fresh validation',
        review_script_sha256=sha(Path(__file__).read_bytes()),
        asset_sha256={a['name']:a['sha256'] for a in publication['assets']})
    (root/'review-verification.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(cells,indent=2))


if __name__=='__main__':
    main()
