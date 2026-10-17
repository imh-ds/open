"""No-fit verification of the completed matched linear-reference pilot."""
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


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    report=load(root/'report.json'); manifest=load(root/'manifest.json')
    assert manifest==load(root.parent/'launch/manifest.json')
    original=load(root.parents[1]/'2026-10-16-lightgbm-effect-pilot/launch/manifest-n200.json')
    assert report['run']=='39004942944' and report['attempt']==1
    assert report['source']=='2e0da27ec17745af37085b02f2627df7756507e4'
    assert report['manifest']==sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
    state=load(root/'run-api.json'); jobs=load(root/'jobs-api.json')['jobs']
    assert state['head_sha']==report['source'] and state['run_attempt']==1
    assert state['event']=='workflow_dispatch' and state['conclusion']=='success'
    assert state['path']=='.github/workflows/linear-effect-pilot.yml'
    assert len(jobs)==64 and sum(j['conclusion']=='success' for j in jobs)==63
    assert {j['name'] for j in jobs if j['name'].startswith('case (')}=={'case ('+s['id']+')' for s in manifest['cases']}
    assert report['execution_complete'] and len(report['cases'])==60
    audited={row['case_id']:row for row in report['cases']}
    assert set(audited)=={s['id'] for s in manifest['cases']}
    bundle_sha=file_sha(args.bundle)
    published=load(root/'publication-verification.json')
    assert published['run']==report['run'] and published['source']==report['source']
    for asset in load(root/'release-api.json')['assets']:
        path=args.bundle if asset['name'].endswith('.zip') else root/asset['name']
        actual=bundle_sha if path==args.bundle else file_sha(path)
        assert asset['digest']=='sha256:'+actual and path.stat().st_size==asset['size']
    for asset in published['assets']:
        assert (bundle_sha if asset['name'].endswith('.zip') else file_sha(root/asset['name']))==asset['sha256']
    inventory=load(root/'inventory.json')[0]
    assert bundle_sha==inventory['sha256']
    details=[]; warnings=Counter()
    with zipfile.ZipFile(args.bundle) as outer:
        assert set(outer.namelist())==set(inventory['members'])
        for name,member in inventory['members'].items():
            raw=outer.read(name); assert sha(raw)==member['sha256']
            row=audited[member['case_id']]; spec=row['spec']; engine=spec['engine']
            assert row['status']=='complete' and not row['fit_cost_unknown']
            assert (row['tuning_fits']==0)==(engine=='ols')
            config=manifest['configurations'][spec['id']]
            expected=original['configurations'][spec['paired_lightgbm_case_id']].copy()
            expected['learner']=dict(expected['learner'],library=[engine])
            assert config==expected
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                prefix='cases/'+spec['id']+'/'
                receipt=json.loads(archive.read(prefix+'receipt.json'))
                assert receipt['spec']==spec and receipt['status']=='complete'
                assert receipt['config']==config and receipt['manifest']==report['manifest']
                assert (receipt['github_run'],receipt['github_sha'],receipt['github_attempt'])==(report['run'],report['source'],'1')
                assert {n[len(prefix):]:sha(archive.read(n)) for n in archive.namelist() if n!=prefix+'receipt.json'}==receipt['files']
                data_sha=sha(archive.read(prefix+'data.csv'))
                assert data_sha==manifest['data_sha256'][spec['id']]==original['data_sha256'][spec['paired_lightgbm_case_id']]
                checkpoint=json.loads(gzip.decompress(archive.read(prefix+'checkpoints/d0-Y/000.json.gz')))
                risks=[r['risk'] for r in checkpoint['risks'] if r['origin']=='analysis']
                full=next(r for r in risks if len(r['predictors'])==4)
                expected_tuning=len(full['folds'])*manifest['tuning_search_sizes'][engine]*config['tuning_cv']['folds']
                assert full['learner']==engine and full['tuning_model_fit_count']==expected_tuning
                for fold in full['folds']:
                    assert set(fold['fitted_parameters'])==(set() if engine=='ols' else {'alpha','l1_ratio'})
                payload=json.loads(gzip.decompress(archive.read(prefix+'network.json.gz')))
                assessments=[a for a in payload['assessment_ledger.json']['assessments'] if a['depth']==0 and a['target']=='Y']
                assert len(assessments)==100 and all(a['status']=='completed' for a in assessments)
                gains=[]; selected=0; passes=0
                for a in assessments:
                    local=a['local']; risk=local['relevance']
                    gain=(risk['baseline_risk']-risk['full_risk'])/risk['baseline_risk']
                    assert abs(gain-local['total_gain_relative'])<1e-12
                    assert local['stop']==(gain<.02)
                    gains.append(gain); passes+=not local['stop']
                    selected+=local['classes'].get('X2',{}).get('kind')=='PROXIMAL_UNIQUE'
                    warnings.update(w['category']+': '+w['message'] for w in a['warnings'])
                assert (selected>=70)==(['X2','Y'] in row['edges'])
                details.append(dict(case_id=spec['id'],engine=engine,effect=spec['effect_percent']/100,
                    median_full_gain_relative=median(gains),root_gain_gate_passes=passes,
                    direct_selections=selected,root_assessments=100))
            actual={tuple(edge) for edge in row['edges']}
            intended={('X1','X2'),('X2','Y')}
            depths={'Y':0,'X2':1,'X1':2}
            false=bool(actual-intended or row['groups'] or set(row['assigned_depth'])-set(depths))
            exact=not false and actual==intended and row['assigned_depth']==depths
            assert false==row['structure']['incorrect'] and exact==row['structure']['exact']
    for cell in report['cells']:
        rows=[r for r in report['cases'] if r['spec']['engine']==cell['engine'] and r['spec']['effect_percent']==round(cell['effect']*100)]
        assert len(rows)==10 and cell['planned']==cell['complete']==10 and cell['unknown']==0
        assert cell['direct_links']==sum(['X2','Y'] in r['edges'] for r in rows)
        assert cell['indirect_links']==sum(['X1','X2'] in r['edges'] for r in rows)
        assert cell['exact_networks']==sum(r['structure']['exact'] for r in rows)
        assert cell['extra_finding_networks']==sum(r['structure']['incorrect'] for r in rows)
    summaries=[]
    for engine in ('elastic_net','ols'):
        rows=[r for r in report['cases'] if r['spec']['engine']==engine]
        for effect in (.2,.3,.4):
            sub=[r for r in details if r['engine']==engine and r['effect']==effect]
            summaries.append(dict(engine=engine,effect=effect,
                median_full_gain_relative=median(r['median_full_gain_relative'] for r in sub),
                root_gain_gate_passes=sum(r['root_gain_gate_passes'] for r in sub),root_assessments=1000,
                median_direct_selections=median(r['direct_selections'] for r in sub)))
    output=dict(checks_passed=True,scientific_fits=0,case_count=60,independent_datasets=30,
        root_assessments=6000,remote_asset_digests_verified=9,source=report['source'],run=report['run'],
        engine_identity_and_tuning_policy_verified_cases=60,cells=report['cells'],root_summaries=summaries,
        recorded_median_analysis_minutes={e:median(r['seconds'] for r in report['cases'] if r['spec']['engine']==e)/60 for e in ('elastic_net','ols')},
        root_warning_counts=dict(warnings),cases=details,review_script_sha256=file_sha(__file__),
        scope='No-fit saved-evidence check; original full checkpoint replay occurred in Actions; no fresh validation')
    (root/'review-verification.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
