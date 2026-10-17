"""Matched Elastic Net/OLS references; versioned separately from LightGBM."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import pandas as pd

import linear_reference_audit as auditor
from lightgbm_effect_pilot import generate as original_generate
import recursive_preview_design as design
import recursive_preview_runner as runner
import tree_recursive_null_evidence as evidence
from open.layer_artifacts import portable
from open.learners.elastic_net import ALPHA_GRID,L1_RATIO_GRID

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'linear-moderate-effects-2026-10-16'
CASE_CAPS={'elastic_net':90,'ols':30}
ORIGINAL=ROOT/'examples/results/2026-10-16-lightgbm-effect-pilot/launch/manifest-n200.json'
PARENT_DIGEST='d5ce001bf45ec0641fabdcda61673cd2222236deee807494ce3c7ffa1be0920f'
RUNTIME = {'numpy':'1.26.4','pandas':'2.2.3','scipy':'1.11.2',
           'scikit-learn':'1.3.0','PyYAML':'6.0.2','matplotlib':'3.9.2',
           'lightgbm':'4.6.0','setuptools':'75.1.0','wheel':'0.45.1'}
FILES = ['configs/default.yaml','pyproject.toml','examples/linear_effect_pilot.py',
         'examples/linear_effect_pilot_protocol.md','examples/linear_reference_audit.py',
         'examples/lightgbm_effect_pilot.py',
         'examples/results/2026-10-16-lightgbm-effect-pilot/launch/manifest-n200.json',
         'examples/recursive_preview_design.py','examples/recursive_preview_runner.py',
         'examples/multiengine_recursive_audit.py','examples/tree_recursive_null_evidence.py',
         '.github/workflows/linear-effect-pilot.yml']
digest = design.digest
write_json = runner.write_json


def cases(n):
    if type(n) is not int or n!=200:
        raise ValueError('this allocation covers N200 only')
    original=load(ORIGINAL)
    if digest(original)!=PARENT_DIGEST:
        raise ValueError('paired LightGBM manifest identity differs')
    return [dict(s,id=s['id'].replace('lgbm-',engine+'-',1),engine=engine,
                 paired_lightgbm_case_id=s['id'])
            for engine in CASE_CAPS for s in original['cases']]


def generate(spec):
    if spec not in cases(spec['n']):
        raise ValueError('unknown pilot case specification')
    source=next(s for s in load(ORIGINAL)['cases'] if s['id']==spec['paired_lightgbm_case_id'])
    return original_generate(source)


def settings(spec):
    config=design.settings(spec)
    return replace(config,learner=replace(config.learner,library=(spec['engine'],)))


def manifest(n):
    specs=cases(n)
    files=sorted(set([*ROOT.joinpath('src/open').rglob('*.py'),
                      *(ROOT/p for p in FILES)]))
    configurations={s['id']:portable(settings(s)) for s in specs}
    hashes={s['id']:hashlib.sha256(generate(s).to_csv(index=False,lineterminator='\n').encode()).hexdigest() for s in specs}
    original=load(ORIGINAL)
    for s in specs:
        expected=load(ORIGINAL)['configurations'][s['paired_lightgbm_case_id']]
        expected['learner']['library']=[s['engine']]
        if configurations[s['id']]!=expected or hashes[s['id']]!=original['data_sha256'][s['paired_lightgbm_case_id']]:
            raise ValueError('reference differs from paired LightGBM data/configuration')
    return dict(schema='open.linear_effect_pilot.v1',study_id=STUDY,sample_size=n,
                cases=specs,python='3.11.9',runtime=RUNTIME,
                sources={p.relative_to(ROOT).as_posix():design.text_hash(p) for p in files},
                configurations=configurations,data_sha256=hashes,
                grids=dict(elastic_net=dict(alpha=list(ALPHA_GRID),l1_ratio=list(L1_RATIO_GRID)),ols={}),
                tuning_search_sizes=dict(elastic_net=25,ols=0),
                parent_lightgbm_manifest=digest(original),
                case_caps_minutes=CASE_CAPS,max_parallel=20,
                reserved_runner_minutes=3871,
                runner_minute_ceiling=4000,scientific_qualification=False,
                threshold_snapshots='posthoc tested comparisons only, not recursive reruns')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_manifest(value):
    if value != manifest(value.get('sample_size')):
        raise ValueError('manifest differs from frozen pilot source/data/configuration')
    return digest(value)


def first_attempt():
    if (os.environ.get('GITHUB_EVENT_NAME')!='workflow_dispatch'
            or os.environ.get('GITHUB_RUN_ATTEMPT')!='1'
            or not os.environ.get('GITHUB_RUN_ID','').isdigit()
            or len(os.environ.get('GITHUB_SHA',''))!=40):
        raise ValueError('pilot requires a first-attempt manual dispatch')


def runtime_check(value):
    if platform.python_version()!=value['python']:
        raise ValueError('Python version differs')
    if {k:importlib.metadata.version(k) for k in value['runtime']}!=value['runtime']:
        raise ValueError('installed runtime differs')
    import open as package
    if 'site-packages' not in Path(package.__file__).parts:
        raise ValueError('scientific workflow must use installed package')


def prepare(n,out):
    first_attempt()
    value=manifest(n)
    runtime_check(value)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'manifest.json',value)
    write_json(out/'producer.json',dict(source=os.environ['GITHUB_SHA'],
               run=os.environ['GITHUB_RUN_ID'],attempt=1,manifest=digest(value)))
    return dict(include=[dict(id=s['id'],case_cap_minutes=CASE_CAPS[s['engine']]) for s in value['cases']])


def producer_check(path,value):
    control=load(Path(path).parent/'producer.json')
    if control!=dict(source=os.environ['GITHUB_SHA'],run=os.environ['GITHUB_RUN_ID'],
                     attempt=1,manifest=digest(value)):
        raise ValueError('authorization run/source/attempt differs')


def execute(path,case_id,out,*,job_started_at,now=None):
    first_attempt()
    value=load(path)
    validate_manifest(value)
    runtime_check(value)
    # Producer identity is checked by the CLI before execution; tests may supply
    # a standalone manifest to exercise deadline handling without a workflow.
    spec=next(s for s in value['cases'] if s['id']==case_id)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    directory=out/'cases'/case_id
    elapsed=max(0,(time.time() if now is None else now)-float(job_started_at))
    # Five minutes of batch-completion grace, then ten for preservation/upload.
    seconds=max(0,CASE_CAPS[spec['engine']]*60-elapsed-900)
    failure=None
    try:
        if seconds<=0:
            directory.mkdir(parents=True)
            write_json(directory/'receipt.json',dict(spec=spec,status='incomplete',
                       failure='budget expired before fitting',fit_cost_unknown=True,
                       manifest=digest(value),scientific_fits=0,budget_expired_before_fit=True,
                       config=portable(settings(spec)),data_sha256=value['data_sha256'][case_id],
                       github_run=os.environ['GITHUB_RUN_ID'],github_sha=os.environ['GITHUB_SHA'],
                       github_attempt='1',python=platform.python_version(),runtime=value['runtime']))
            raise TimeoutError('budget expired before fitting')
        runner.run_case(spec,directory,value,data=generate(spec),config=settings(spec),seconds=seconds)
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)[:2000]
        if not directory.exists():
            directory.mkdir(parents=True)
            write_json(directory/'failure.json',dict(failure=failure))
    packed=evidence.pack_case(directory,case_id,out/'raw/case.zip')
    result=dict(case_id=case_id,spec=spec,manifest=digest(value),
                source=os.environ['GITHUB_SHA'],run=os.environ['GITHUB_RUN_ID'],attempt=1,
                status='complete' if failure is None else 'incomplete',
                failure=failure,fit_cost_unknown=failure is not None,
                archive_sha256=packed['sha256'],archive_bytes=packed['bytes'],
                soft_compute_seconds=seconds)
    write_json(out/'control.json',result)
    return result


def preserve_interrupted(path,case_id,out):
    """Pack checkpoints after a timed-out fit step; never restart an assessment."""
    first_attempt()
    value=load(path)
    validate_manifest(value)
    producer_check(path,value)
    spec=next(s for s in value['cases'] if s['id']==case_id)
    out=Path(out)
    if (out/'control.json').is_file():
        return load(out/'control.json')
    directory=out/'cases'/case_id
    if not directory.is_dir():
        return dict(case_id=case_id,status='missing',failure='no case evidence was created')
    packed=evidence.pack_case(directory,case_id,out/'raw/case.zip')
    control=dict(case_id=case_id,spec=spec,manifest=digest(value),
        source=os.environ['GITHUB_SHA'],run=os.environ['GITHUB_RUN_ID'],attempt=1,
        status='incomplete',fit_cost_unknown=True,failure='fit step interrupted before final control',
        archive_sha256=packed['sha256'],archive_bytes=packed['bytes'],salvaged=True)
    write_json(out/'control.json',control)
    return control


def summarize(planned,rows):
    expected={s['id'] for s in planned}
    found={}
    for row in rows:
        if row['case_id'] not in expected:
            raise ValueError('foreign case identity')
        if row['case_id'] in found:
            raise ValueError('duplicate case identity')
        found[row['case_id']]=row
    result=[]
    for engine,effect in ((e,r) for e in CASE_CAPS for r in (20,30,40)):
        specs=[s for s in planned if s['effect_percent']==effect and s['engine']==engine]
        complete=[found[s['id']] for s in specs if s['id'] in found and found[s['id']]['status']=='complete']
        result.append(dict(engine=engine,effect=effect/100,planned=len(specs),complete=len(complete),
             unknown=len(specs)-len(complete),
             direct_links=sum(['X2','Y'] in r['edges'] for r in complete),
             indirect_links=sum(['X1','X2'] in r['edges'] for r in complete),
             indirect_target_tested=sum(any(c['target']=='X2' for c in r.get('candidate_comparisons',[])) for r in complete),
             exact_networks=sum(r['structure']['exact'] for r in complete),
             extra_finding_networks=sum(r['structure']['incorrect'] for r in complete)))
    return result


def candidate_rows(spec,payload):
    rows=[]
    for layer in payload['layer_evidence.json']['layers']:
        for target in layer['evidence']:
            for candidate in target['candidates_evidence']:
                counts=candidate['classes'].get('PROXIMAL_UNIQUE',{})
                rows.append(dict(case_id=spec['id'],engine=spec['engine'],effect=spec['effect_percent']/100,
                    depth=layer['depth'],target=target['target'],source=candidate['source'],
                    truth=[candidate['source'],target['target']] in [['X1','X2'],['X2','Y']],
                    proximal_selected=counts.get('selected',0),
                    planned=counts.get('planned',len(target['assessments'])),
                    completed=counts.get('completed',len(target['assessments'])),
                    scope='already-tested comparisons only; no alternate recursive rerun'))
    return rows


def audit_saved(spec,folder,value):
    folder=Path(folder)
    control=load(folder/'control.json')
    expected=dict(case_id=spec['id'],spec=spec,manifest=digest(value),
                  source=os.environ['GITHUB_SHA'],run=os.environ['GITHUB_RUN_ID'],attempt=1)
    if any(control.get(k)!=v for k,v in expected.items()):
        raise ValueError('case control provenance differs')
    archive=folder/'raw/case.zip'
    if evidence.sha256_file(archive)!=control['archive_sha256']:
        raise ValueError('case raw archive checksum differs')
    store=auditor.Store(archive,spec['id'])
    try:
        receipt=store.json('receipt.json')
        if (receipt['github_run'],receipt['github_sha'],receipt['github_attempt'])!=(
                os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA'],'1'):
            raise ValueError('raw receipt producer differs')
        if receipt.get('budget_expired_before_fit'):
            if (set(store.names)!={'receipt.json'} or receipt.get('scientific_fits')!=0
                    or receipt.get('spec')!=spec or receipt.get('manifest')!=digest(value)
                    or receipt.get('config')!=portable(settings(spec))
                    or receipt.get('data_sha256')!=value['data_sha256'][spec['id']]
                    or receipt.get('python')!=value['python'] or receipt.get('runtime')!=value['runtime']
                    or receipt.get('status')!='incomplete' or control['status']!='incomplete'):
                raise ValueError('zero-fit budget-expiry receipt differs')
            return dict(case_id=spec['id'],status='incomplete',failure=control.get('failure'),
                        scientific_fits=0,started_assessments=0,finished_assessments=0,
                        fit_cost_unknown=False,provenance_verified=True)
        row=auditor.audit_case(spec,store,value,data=generate(spec),config=settings(spec))
        row['case_id']=spec['id']
        if control['status']!='complete':
            row.update(raw_receipt_status=row['status'],status='incomplete',
                       failure=control.get('failure'),provenance_verified=True)
            return row
        if row['status']!='complete' or row['fit_cost_unknown']:
            raise ValueError('completed control differs from raw completion')
        row['structure']=design.structure(spec,row['edges'],row['groups'],row['assigned_depth'])
        row['case_id']=spec['id']
        payload=store.compressed('network.json.gz')
        row['candidate_comparisons']=candidate_rows(spec,payload)
        return row
    finally:
        store.close()


def aggregate(path,download,out):
    first_attempt()
    value=load(path)
    validate_manifest(value)
    runtime_check(value)
    producer_check(path,value)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    rows,archives=[],[]
    for spec in value['cases']:
        folder=Path(download)/('linear-case-'+spec['id'])
        try:
            row=audit_saved(spec,folder,value)
        except Exception as exc:
            row=dict(case_id=spec['id'],status='audit_failed',failure=type(exc).__name__+': '+str(exc)[:2000])
        rows.append(row)
        archive=folder/'raw/case.zip'
        if archive.is_file():
            archives.append(dict(case_id=spec['id'],archive=archive,sha256=evidence.sha256_file(archive)))
    report=dict(study=STUDY,sample_size=value['sample_size'],run=os.environ['GITHUB_RUN_ID'],
                source=os.environ['GITHUB_SHA'],attempt=1,manifest=digest(value),
                scientific_qualification=False,execution_complete=all(r['status']=='complete' for r in rows),
                cells=summarize(value['cases'],rows),cases=rows)
    write_json(out/'report.json',report)
    write_json(out/'manifest.json',value)
    (out/'protocol.md').write_bytes((ROOT/'examples/linear_effect_pilot_protocol.md').read_bytes())
    candidates=[c for r in rows for c in r.get('candidate_comparisons',[])]
    pd.DataFrame(candidates).to_csv(out/'candidate-comparisons.csv',index=False)
    snapshots=[]
    for engine,effect in ((e,r) for e in CASE_CAPS for r in (20,30,40)):
        for target in ('Y','X2'):
            subset=[c for c in candidates if c['engine']==engine and c['effect']==effect/100 and c['target']==target]
            for cutoff in range(70,91,2):
                kept=[c for c in subset if c['proximal_selected']*100 >= cutoff*c['planned']]
                snapshots.append(dict(engine=engine,effect=effect/100,target=target,cutoff_percent=cutoff,
                    tested_targets=len({c['case_id'] for c in subset}),
                    tested_true_candidates=sum(c['truth'] for c in subset),
                    tested_false_candidates=sum(not c['truth'] for c in subset),
                    retained_true_candidates=sum(c['truth'] for c in kept),
                    retained_false_candidates=sum(not c['truth'] for c in kept),
                    scope='posthoc PROXIMAL_UNIQUE comparisons, excludes shared groups; not full recursive reruns'))
    pd.DataFrame(snapshots).to_csv(out/'retention-snapshots.csv',index=False)
    summary=['# Matched Elastic Net and OLS effect pilot', '', f"N={value['sample_size']}; exploratory, ten planned datasets per effect/engine.",
             '', '| Engine | r | Complete /10 | Unknown | Direct /10 | Indirect /10 | Indirect target tested /10 | Exact /10 | Extra-finding networks /10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in report['cells']:
        summary.append('| '+ ' | '.join(str(c[k]) for k in ['engine','effect','complete','unknown','direct_links','indirect_links','indirect_target_tested','exact_networks','extra_finding_networks'])+' |')
    summary+=['','Unknown cases remain in planned denominators; their scientific outcomes are unavailable.',
              'An indirect link may be untested when the direct parent was not retained.',
              'Zero observed errors in ten datasets does not demonstrate error control.',
              'Saved candidate comparisons describe the tested targets, not alternate full-recursive networks.']
    (out/'README.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    if archives:
        assets=evidence.build_release_bundles(archives,out/'bundles',cell_name=f'linear-n{value["sample_size"]}')
        evidence.write_inventory(out/'inventory.json',assets)
    else:
        write_json(out/'inventory.json',[])
    return report


def publish(out):
    first_attempt()
    out=Path(out)
    report=load(out/'report.json')
    if (report['run'],report['source'],report['attempt'])!=(os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA'],1):
        raise ValueError('publication producer differs')
    tag=f'{STUDY}-n{report["sample_size"]}-{report["run"]}'
    def gh(*args):
        subprocess.run(['gh',*map(str,args)],check=True)
    gh('release','create',tag,'--target',report['source'],'--prerelease',
       '--title',f'Matched Elastic Net and OLS moderate-effect pilot N={report["sample_size"]}',
       '--notes-file',out/'README.md')
    assets=[*sorted(p for p in out.iterdir() if p.is_file()),*sorted((out/'bundles').glob('*.zip'))]
    verified=[]
    remote=out/'remote-verification'
    remote.mkdir(exist_ok=False)
    for asset in assets:
        gh('release','upload',tag,asset)
        gh('release','download',tag,'--pattern',asset.name,'--dir',remote)
        evidence.verify_uploaded_asset(asset,remote/asset.name)
        verified.append(dict(name=asset.name,sha256=evidence.sha256_file(asset)))
        (remote/asset.name).unlink()
    write_json(out/'publication-verification.json',dict(tag=tag,run=report['run'],source=report['source'],assets=verified))
    gh('release','upload',tag,out/'publication-verification.json')
    gh('release','download',tag,'--pattern','publication-verification.json','--dir',remote)
    evidence.verify_uploaded_asset(out/'publication-verification.json',remote/'publication-verification.json')
    return dict(tag=tag,verified_assets=len(verified)+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--n',type=int,required=True); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('case'); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--id',required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--job-started',type=float,required=True)
    p=sub.add_parser('preserve'); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--id',required=True); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('aggregate'); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--download',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('publish'); p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        print(json.dumps(prepare(args.n,args.out)))
    elif args.command=='case':
        producer_check(args.manifest,load(args.manifest))
        result=execute(args.manifest,args.id,args.out,job_started_at=args.job_started)
        print(json.dumps(result)); raise SystemExit(0 if result['status']=='complete' else 1)
    elif args.command=='aggregate':
        result=aggregate(args.manifest,args.download,args.out)
        print(json.dumps(result['cells']))
    elif args.command=='preserve':
        print(json.dumps(preserve_interrupted(args.manifest,args.id,args.out)))
    else:
        print(json.dumps(publish(args.out)))


if __name__=='__main__':
    main()
