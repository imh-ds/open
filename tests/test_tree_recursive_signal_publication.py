import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


@pytest.mark.parametrize('omit_verified_archive',[False,True])
def test_publication_binds_every_verified_case_archive_and_distinct_inventory(tmp_path,monkeypatch,omit_verified_archive):
    import tree_recursive_signal_actions as a
    import qualification_artifacts
    manifest,digest = a.checked_manifest()
    for name,value in {'GITHUB_RUN_ATTEMPT':'1','GITHUB_EVENT_NAME':'workflow_dispatch',
                       'GITHUB_RUN_ID':'123','GITHUB_SHA':'a'*40,'GITHUB_REPOSITORY':'imh-ds/open'}.items():
        monkeypatch.setenv(name,value)
    monkeypatch.setattr(a,'checked_manifest',lambda *args,**kwargs:(manifest,digest))
    report={'run_id':'123','source_sha':'a'*40,'manifest_sha256':digest,'cells':[]}
    prepared={}
    for n in (80,200):
        for engine in a.study.ENGINES:
            name=f'tree-signal-publication-{engine}-n{n}'
            folder=tmp_path/'inputs'/name
            folder.mkdir(parents=True)
            specs=[s for s in manifest['cases'] if s['sample_size']==n and s['engine']==engine]
            records,rows=[],[]
            for i,spec in enumerate(specs):
                raw=tmp_path/'raw'/f"{spec['id']}.zip"
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(spec['id'].encode())
                sha=a.evidence.sha256_file(raw)
                rows.append(dict(case_id=spec['id'],status='complete',fit_cost_unknown=False,archive_sha256=sha))
                if not (omit_verified_archive and n==80 and engine=='random_forest' and i==54):
                    records.append(dict(case_id=spec['id'],archive=raw,sha256=sha))
            assets=a.evidence.build_release_bundles(records,folder/'bundles',cell_name=f'tree-signal-{engine}-n{n}')
            inventory=a.evidence.write_inventory(folder/'archive-inventory.json',assets)
            a.write_json(folder/'publication.json',dict(run_id='123',source_sha='a'*40,manifest_sha256=digest,
                         engine=engine,sample_size=n,assets=inventory))
            report['cells'].append(dict(engine=engine,sample_size=n,cases=rows))
            prepared[name]=folder
    out=tmp_path/'publication'
    out.mkdir()
    a.write_json(out/'study-report.json',report)
    for name in ('interpretation.md','study-manifest.json','study-protocol.md'):
        (out/name).write_text('metadata')
    def download(run,names,path,**kwargs):
        for name in names:
            shutil.copytree(prepared[name],Path(path)/name)
    monkeypatch.setattr(qualification_artifacts,'download_artifacts',download)
    uploaded={}
    def gh(argv,**kwargs):
        if argv[2]=='upload':
            path=Path(argv[4])
            assert path.name not in uploaded, 'release assets must have distinct filenames'
            uploaded[path.name]=path.read_bytes()
        elif argv[2]=='download':
            name=argv[argv.index('--pattern')+1]
            directory=Path(argv[argv.index('--dir')+1]); directory.mkdir(parents=True,exist_ok=True)
            (directory/name).write_bytes(uploaded[name])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a.subprocess,'run',gh)
    if omit_verified_archive:
        with pytest.raises(ValueError,match='verified|archive|coverage'):
            a.publish(out)
    else:
        assert a.publish(out)['verified']
        assert len([n for n in uploaded if n.startswith('inventory-')])==6
        assert len([n for n in uploaded if n.endswith('.zip')])==6
