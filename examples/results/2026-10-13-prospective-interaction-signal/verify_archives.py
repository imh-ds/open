"""Verify and preserve every raw shard from the original Actions artifact; no fits."""
from pathlib import Path
import argparse,json,hashlib,zipfile,io

def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True,type=Path);args=parser.parse_args()
    folder=Path(__file__).resolve().parent;root=folder.parents[2]
    audit=json.loads((folder/'actions-audit.json').read_text())
    manifest=json.loads((root/'examples/study_manifests/prospective-interaction.json').read_text())
    selection=json.loads((root/'examples/study_manifests/prospective-interaction-selection.json').read_text())
    identity=fp(manifest);selected_identity=fp(selection);selected_ids=set(selection['selected_ids'])
    assert identity==audit['manifest'] and selected_identity==audit['selection']
    for path,digest in manifest['sources'].items():assert sha((root/path).read_bytes().replace(b'\r\n',b'\n'))==digest,path
    digest=hashlib.sha256()
    with args.artifact.open('rb') as f:
        while data:=f.read(8*1024*1024):digest.update(data)
    assert args.artifact.stat().st_size==2905876229 and digest.hexdigest()=='e6d4b30230066ca94589065566860c18e373fe1335c6b453846032ec6387954a'
    expected=[c for c in manifest['tasks'] if c['id'] in selected_ids]
    assert len(expected)==10800
    anchors={r['path'].replace(chr(92),'/').split('/')[-2].rsplit('-',1)[-1]:r for r in audit['merge']['archives']}
    assert set(anchors)=={f'{i:03d}' for i in range(135)}
    raw=folder/'raw';raw.mkdir(exist_ok=True);files=0;seen=set();cost=refits=warnings=0;retained=[]
    with zipfile.ZipFile(args.artifact) as outer:
        assert len(set(outer.namelist()))==len(outer.namelist())
        for name in ('audit.json','dataset-metrics.json'):
            assert outer.read('.superpowers/prospective-interaction/'+name)==(folder/('actions-'+name)).read_bytes()
        for i in range(135):
            shard=f'{i:03d}';prefix=f'.superpowers/prospective-interaction-downloads/prospective-interaction-signal-shard-{shard}/'
            data=outer.read(prefix+'raw-shard.zip');receipt_data=outer.read(prefix+'receipt.json');receipt=json.loads(receipt_data)
            assert len(data)==receipt['bytes']==anchors[shard]['bytes'] and sha(data)==receipt['sha256']==anchors[shard]['sha256']
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                inventory={r['path']:r for r in receipt['inventory']};assert len(inventory)==receipt['files']==len(archive.namelist())
                assert set(archive.namelist())==set(inventory)
                tasks=expected[i*80:(i+1)*80]
                stamp=json.loads(archive.read(f'shard-signal-{shard}.json'))
                assert stamp==dict(id=shard,stage='signal',tasks=tasks,manifest=identity,selection=selected_identity)
                for name,entry in inventory.items():
                    contents=archive.read(name);assert len(contents)==entry['bytes'] and sha(contents)==entry['sha256'];files+=1
                for c in tasks:
                    assert c['id'] not in seen;seen.add(c['id']);context=fp(dict(task=c,manifest=identity,selection=selected_identity))
                    value=json.loads(archive.read(c['id']+'/result.json'));assert value['task']==c and value['context']==context and value['status']=='complete'
                    warnings+=len(value['warnings'])
                    attempt=json.loads(archive.read(c['id']+'/attempts.json'));assert attempt['context']==context
                    for a in attempt['attempts']:
                        assert a['status']=='complete' and not a['unknown_fit_cost'] and not a['unknown_verification_cost']
                        cost+=a['fits'];refits+=a['verification_refits']
            target=raw/f'shard-{shard}.zip';target.write_bytes(data);target.with_suffix('.json').write_bytes(receipt_data)
            retained.append(dict(shard=shard,bytes=receipt['bytes'],sha256=receipt['sha256'],files=receipt['files']))
            if i%15==0 or i==134:print('Verified and retained shard',shard,flush=True)
    assert seen==set(selection['selected_ids']) and cost==audit['known_attempt_cost']['fits'] and refits==audit['known_attempt_cost']['verification_refits'] and warnings==audit['warning_count']==0
    result=dict(artifact_id=11845895297,artifact_bytes=args.artifact.stat().st_size,artifact_sha256=digest.hexdigest(),source='110ddfd617e8d9694d27688998d1d4cb1f719700',sources_checked=len(manifest['sources']),datasets=len(seen),shards=len(retained),raw_files=files,fits=cost,verification_refits=refits,model_fits_performed=0,scope='Independent provenance, inventory, hashes, ZIP CRCs, task contexts and attempt costs; arithmetic bounds checked separately. Saved prediction and curve arithmetic were audited on Actions, not independently refitted here.',archives=retained)
    (folder/'raw-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified full raw evidence:',len(seen),'datasets,',files,'files; no new fits',flush=True)
if __name__=='__main__':main()
