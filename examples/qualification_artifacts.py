"""Fetch every artifact page, then download exact required names without fitting.

The original combined CI audit received only 100 of 112 producer artifacts from
its wildcard download action. Keep frozen producer workflows unchanged; use this
separate recovery utility for complete, checked input transport. The scientific
auditors must still verify archive and receipt fingerprints after download.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def github_api(path):
    return json.loads(subprocess.check_output(['gh','api','repos/imh-ds/open/'+path]))


def select_artifacts(run_id, names, *, api=github_api, allow_missing=False):
    if not str(run_id).isdigit() or not names or len(set(names)) != len(names):
        raise ValueError('requires one run and distinct required artifact names')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',n) for n in names):
        raise ValueError('unsafe required artifact name')
    records, ids, total, page = [], set(), None, 1
    while True:
        response = api(f'actions/runs/{run_id}/artifacts?per_page=100&page={page}')
        count = response['total_count']
        if type(count) is not int or count < 0 or total is not None and total != count:
            raise ValueError('artifact inventory changed while paging')
        total = count
        batch = response['artifacts']
        for record in batch:
            if record['id'] in ids:
                raise ValueError('duplicate artifact identity across pages')
            ids.add(record['id'])
        records.extend(batch)
        if len(records) == total:
            break
        if len(batch) != 100 or len(records) > total:
            raise ValueError('incomplete artifact inventory; do not aggregate partial downloads')
        page += 1
    chosen = []
    for name in names:
        matches = [r for r in records if r['name'] == name]
        if allow_missing and (not matches or len(matches)==1 and matches[0]['expired']):
            continue
        if len(matches) != 1 or matches[0]['expired']:
            raise ValueError(f'required artifact missing, duplicate or expired: {name}')
        chosen.append(matches[0])
    return chosen


def download_artifacts(run_id, names, output, *, api=github_api, run=subprocess.run, allow_missing=False):
    selected = select_artifacts(run_id, names, api=api, allow_missing=allow_missing)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Check all target directories before any download, never overwrite evidence.
    if any((output/r['name']).exists() for r in selected):
        raise ValueError('artifact target already exists; preserve previous download')
    inventory=dict(run=str(run_id),artifacts=selected)
    if allow_missing:
        found={r['name'] for r in selected}
        inventory.update(missing=[n for n in names if n not in found],failed=[])
    with (output/'download-inventory.json').open('x',encoding='utf-8') as stream:
        json.dump(inventory,stream,indent=2)
        stream.write('\n')
    for record in selected:
        directory = output/record['name']
        try:
            run(['gh','run','download',str(run_id),'--repo','imh-ds/open',
                 '--name',record['name'],'--dir',str(directory)],check=True)
        except subprocess.CalledProcessError:
            if not allow_missing:
                raise
            inventory['failed'].append(record['name'])
            (output/'download-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    if allow_missing and (inventory['missing'] or inventory['failed']):
        raise ValueError('some artifacts missing/expired or failed transfer; available evidence retained')
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--names-file',required=True,type=Path,help='JSON list of exact required names')
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    result = download_artifacts(args.run,json.loads(args.names_file.read_text()),args.out)
    print(f'Downloaded {len(result)} required artifacts from the complete paginated inventory.')
