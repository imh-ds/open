"""Preserve downloaded evidence and verify byte inventories; performs no model fits.

Run from the repository root after downloading all run artifacts into the named
.superpowers directory. Scientific arithmetic was independently replayed by CI
for completed cases; this script verifies those exact archives, not a second
scientific replay. Incomplete cases remain unknown.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
DOWNLOAD = Path('.superpowers/qualification-noise-download-38655036215')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    raw = OUT / 'raw'
    raw.mkdir(exist_ok=True)
    audit_source = next((DOWNLOAD / 'qualification-noise-audit').rglob('audit.json'))
    report = json.loads(audit_source.read_text())
    expected = {r['spec']['index']: r for r in report['cases']}
    assert len(expected) == report['planned'] == 110
    assert report['github_run'] == '38655036215' and not report['qualified']
    failures, rows, seen = Counter(), [], set()
    for control_path in sorted(DOWNLOAD.rglob('control.json')):
        control = json.loads(control_path.read_text())
        index = control['index']
        assert index not in seen
        seen.add(index)
        row = expected[index]
        assert control['spec'] == row['spec']
        assert control['github_run'] == report['github_run']
        assert control['github_sha'] == report['github_sha']
        assert control['github_attempt'] == report['github_attempt'] == '1'
        assert control['manifest'] == report['manifest']
        archive = control_path.parent / 'raw/case.zip'
        assert sha(archive.read_bytes()) == row['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            names = z.namelist()
            assert len(names) == len(set(names))
            prefix = 'cases/' + control['spec']['id'] + '/'
            assert all(n.startswith(prefix) and '..' not in Path(n).parts for n in names)
            receipt_bytes = z.read(prefix + 'receipt.json')
            assert sha(receipt_bytes) == row['receipt_sha256']
            receipt = json.loads(receipt_bytes)
            assert receipt['spec'] == control['spec']
            for key in ('github_run', 'github_sha', 'github_attempt', 'manifest'):
                assert receipt[key] == control[key]
            actual = {n[len(prefix):]: sha(z.read(n)) for n in names
                      if not n.endswith('/') and n != prefix + 'receipt.json'}
            assert actual == receipt['files']
            targets = Counter()
            for name in names:
                if '/checkpoints/' in name and name.endswith('.json.gz'):
                    payload = json.loads(gzip.decompress(z.read(name)))
                    a = payload['assessment']
                    targets[f"depth {a['depth']}: {a['target']}"] += 1
            if control['status'] != 'complete':
                failures[control['failure']] += 1
            rows.append(dict(index=index, id=control['spec']['id'], n=control['spec']['n'],
                             status=receipt['status'], failure=control.get('failure'),
                             seconds=receipt['seconds'], checkpoint_targets=dict(targets),
                             started=row['started_assessments'], finished=row['finished_assessments']))
        shutil.copyfile(archive, raw / f'case-{index}.zip')
        shutil.copyfile(control_path, raw / f'control-{index}.json')
    assert seen == set(expected)
    for folder in ('qualification-noise-authorization', 'qualification-noise-audit'):
        for source in (DOWNLOAD / folder).rglob('*'):
            if source.is_file():
                target = raw / folder / source.relative_to(DOWNLOAD / folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    result = dict(run=report['github_run'], verified_archives=len(rows),
                  complete=report['complete'], failure_counts=dict(failures),
                  qualified=report['qualified'], cells=report['cells'], cases=rows,
                  verification='Local CRC, archive SHA256, receipt SHA256, file inventories and provenance; '
                  'completed-case arithmetic replay is the preserved CI audit. No local fits or retries.')
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    hashes = {p.relative_to(OUT).as_posix(): sha(p.read_bytes())
              for p in sorted(raw.rglob('*')) if p.is_file()}
    (OUT / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(archives=len(rows), failures=dict(failures), raw_bytes=sum(
        p.stat().st_size for p in raw.rglob('*') if p.is_file()))))


if __name__ == '__main__':
    main()
