"""Reassemble and verify retained study evidence without GitHub access.

Usage: python verify_archive.py --output PATH [--extract]
Does not refit models or alter the frozen study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    receipt = json.loads((here/'receipt.json').read_text())['retained_archive']
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'raw-evidence.zip'
    if target.exists():
        raise ValueError('output archive already exists; choose a fresh output directory')
    digest = hashlib.sha256()
    with target.open('wb') as output:
        for part in receipt['parts']:
            data = (here/part['file']).read_bytes()
            assert len(data)==part['bytes']
            assert hashlib.sha256(data).hexdigest()==part['sha256']
            output.write(data)
            digest.update(data)
    assert target.stat().st_size==receipt['bytes']
    assert digest.hexdigest()==receipt['sha256']
    inventory = json.loads((here/'retained-inventory.json').read_text())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==receipt['files']==len(inventory)
        assert set(archive.namelist())=={item['path'] for item in inventory}
        for item in inventory:
            data = archive.read(item['path'])
            assert len(data)==item['bytes']
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            destination = (args.output/item['path']).resolve()
            assert destination.is_relative_to(args.output.resolve())
            if args.extract:
                if destination.exists():
                    raise ValueError('extraction target already exists: '+str(destination))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    print(f"Verified {receipt['files']} files; archive SHA256 {digest.hexdigest()}")


if __name__=='__main__':
    main()
