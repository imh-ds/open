"""Reconstruct the four byte-identical archives sent as small transport pieces."""
from pathlib import Path
import hashlib,json,zipfile

def assemble(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    manifest=json.loads((root/'archive-pieces/manifest.json').read_text())
    for item in manifest['archives']:
        target=root/item['path']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
            continue
        pieces=[]
        for piece in item['pieces']:
            data=(root/piece['path']).read_bytes()
            assert len(data)==piece['bytes'] and hashlib.sha256(data).hexdigest()==piece['sha256'],piece['path']
            pieces.append(data)
        data=b''.join(pieces)
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
        print('Reassembled and verified',item['path'])

if __name__=='__main__':assemble()
