"""Reassemble and verify the exact native raw archive without changing parts."""
import hashlib,json,zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent;m=json.loads((root/'reassembly.json').read_text())
    pieces=[]
    for entry in m['parts']:
        data=(root/entry['path']).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('part digest differs')
        pieces.append(data)
    data=b''.join(pieces)
    if len(data)!=m['bytes'] or hashlib.sha256(data).hexdigest()!=m['sha256']:raise ValueError('archive digest differs')
    output=root/m['archive']
    if output.exists() and output.read_bytes()!=data:raise ValueError('existing archive differs; refusing replacement')
    output.write_bytes(data)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('archive CRC differs')
    print('Verified exact archive:',output)


if __name__=='__main__':main()
