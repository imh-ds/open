import sys,importlib.util,zipfile
from pathlib import Path
import pytest


def load():
    p=Path(__file__).parents[1]/'examples/prospective_interaction_archive.py'
    assert p.exists(),'bounded compressed evidence pipeline missing'
    sys.path.insert(0,str(p.parent));s=importlib.util.spec_from_file_location('prospective_archive',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_pack_preserves_raw_bytes_and_hashes(tmp_path):
    m=load();raw=tmp_path/'raw';raw.mkdir();(raw/'checkpoint.json').write_text('{"raw":1}')
    out=tmp_path/'out';r=m.pack(raw,out)
    with zipfile.ZipFile(out/'raw-shard.zip') as z:assert z.read('checkpoint.json')==(raw/'checkpoint.json').read_bytes()
    assert r['raw_bytes']==9 and r['files']==1


def test_extraction_rejects_escaping_paths(tmp_path):
    m=load();p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','bad')
    with pytest.raises(ValueError):m.extract_checked(p,tmp_path/'scratch')
    assert not (tmp_path/'escape').exists()


def test_stream_duplicate_receipt_and_final_counts(tmp_path,monkeypatch):
    m=load();root=tmp_path/'input';root.mkdir();raw=root/'raw';raw.mkdir();(raw/'sample').write_text('one')
    m.pack(raw,root/'one');m.pack(raw,root/'two')
    archives=m.archives([root/'one',root/'two']);assert len(archives)==2
    unique,receipt=m.unique_archives(archives);assert len(unique)==1 and receipt['identical_archive_duplicates']==1
    (root/'two'/'receipt.json').write_text('{}')
    with pytest.raises(ValueError):m.unique_archives(archives)
