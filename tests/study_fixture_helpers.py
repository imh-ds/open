"""Temporary study contexts for unit tests, never scientific calibration evidence."""
from functools import lru_cache


def isolate_caches(module, monkeypatch):
    # freeze() clears module caches; keep shared originals intact on teardown.
    for name, value in list(vars(module).items()):
        if callable(getattr(value, 'cache_parameters', None)):
            monkeypatch.setattr(module, name,
                lru_cache(**value.cache_parameters())(value.__wrapped__))


def isolate_pair_calibration(old, tmp, monkeypatch):
    monkeypatch.setattr(old, 'MANIFESTS',
        {s: tmp/f'{s}.json' for s in ('calibration', 'confirmation')})
    monkeypatch.setattr(old, 'THRESHOLDS', tmp/'thresholds.json')
    isolate_caches(old, monkeypatch)
    old.freeze('calibration', False)
    rows = [dict(id=c['id'], cell=c['cell'], scenario=c['scenario'],
                 score=0., result_sha256='a'*64)
            for c in old.rules.tasks('calibration')]
    old.io.write(old.THRESHOLDS, old.make_rule(rows))
    old.freeze('confirmation', False)


def isolate_prospective(study, tmp, monkeypatch):
    isolate_pair_calibration(study.old, tmp/'pair', monkeypatch)
    for name, filename in (('MANIFEST', 'manifest.json'),
                           ('SELECTION', 'selection.json'), ('STAGE_A', 'null.json')):
        monkeypatch.setattr(study, name, tmp/filename)
    isolate_caches(study, monkeypatch)
    study.freeze(False)
