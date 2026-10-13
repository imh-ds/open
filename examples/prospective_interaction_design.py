"""Immutable prospective universe and auditable null-first selection."""
from functools import lru_cache
import math
import confidence_diagnostic as d

CELLS=((300,5,0.),(300,5,.7),(600,5,0.),(600,5,.7),(300,15,.7),(300,15,.9))
BUDGET_MINUTES=6450


def cell_ids():return [d.previous.rules.cell_id(*c) for c in CELLS]


@lru_cache(None)
def _tasks():
    result=[]
    for stage in ('null','signal'):
        for n,p,rho in CELLS:
            if stage=='null':groups=[('noise',0.,0.,1200,'noise'),('additive',0.,0.,1200,'additive')]
            else:
                qs=(.002,.005) if rho==.9 else (.01,.04)
                groups=[('mixed',q,d.matched_coefficient(q,rho),450,f'matched-q{q:g}') for q in qs]
                groups += [('mixed',0.,.8,450,'mixed-b0.8'),('pure',0.,.8,450,'pure-b0.8')]
            for scenario,q,b,count,benchmark in groups:
                for rep in range(count):
                    i=len(result);cell=d.previous.rules.cell_id(n,p,rho)
                    result.append(dict(id=f'prospective--{stage}--{cell}--{benchmark}-r{rep:04d}',
                        stage=stage,cell=cell,n=n,p=p,rho=rho,scenario=scenario,target_gap=q,b=b,
                        benchmark=benchmark,replicate=rep,seed=608000000+i,test_seed=618000000+i,split_seed=628000000+i))
    return tuple(result)


def tasks():return [dict(c) for c in _tasks()]


def stage_tasks(stage,selection=None):
    if stage not in ('null','signal'):raise ValueError('unknown prospective stage')
    if stage=='signal' and selection is None:raise ValueError('signal stage requires frozen selection')
    cells=cell_ids() if stage=='null' else selection['cells']
    if len(set(cells))!=len(cells) or not set(cells)<=set(cell_ids()):raise ValueError('invalid selected cells')
    return [dict(c) for c in _tasks() if c['stage']==stage and c['cell'] in cells]


def shards(stage,selection=None):return [f'{i:03d}' for i in range(math.ceil(len(stage_tasks(stage,selection))/80))]


def shard_tasks(stage,shard,selection=None):
    if shard not in shards(stage,selection):raise ValueError('foreign prospective shard')
    i=int(shard);return stage_tasks(stage,selection)[80*i:80*(i+1)]


def seed_values(value):
    if isinstance(value,dict):
        for key,v in value.items():
            if key in ('seed','test_seed','split_seed'):
                if type(v) is not int:raise ValueError('invalid historical seed field')
                yield v
            else:yield from seed_values(v)
    elif isinstance(value,(list,tuple)):
        for v in value:yield from seed_values(v)


def audit_seed_collisions(planned,historical):
    ids=[c['id'] for c in planned];values=list(seed_values(planned));prior=set(seed_values(historical))
    if len(set(ids))!=len(ids) or len(values)!=3*len(planned) or len(set(values))!=len(values) or set(values)&prior:
        raise ValueError('duplicate identity or seed collision')
    return dict(planned=len(planned),unique_seeds=len(values),historical_unique_seeds=len(prior),collisions=0)


def make_selection(manifest,audit):
    identity=manifest.get('identity',d.previous.io.fingerprint(manifest))
    if (audit.get('stage')!='null' or audit.get('manifest')!=identity or audit.get('complete')!=14400 or
        audit.get('missing')!=0 or audit.get('audit_errors') or audit.get('planned',14400)!=14400):
        raise ValueError('selection requires complete matching audited null evidence')
    cells=audit['passed_cells']
    if cells!=[c for c in cell_ids() if c in cells]:raise ValueError('invalid passed condition order')
    signal=[c for c in _tasks() if c['stage']=='signal']
    return dict(schema='open.prospective_selection.v1',manifest=identity,
        stage_a_audit=d.previous.io.fingerprint(audit),cells=cells,
        selected_ids=[c['id'] for c in signal if c['cell'] in cells],
        skipped_ids=[c['id'] for c in signal if c['cell'] not in cells],
        skip_reason='both null-class gates not passed',multiplicity=dict(false=36,detection=24,shape=24))


def validate_selection(selection,manifest,stage_a_audit):
    if selection!=make_selection(manifest,stage_a_audit):raise ValueError('selection provenance or planned identities differ')
    return selection['cells']
