"""Approved D-043 development universe, separate from historical confirmation."""
from functools import lru_cache
import math
from prospective_interaction_design import audit_seed_collisions


@lru_cache(None)
def _tasks():
    result=[]
    for stage in ('calibration','development'):
        for n in (150,300,600):
            for gamma in (.1,.3,.5):
                for reliability in (1.,.8):
                    cell=f'n{n}-g{gamma:g}-R{reliability:g}'
                    groups=[('noise','none',0.),('observed-additive','none',0.)]
                    if stage=='development':
                        groups += [(f'{shape}-b{b:g}',shape,b) for shape in ('product','tanh') for b in (.2,.4)]
                        groups += [('product-b0.8','product',.8)]
                        if reliability==.8:groups += [('latent-additive','none',0.)]
                    for benchmark,shape,b in groups:
                        for rep in range(100 if stage=='calibration' else 80):
                            i=len(result)
                            result.append(dict(id=f'realistic--{stage}--{cell}--{benchmark}-r{rep:03d}',
                                stage=stage,cell=cell,n=n,p=5,gamma=gamma,reliability=reliability,
                                rho=gamma*reliability,benchmark=benchmark,shape=shape,b=b,replicate=rep,
                                seed=708000000+i,test_seed=718000000+i,split_seed=728000000+i))
    return tuple(result)


def tasks():return [dict(c) for c in _tasks()]


def stage_tasks(stage):
    if stage not in ('calibration','development'):raise ValueError('unknown realistic stage')
    return [dict(c) for c in _tasks() if c['stage']==stage]


def shards(stage):return [f'{i:03d}' for i in range(math.ceil(len(stage_tasks(stage))/80))]


def shard_tasks(stage,shard):
    if shard not in shards(stage):raise ValueError('foreign realistic shard')
    i=int(shard);return stage_tasks(stage)[80*i:80*(i+1)]
