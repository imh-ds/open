"""No-fit recovery of D-043's exact-equality diagnostic audit failure.

The frozen generators, models, cutoffs and original evidence stay unchanged.
Only recomputed diagnostic teacher values admit bounded floating-point roundoff.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,hashlib,os,json,time
import numpy as np
import realistic_interaction_study as study
import realistic_interaction_report as report
import prospective_interaction_archive as archive

ABSOLUTE_TOLERANCE=1e-12


def validate_population(expected,saved):
    if not isinstance(saved,dict) or set(saved)!=set(expected):raise ValueError('teacher schema differs')
    for key in ('scope','ideal_additive'):
        if saved[key]!=expected[key]:raise ValueError('teacher metadata differs')
    maximum=0.;fields={}
    for key in ('predicted','surfaces','mse'):
        a=np.asarray(expected[key]);b=np.asarray(saved[key])
        if a.shape!=b.shape or b.dtype.kind not in 'fiu' or not np.isfinite(b).all() or not np.isfinite(a).all():
            raise ValueError('teacher numeric shape, type or finiteness differs: '+key)
        if key=='mse' and (b.shape!=() or float(b)<0):raise ValueError('invalid teacher risk')
        delta=float(np.max(np.abs(a-b),initial=0.))
        if delta>ABSOLUTE_TOLERANCE:raise ValueError('teacher discrepancy exceeds absolute tolerance: '+key)
        fields[key]=delta;maximum=max(maximum,delta)
    return dict(max_absolute_difference=maximum,fields=fields)


@contextmanager
def numeric_audit():
    original=study.audit_population;compare=study.old.models.compare;audit_model=study.old.models.audit_comparison
    stats=dict(comparisons=0,roundoff_comparisons=0,max_absolute_difference=0.,by_task={})
    def bounded(c,data,v,o):
        result=validate_population(study.population_record(c,data,v),o)
        stats['comparisons']+=1
        delta=result['max_absolute_difference'];stats['max_absolute_difference']=max(stats['max_absolute_difference'],delta)
        if delta:
            stats['roundoff_comparisons']+=1
            stats['by_task'][c['id']]=max(stats['by_task'].get(c['id'],0.),delta)
    def forbidden(*args,**kwargs):raise ValueError('recovery forbids model fitting')
    def no_refit(*args,**kwargs):
        if kwargs.get('refit') or (len(args)>6 and args[6]):raise ValueError('recovery forbids verification refits')
        return audit_model(*args,**kwargs)
    study.audit_population=bounded;study.old.models.compare=forbidden;study.old.models.audit_comparison=no_refit
    try:yield stats
    finally:study.audit_population=original;study.old.models.compare=compare;study.old.models.audit_comparison=audit_model


def recover(inputs,output):
    study.freeze();study.thresholds();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with numeric_audit() as stats:audit=report.aggregate('development',inputs,output)
    proof=dict(schema='open.realistic_numeric_recovery.v1',status='complete' if audit['complete']==10800 and not audit['missing'] and not audit['audit_errors'] else 'incomplete',
        original_run=38482368226,original_source='2005c0aea61f963386569bdc83d26be0d60efc04',
        frozen_manifest=study.identity(),thresholds=study.io.fingerprint(study.thresholds()),absolute_tolerance=ABSOLUTE_TOLERANCE,relative_tolerance=0.,
        new_fits=0,new_refits=0,statistics=stats,raw_archives=audit['merge']['archives'],
        recovery_source=os.environ.get('GITHUB_SHA'),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        scope='All original file digests, model/CV/risk/curve/identity checks retained. Only diagnostic teacher recomputation permits absolute roundoff <=1e-12. Saved values and all decisions use untouched original evidence.')
    study.io.write(output/'recovery.json',proof)
    return audit,proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit,proof=recover([a.input],a.output)
    print(json.dumps(dict(planned=audit['planned'],complete=audit['complete'],missing=audit['missing'],errors=len(audit['audit_errors']),max_roundoff=proof['statistics']['max_absolute_difference'],new_fits=0,new_refits=0)))
    if proof['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
