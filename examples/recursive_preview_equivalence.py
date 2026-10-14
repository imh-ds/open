"""Numerical equivalence for derived floats; exact scientific decisions and inputs."""
import math

from open.layer_artifacts import portable

RTOL = 1e-12
ATOL = 1e-14
EXACT_FLOAT_CONTEXTS = {'fitted_parameters', 'observed', 'selection_frequency',
                        'evaluation_frequency', 'probability'}


def compare_payloads(actual, expected):
    report = dict(equivalent=True, relative_tolerance=RTOL, absolute_tolerance=ATOL,
                  changed_float_fields=0, max_absolute_difference=0., mismatches=[])
    def mismatch(path, reason):
        report['equivalent'] = False
        report['mismatches'].append(dict(path='/'.join(map(str,path)), reason=reason))
    def walk(a, b, path=()):
        if type(a) is not type(b):
            mismatch(path,'type differs')
        elif isinstance(a,dict):
            if set(a) != set(b):
                mismatch(path,'mapping keys differ')
            for key in sorted(a.keys() & b.keys()):
                walk(a[key],b[key],(*path,key))
        elif isinstance(a,list):
            if len(a) != len(b):
                mismatch(path,'sequence length differs')
            for i,(x,y) in enumerate(zip(a,b)):
                walk(x,y,(*path,i))
        elif isinstance(a,float):
            if not math.isfinite(a) or not math.isfinite(b):
                mismatch(path,'nonfinite numeric value')
            elif a != b:
                report['changed_float_fields'] += 1
                report['max_absolute_difference'] = max(report['max_absolute_difference'],abs(a-b))
                if set(path)&EXACT_FLOAT_CONTEXTS:
                    mismatch(path,'input/parameter/frequency must match exactly')
                elif not math.isclose(a,b,rel_tol=RTOL,abs_tol=ATOL):
                    mismatch(path,'derived numeric value differs beyond roundoff tolerance')
        elif a != b:
            mismatch(path,'discrete value differs')
    walk(actual,expected)
    return report


def compare_networks(actual, expected):
    report = compare_payloads(portable(actual.layers),portable(expected.layers))
    # Runtime, Git revision and memory provenance naturally differ between executions.
    # Every configuration, sample, split and graph construction decision remains exact.
    for name in ('config','variables','plans','assigned_depth','parents','stop_reasons','status'):
        if getattr(actual,name) != getattr(expected,name):
            report['equivalent'] = False
            report['mismatches'].append(dict(path=name,reason='network identity/decision differs'))
    return report
