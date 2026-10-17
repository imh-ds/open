"""Explicit OLS zero-search policy around the unchanged historical auditor."""
from contextlib import contextmanager

import multiengine_recursive_audit as legacy

Store=legacy.Store


@contextmanager
def policy():
    valid,expected=legacy.valid_parameters,legacy.expected_tuning

    def valid_parameters(params,engine):
        return isinstance(params,dict) and not params if engine=='ols' else valid(params,engine)

    def expected_tuning(risk,config,tuned):
        return 0 if config.learner.library==('ols',) else expected(risk,config,tuned)

    legacy.valid_parameters,legacy.expected_tuning=valid_parameters,expected_tuning
    try:
        yield
    finally:
        legacy.valid_parameters,legacy.expected_tuning=valid,expected


def audit_case(*args,**kwargs):
    # Aggregation replays cases sequentially; do not share this scope with threads.
    with policy():
        return legacy.audit_case(*args,**kwargs)
