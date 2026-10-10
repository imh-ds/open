"""Threshold sensitivity by independent reconstruction of every layer."""
from dataclasses import replace
import math
from numbers import Real

from open.layer_stability import fit_layer_consensus


def fit_layer_sensitivity(data, root, variables, config, *, thresholds=(.60,.70,.80), n_resamples=None):
    values = tuple(thresholds) + (config.thresholds.stability,)
    if any(isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in values):
        raise ValueError('thresholds must be finite non-bool numbers in [0, 1]')
    return tuple(fit_layer_consensus(data,root,variables,
        replace(config,thresholds=replace(config.thresholds,stability=float(p))),n_resamples=n_resamples)
        for p in sorted(set(values)))
