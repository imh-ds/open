"""Small valid local results for layer orchestration contract tests."""
from dataclasses import replace

import pandas as pd

from open.config import (AnalysisConfig, OpenConfig, PerformanceCVConfig,
                         StabilityConfig, TuningCVConfig)
from open.grouping import SharedComponent
from open.local_target import LocalClass, LocalResult, VariableResult
from open.relevance import RelevanceContrast, RelevanceResult
from open.splits import make_split_plan


def make_config(*, root="y", seed=20, n_resamples=10, max_depth=3):
    return OpenConfig(AnalysisConfig(root, seed, max_depth),
                      performance_cv=PerformanceCVConfig(2, 1), tuning_cv=TuningCVConfig(2),
                      stability=StabilityConfig(n_resamples=n_resamples, scope="layer_consensus"))


def make_local(target, candidates, *, unique=(), stop=False, baseline=1.,
               raw_unique=None, shared=(), screened=None):
    raw_unique = raw_unique or {}
    screened = screened or {}
    gain = .01 if stop else .5
    raw = {n: raw_unique.get(n, .2 if n in unique else -.01) for n in candidates}
    relevance = RelevanceResult(gain, gain / baseline if baseline > 0 else None,
        "LOW_TOTAL_GAIN" if stop else None,
        {n: RelevanceContrast(.2, None if stop else .4) for n in candidates},
        {n: RelevanceContrast(raw[n], None if stop else raw[n] / gain) for n in candidates},
        baseline, baseline - gain)
    groups = tuple(SharedComponent(tuple(sorted(m)), True, .3, .6, None) for m in shared)
    classes = {} if stop else {
        n: VariableResult(LocalClass.PROXIMAL_UNIQUE if n in unique else
            LocalClass.SHARED_GROUP_MEMBER if any(n in m for m in shared) else
            LocalClass.SCREENED_BY_PARTNER if n in screened else LocalClass.NEGLIGIBLE,
            screened_by_partner=tuple(screened.get(n, ())),
            shared_groups=tuple(tuple(sorted(m)) for m in shared if n in m)) for n in candidates}
    promoted = frozenset(n for n, v in classes.items() if v.kind in
                        (LocalClass.PROXIMAL_UNIQUE, LocalClass.SHARED_GROUP_MEMBER))
    return LocalResult(stop, "LOW_TOTAL_GAIN" if stop else None, target, gain,
        gain / baseline if baseline > 0 else None, relevance, None,
        {n: c.normalized for n, c in relevance.broad.items()},
        {n: c.normalized for n, c in relevance.unique.items()}, None,
        groups, (), (), classes, promoted, n_observed_target_rows=16,
        n_missing_target_rows=0, input_candidates=tuple(candidates))


def make_sample(index=0):
    from open.layer_types import PlannedSample
    config = make_config(seed=20 + index)
    split = make_split_plan(pd.DataFrame({"y": range(16)}), config)
    return PlannedSample(index, 20 + index, tuple(range(16)), split, None)


def make_assessment(local, *, index=0, depth=0):
    from open.layer_types import LocalAssessment
    return LocalAssessment(depth, local.target, local.input_candidates, make_sample(index),
                           "completed", local, (), None, None, 16, 0)


def make_failed_assessment(target, candidates, *, index=0, depth=0, status="failed"):
    from open.layer_types import LocalAssessment
    return LocalAssessment(depth, target, candidates, make_sample(index), status, None, (),
                           "ExampleError", "example", 16, 0)


def install_scripted_analyzer(monkeypatch, *, base_seed=20, selections):
    import open.layer_stability as module
    calls = []
    def analyze(data, target, candidates, config, split_plan):
        calls.append((target, candidates, config.analysis.random_seed, config.thresholds.stability))
        chosen = selections.get((target, config.analysis.random_seed - base_seed), ())
        local = make_local(target, candidates, unique=tuple(n for n in chosen if n in candidates))
        return replace(local, n_observed_target_rows=int(data[target].notna().sum()),
                       n_missing_target_rows=int(data[target].isna().sum()))
    monkeypatch.setattr(module, "analyze_target", analyze)
    return calls


def save_root_only_bundle(path):
    from open.layer_stability import fit_layer_consensus
    from open.layer_io import write_layer_outputs, load_layer_bundle
    from open.metadata import validate_metadata
    frame = pd.DataFrame({"y": range(20)})
    metadata = validate_metadata(frame.columns, [{"name":"y","type":"continuous","role":"root"}], "y")
    result = fit_layer_consensus(frame,"y",tuple(frame),make_config())
    write_layer_outputs(result,path,metadata=metadata)
    return load_layer_bundle(path)
