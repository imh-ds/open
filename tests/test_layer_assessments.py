from dataclasses import replace
import warnings
import numpy as np
import pandas as pd
import pytest

import open.layer_stability as module
from layer_helpers import make_config, make_local


def frame(n=20):
    return pd.DataFrame({"y": np.arange(n, dtype=float), "x": np.arange(n, dtype=float),
                         "z": np.arange(n, dtype=float)})


def test_planned_samples_match_historical_sampling_positions_and_override():
    data, config = frame(), make_config()
    plans = module.make_layer_sample_plans(data, config)
    assert len(plans) == 10
    expected = tuple(sorted(int(i) for i in np.random.default_rng(20).choice(20, size=16, replace=False)))
    assert plans[0].sampled_positions == expected
    assert plans[0].split_plan.random_seed == 20
    assert len(module.make_layer_sample_plans(data, config, n_resamples=4)) == 4
    assert config.stability.n_resamples == 10


@pytest.mark.parametrize("budget,seed", [(True,20),(0,20),(3,-1),(3,2**32-1)])
def test_invalid_budgets_or_seed_range_fail_before_sampling(budget, seed):
    with pytest.raises(ValueError):
        module.make_layer_sample_plans(frame(), make_config(seed=seed), n_resamples=budget)


def test_same_plans_and_candidates_reused_for_each_target(monkeypatch):
    config, data, calls = make_config(n_resamples=3), frame(), []
    plans = module.make_layer_sample_plans(data, config)
    def analyze(data, target, candidates, config, split_plan):
        calls.append((target, candidates, config.analysis.random_seed, split_plan))
        return make_local(target, candidates)
    monkeypatch.setattr(module, "analyze_target", analyze)
    for target in ("y","z"):
        records = module.assess_layer_target(data, depth=0, target=target, candidates=("x",), plans=plans, config=config)
        assert len(records) == 3 and all(r.status == "completed" for r in records)
    assert all(c[1] == ("x",) for c in calls)
    assert all(calls[i][3] is calls[i+3][3] is plans[i].split_plan for i in range(3))


def test_partial_missing_outcomes_are_counted_and_never_imputed(monkeypatch):
    config, data = make_config(n_resamples=2), frame()
    plans = module.make_layer_sample_plans(data, config)
    data.loc[plans[0].sampled_positions[0], "x"] = np.nan
    seen = []
    def analyze(data, target, candidates, config, split_plan):
        seen.append(data[target].isna().sum())
        return replace(make_local(target, candidates), n_observed_target_rows=int(data[target].notna().sum()),
                       n_missing_target_rows=int(data[target].isna().sum()))
    monkeypatch.setattr(module, "analyze_target", analyze)
    records = module.assess_layer_target(data, depth=1, target="x", candidates=("z",), plans=plans, config=config)
    assert seen[0] == 1 and records[0].n_missing_target_rows == 1
    assert records[0].n_observed_target_rows == 15
    assert all(r.status == "completed" for r in records)


def test_infeasible_target_and_global_sample_keep_every_planned_attempt(monkeypatch):
    config, data = make_config(n_resamples=3), frame()
    data["x"] = np.nan
    plans = module.make_layer_sample_plans(data, config)
    monkeypatch.setattr(module, "analyze_target", lambda *a, **k: pytest.fail("must not fit"))
    records = module.assess_layer_target(data, depth=1, target="x", candidates=("z",), plans=plans, config=config)
    assert len(records) == 3 and all(r.status == "infeasible" for r in records)
    assert all(r.n_observed_target_rows == 0 for r in records)
    plans = module.make_layer_sample_plans(frame(4), config)
    assert all(p.split_plan is None for p in plans)
    records = module.assess_layer_target(frame(4), depth=0, target="y", candidates=("x",), plans=plans, config=config)
    assert all(r.status == "infeasible" for r in records)


def test_failures_do_not_retry_or_erase_warnings_and_low_gain_counts(monkeypatch):
    config, data, calls = make_config(n_resamples=3), frame(), []
    plans = module.make_layer_sample_plans(data, config)
    def analyze(data, target, candidates, config, split_plan):
        calls.append(config.analysis.random_seed)
        if config.analysis.random_seed == 21:
            warnings.warn("before failure", UserWarning)
            raise RuntimeError("example failure")
        return make_local(target, candidates, stop=True)
    monkeypatch.setattr(module, "analyze_target", analyze)
    with pytest.warns(UserWarning, match="before failure"):
        records = module.assess_layer_target(data, depth=0, target="y", candidates=("x",), plans=plans, config=config)
    assert calls == [20,21,22]
    assert [r.status for r in records] == ["completed","failed","completed"]
    assert records[1].warnings[0].message == "before failure"
    assert records[1].failure_type == "RuntimeError"
    monkeypatch.setattr(module, "analyze_target", lambda *a,**k: (_ for _ in ()).throw(RuntimeError("x"*3000)))
    failed = module.assess_layer_target(data, depth=0, target="y", candidates=("x",), plans=plans, config=config)
    assert len(failed) == 3 and all(r.status == "failed" and len(r.failure_message) == 2000 for r in failed)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_control_interrupts_propagate(monkeypatch, exception):
    config, data = make_config(n_resamples=1), frame()
    monkeypatch.setattr(module, "analyze_target", lambda *a,**k: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        module.assess_layer_target(data, depth=0, target="y", candidates=("x",),
                                  plans=module.make_layer_sample_plans(data, config), config=config)
