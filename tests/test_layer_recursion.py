from dataclasses import replace
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest

import open.layer_stability as module
from open.stability import fit_stable_open, OPENResult
from open.layer_types import LayeredOPENResult
from layer_helpers import make_config, make_local, install_scripted_analyzer


def data(*names):
    return pd.DataFrame({n: np.arange(20,dtype=float) for n in names})


def fit(frame, config=None, **kwargs):
    return module.fit_layer_consensus(frame, "y", tuple(frame), config or make_config(), **kwargs)


def test_fixed_layers_give_every_upstream_target_all_ten_assessments(monkeypatch):
    selections = {("y",b): ("a","b") if b<6 else ("a",) for b in range(10)}
    selections.update({("a",b):("b",) for b in range(10)})
    selections.update({("b",b):("x",) for b in range(10)})
    calls = install_scripted_analyzer(monkeypatch, selections=selections)
    result = fit(data("y","a","b","x"))
    assert result.assigned_depth == {"y":0,"a":1,"b":2,"x":3}
    assert {len(t.assessments) for layer in result.layers for t in layer.evidence} == {10}
    assert len(calls) == 30
    assert all(c[1] == ("b","x") for c in calls if c[0] == "a")
    assert all(c[1] == ("x",) for c in calls if c[0] == "b")
    assert result.stop_reasons["x"] == "MAX_DEPTH"


def test_synchronous_candidates_multi_parent_and_order_invariance(monkeypatch):
    choices = {(t,b): ("a","c") if t=="y" else ("x",) for t in ("y","a","c") for b in range(10)}
    install_scripted_analyzer(monkeypatch, selections=choices)
    first = fit(data("y","a","c","x"))
    second = fit(data("x","c","a","y"))
    assert first.assigned_depth == second.assigned_depth == {"y":0,"a":1,"c":1,"x":2}
    assert first.parents == second.parents == {"a":("y",),"c":("y",),"x":("a","c")}
    assert first.layers[1].candidates == ("x",)
    assert first.stop_reasons["x"] == "NO_UNASSIGNED"


def test_shared_group_promotes_all_members_without_unique_arrows(monkeypatch):
    monkeypatch.setattr(module, "analyze_target", lambda d,t,c,config,s: make_local(t,c,shared=(("a","b"),)))
    result = fit(data("y","a","b"))
    assert result.assigned_depth == {"y":0,"a":1,"b":1}
    assert result.layers[0].evidence[0].relations[0].relation == "SHARED_GROUP"
    assert len(result.layers[0].evidence[0].relations) == 1


def test_incomplete_layer_preserves_prefix_and_finishes_all_siblings(monkeypatch):
    calls = []
    def analyze(d,t,c,config,s):
        calls.append(t)
        return make_local(t,c,unique=("a","c") if t=="y" else ("x",))
    monkeypatch.setattr(module, "analyze_target", analyze)
    frame = data("y","a","c","x")
    frame["a"] = np.nan
    result = fit(frame)
    assert result.status == "INCOMPLETE_LAYER"
    assert result.assigned_depth == {"y":0,"a":1,"c":1}
    assert calls.count("c") == 10 and "a" not in calls
    assert len(result.layers[1].evidence) == 2 and not result.layers[1].finalized
    assert result.layers[1].new_nodes == ()
    a = next(t for t in result.layers[1].evidence if t.target=="a")
    assert all(r.status=="infeasible" for r in a.assessments)


def test_all_failed_root_returns_diagnostics_and_no_promotions(monkeypatch):
    monkeypatch.setattr(module,"analyze_target",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("test")))
    result = fit(data("y","x"),n_resamples=3)
    assert result.status == "INCOMPLETE_LAYER" and result.assigned_depth == {"y":0}
    assert result.effective_resamples == 3 and result.config.stability.n_resamples == 10
    assert len(result.layers[0].evidence[0].assessments) == 3


def test_ordinary_stops_and_depth_boundary(monkeypatch):
    calls = install_scripted_analyzer(monkeypatch,selections={})
    assert fit(data("y","x")).stop_reasons == {"y":"NO_CONSENSUS_PROMOTION"}
    assert fit(data("y")).stop_reasons == {"y":"NO_UNASSIGNED"}
    calls = install_scripted_analyzer(monkeypatch,selections={("y",b):("x",) for b in range(10)})
    result = fit(data("y","x"),make_config(max_depth=1))
    assert len(calls)==10 and result.stop_reasons["x"]=="MAX_DEPTH"


def test_constant_target_is_valid_low_gain_and_warnings_are_preserved():
    frame = pd.DataFrame({"y":np.ones(20),"x":np.arange(20,dtype=float)})
    with warnings.catch_warnings(record=True):
        result = fit(frame, make_config(n_resamples=1))
    assert result.status == "COMPLETE"
    assert result.layers[0].evidence[0].assessments[0].local.stop
    assert result.layers[0].evidence[0].complete


def test_public_dispatch_and_tracing_ownership(monkeypatch):
    install_scripted_analyzer(monkeypatch,selections={})
    frame = data("y","x")
    tracemalloc.start()
    try:
        result = fit_stable_open(frame,"y",tuple(frame),make_config(),n_resamples=1)
        assert isinstance(result,LayeredOPENResult) and tracemalloc.is_tracing()
        assert result.provenance.peak_traced_memory_bytes is None
    finally:
        tracemalloc.stop()
    legacy = replace(make_config(n_resamples=1), stability=replace(make_config().stability,scope="whole_pipeline",n_resamples=1))
    result = fit_stable_open(frame,"y",tuple(frame),legacy)
    assert isinstance(result,OPENResult)


@pytest.mark.parametrize("kind", ["root","type","infinity"])
def test_bad_inputs_are_rejected_before_assessment(kind):
    frame = data("y","x")
    if kind == "root":
        frame["y"] = np.nan
    elif kind == "type":
        frame["x"] = "bad"
    else:
        frame["x"] = np.inf
    with pytest.raises(ValueError):
        fit(frame)
