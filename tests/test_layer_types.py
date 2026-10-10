from dataclasses import replace
import importlib

import pandas as pd
import pytest

from open.config import StabilityConfig, parse_config, dump_config, load_config
from layer_helpers import make_config, make_local, make_sample, make_assessment


@pytest.fixture
def types():
    return importlib.import_module("open.layer_types")


def test_scope_omission_round_trip_and_positional_compatibility(tmp_path):
    assert StabilityConfig("subsample", .8, 10, 20).scope == "whole_pipeline"
    config = make_config()
    dump_config(config, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == config
    assert parse_config({"analysis": {"root": "y", "random_seed": 20}}).stability.scope == "whole_pipeline"
    with pytest.raises(ValueError):
        StabilityConfig(scope="ambiguous")


def test_layer_scope_cannot_silently_use_historical_builders():
    from open.stability import run_stability_batch
    from open.recurse import build_network_once
    frame = pd.DataFrame({"y": range(20), "x": range(20)})
    for fn in (run_stability_batch, build_network_once):
        with pytest.raises(NotImplementedError, match="layer"):
            fn(frame, "y", tuple(frame), make_config())


def test_incomplete_count_does_not_publish_a_final_frequency(types):
    count = types.SelectionCount(1, 3, 2)
    assert count.probability is None
    assert count.provisional_probability == .5
    assert types.SelectionCount(0, 3, 0).provisional_probability is None
    assert types.SelectionCount(2, 3, 3).probability == 2/3


@pytest.mark.parametrize("args", [(True, 3, 3), (0, 0, 0), (-1, 3, 3),
                                  (3, 3, 2), (1, 2, 3), (1, float("nan"), 2)])
def test_invalid_count_contracts(types, args):
    with pytest.raises(ValueError):
        types.SelectionCount(*args)


@pytest.mark.parametrize("count,value", [(0, .2), (1, None), (True, .2), (1, float("inf"))])
def test_strength_availability_and_finiteness(types, count, value):
    with pytest.raises(ValueError):
        types.StrengthSummary(count, value, 0, None, 0, None, "test")


def test_sample_and_assessment_contracts(types):
    sample = make_sample()
    with pytest.raises(ValueError):
        replace(sample, sampled_positions=(1, 0))
    with pytest.raises(ValueError):
        replace(sample, infeasibility_reason="no split")
    record = make_assessment(make_local("y", ("x",), unique=("x",)))
    for change in ({"target": "z"}, {"candidates": ("z",)}, {"status": "failed"},
                   {"failure_type": "wrong"}, {"n_missing_target_rows": -1}):
        with pytest.raises(ValueError):
            replace(record, **change)


def test_local_mapping_is_detached_and_frozen(types):
    local = make_local("y", ("x",), unique=("x",))
    record = make_assessment(local)
    local.classes.clear()
    assert "x" in record.local.classes
    with pytest.raises(TypeError):
        record.local.classes["x"] = None


def test_duplicate_assessments_and_empty_targets_are_invalid(types):
    record = make_assessment(make_local("y", ("x",)))
    for records in ((), (record, record)):
        with pytest.raises(ValueError):
            types.TargetEvidence(0, "y", ("x",), records, (), (), {})


def test_layer_result_cannot_have_unjustified_nodes_or_incomplete_status(types):
    provenance = types.FitProvenance(None, "test", 0., None, "test", None, {}, {}, {})
    args = (make_config(), ("y",), (make_sample(),), (), {"y": 0}, {}, {"y":"NO_UNASSIGNED"}, "COMPLETE", provenance)
    result = types.LayeredOPENResult(*args)
    assert result.effective_resamples == 1
    with pytest.raises(ValueError):
        replace(result, assigned_depth={"y":0, "x":1}, parents={"x":("y",)})
    with pytest.raises(ValueError):
        replace(result, status="INCOMPLETE_LAYER")
    with pytest.raises(TypeError):
        result.assigned_depth["x"] = 1
