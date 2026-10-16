import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"examples"))

@pytest.mark.parametrize("engine", ["gbm", "random_forest"])
@pytest.mark.parametrize("edges,depths,extra,exact", [
    ([("X2", "Y"), ("X1", "X2")], {"Y": 0, "X2": 1, "X1": 2}, False, True),
    ([("X2", "Y")], {"Y": 0, "X2": 1}, False, False),
    ([("X2", "Y"), ("Z1", "Y")], {"Y": 0, "X2": 1, "Z1": 1}, True, False),
    ([("X2", "Y"), ("X1", "X2")], {"Y": 0, "X2": 1, "X1": 1}, False, False),
])
def test_chain_audit_scores_actual_network_and_depths(tmp_path, monkeypatch, edges, depths, extra, exact, engine):
    import tree_recursive_signal_audit as audit
    saved = {"status": "complete", "fit_cost_unknown": False, "structure": None,
             "edges": edges, "groups": [], "assigned_depth": depths}
    monkeypatch.setattr(audit.legacy, "audit_case", lambda *a, **k: copy.deepcopy(saved))
    row = audit.audit_case({"id": "case", "engine": engine, "scenario": "chain", "n": 80},
                          SimpleNamespace(location=tmp_path),
                          {"schema": audit.SCHEMA, "grid_policy": audit.POLICY, "engines": [engine],
                           "grids": {"random_forest": audit.rf_singleton.NOMINAL_GRID}})
    assert row["false_network"] is extra
    assert row["exact_network"] is exact


def test_signal_gate_requires_50_exact_and_at_most_one_extra_and_no_unknowns():
    import qualify_tree_recursive_signal as gate
    expected = [dict(case_id=f"case-{r}", engine="gbm", sample_size=80, repetition=r,
                     manifest_sha256="a"*64, source_sha="b"*40, study_id="signal", attempt=1)
                for r in range(1, 56)]
    rows = [dict(e, status="complete", fit_cost_unknown=False, archive_sha256="c"*64,
                 false_network=False, exact_network=i < 50) for i, e in enumerate(expected)]
    assert gate.summarize_cell(expected, rows)["passed"]
    rows[49]["exact_network"] = False
    assert not gate.summarize_cell(expected, rows)["passed"]
    rows[49]["exact_network"] = True
    rows[0]["false_network"] = True
    rows[0]["exact_network"] = False
    rows[50]["exact_network"] = True
    assert gate.summarize_cell(expected, rows)["passed"]
    rows[1]["false_network"] = True
    rows[1]["exact_network"] = False
    rows[51]["exact_network"] = True
    assert not gate.summarize_cell(expected, rows)["passed"]
    rows[1]["false_network"] = False
    assert not gate.summarize_cell(expected, rows[:-1])["passed"]
