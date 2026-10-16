import sys
import hashlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))


def expected(engine="gbm", sample_size=80, count=55):
    return [
        {
            "case_id": f"{engine}-n{sample_size}-r{i:02}",
            "engine": engine,
            "sample_size": sample_size,
            "repetition": i,
            "manifest_sha256": "a" * 64,
            "study_id": "tree-null-study",
            "source_sha": "b" * 40,
            "attempt": 1,
        }
        for i in range(1, count + 1)
    ]


def audited(expected_rows, incorrect=0):
    return [
        {
            **row,
            "status": "complete",
            "fit_cost_unknown": False,
            "false_network": i < incorrect,
            "archive_sha256": hashlib.sha256(f"archive-{i}".encode()).hexdigest(),
        }
        for i, row in enumerate(expected_rows)
    ]


@pytest.mark.parametrize(
    "incorrect,passes,upper",
    [(0, True, 0.0649), (1, True, 0.0972), (2, False, 0.1253)],
)
def test_exact_per_cell_false_network_gate(incorrect, passes, upper):
    import qualify_tree_recursive_validation as qualify

    plan = expected()
    cell = qualify.summarize_cell(plan, audited(plan, incorrect))
    assert cell["planned"] == 55
    assert cell["complete"] == 55
    assert cell["unknown"] == 0
    assert cell["incorrect"] == incorrect
    assert cell["incorrect_interval"]["upper"] == pytest.approx(upper, abs=0.0001)
    assert cell["passed"] is passes


def test_unknown_case_stays_in_denominator_and_blocks_cell():
    import qualify_tree_recursive_validation as qualify

    plan = expected()
    rows = audited(plan)
    rows[-1]["status"] = "incomplete"
    rows[-1]["fit_cost_unknown"] = True
    rows[-1]["false_network"] = None
    cell = qualify.summarize_cell(plan, rows[:-1] + [rows[-1]])
    assert cell["planned"] == 55
    assert cell["complete"] == 54
    assert cell["unknown"] == 1
    assert cell["passed"] is False


@pytest.mark.parametrize("problem", ["duplicate", "foreign", "provenance"])
def test_identity_or_provenance_errors_are_rejected(problem):
    import qualify_tree_recursive_validation as qualify

    plan = expected()
    rows = audited(plan)
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "foreign":
        rows[0]["case_id"] = "foreign-case"
    else:
        rows[0]["source_sha"] = "wrong-source"
    with pytest.raises(ValueError):
        qualify.summarize_cell(plan, rows)


@pytest.mark.parametrize("field,value", [("attempt", True), ("repetition", True),
                                         ("manifest_sha256", "not-a-sha"),
                                         ("source_sha", "not-a-git-sha")])
def test_boolean_or_malformed_frozen_identity_is_rejected(field, value):
    import qualify_tree_recursive_validation as qualify

    plan = expected()
    for row in plan:
        row[field] = value
    with pytest.raises(ValueError):
        qualify.summarize_cell(plan, audited(plan))


def test_study_gate_requires_both_waves_and_bound_provenance():
    import qualify_tree_recursive_validation as qualify

    def wave(sample_size, *, source="b" * 40):
        cases = []
        for engine in ("random_forest", "gbm", "lightgbm"):
            plan = expected(engine, sample_size)
            cases.append(qualify.summarize_cell(plan, audited(plan)))
        return {
            "sample_size": sample_size,
            "manifest_sha256": "a" * 64,
            "study_id": "tree-null-study",
            "source_sha": source,
            "run_id": "123456",
            "wave_id": f"n{sample_size}",
            "attempt": 1,
            "transport_verified": True,
            "cells": cases,
        }

    result = qualify.qualify_study(wave(80), wave(200))
    assert result["engine_eligibility"] == {
        "random_forest": True,
        "gbm": True,
        "lightgbm": True,
    }
    with pytest.raises(ValueError, match="source"):
        qualify.qualify_study(wave(80), wave(200, source="c" * 40))
