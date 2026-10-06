import pytest

from open.metadata import validate_metadata


def records(names=None, **overrides):
    base = {
        "y": {"role": "root", "type": "continuous", "display_label": "Outcome", "scale_items": ["i1", "i2"]},
        "x": {"role": "candidate", "type": "continuous", "display_label": "Predictor X"},
        "rank": {"role": "candidate", "type": "ordinal"},
        "sid": {"role": "id", "type": "continuous"},
        "fold": {"role": "exclude", "type": "categorical", "exclude_reason": "fold label"},
        "item_total": {"role": "candidate", "type": "continuous", "scale_items": ["i1", "i2"]},
        "other_total": {"role": "candidate", "type": "continuous", "scale_items": ["i2", "i3"]},
    }
    base.update(overrides)
    names = list(base) if names is None else names
    return [{"name": name, **base[name]} for name in names]


def test_selects_supported_predictors_and_marks_target_item_overlap():
    result = validate_metadata(
        ["y", "x", "rank", "sid", "fold", "item_total", "other_total"],
        records(["y", "x", "rank", "sid", "fold", "item_total", "other_total"]), "y"
    )
    assert result.root == "y"
    assert result.candidates == ("x", "rank", "item_total", "other_total")
    flagged = next(row for row in result.variables if row["name"] == "item_total")
    assert flagged["measurement_overlap"] is True
    root = next(row for row in result.variables if row["name"] == "y")
    other = next(row for row in result.variables if row["name"] == "other_total")
    assert root["measurement_overlap"] is True
    assert other["measurement_overlap"] is True
    assert next(row for row in result.variables if row["name"] == "x")["display_label"] == "Predictor X"


def test_duplicate_columns_fail_with_useful_error():
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_metadata(["y", "x", "x"], records(["y", "x"]), "y")


def test_missing_root_fails_with_useful_error():
    with pytest.raises(ValueError, match="not present"):
        validate_metadata(["x"], records(["x"]), "y")


def test_target_item_overlap_is_excluded():
    data = records(["y", "i1"], i1={"role": "candidate", "type": "continuous"})
    result = validate_metadata(["y", "i1"], data, "y")
    assert result.candidates == ()


def test_predictor_to_predictor_item_overlap_flags_both_without_root_overlap():
    data = [
        {"name": "y", "role": "root", "type": "continuous", "scale_items": ["root_item"]},
        {"name": "x1", "role": "candidate", "type": "continuous", "scale_items": ["shared"]},
        {"name": "x2", "role": "candidate", "type": "ordinal", "scale_items": ["shared", "other"]},
    ]
    result = validate_metadata(["y", "x1", "x2"], data, "y")
    overlap = {row["name"]: row["measurement_overlap"] for row in result.variables}
    assert overlap["x1"] is True
    assert overlap["x2"] is True
    assert overlap["y"] is False


def test_item_level_mode_is_rejected():
    with pytest.raises(ValueError, match="item-level analysis is unsupported"):
        validate_metadata(["y", "x"], records(["y", "x"]), "y", mode="item_level")


def test_post_outcome_predictor_is_rejected_in_prospective_mode():
    data = records(["y", "x"], x={"role": "candidate", "type": "continuous", "time_index": 3})
    data[0]["time_index"] = 2
    with pytest.raises(ValueError, match="post-outcome"):
        validate_metadata(["y", "x"], data, "y", prospective=True)


def test_target_derivatives_and_explicit_excludes_are_omitted():
    data = records(["y", "x", "rank"],
        x={"role": "candidate", "type": "continuous", "derived_from": "y"},
        rank={"role": "candidate", "type": "ordinal", "exclude_reason": "prespecified"},
    )
    result = validate_metadata(["y", "x", "rank"], data, "y")
    assert result.candidates == ()


def test_unsupported_candidate_type_fails_explicitly():
    data = records(["y", "binary"], binary={"role": "candidate", "type": "binary"})
    with pytest.raises(ValueError, match="unsupported type 'binary'"):
        validate_metadata(["y", "binary"], data, "y")

