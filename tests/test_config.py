"""Configuration contract for the first supported OPEN path."""

from dataclasses import asdict
from pathlib import Path

import pytest

from open.config import dump_config, load_config, parse_config


DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_documented_defaults() -> None:
    config = load_config(DEFAULT)
    assert config.analysis.max_depth == 3
    assert config.analysis.random_seed == 20261004
    assert config.outcome.type == "continuous"
    assert config.outcome.loss == "mse"
    assert config.learner.mode == "fixed"
    assert config.learner.library == ("elastic_net",)
    assert config.learner.retune_ablation is False
    assert (config.performance_cv.folds, config.performance_cv.repeats) == (5, 2)
    assert config.tuning_cv.folds == 3
    assert asdict(config.thresholds) == {
        "min_total_gain_relative": 0.02,
        "broad_relative": 0.05,
        "unique_relative": 0.05,
        "group_relative": 0.10,
        "dependence_pair": 0.70,
        "stability": 0.70,
    }
    assert (config.stability.fraction, config.stability.n_resamples) == (0.80, 100)
    assert config.stability.final_resamples == 500


@pytest.mark.parametrize("value", [-0.01, 1.01, True, "0.5"])
def test_invalid_threshold(value: object) -> None:
    with pytest.raises(ValueError, match="thresholds.broad_relative"):
        parse_config({"analysis": {"root": "Anxiety", "random_seed": 1},
                      "thresholds": {"broad_relative": value}})


@pytest.mark.parametrize("section,field,value", [
    ("outcome", "type", "binary"),
    ("outcome", "loss", "mae"),
    ("learner", "mode", "per_target"),
    ("learner", "library", ["elastic_net", "random_forest"]),
    ("learner", "retune_ablation", True),
    ("performance_cv", "grouped", True),
    ("dependence", "method", "mixed"),
])
def test_deferred_mode_rejected(section: str, field: str, value: object) -> None:
    with pytest.raises(ValueError, match=section):
        parse_config({"analysis": {"root": "Anxiety", "random_seed": 1},
                      section: {field: value}})


def test_required_root_seed_and_positive_counts() -> None:
    with pytest.raises(ValueError, match="analysis section is required"):
        parse_config({})
    with pytest.raises(ValueError, match="analysis"):
        parse_config({"analysis": {"root": ["Anxiety"], "random_seed": 1}})
    with pytest.raises(ValueError, match="analysis"):
        parse_config({"analysis": {"root": "Anxiety"}})
    with pytest.raises(ValueError, match="performance_cv.folds"):
        parse_config({"analysis": {"root": "Anxiety", "random_seed": 1},
                      "performance_cv": {"folds": 0}})


def test_yaml_round_trip(tmp_path: Path) -> None:
    original = load_config(DEFAULT)
    target = tmp_path / "round-trip.yaml"
    dump_config(original, target)
    assert load_config(target) == original
