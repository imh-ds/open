"""Coordinated multi-outcome exploration and confirmation workflow."""
import importlib

import numpy as np
import pandas as pd
import pytest


def _data(n=1200, seed=41):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame({
        "Y1": 0.8 * x1 + rng.normal(size=n),
        "Y2": 0.7 * x2 + rng.normal(size=n),
        "X1": x1,
        "X2": x2,
        "X3": rng.normal(size=n),
    })


def _blank_exploration(outcome):
    from open.explore import ExploreResult

    empty = pd.DataFrame()
    return ExploreResult(empty, empty, empty, empty, empty, {},
                         {"outcome": outcome, "engine": "ols", "controls": ()}, empty)


def test_public_api_exposes_multi_outcome_workflow():
    api = importlib.import_module("open.api")
    module = importlib.import_module("open.multi_outcome")

    assert api.explore_and_confirm_outcomes is module.explore_and_confirm_outcomes
    assert api.MultiOutcomeResult is module.MultiOutcomeResult
    assert "explore_and_confirm_outcomes" in api.__all__
    assert "MultiOutcomeResult" in api.__all__


def test_two_outcomes_return_separate_split_results_and_aggregate_tables():
    from open.api import ExploreConfig, SplitResult, explore_and_confirm_outcomes
    from open.multi_outcome import MultiOutcomeResult

    data = _data()
    with _no_warning_context():
        result = explore_and_confirm_outcomes(
            data, ["Y1", "Y2"], config=ExploreConfig(folds=3, tuning_folds=2,
                                                      max_depth=1, interactions=False),
            split_seed=7,
        )

    assert isinstance(result, MultiOutcomeResult)
    assert list(result.by_outcome) == ["Y1", "Y2"]
    assert all(isinstance(item, SplitResult) for item in result.by_outcome.values())
    assert set(result.confirmation.root_outcome) <= {"Y1", "Y2"}
    assert set(result.fit.root_outcome) <= {"Y1", "Y2"}
    assert result.settings["outcomes"] == ("Y1", "Y2")
    assert result.settings["correction_scope"] == "all_outcomes"
    assert "Y1" in result.summary() and "Y2" in result.summary()


class _no_warning_context:
    """A context manager that keeps expected low-N calibration warnings local."""

    def __enter__(self):
        import warnings
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore", UserWarning)

    def __exit__(self, *exc):
        return self._warnings.__exit__(*exc)


@pytest.mark.parametrize(
    "outcomes, message",
    [([], "at least one"), (["Y1", "Y1"], "distinct"), (["absent"], "unknown")],
)
def test_rejects_empty_repeated_or_unknown_outcomes(outcomes, message):
    from open.multi_outcome import explore_and_confirm_outcomes

    with pytest.raises(ValueError, match=message):
        explore_and_confirm_outcomes(_data(), outcomes)


def test_rejects_missing_outcome_values_before_splitting(monkeypatch):
    import open.multi_outcome as multi

    def unexpected_split(*args, **kwargs):
        pytest.fail("validation should happen before splitting")

    monkeypatch.setattr(multi, "split_sample", unexpected_split)
    data = _data()
    data.loc[3, "Y2"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        multi.explore_and_confirm_outcomes(data, ["Y1", "Y2"])


def test_rejects_selected_outcome_in_explicit_predictors():
    from open.multi_outcome import explore_and_confirm_outcomes

    with pytest.raises(ValueError, match="outcomes.*predictors|predictors.*outcomes"):
        explore_and_confirm_outcomes(_data(), ["Y1", "Y2"], predictors=["X1", "Y2"])


def test_shared_split_config_fold_ids_and_predictor_exclusion(monkeypatch):
    import open.multi_outcome as multi
    from open.api import ExploreConfig

    data = _data()
    data.index = np.tile(np.arange(100), 12)  # duplicate caller indexes must not define row identity
    config = ExploreConfig(folds=3, tuning_folds=2, max_depth=1, interactions=False, seed=9)
    split_calls = []
    explore_calls = []
    original_split = multi.split_sample
    original_explore = multi.explore

    def split_spy(frame, *args, **kwargs):
        split_calls.append(frame.copy())
        return original_split(frame, *args, **kwargs)

    def explore_spy(frame, outcome, predictors, **kwargs):
        explore_calls.append((frame.copy(), outcome, tuple(predictors), kwargs["config"]))
        return original_explore(frame, outcome, predictors, **kwargs)

    monkeypatch.setattr(multi, "split_sample", split_spy)
    monkeypatch.setattr(multi, "explore", explore_spy)

    with _no_warning_context():
        result = multi.explore_and_confirm_outcomes(data, ["Y1", "Y2"], config=config, split_seed=9)

    assert len(split_calls) == 1
    assert split_calls[0].index.equals(pd.RangeIndex(len(data)))
    assert len(explore_calls) == 2
    assert all(call[2] == ("X1", "X2", "X3") for call in explore_calls)
    assert all(call[3] is config for call in explore_calls)
    assert explore_calls[0][0].index.equals(explore_calls[1][0].index)
    first_folds = result.by_outcome["Y1"].exploration.folds
    second_folds = result.by_outcome["Y2"].exploration.folds
    assert first_folds.fold.tolist() == second_folds.fold.tolist()


@pytest.mark.parametrize("scope", ["all_outcomes", "per_outcome"])
@pytest.mark.parametrize("correction", ["holm", "bh"])
def test_applies_requested_correction_to_raw_testable_pvalues(monkeypatch, scope, correction):
    import open.multi_outcome as multi
    from open.confirm import SplitResult, _adjust

    raw = {"Y1": [0.01, 0.01, np.nan], "Y2": [0.02, 0.05]}
    monkeypatch.setattr(multi, "split_sample", lambda data, *a, **k: (data, data))
    monkeypatch.setattr(multi, "explore", lambda data, outcome, *a, **k: _blank_exploration(outcome))

    def confirm_stub(exploration, data, **kwargs):
        outcome = exploration.settings["outcome"]
        p = raw[outcome]
        return pd.DataFrame({"p": p, "p_adjusted": p, "confirmed": [False] * len(p),
                             "engine": ["ols"] * len(p),
                             "note": [""] * len(p)})

    monkeypatch.setattr(multi, "confirm", confirm_stub)
    monkeypatch.setattr(multi, "confirmation_fit", lambda *a, **k: pd.DataFrame({"r2_network": [0.1]}))

    result = multi.explore_and_confirm_outcomes(
        _data(), ["Y1", "Y2"], predictors=["X1"], correction=correction, correction_scope=scope,
    )
    all_raw = np.array([0.01, 0.01, 0.02, 0.05])
    expected_global = _adjust(all_raw, correction)
    expected_by_outcome = {
        "Y1": _adjust(np.array([0.01, 0.01]), correction),
        "Y2": _adjust(np.array([0.02, 0.05]), correction),
    }
    offset = 0
    for outcome in ("Y1", "Y2"):
        item = result.by_outcome[outcome]
        assert isinstance(item, SplitResult)
        assert item.confirmation.root_outcome.eq(outcome).all()
        tested = item.confirmation.p.notna()
        count = int(tested.sum())
        expected = (expected_global[offset:offset + count] if scope == "all_outcomes"
                    else expected_by_outcome[outcome])
        np.testing.assert_allclose(item.confirmation.loc[tested, "p_adjusted"], expected)
        aggregate_tested = result.confirmation.root_outcome.eq(outcome) & result.confirmation.p.notna()
        np.testing.assert_allclose(result.confirmation.loc[aggregate_tested, "p_adjusted"], expected)
        assert item.confirmation.loc[~tested, "p_adjusted"].isna().all()
        offset += count
    assert result.settings["correction_scope"] == scope
    assert result.settings["correction"] == correction


def test_correction_handles_empty_and_single_outcome_families(monkeypatch):
    import open.multi_outcome as multi
    from open.confirm import _adjust

    monkeypatch.setattr(multi, "split_sample", lambda data, *a, **k: (data, data))
    monkeypatch.setattr(multi, "explore", lambda data, outcome, *a, **k: _blank_exploration(outcome))
    p_values = {"Y1": [0.02, 0.04], "Y2": []}

    def confirm_stub(exploration, data, **kwargs):
        p = p_values[exploration.settings["outcome"]]
        return pd.DataFrame({"p": p, "p_adjusted": p, "confirmed": [False] * len(p),
                             "engine": ["ols"] * len(p)})

    monkeypatch.setattr(multi, "confirm", confirm_stub)
    monkeypatch.setattr(multi, "confirmation_fit", lambda *a, **k: pd.DataFrame())
    single = multi.explore_and_confirm_outcomes(_data(), ["Y1"], predictors=["X1"])
    np.testing.assert_allclose(single.confirmation.p_adjusted, _adjust(np.array([0.02, 0.04]), "holm"))

    empty = multi.explore_and_confirm_outcomes(_data(), ["Y2"], predictors=["X1"])
    assert empty.confirmation.empty

