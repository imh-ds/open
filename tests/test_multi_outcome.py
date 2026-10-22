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


def _confirmation_exploration(outcome):
    from open.explore import ExploreResult

    links = pd.DataFrame([{
        "depth": 1, "target": outcome, "source": "X", "kind": "unique", "tier": "kept",
        "pass_rate": 1.0, "r2_median": 0.2,
    }])
    folds = pd.DataFrame([{
        "depth": 1, "target": outcome, "source": "X", "contrast": "unique", "noise": False,
    }])
    fit = pd.DataFrame([{
        "depth": 1, "target": outcome, "n": 100, "network": ("X",), "network_size": 1,
        "pool_size": 1, "r2_network": 0.2,
    }])
    empty = pd.DataFrame()
    return ExploreResult(empty, links, empty, folds, empty, {},
                         {"outcome": outcome, "engine": "ols", "controls": (), "tuning_folds": 2}, fit)


def _assert_same_exploration(actual, expected):
    assert actual.settings == expected.settings
    assert actual.stops == expected.stops
    for name in ("nodes", "links", "pairs", "folds", "bars", "fit"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))


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
    original_explore = multi._explore_with_shared_context

    def split_spy(frame, *args, **kwargs):
        split_calls.append(frame.copy())
        return original_split(frame, *args, **kwargs)

    def explore_spy(frame, outcome, predictors, **kwargs):
        explore_calls.append((frame.copy(), outcome, tuple(predictors), kwargs["config"]))
        return original_explore(frame, outcome, predictors, **kwargs)

    monkeypatch.setattr(multi, "split_sample", split_spy)
    monkeypatch.setattr(multi, "_explore_with_shared_context", explore_spy)

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
    monkeypatch.setattr(multi, "_explore_with_shared_context", lambda data, outcome, *a, **k: _blank_exploration(outcome))

    def confirm_stub(exploration, data, **kwargs):
        outcome = exploration.settings["outcome"]
        p = raw[outcome]
        return pd.DataFrame({"p": p, "p_adjusted": p, "confirmed": [False] * len(p),
                             "engine": ["ols"] * len(p),
                             "note": [""] * len(p)})

    monkeypatch.setattr(multi, "_confirm_with_shared_folds", confirm_stub)
    monkeypatch.setattr(multi, "_confirmation_fit_with_shared_folds",
                        lambda *a, **k: pd.DataFrame({"r2_network": [0.1]}))

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
    monkeypatch.setattr(multi, "_explore_with_shared_context", lambda data, outcome, *a, **k: _blank_exploration(outcome))
    p_values = {"Y1": [0.02, 0.04], "Y2": []}

    def confirm_stub(exploration, data, **kwargs):
        p = p_values[exploration.settings["outcome"]]
        return pd.DataFrame({"p": p, "p_adjusted": p, "confirmed": [False] * len(p),
                             "engine": ["ols"] * len(p)})

    monkeypatch.setattr(multi, "_confirm_with_shared_folds", confirm_stub)
    monkeypatch.setattr(multi, "_confirmation_fit_with_shared_folds", lambda *a, **k: pd.DataFrame())
    single = multi.explore_and_confirm_outcomes(_data(), ["Y1"], predictors=["X1"])
    np.testing.assert_allclose(single.confirmation.p_adjusted, _adjust(np.array([0.02, 0.04]), "holm"))

    empty = multi.explore_and_confirm_outcomes(_data(), ["Y2"], predictors=["X1"])
    assert empty.confirmation.empty


def test_row_aligned_fold_plans_are_reproducible_and_follow_row_identity():
    from open.confirm import _aligned_confirmation_folds

    rows = np.arange(31, 55)
    plan = _aligned_confirmation_folds(rows, seed=19)
    reversed_plan = _aligned_confirmation_folds(rows[::-1], seed=19)
    repeated_plan = _aligned_confirmation_folds(rows, seed=19)
    for name in ("k10", "half"):
        by_row = dict(zip(rows, plan[name]))
        reversed_by_row = dict(zip(rows[::-1], reversed_plan[name]))
        assert by_row == reversed_by_row
        np.testing.assert_array_equal(plan[name], repeated_plan[name])
        assert set(plan[name]) <= (set(range(10)) if name == "k10" else {0, 1})

    other_rows = np.arange(32, 56)
    other = _aligned_confirmation_folds(other_rows, seed=19)
    other_again = _aligned_confirmation_folds(other_rows, seed=19)
    for name in ("k10", "half"):
        np.testing.assert_array_equal(other[name], other_again[name])
        assert set(other[name]) <= (set(range(10)) if name == "k10" else {0, 1})


def test_multi_outcome_confirmation_and_fit_share_folds_for_same_rows(monkeypatch):
    import open.confirm as confirmation
    import open.explore as exploration
    import open.multi_outcome as multi

    rng = np.random.default_rng(23)
    n = 600
    x = rng.normal(size=n)
    data = pd.DataFrame({"Y1": x + rng.normal(size=n), "Y2": -x + rng.normal(size=n), "X": x})
    data.loc[::9, "X"] = np.nan
    data.index = np.tile(np.arange(20), n // 20)  # batch code must replace caller labels with row positions
    monkeypatch.setattr(multi, "_explore_with_shared_context", lambda frame, outcome, *a, **k: _confirmation_exploration(outcome))

    gcm_calls = []

    def gcm_stub(X, T, folds, dseed, columns=None):
        gcm_calls.append(({key: value.copy() for key, value in folds.items()}, dseed))
        return np.full(len(columns), 0.2)

    monkeypatch.setattr(confirmation, "gcm_pcm_pvalues", gcm_stub)
    fit_calls = []

    def network_fit_stub(data, target, pool, network, fold_ids, *args, **kwargs):
        fit_calls.append(np.asarray(fold_ids).copy())
        return {"network": tuple(network), "network_size": len(network), "pool_size": len(pool),
                "r2_network": 0.3, "r2_pool": 0.4, "share_of_pool": 0.75}

    monkeypatch.setattr(exploration, "network_fit", network_fit_stub)
    with _no_warning_context():
        multi.explore_and_confirm_outcomes(
            data, ["Y1", "Y2"], predictors=["X"], confirm_engine="lightgbm",
        )

    assert len(gcm_calls) == 2
    for key in ("k10", "half"):
        np.testing.assert_array_equal(gcm_calls[0][0][key], gcm_calls[1][0][key])
    assert gcm_calls[0][1] != gcm_calls[1][1]  # model sub-seeds remain target-specific
    assert len(fit_calls) == 2
    np.testing.assert_array_equal(fit_calls[0], fit_calls[1])


def test_batch_reuses_identical_deeper_assessment_and_matches_scalar_results(monkeypatch):
    import open.explore as exploration
    import open.multi_outcome as multi
    from open.confirm import split_sample
    from open.explore import ExploreConfig

    rng = np.random.default_rng(51)
    n = 1200
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    data = pd.DataFrame({
        "Y1": 8 * x1 + 0.05 * rng.normal(size=n),
        "Y2": -6 * x1 + 0.05 * rng.normal(size=n),
        "X1": x1, "X2": x2, "X3": x3,
    })
    config = ExploreConfig(folds=3, tuning_folds=2, max_depth=2, interactions=False, seed=12)
    assessments = []
    original_assess = exploration.assess_target

    def assess_spy(*args, **kwargs):
        assessments.append((args[1], tuple(args[2]), args[6]))
        return original_assess(*args, **kwargs)

    monkeypatch.setattr(exploration, "assess_target", assess_spy)
    with _no_warning_context():
        result = multi.explore_and_confirm_outcomes(data, ["Y1", "Y2"], config=config, split_seed=17)

    batch_assessments = list(assessments)
    shared_target_calls = [call for call in batch_assessments if call == ("X1", ("X2", "X3"), 2)]
    assert len(shared_target_calls) == 1
    ids1 = result.by_outcome["Y1"].exploration.assessment_ids
    ids2 = result.by_outcome["Y2"].exploration.assessment_ids
    assert ids1[(2, "X1")] == ids2[(2, "X1")]
    assert ids1[(1, "Y1")] != ids2[(1, "Y2")]

    explore_part, _ = split_sample(data, seed=17)
    for outcome in ("Y1", "Y2"):
        uncached = exploration.explore(explore_part, outcome, ("X1", "X2", "X3"), config=config)
        assert uncached.assessment_ids == {}
        _assert_same_exploration(result.by_outcome[outcome].exploration, uncached)


def test_batch_does_not_reuse_assessment_when_depth_or_pool_differs(monkeypatch):
    import open.explore as exploration
    import open.multi_outcome as multi
    from open.confirm import split_sample
    from open.explore import ExploreConfig

    data = _data()
    data["Z"] = np.random.default_rng(4).normal(size=len(data))
    data["W"] = np.random.default_rng(5).normal(size=len(data))
    config = ExploreConfig(folds=2, tuning_folds=2, max_depth=3, interactions=False)
    assessments = []
    promote = {("Y1", 1): "X1", ("Y2", 1): "Z", ("Z", 2): "X1"}

    def assess_stub(frame, target, candidates, fold_ids, engine, config, depth, controls=()):
        candidates = tuple(candidates)
        assessments.append((target, candidates, depth))
        selected = promote.get((target, depth))
        links = pd.DataFrame([
            {"depth": depth, "target": target, "source": candidate, "kind": "unique",
             "tier": "kept" if candidate == selected else "dropped", "members": (candidate,)}
            for candidate in candidates
        ])
        return pd.DataFrame(), links, pd.DataFrame()

    def network_fit_stub(frame, target, pool, network, fold_ids, engine, config, depth, controls=()):
        return {"depth": depth, "target": target, "pool_size": len(pool), "network_size": len(network),
                "network": tuple(network), "r2_network": 0.1, "r2_pool": 0.2,
                "share_of_pool": 0.5, "folds": config.folds}

    empty_confirmation = pd.DataFrame({"p": [], "engine": []})
    monkeypatch.setattr(exploration, "assess_target", assess_stub)
    monkeypatch.setattr(exploration, "network_fit", network_fit_stub)
    monkeypatch.setattr(multi, "_confirm_with_shared_folds", lambda *a, **k: empty_confirmation.copy())
    monkeypatch.setattr(multi, "_confirmation_fit_with_shared_folds", lambda *a, **k: pd.DataFrame())

    result = multi.explore_and_confirm_outcomes(
        data, ["Y1", "Y2"], predictors=["X1", "Z", "W"], config=config,
    )
    batch_assessments = list(assessments)
    target_x_calls = [call for call in batch_assessments if call[0] == "X1"]
    assert target_x_calls == [("X1", ("Z", "W"), 2), ("X1", ("W",), 3)]
    assert (result.by_outcome["Y1"].exploration.assessment_ids[(2, "X1")] !=
            result.by_outcome["Y2"].exploration.assessment_ids[(3, "X1")])

    explore_part, _ = split_sample(data.reset_index(drop=True), seed=0)
    for outcome in ("Y1", "Y2"):
        uncached = exploration.explore(explore_part, outcome, ("X1", "Z", "W"), config=config)
        _assert_same_exploration(result.by_outcome[outcome].exploration, uncached)


@pytest.mark.parametrize("role", ["predictors", "controls", "covariates"])
def test_validates_model_columns_before_hashing(role):
    from open.multi_outcome import explore_and_confirm_outcomes

    data = _data()
    data["Bad"] = pd.Series([[1]] * len(data), dtype=object)
    options = {"predictors": ["X1"]}
    options[role] = ["Bad"]
    with pytest.raises(ValueError, match="numeric"):
        explore_and_confirm_outcomes(data, ["Y1", "Y2"], **options)


def test_correction_provenance_survives_independent_inspection():
    from open.api import ExploreConfig, explore_and_confirm_outcomes

    data = _data()
    data["Y2"] += 1.5 * data.X1
    result = explore_and_confirm_outcomes(
        data, ["Y1", "Y2"], config=ExploreConfig(folds=3, max_depth=1, interactions=False),
    )
    assert result.confirmation.attrs == {
        "correction": "holm", "correction_scope": "all_outcomes", "alpha": 0.05,
        "links_tested": int(result.confirmation.p.notna().sum()),
    }
    assert all("all selected outcomes" in item.summary() for item in result.by_outcome.values())


@pytest.mark.parametrize("sample", ["empty", "small", "constant"])
def test_batch_retains_unsupported_confirmation_fits(sample):
    from open.api import ExploreConfig, explore_and_confirm_outcomes
    from open.confirm import split_sample

    data = _data()
    _, heldout = split_sample(data, seed=0)
    if sample == "constant":
        data.loc[heldout.index, "Y1"] = 1.0
    else:
        unavailable = heldout.index if sample == "empty" else heldout.index[5:]
        data.loc[unavailable, "X1"] = np.nan
    with _no_warning_context():
        result = explore_and_confirm_outcomes(
            data, ["Y1", "Y2"], config=ExploreConfig(folds=3, max_depth=1, interactions=False),
        )
    y1 = result.fit[result.fit.root_outcome.eq("Y1")]
    assert len(y1) == 1
    assert y1.r2_network.isna().all() and y1.r2_pool.isna().all()
    assert y1.note.str.contains("not evaluated").all()
    assert "not evaluated" in result.by_outcome["Y1"].summary()
    if sample == "empty":
        assert result.confirmation.p.isna().all()
        assert result.confirmation.n.eq(0).all()


@pytest.mark.parametrize("eligible", [0, 5])
def test_batch_skips_lightgbm_test_without_eligible_rows(monkeypatch, eligible):
    from open.api import ExploreConfig, explore_and_confirm_outcomes
    from open.confirm import split_sample

    data = _data()
    _, heldout = split_sample(data, seed=0)
    data.loc[heldout.index[eligible:], "X1"] = np.nan
    monkeypatch.setattr("open.confirm.gcm_pcm_pvalues",
                        lambda *a, **k: pytest.fail("no GCM/PCM fit is possible without rows"))
    with _no_warning_context():
        result = explore_and_confirm_outcomes(
            data, ["Y1", "Y2"], confirm_engine="lightgbm",
            config=ExploreConfig(folds=3, max_depth=1, interactions=False),
        )
    assert result.confirmation.p.isna().all()
    assert result.confirmation.note.str.contains("no complete|too few complete").all()
