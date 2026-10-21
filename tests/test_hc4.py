"""HC4 rejects unidentifiable designs and is invariant to measurement units."""

import numpy as np
import pytest

from open.confirm import hc4_pvalues


@pytest.mark.parametrize("case,reason", [
    ("constant", "rank"),
    ("duplicate", "rank"),
    ("dummy_set", "rank"),
    ("empty", "no.*rows"),
    ("saturated", "degrees of freedom"),
    ("underdetermined", "degrees of freedom"),
    ("nan_predictor", "finite"),
    ("infinite_predictor", "finite"),
    ("infinite_outcome", "finite"),
    ("unit_leverage", "leverage"),
])
def test_hc4_rejects_untestable_designs(case, reason):
    r = np.random.default_rng(31)
    X = r.normal(size=(300, 2))
    y = X[:, 0] + r.normal(size=300)
    if case == "constant":
        X = np.column_stack([X, np.ones(300)])
    elif case == "duplicate":
        X = np.column_stack([X, X[:, 0]])
    elif case == "dummy_set":
        X = np.eye(3)[np.arange(300) % 3]
    elif case == "empty":
        X, y = X[:0], y[:0]
    elif case == "saturated":
        X, y = X[:3], y[:3]
    elif case == "underdetermined":
        X, y = X[:2], y[:2]
    elif case == "nan_predictor":
        X[0, 0] = np.nan
    elif case == "infinite_predictor":
        X[0, 0] = np.inf
    elif case == "infinite_outcome":
        y[0] = np.inf
    elif case == "unit_leverage":
        X = np.column_stack([X, np.arange(300) == 0])
    with pytest.raises(ValueError, match=reason):
        hc4_pvalues(X, y)


@pytest.mark.parametrize("case", ["one_dimensional_X", "two_dimensional_y", "row_mismatch"])
def test_hc4_rejects_malformed_array_shapes(case):
    X = np.arange(30.).reshape(10, 3)
    y = np.arange(10.)
    if case == "one_dimensional_X":
        X = X[:, 0]
    elif case == "two_dimensional_y":
        y = y[:, None]
    else:
        y = y[:-1]
    with pytest.raises(ValueError, match="shape|dimensional|rows"):
        hc4_pvalues(X, y)


@pytest.mark.parametrize("scales", [(1e-150, 1e150), (1e12, 1e-12)])
def test_hc4_is_invariant_to_predictor_units(scales):
    r = np.random.default_rng(14)
    X = r.normal(size=(300, 2))
    y = .2 * X[:, 0] + r.normal(size=300) * (1 + X[:, 1] ** 2)
    reference = hc4_pvalues(X, y)
    changed_units = hc4_pvalues(X * scales, y)
    assert np.isfinite(changed_units).all()
    np.testing.assert_allclose(changed_units, reference, rtol=1e-10, atol=1e-12)


def test_hc4_is_invariant_to_predictor_origins():
    r = np.random.default_rng(14)
    X = r.normal(size=(300, 2))
    y = .2 * X[:, 0] + r.normal(size=300) * (1 + X[:, 1] ** 2)
    np.testing.assert_allclose(hc4_pvalues(X + [1e6, -1e6], y), hc4_pvalues(X, y),
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_hc4_is_invariant_to_outcome_units(scale):
    r = np.random.default_rng(14)
    X = r.normal(size=(300, 2))
    y = .2 * X[:, 0] + r.normal(size=300) * (1 + X[:, 1] ** 2)
    reference = hc4_pvalues(X, y)
    changed_units = hc4_pvalues(X, y * scale)
    assert np.isfinite(changed_units).all()
    np.testing.assert_allclose(changed_units, reference, rtol=1e-10, atol=1e-12)
