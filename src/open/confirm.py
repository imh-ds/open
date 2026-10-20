"""Confirm an exploratory network on independent data.

The links kept by ``open.explore`` (optionally also the flagged ones) are
frozen. On data the exploration never saw, each frozen target is regressed on
the same candidate pool it was explored with, and only the frozen links are
tested. The recursion is not re-run.

Tests use the calibrated settings from the reference studies:

- ``"ols"``: OLS with HC4 standard errors and a t(n - k) reference. Calibrated
  from N >= 275 (Phase A').
- ``"lightgbm"``: the combined cross-fitted GCM/PCM test,
  min(1, 2 * min(p_GCM, p_PCM)), with K = 10 folds and fold-cluster (CR1)
  variance on a t(9) reference ("G2"). LightGBM settings are Phase B's
  configuration F3. Calibrated from N >= 300 (Phase B).

Shared groups and interactions are not tested: no group or interaction test
has been calibrated yet. They are listed with that reason.

The functions below reproduce Phase B's frozen implementation; the identity is
checked in ``tests/test_confirm.py``.
"""
from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from open.explore import ExploreResult

CALIBRATED_FROM = {"ols": 275, "lightgbm": 300}
GCM_FOLDS = 10


# ---------------------------------------------------------------- OLS + HC4

def hc4_pvalues(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-sided HC4 p-values for the slopes of OLS of y on X (intercept added)."""
    A = np.column_stack([np.ones(len(X)), X])
    n, k = A.shape
    inv = np.linalg.inv(A.T @ A)
    C = inv @ A.T
    beta = C @ y
    e = y - A @ beta
    h = np.einsum("ij,ji->i", A, C)
    w = e ** 2 / (1 - h) ** np.minimum(4.0, h / (k / n))
    t = beta / np.sqrt((C ** 2) @ w)
    return 2 * stats.t.sf(np.abs(t[1:]), n - k)


# ---------------------------------------------------------------- LightGBM GCM/PCM (Phase B, F3, G2)

def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(x) for x in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _fit_predict(Xtr, ytr, Xtests, seed):
    """Phase B configuration F3: LightGBM with early stopping on an inner 20% split."""
    import lightgbm as lgb
    n_tr = len(ytr)
    kw = dict(num_leaves=15, learning_rate=0.05, min_child_samples=max(20, math.ceil(0.01 * n_tr)),
              subsample=0.8, subsample_freq=1, reg_lambda=1.0, n_estimators=1000, num_threads=1,
              deterministic=True, force_col_wise=True, verbose=-1, random_state=seed)
    perm = np.random.default_rng(seed).permutation(n_tr)
    n_val = math.ceil(0.2 * n_tr)
    val, fit = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    m = lgb.LGBMRegressor(**kw)
    m.fit(Xtr[fit], ytr[fit], eval_set=[(Xtr[val], ytr[val])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    best = int(m.best_iteration_ or 1000)
    return [m.predict(Z, num_iteration=best) for Z in Xtests]


def _gcm_residuals(X, T, fold, K, dseed, engine="F3"):
    n, p = X.shape
    eT, eX = np.empty((p, n)), np.empty((p, n))
    for j in range(p):
        C = np.delete(X, j, axis=1)
        for k in range(K):
            test = fold == k
            train = ~test
            fT, = _fit_predict(C[train], T[train], [C[test]], _seed(dseed, engine, K, k, "gcmT", j))
            fX, = _fit_predict(C[train], X[train, j], [C[test]], _seed(dseed, engine, K, k, "gcmX", j))
            eT[j, test] = T[test] - fT
            eX[j, test] = X[test, j] - fX
    return eT, eX


def _pcm_products(X, T, half, fold, K, dseed, engine="F3"):
    n, p = X.shape
    out = []
    for h in (0, 1):
        A, B = half == h, half != h
        gA, gB = _fit_predict(X[A], T[A], [X[A], X[B]], _seed(dseed, engine, K, h, "pcm_g", -1))
        fB = fold[B]
        TB = T[B]
        Rh = np.empty((p, int(B.sum())))
        for j in range(p):
            C = np.delete(X, j, axis=1)
            mB, = _fit_predict(C[A], gA, [C[B]], _seed(dseed, engine, K, h, "pcm_m", j))
            hB = gB - mB
            CB = C[B]
            for k in range(K):
                test = fB == k
                train = ~test
                fT, = _fit_predict(CB[train], TB[train], [CB[test]], _seed(dseed, engine, K, h, f"pcmT{k}", j))
                fH, = _fit_predict(CB[train], hB[train], [CB[test]], _seed(dseed, engine, K, h, f"pcmH{k}", j))
                Rh[j, test] = (TB[test] - fT) * (hB[test] - fH)
        out.append((Rh, fB))
    return out


def _stat_cluster(R, fold, K, two_sided=True):
    """Fold-cluster-robust (CR1) t for the mean of R, referred to t(K - 1)."""
    n = R.shape[-1]
    Rbar = R.mean(axis=-1, keepdims=True)
    sums = np.stack([(R - Rbar)[..., fold == k].sum(axis=-1) for k in range(K)], axis=-1)
    V = K / (K - 1) * (sums ** 2).sum(axis=-1) / n ** 2
    t = Rbar[..., 0] / np.sqrt(V)
    return 2 * stats.t.sf(np.abs(t), K - 1) if two_sided else stats.t.sf(t, K - 1)


def _folds(n, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for key, k in (("k10", GCM_FOLDS), ("half", 2)):
        perm = rng.permutation(n)
        f = np.empty(n, dtype=np.int64)
        f[perm] = np.arange(n) % k
        out[key] = f
    return out


def gcm_pcm_pvalues(X: np.ndarray, T: np.ndarray, folds: dict, dseed: int) -> np.ndarray:
    """Combined GCM/PCM p-value per column of X (Phase B, F3 with G2)."""
    eT, eX = _gcm_residuals(X, T, folds["k10"], GCM_FOLDS, dseed)
    p_gcm = _stat_cluster(eT * eX, folds["k10"], GCM_FOLDS)
    parts = _pcm_products(X, T, folds["half"], folds["k10"], GCM_FOLDS, dseed)
    ps = [_stat_cluster(Rh, fB, GCM_FOLDS, two_sided=False) for Rh, fB in parts]
    p_pcm = np.minimum(1.0, 2 * np.minimum(ps[0], ps[1]))
    return np.minimum(1.0, 2 * np.minimum(p_gcm, p_pcm))


# ---------------------------------------------------------------- confirmation

def frozen_network(result: ExploreResult, include_flagged: bool = False) -> pd.DataFrame:
    """The links to confirm, each with the candidate pool its target was explored with."""
    tiers = ["kept", "flagged"] if include_flagged else ["kept"]
    links = result.links[result.links.tier.isin(tiers)].copy()
    f = result.folds
    pools = (f[(f.contrast == "unique") & ~f.noise].groupby(["depth", "target"]).source
             .apply(lambda s: tuple(dict.fromkeys(s))).to_dict())
    links["pool"] = [pools[(d, t)] for d, t in zip(links.depth, links.target)]
    return links.reset_index(drop=True)


def _adjust(p: np.ndarray, method: str) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[i]))
            adj[i] = running
    elif method == "bh":
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, p[i] * m / (rank + 1))
            adj[i] = min(1.0, running)
    else:
        raise ValueError("correction must be 'holm' or 'bh'")
    return adj


def confirm(result: ExploreResult, data: pd.DataFrame, engine: str | None = None,
            include_flagged: bool = False, correction: str = "holm", alpha: float = 0.05,
            seed: int = 0) -> pd.DataFrame:
    """Test the frozen links of ``result`` on independent ``data``.

    ``engine`` defaults to LightGBM for a LightGBM network and to OLS otherwise
    (an Elastic-Net network is confirmed with OLS + HC4). Rows with a missing
    value in a target or its pool are dropped for that target. Returns one row
    per frozen link, with the raw and adjusted p-value and whether it is confirmed.
    """
    if engine is None:
        engine = "lightgbm" if result.settings.get("engine") == "lightgbm" else "ols"
    if engine not in CALIBRATED_FROM:
        raise ValueError(f"confirmation engine must be one of {tuple(CALIBRATED_FROM)}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    links = frozen_network(result, include_flagged)
    rows = []
    for (depth, target), group in links.groupby(["depth", "target"], sort=False):
        pool = group.pool.iloc[0]
        missing = [c for c in (target, *pool) if c not in data]
        if missing:
            raise ValueError(f"confirmation data lacks columns {missing}")
        frame = data[[target, *pool]].apply(pd.to_numeric, errors="raise").dropna()
        n = len(frame)
        X, T = frame[list(pool)].to_numpy(float), frame[target].to_numpy(float)
        note = ""
        if n < CALIBRATED_FROM[engine]:
            note = f"N = {n} is below the calibrated minimum {CALIBRATED_FROM[engine]}"
            warnings.warn(f"{target}: {note}", stacklevel=2)
        testable = group.kind == "unique"
        if testable.any():
            if engine == "ols":
                p_all = hc4_pvalues(X, T)
            else:
                dseed = _seed(seed, "confirm", depth, target)
                p_all = gcm_pcm_pvalues(X, T, _folds(n, dseed), dseed)
            p_of = dict(zip(pool, p_all))
        for _, r in group.iterrows():
            row = dict(depth=depth, target=target, source=r.source, kind=r.kind, explore_tier=r.tier,
                       explore_pass_rate=r.pass_rate, explore_r2=r.r2_median, engine=engine, n=n,
                       pool_size=len(pool), p=np.nan, note=note)
            if r.kind == "unique":
                row["p"] = float(p_of[r.source])
            else:
                row["note"] = f"not tested: no calibrated {r.kind} test"
            rows.append(row)
    out = pd.DataFrame(rows, columns=["depth", "target", "source", "kind", "explore_tier", "explore_pass_rate",
                                      "explore_r2", "engine", "n", "pool_size", "p", "note"])
    tested = out.p.notna()
    out["p_adjusted"] = np.nan
    if tested.any():
        out.loc[tested, "p_adjusted"] = _adjust(out.loc[tested, "p"].to_numpy(), correction)
    out["confirmed"] = out.p_adjusted < alpha
    out.attrs.update(correction=correction, alpha=alpha, links_tested=int(tested.sum()))
    return out
