"""A regression-style table of the effects in a discovered network.

``effects_table`` puts one row on every link (and every declared block) with its predictive effect, the
evidence behind it and, when confirmation is supplied, the confirmation result, plus each target's R². It
holds the same numbers the graph can leave out (``GraphStyle(show_r2=False, edge_labels=None)``).

It reports what the procedure measures: held-out R² lost when a predictor is removed (ΔR²), how many folds
passed the noise bar, and the confirmation p-value. It does not report coefficients, standard errors or
direction, which are not saved yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EFFECT_COLUMNS = [
    "outcome", "target", "depth", "predictor", "kind", "block", "tier",
    "delta_r2", "delta_r2_min", "delta_r2_max", "noise_bar", "above_noise", "folds_passed", "pass_rate",
    "p", "p_adjusted", "confirmed", "confirmation_note",
    "target_r2_discovery", "target_r2_pool", "target_r2_confirmation",
]


def effects_table(result, confirmation: pd.DataFrame | None = None, confirmation_fit: pd.DataFrame | None = None,
                  *, include: str = "drawn") -> pd.DataFrame:
    """One row per link, sorted by depth, target and predictor.

    ``result`` may be a discovery result, a split result (its confirmation and fit are used) or a
    multi-outcome result (one block of rows per outcome). ``include`` is ``"drawn"`` (kept and flagged links,
    the default), ``"kept"``, or ``"all"`` (also dropped and hidden relations, and the members' own scores in
    declared blocks).

    Columns: ``delta_r2`` is the median held-out R² lost when the predictor is removed (gained, for an
    interaction), with its range over folds; ``noise_bar`` the bar it must beat; ``folds_passed`` how often it
    did. ``p``, ``p_adjusted`` and ``confirmed`` come from the confirmation test where one exists (unique
    links only; blocks, groups and interactions have none, and ``confirmation_note`` says so). The
    ``target_r2_*`` columns are the target's cross-validated R² from its kept predictors and from its whole pool,
    and on the confirmation sample.
    """
    if include not in ("drawn", "kept", "all"):
        raise ValueError('include must be "drawn", "kept" or "all"')
    by_outcome = getattr(result, "by_outcome", None)
    if by_outcome is not None:                                  # multi-outcome result
        parts = [effects_table(r, include=include) for r in by_outcome.values()]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EFFECT_COLUMNS)
    if hasattr(result, "exploration"):                          # split result
        confirmation = result.confirmation if confirmation is None else confirmation
        confirmation_fit = result.fit if confirmation_fit is None else confirmation_fit
        result = result.exploration

    links = result.links
    if not len(links):
        return pd.DataFrame(columns=EFFECT_COLUMNS)
    if include == "kept":
        links = links[links.tier == "kept"]
    elif include == "drawn":
        links = links[links.tier.isin(["kept", "flagged"])]
    settings = result.settings
    n_folds = int(settings.get("folds", 10)) * int(settings.get("repeats", 1))

    rows = pd.DataFrame({
        "outcome": settings.get("outcome"),
        "target": links.target.to_numpy(),
        "depth": links.depth.to_numpy(),
        "predictor": links.source.to_numpy(),
        "kind": np.where(links.get("origin", pd.Series("", index=links.index)).eq("declared").to_numpy(),
                         "block", links.kind.to_numpy()),
        "block": links["block"].to_numpy() if "block" in links else None,
        "tier": links.tier.to_numpy(),
        "delta_r2": links.r2_median.to_numpy(),
        "delta_r2_min": links.r2_min.to_numpy(),
        "delta_r2_max": links.r2_max.to_numpy(),
        "noise_bar": links.noise_bar.to_numpy(),
        "above_noise": links.r2_above_noise.to_numpy(),
        "folds_passed": [f"{int(round(p * n_folds))}/{n_folds}" for p in links.pass_rate],
        "pass_rate": links.pass_rate.to_numpy(),
    })

    for column in ("p", "p_adjusted", "confirmed", "confirmation_note"):
        rows[column] = np.nan if column != "confirmation_note" else ""
    if confirmation is not None and len(confirmation):
        key = ["depth", "target", "source"]
        merged = rows.merge(
            confirmation.rename(columns={"source": "predictor", "note": "confirmation_note"})[
                ["depth", "target", "predictor", "p", "p_adjusted", "confirmed", "confirmation_note"]],
            on=["depth", "target", "predictor"], how="left", suffixes=("_x", ""))
        for column in ("p", "p_adjusted", "confirmed", "confirmation_note"):
            rows[column] = merged[column].to_numpy()
        rows["confirmation_note"] = rows["confirmation_note"].fillna("")
    elif confirmation is None:
        rows["confirmation_note"] = ""
    rows["confirmed"] = rows["confirmed"].astype(object).where(rows["p"].notna(), np.nan)   # untested is not "no"
    rows.loc[rows.kind.isin(["block", "shared", "interaction"]) & rows.confirmation_note.eq(""),
             "confirmation_note"] = "not tested: no calibrated group test"

    fit = result.fit
    discovery = {(int(r.depth), r.target): (r.r2_network, r.r2_pool) for r in fit.itertuples()}
    confirmed_fit = ({(int(r.depth), r.target): r.r2_network for r in confirmation_fit.itertuples()}
                     if confirmation_fit is not None and len(confirmation_fit) else {})
    rows["target_r2_discovery"] = [discovery.get((int(d), t), (np.nan, np.nan))[0]
                                   for d, t in zip(rows.depth, rows.target)]
    rows["target_r2_pool"] = [discovery.get((int(d), t), (np.nan, np.nan))[1]
                              for d, t in zip(rows.depth, rows.target)]
    rows["target_r2_confirmation"] = [confirmed_fit.get((int(d), t), np.nan)
                                      for d, t in zip(rows.depth, rows.target)]
    order = rows.sort_values(["depth", "target", "predictor"], kind="stable").reset_index(drop=True)
    return order[EFFECT_COLUMNS]
