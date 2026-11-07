"""A regression-style table of the effects in a discovered network.

``effects_table`` puts one row on every link (and every declared block) with its predictive effect, the
evidence behind it and, when confirmation is supplied, the confirmation result, plus each target's R². It
holds the same numbers the graph can leave out (``GraphStyle(show_r2=False, edge_labels=None)``).

It reports what the procedure measures: held-out R² lost when a predictor is removed (ΔR²), how many folds
passed the noise bar, and the confirmation p-value. For linear learners (OLS, Elastic-Net) it also reports the
signed effect, from two different sources that are kept apart:

* ``estimate`` / ``estimate_std`` / ``direction`` come from the discovery sample: the coefficient of the predictor
  in the target's full-pool model, as the median over the training-fold fits (per original unit, and per SD of the
  predictor). ``direction`` is "positive" or "negative" when every fit has that sign, otherwise "mixed" (or
  "zero"); ``direction_agreement`` is the share of fits with the majority sign. It describes the fitted model;
  it was not tested, and Elastic-Net values are penalized. ``direction_basis`` is "coefficient" for these.
* For learners without coefficients (LightGBM) ``direction_basis`` is "ale": the sign comes from the average
  local effect (ALE) profile of the fitted model, judged against the model's own permuted noise columns.
  ``estimate`` is empty and ``profile_net`` is the median net change of the profile over the predictor's range
  (outcome units). "mixed" there includes a profile that rises and falls.
* ``slope`` and its interval come from the confirmation sample: the frozen full-pool OLS slope with its HC4
  standard error, for links the confirmation tested with OLS. The interval is a 95% pointwise t interval; the
  adjusted p-value is not a simultaneous interval.
* Standardization. ``estimate_std`` scales the predictor only (outcome units per SD of the predictor) and is NOT a
  standardized beta. ``estimate_beta`` (discovery) and ``beta`` (confirmation) scale both: B x SD(predictor) /
  SD(outcome). Discovery uses the SD of the target in each training fold and takes the median over fits;
  confirmation uses the sample SDs (n - 1) of the same complete rows as the slope. ``beta_se`` and the interval
  bounds are the slope's, multiplied by the same factor, so beta has the same p-value as its slope. All are empty
  where an SD is zero.

Blocks, groups and interactions have no single direction, and learners without coefficients (LightGBM) have none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EFFECT_COLUMNS = [
    "outcome", "target", "depth", "predictor", "kind", "block", "tier",
    "delta_r2", "delta_r2_min", "delta_r2_max", "noise_bar", "above_noise", "folds_passed", "pass_rate",
    "p", "p_adjusted", "confirmed", "confirmation_note",
    "estimate", "estimate_std", "estimate_beta", "direction", "direction_agreement", "direction_model",
    "direction_basis", "profile_net",
    "slope", "slope_se", "slope_ci_low", "slope_ci_high",
    "beta", "beta_se", "beta_ci_low", "beta_ci_high",
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
    for column in ("slope", "slope_se", "slope_ci_low", "slope_ci_high",
                   "beta", "beta_se", "beta_ci_low", "beta_ci_high"):
        rows[column] = np.nan
    if confirmation is not None and len(confirmation):
        renamed = confirmation.rename(columns={"source": "predictor", "note": "confirmation_note",
                                               "coef": "slope", "se": "slope_se", "ci_low": "slope_ci_low",
                                               "ci_high": "slope_ci_high"})
        wanted = ["depth", "target", "predictor", "p", "p_adjusted", "confirmed", "confirmation_note"]
        wanted += [c for c in ("slope", "slope_se", "slope_ci_low", "slope_ci_high",
                               "beta", "beta_se", "beta_ci_low", "beta_ci_high") if c in renamed]
        merged = rows.merge(renamed[wanted], on=["depth", "target", "predictor"], how="left", suffixes=("_x", ""))
        for column in wanted[3:]:
            rows[column] = merged[column].to_numpy()
        rows["confirmation_note"] = rows["confirmation_note"].fillna("")
    elif confirmation is None:
        rows["confirmation_note"] = ""
    rows["confirmed"] = rows["confirmed"].astype(object).where(rows["p"].notna(), np.nan)   # untested is not "no"
    rows.loc[rows.kind.isin(["block", "shared", "interaction"]) & rows.confirmation_note.eq(""),
             "confirmation_note"] = "not tested: no calibrated group test"

    signed = getattr(result, "directions", None)
    for column in ("estimate", "estimate_std", "estimate_beta", "direction_agreement", "profile_net"):
        rows[column] = np.nan
    rows["direction"], rows["direction_model"], rows["direction_basis"] = "", "", ""
    if signed is not None and len(signed):
        lookup = {(int(r.depth), r.target, r.source): r for r in signed.itertuples()}
        for i, row in rows.iterrows():
            found = lookup.get((int(row.depth), row.target, row.predictor)) if row.kind == "unique" else None
            if found is not None:
                rows.loc[i, ["estimate", "estimate_std", "estimate_beta", "direction_agreement", "profile_net"]] = [
                    found.coef, found.coef_std, found.beta, found.agreement, found.ale_net]
                rows.loc[i, "direction"], rows.loc[i, "direction_model"] = found.direction, found.model
                rows.loc[i, "direction_basis"] = found.basis
    rows.loc[rows.kind.isin(["block", "shared"]), "direction"] = "none: a joint effect has no single sign"
    rows.loc[rows.kind.eq("interaction"), "direction"] = "none: not defined for an interaction"
    rows.loc[rows.kind.eq("unique") & rows.direction.eq(""), "direction"] = "unavailable: no signed effect for this learner"

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
