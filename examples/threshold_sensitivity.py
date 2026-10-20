"""Threshold sensitivity of the exploratory validation study (critique item 6).

Re-scores the validation study's per-link records (release
``exploratory-validation-2026-10-19-39184696821``) under other cut-offs,
without re-running anything:

- keep in {.7, .8, .9}, flag in {.4, .5}, hide floor .001 on or off.

Only depth-1 links (into Y) are re-scored. Changing ``keep`` changes which
variables are promoted, so depth-2 links that a looser rule would test were
never computed. False links below 40% of folds were not recorded, which does
not matter here because every flag value is at least .4.

Usage: python examples/threshold_sensitivity.py RAW_DIR OUT_DIR
(RAW_DIR is the unzipped release; it holds ``cells/*/*-links.csv``.)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open.explore import tier  # noqa: E402

KEEP = (.7, .8, .9)
FLAG = (.4, .5)
HIDE = (.001, None)
CELL = ["engine", "n", "shape"]


def retier(links: pd.DataFrame, keep: float, flag: float, hide: float | None) -> pd.Series:
    fold = links.pass_rate.map(lambda r: "absent" if np.isnan(r) else tier(r, keep, flag))
    if hide is not None:
        fold = fold.mask(fold.isin(["kept", "flagged"]) & (links.r2_median < hide), "hidden")
    return fold


def analyse(raw: Path):
    links = pd.concat([pd.read_csv(p) for p in sorted(raw.rglob("*-links.csv"))], ignore_index=True)
    datasets = pd.concat([pd.read_csv(p) for p in sorted(raw.rglob("*-datasets.csv"))], ignore_index=True)
    links = links[links.link.str.endswith("->Y")].copy()
    links["family"] = np.where(links.kind == "interaction", "interaction", "direct")

    # The default setting must reproduce the recorded tiers.
    check = retier(links, .8, .4, .001)
    recorded = links.tier.where(links.pass_rate.notna(), "absent")
    mismatch = int((check != recorded).sum())

    n_sets = datasets.groupby(CELL).size().rename("datasets")
    true_rows, false_rows = [], []
    for keep, flag, hide in itertools.product(KEEP, FLAG, HIDE):
        setting = dict(keep=keep, flag=flag, hide_floor="on" if hide else "off")
        t = links.assign(new=retier(links, keep, flag, hide))
        tr = t[t.true]
        rates = (tr.assign(kept=tr.new == "kept", flagged=tr.new == "flagged")
                 .groupby(CELL + ["link"])[["kept", "flagged"]].mean().reset_index())
        true_rows.append(rates.assign(**setting))
        fl = t[~t.true]
        counts = (fl.assign(kept=fl.new == "kept", flagged=fl.new == "flagged")
                  .groupby(CELL + ["family"])[["kept", "flagged"]].sum())
        counts = (counts.unstack("family", fill_value=0)
                  .reindex(pd.MultiIndex.from_frame(n_sets.reset_index()[CELL]), fill_value=0))
        counts.columns = [f"false_{a}_{b}" for a, b in counts.columns]
        counts = counts.div(n_sets.values, axis=0).add_suffix("_per_dataset").reset_index()
        false_rows.append(counts.assign(**setting))
    return pd.concat(true_rows, ignore_index=True), pd.concat(false_rows, ignore_index=True), mismatch


def summary(true: pd.DataFrame, false: pd.DataFrame) -> pd.DataFrame:
    """Pooled over shapes and N, by engine and setting."""
    keys = ["engine", "keep", "flag", "hide_floor"]
    t = true.pivot_table(index=keys, columns="link", values="kept", aggfunc="mean").add_prefix("kept_")
    f = false.groupby(keys)[[c for c in false.columns if c.startswith("false_")]].mean()
    return t.join(f).round(3).reset_index()


def summary_by_n(true: pd.DataFrame, false: pd.DataFrame) -> pd.DataFrame:
    """Pooled over shapes and true links, by engine, N and setting."""
    keys = ["engine", "n", "keep", "flag", "hide_floor"]
    t = true.assign(either=true.kept + true.flagged).groupby(keys).agg(
        true_kept=("kept", "mean"), true_kept_or_flagged=("either", "mean"))
    f = false.groupby(keys).mean(numeric_only=True)
    f = pd.DataFrame({"false_kept": f.false_kept_direct_per_dataset + f.false_kept_interaction_per_dataset,
                      "false_kept_interaction": f.false_kept_interaction_per_dataset,
                      "false_dotted_direct": f.false_flagged_direct_per_dataset,
                      "false_dotted_interaction": f.false_flagged_interaction_per_dataset})
    return t.join(f).round(3).reset_index()


def main(argv=None):
    raw, out = (Path(a) for a in (argv or sys.argv[1:]))
    out.mkdir(parents=True, exist_ok=True)
    true, false, mismatch = analyse(raw)
    if mismatch:
        raise SystemExit(f"default setting does not reproduce {mismatch} recorded tiers")
    true.to_csv(out / "true-link-rates-by-setting.csv", index=False)
    false.to_csv(out / "false-links-by-setting.csv", index=False)
    s = summary(true, false)
    s.to_csv(out / "summary-by-engine.csv", index=False)
    summary_by_n(true, false).to_csv(out / "summary-by-engine-n.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
