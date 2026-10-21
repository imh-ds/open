"""Report tables for study v2-R beyond ``study_v2r.py aggregate``.

- ``report-depth1-agreement.csv``: for true depth-1 links, agreement between
  the two seeds at R = 1 and R = 5: kept again, kept-or-flagged again, and
  exactly the same tier.
- ``report-reproducibility-vs-v2.csv`` (with the study v2 raw release): the
  R = 1 runs here against study v2's runs of the same datasets and seeds,
  which used the code before ``repeats`` existed, on other runners.

Usage: python examples/study_v2r_report.py SHARDS_DIR OUT_DIR [V2_SHARDS_DIR]
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

KEY = ["engine", "n", "shape", "rep", "depth", "target", "source", "kind"]
TESTED = ["kept", "flagged", "hidden", "dropped"]


def _read(root, pattern):
    paths = sorted(Path(root).rglob(pattern))
    return pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in paths], ignore_index=True)


def depth1_agreement(links):
    rows = []
    for R in (1, 5):
        a = links[(links.run == f"r{R}s0") & links.tier.isin(TESTED) & (links.status == "designed")
                  & (links.depth == 1)]
        b = links[(links.run == f"r{R}s1") & links.tier.isin(TESTED) & (links.depth == 1)]
        m = a[KEY + ["tier"]].merge(b[KEY + ["tier"]], on=KEY, suffixes=("_1", "_2"))
        m["kf1"], m["kf2"] = m.tier_1.isin(["kept", "flagged"]), m.tier_2.isin(["kept", "flagged"])
        for e, g in m.groupby("engine"):
            rows.append(dict(engine=e, R=R, links=len(g), kept_first_seed=int((g.tier_1 == "kept").sum()),
                             kept_again=(g[g.tier_1 == "kept"].tier_2 == "kept").mean(),
                             kept_or_flagged_again=g[g.kf1].kf2.mean(),
                             same_tier=(g.tier_1 == g.tier_2).mean()))
    return pd.DataFrame(rows).round(4)


def reproducibility(links, v2_root):
    first = _read(v2_root, "*-25-correlated-normal-*/*-links.csv.gz")
    second = _read(v2_root, "*-25-correlated-normal-*/*-seed2.csv.gz")
    rows = []
    for run, ref in (("r1s0", first[first.rep < 50]), ("r1s1", second)):
        a = links[links.run == run][KEY + ["tier", "pass_rate", "r2_median"]]
        m = a.merge(ref[KEY + ["tier", "pass_rate", "r2_median"]], on=KEY, how="outer", suffixes=("", "_v2"),
                    indicator=True)
        for e, g in m.groupby("engine"):
            both = g[g._merge == "both"]
            changed = both[both.tier != both.tier_v2]
            rows.append(dict(run=run, engine=e, links=len(both), only_in_one=int((g._merge != "both").sum()),
                             pass_rate_differs=int(((both.pass_rate - both.pass_rate_v2).abs() > 1e-12).sum()),
                             tier_differs=len(changed),
                             tier_differs_drawn=int((changed.tier.isin(["kept", "flagged"])
                                                     | changed.tier_v2.isin(["kept", "flagged"])).sum()),
                             max_abs_r2_difference=float((both.r2_median - both.r2_median_v2).abs().max())))
    return pd.DataFrame(rows)


def main(argv=None):
    args = [Path(a) for a in (argv or sys.argv[1:])]
    shards, out = args[0], args[1]
    out.mkdir(parents=True, exist_ok=True)
    links = _read(shards, "*-links.csv.gz")
    depth1_agreement(links).to_csv(out / "report-depth1-agreement.csv", index=False)
    if len(args) > 2:
        reproducibility(links, args[2]).to_csv(out / "report-reproducibility-vs-v2.csv", index=False)
    print(f"wrote report tables to {out}")


if __name__ == "__main__":
    main()
