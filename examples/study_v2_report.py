"""Report tables for study v2: every number quoted in its interpretation.

Reads the raw shard folders (release ``study-v2-2026-10-20-39397875389``) and
the tables written by ``study_v2.py aggregate``, and writes ``report-*.csv``.

Usage: python examples/study_v2_report.py SHARDS_DIR AGG_DIR OUT_DIR
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

CELL = ["engine", "n", "pool", "corr", "ptype", "shape"]
TESTED = ["kept", "flagged", "hidden", "dropped"]
# In U-shape cells A and B depend on K², so B (and P_A) improve a *linear* fit of A beyond K
# although A and B are independent given K: false by the graph, genuine for a linear model.
NONLINEAR_PROXY = {("A", "B"), ("B", "A"), ("A", "P_A"), ("B", "P_A")}


def _read(root, table):
    paths = sorted(Path(root).rglob(f"*-{table}.csv*"))
    return pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in paths], ignore_index=True)


def _cell_label(d):
    return (d.n.astype(str) + "/" + d.pool.astype(str) + "/" + d["corr"].str[:4] + "/" + d.ptype.str[:4]
            + "/" + d["shape"].str[:3])


def main(argv=None):
    shards, agg, out = (Path(a) for a in (argv or sys.argv[1:]))
    out.mkdir(parents=True, exist_ok=True)
    datasets = _read(shards, "datasets")
    assert len(datasets) == len(datasets.drop_duplicates(CELL + ["rep"])) == 3300, "expected 3,300 unique datasets"

    # 1. False links per tested link, by source class (item 3: proxies vs independent nulls).
    f = pd.read_csv(agg / "exploration-false.csv", keep_default_na=False, na_values=[""])
    u = f[f.kind == "unique"].groupby(["engine", "corr", "source_class"])[["kept", "flagged", "false_tested"]].sum()
    u["kept_rate"], u["flagged_rate"] = u.kept / u.false_tested, u.flagged / u.false_tested
    u.round(4).reset_index().to_csv(out / "report-false-by-source-class.csv", index=False)

    # 2. False links per network, by cell (item 4).
    f["family"] = np.where(f.kind == "interaction", "interaction", "direct")
    net = f.groupby(CELL + ["family"])[["false_kept_per_network", "false_flagged_per_network"]].sum().unstack("family")
    net.columns = [f"{a.removesuffix('_per_network')}_{b}" for a, b in net.columns]
    net.round(3).reset_index().to_csv(out / "report-false-per-network.csv", index=False)

    # 3. Null p-value calibration in confirmation (item 1), by engine, shape and link group.
    conf = _read(shards, "confirmation")
    nulls = conf[(conf.status == "false") & conf.p.notna()].copy()
    nulls["group"] = np.where([(t, s) in NONLINEAR_PROXY for t, s in zip(nulls.target, nulls.source)],
                              "A/B nonlinear proxy", "other")
    cal = nulls.groupby(["engine", "shape", "group"]).p.agg(
        tested="size", below_05=lambda p: (p < .05).mean(), below_01=lambda p: (p < .01).mean(),
        p_equal_1=lambda p: (p == 1).mean(), ks_uniform_p=lambda p: stats.kstest(p, "uniform").pvalue)
    cal.round(4).reset_index().to_csv(out / "report-null-calibration.csv", index=False)
    fw = pd.read_csv(agg / "confirmation-familywise.csv")
    fw.groupby(["engine", "shape"])[[c for c in fw.columns if c.startswith("familywise")]].agg(["mean", "max"]) \
        .round(3).to_csv(out / "report-familywise-by-shape.csv")

    # 4. Designed-link keep rates (exploration) and kept-and-confirmed rates, wide by link.
    des = pd.read_csv(agg / "exploration-designed.csv")
    des["cell"] = _cell_label(des)
    des["kept_or_flagged"] = des.kept + des.flagged
    des.pivot_table(index=["engine", "cell"], columns="link", values=["kept", "kept_or_flagged"]).round(2) \
        .to_csv(out / "report-designed-keep.csv")
    pw = pd.read_csv(agg / "confirmation-power.csv")
    pw["cell"] = _cell_label(pw)
    pw.pivot_table(index=["engine", "cell"], columns="link", values="confirmed_kept_only").round(2) \
        .to_csv(out / "report-kept-and-confirmed.csv")

    # 5. Comparators (item 12).
    cf = pd.read_csv(agg / "comparators-false.csv", keep_default_na=False, na_values=[""])
    cd = pd.read_csv(agg / "comparators-designed.csv")
    cd = cd[cd.link != "C*D->Y"]   # no comparator models interactions
    summ = (cf[cf.depth == "all"].set_index(["n", "pool", "shape", "method"]).false_kept_per_network
            .to_frame().join(cd.groupby(["n", "pool", "shape", "method"]).kept_rate.mean()
                             .rename("mean_designed_direct_kept")))
    summ.round(3).reset_index().to_csv(out / "report-comparators.csv", index=False)

    # 6. Second seed (item 5): replication of kept links and its dependence on the pass rate.
    s1, s2 = _read(shards, "links"), _read(shards, "seed2")
    s1 = s1.merge(s2[CELL + ["rep"]].drop_duplicates(), on=CELL + ["rep"])
    k = CELL + ["rep", "depth", "target", "source", "kind"]
    m = s1[s1.tier.isin(TESTED)][k + ["tier", "pass_rate", "status"]].merge(
        s2[s2.tier.isin(TESTED)][k + ["tier", "pass_rate"]], on=k, suffixes=("_1", "_2"))
    lin1 = m[(m.depth == 1) & (m["shape"] == "linear")]
    rep = lin1[lin1.tier_1 == "kept"].groupby(["engine", "status"]).tier_2.agg(
        kept_with_seed_1="size", kept_again=lambda s: (s == "kept").mean())
    rep.round(3).reset_index().to_csv(out / "report-seed-replication.csv", index=False)
    des1 = lin1[lin1.status == "designed"]
    pd.crosstab(des1.pass_rate_1.round(1), des1.tier_2, normalize="index").round(3) \
        .to_csv(out / "report-seed-by-pass-rate.csv")
    pd.read_csv(agg / "seed-agreement.csv").to_csv(out / "report-seed-agreement.csv", index=False)

    # 7. Runtime.
    secs = datasets.filter(like="seconds_").fillna(0)
    pd.DataFrame([dict(datasets=len(datasets), single_core_minutes=round(secs.sum().sum() / 60),
                       runner_minutes_4_workers=round(secs.sum().sum() / 60 / 4))]) \
        .to_csv(out / "report-runtime.csv", index=False)
    print(f"wrote {len(list(out.glob('report-*.csv')))} report tables to {out}")


if __name__ == "__main__":
    main()
