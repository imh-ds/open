"""Study v2-R: does pooling repeated 10-fold splits (``repeats``) steady the network?

Plan: outline 2026-10-20-repeated-splits-plan. Reuses study v2's generator,
truth scoring and dataset seeds (``study_v2.py``).

Cells: engine (ols, elastic_net, lightgbm) x total N (1000, 2000) x shape
(linear, u_shape), pool 25, correlated normal predictors: 12 cells, datasets
0-49 (the same datasets as study v2).

Each dataset is split 50/50 as in study v2, and its exploration half is explored
four times (one task each):

- ``r1s0``, ``r1s1``: repeats = 1 with seeds 0 and 1 (study v2's two seeds);
- ``r5s0``, ``r5s1``: repeats = 5 with seeds 0 and 1.

``r1s0`` and ``r5s0`` are also confirmed on the other half (kept and flagged
links, Holm). R = 5 runs store their depth-1 per-fold values, so depth-1 tiers
for R = 3 can be recomputed from repeats 0-2 (deeper links cannot: R changes
what is promoted).

Success criteria (set before the run):
- true links kept again under the other seed >= 90% at R = 5;
- false solid links per network not above R = 1 beyond Monte Carlo error;
- true-link keep rates not below R = 1 by more than .05.

Commands:
  probe --out DIR [--workers W] [--engines E1,E2]
  run --engine E --n N --shape S --reps LO:HI --runs R1,R2 --out DIR [--workers W]
  aggregate --inputs DIR --out DIR
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import study_v2 as sv  # noqa: E402
from open.confirm import _adjust, confirm, frozen_network, split_sample  # noqa: E402
from open.explore import ExploreConfig, explore, tier  # noqa: E402

ENGINES = sv.ENGINES
SIZES = sv.SIZES
SHAPES = sv.SHAPES
REPS = 50
RUNS = {"r1s0": (1, 0), "r1s1": (1, 1), "r5s0": (5, 0), "r5s1": (5, 1)}
CONFIRMED = ("r1s0", "r5s0")
CELLS = [dict(n=n, pool=25, corr="correlated", ptype="normal", shape=s) for n in SIZES for s in SHAPES]
CELL = ["engine", "n", "pool", "corr", "ptype", "shape"]


def run_task(task):
    engine, cond, rep, run = task
    repeats, seed = RUNS[run]
    dseed = sv.dataset_seed(cond, rep)
    explore_half, confirm_half = split_sample(sv.generate(cond, dseed), .5, dseed)
    base = dict(engine=engine, **cond, rep=rep, seed=dseed, run=run, repeats=repeats)
    out = dict(base=base, seconds={})
    t = time.perf_counter()
    result = explore(explore_half, "Y", engine=engine,
                     config=ExploreConfig(max_depth=3, repeats=repeats, seed=seed))
    out["seconds"]["explore"] = time.perf_counter() - t
    out["depth_reached"] = int(result.nodes.depth.max())
    scored = sv.score_open(result, cond)
    out["links"] = scored
    out["designed"] = pd.DataFrame(sv.designed_outcomes(scored, cond))
    out["confirmation"] = pd.DataFrame()
    if run in CONFIRMED:
        t = time.perf_counter()
        conf = confirm(result, confirm_half, include_flagged=True, seed=dseed)
        out["seconds"]["confirm"] = time.perf_counter() - t
        frozen = frozen_network(result, include_flagged=True)
        g, design = sv.graph(cond), sv._design_keys(cond)
        info = {(r.depth, r.target, r.source, r.kind): (tuple(r.members), r.pool) for r in frozen.itertuples()}
        conf["status"] = [sv.status(g, design, r.target, r.kind, *info[(r.depth, r.target, r.source, r.kind)])
                          for r in conf.itertuples()]
        conf["p_adjusted_kept_only"] = np.nan
        kept = conf.p.notna() & (conf.explore_tier == "kept")
        if kept.any():
            conf.loc[kept, "p_adjusted_kept_only"] = _adjust(conf.loc[kept, "p"].to_numpy(), "holm")
        out["confirmation"] = conf
    f = result.folds
    out["folds1"] = (f[(f.depth == 1) & (f.contrast == "unique")][["repeat", "source", "fold", "noise", "value"]]
                     if repeats > 1 else pd.DataFrame())
    return out


def _map(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_task, tasks))


TABLES = ("links", "designed", "confirmation", "folds1")


def _write(out, records, name):
    out.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        frames = [rec[table].assign(**rec["base"]) for rec in records if len(rec[table])]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / f"{name}-{table}.csv.gz", index=False)
    pd.DataFrame([dict(rec["base"], depth_reached=rec["depth_reached"],
                       **{f"seconds_{k}": v for k, v in rec["seconds"].items()}) for rec in records]) \
        .to_csv(out / f"{name}-datasets.csv", index=False)


def probe(out, workers, engines=ENGINES):
    # Longest tasks first, so the pool is not left waiting on one long task at the end.
    tasks = [(e, c, 0, r) for r in ("r5s0", "r5s1", "r1s0", "r1s1") for e in engines for c in CELLS]
    started = time.perf_counter()
    records = _map(tasks, workers)
    _write(out, records, "probe")
    per = pd.DataFrame([dict(engine=r["base"]["engine"], n=r["base"]["n"], shape=r["base"]["shape"],
                             run=r["base"]["run"], depth_reached=r["depth_reached"], **r["seconds"])
                        for r in records])
    (out / "probe.json").write_text(json.dumps(dict(workers=workers, wall_seconds=time.perf_counter() - started,
                                                    rows=per.to_dict("records")), indent=2) + "\n")
    print(per.round(1).to_string(index=False))


def run(out, engine, cond, lo, hi, runs, workers):
    tasks = [(engine, cond, rep, r) for r in runs for rep in range(lo, hi)]
    records = _map(tasks, workers)
    _write(out, records, f"{engine}-n{cond['n']}-{cond['shape']}-{'-'.join(runs)}-r{lo:04d}-{hi:04d}")


# ---------------------------------------------------------------- aggregate

def _read(inputs, table):
    paths = sorted(Path(inputs).rglob(f"*-{table}.csv*"))
    frames = [pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in paths]
    return pd.concat(frames, ignore_index=True) if paths else pd.DataFrame()


def _kept_sets(links):
    k = links[links.tier == "kept"]
    return {key: set(zip(g.depth, g.target, g.source, g.kind)) for key, g in k.groupby(CELL + ["rep"])}


def _stability(links, first, second, label):
    """Seed agreement between two runs with the same R."""
    a, b = links[links.run == first], links[links.run == second]
    k = CELL + ["rep", "depth", "target", "source", "kind"]
    tested = ["kept", "flagged", "hidden", "dropped"]
    m = a[a.tier.isin(tested)][k + ["tier", "status"]].merge(b[b.tier.isin(tested)][k + ["tier"]], on=k,
                                                            how="left", suffixes=("_1", "_2"))
    m["tier_2"] = m.tier_2.fillna("not_tested")
    rows = []
    for (cell, status, depth1), g in m[m.tier_1 == "kept"].groupby(
            [m.engine + "|" + m.n.astype(str) + "|" + m["shape"], "status", m.depth == 1]):
        rows.append(dict(R=label, cell=cell, status=status, depth="1" if depth1 else "2-3",
                         kept_first_seed=len(g), kept_again=float((g.tier_2 == "kept").mean())))
    rep = pd.DataFrame(rows, columns=["R", "cell", "status", "depth", "kept_first_seed", "kept_again"])
    ka, kb = _kept_sets(a), _kept_sets(b)
    jac = []
    for key in a[CELL + ["rep"]].drop_duplicates().itertuples(index=False, name=None):
        sa, sb = ka.get(key, set()), kb.get(key, set())
        jac.append(dict(zip(CELL + ["rep"], key), R=label,
                        kept_jaccard=len(sa & sb) / len(sa | sb) if sa | sb else 1.0, both_empty=not (sa | sb)))
    return rep, pd.DataFrame(jac)


def _depth1_by_r(f1, links_r5):
    """Depth-1 fold tiers for R = 1, 3, 5 from the R = 5 runs' per-fold values."""
    status = (links_r5[(links_r5.depth == 1) & (links_r5.kind == "unique")]
              .drop_duplicates(CELL + ["rep", "source"]).set_index(CELL + ["rep", "source"]).status)
    rows = []
    for key, d in f1.groupby(CELL + ["rep", "run"]):
        for R in (1, 3, 5):
            sub = d[d.repeat < R]
            bar = np.quantile(sub[sub.noise].value.to_numpy(), .95)
            real = sub[~sub.noise]
            for s, rate in (real.value > bar).groupby(real.source).mean().items():
                rows.append(dict(zip(CELL + ["rep", "run"], key), R=R, source=s, pass_rate=rate,
                                 fold_tier=tier(rate, .8, .4)))
    t = pd.DataFrame(rows)
    t["status"] = [status.get(tuple(k)) for k in t[CELL + ["rep", "source"]].itertuples(index=False, name=None)]
    return t


def aggregate(inputs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = _read(inputs, "datasets")
    expected = len(CELLS) * len(ENGINES) * REPS * len(RUNS)
    counts = dict(tasks=len(datasets), expected=expected,
                  unique=len(datasets.drop_duplicates(CELL + ["rep", "run"])))
    (out / "integrity.json").write_text(json.dumps(counts, indent=2) + "\n")

    links = _read(inputs, "links")
    n_sets = datasets.groupby(CELL + ["run"]).size().rename("datasets")

    # True-link keep rates and false links per network, by run.
    des = _read(inputs, "designed")
    rates = des.assign(kept=des.tier == "kept", flagged=des.tier == "flagged") \
        .groupby(CELL + ["run", "link"])[["kept", "flagged"]].mean().reset_index()
    rates.to_csv(out / "designed-by-run.csv", index=False)
    false = links[links.status == "false"].assign(
        family=lambda d: np.where(d.kind == "interaction", "interaction", "direct"))
    fn = (false.assign(kept=false.tier == "kept", flagged=false.tier == "flagged")
          .groupby(CELL + ["run", "family"])[["kept", "flagged"]].sum().unstack("family", fill_value=0))
    fn.columns = [f"false_{a}_{b}" for a, b in fn.columns]
    fn = fn.reindex(n_sets.index, fill_value=0).div(n_sets, axis=0).add_suffix("_per_network")
    fn.round(4).reset_index().to_csv(out / "false-by-run.csv", index=False)

    # Seed stability for R = 1 and R = 5.
    reps, jacs = zip(*(_stability(links, f"r{R}s0", f"r{R}s1", R) for R in (1, 5)))
    pd.concat(reps).round(4).to_csv(out / "stability-kept-again.csv", index=False)
    jac = pd.concat(jacs)
    jac.groupby(CELL + ["R"]).agg(kept_jaccard_mean=("kept_jaccard", "mean"),
                                  both_empty=("both_empty", "mean")).round(3).reset_index() \
        .to_csv(out / "stability-jaccard.csv", index=False)

    # Depth 1 for R = 1, 3, 5 (fold tiers, unique links).
    f1 = _read(inputs, "folds1")
    if len(f1):
        d1 = _depth1_by_r(f1, links[links.run.isin(["r5s0", "r5s1"])])
        # Repeat 0 of an R = 5 run is the R = 1 run with the same seed: its depth-1 tiers must match.
        r1 = links[(links.depth == 1) & (links.kind == "unique") & links.run.isin(["r1s0", "r1s1"])]
        r1 = r1.assign(run=r1.run.str.replace("r1", "r5"))[CELL + ["rep", "run", "source", "fold_tier"]]
        chk = d1[d1.R == 1].merge(r1, on=CELL + ["rep", "run", "source"], suffixes=("", "_r1"))
        counts.update(depth1_r1_checked=len(chk), depth1_r1_mismatch=int((chk.fold_tier != chk.fold_tier_r1).sum()))
        (out / "integrity.json").write_text(json.dumps(counts, indent=2) + "\n")
        d1.assign(kept=d1.fold_tier == "kept", flagged=d1.fold_tier == "flagged") \
            .groupby(CELL + ["R", "status"])[["kept", "flagged"]].mean().round(4).reset_index() \
            .to_csv(out / "depth1-by-R.csv", index=False)
        k = CELL + ["rep", "R", "source"]
        p = d1[d1.run == "r5s0"][k + ["fold_tier", "status"]].merge(d1[d1.run == "r5s1"][k + ["fold_tier"]],
                                                                   on=k, suffixes=("_1", "_2"))
        p[p.fold_tier_1 == "kept"].groupby(["engine", "R", "status"]).fold_tier_2 \
            .agg(kept_first_seed="size", kept_again=lambda s: (s == "kept").mean()).round(4).reset_index() \
            .to_csv(out / "depth1-stability-by-R.csv", index=False)

    # Confirmation for R = 1 and R = 5.
    conf = _read(inputs, "confirmation")
    if len(conf):
        c = conf[conf.p.notna()].assign(confirmed_kept_only=lambda d: d.p_adjusted_kept_only < .05,
                                        confirmed_all=lambda d: d.p_adjusted < .05)
        fw = (c[c.status == "false"].groupby(CELL + ["run", "rep"])[["confirmed_kept_only", "confirmed_all"]]
              .any().groupby(CELL + ["run"]).sum())
        idx = n_sets[n_sets.index.get_level_values("run").isin(CONFIRMED)].index
        fw = fw.reindex(idx, fill_value=0).div(n_sets.reindex(idx), axis=0).add_prefix("familywise_false_")
        fw.round(3).reset_index().to_csv(out / "confirmation-familywise-by-run.csv", index=False)
        nulls = c[c.status == "false"]
        nulls.groupby(["engine", "shape", "run"]).p.agg(
            tested="size", below_05=lambda p: (p < .05).mean(),
            ks_uniform_p=lambda p: stats.kstest(p, "uniform").pvalue if len(p) >= 5 else np.nan) \
            .round(4).reset_index().to_csv(out / "null-pvalues-by-run.csv", index=False)

    # The three success criteria, per cell.
    rep_all = pd.concat(reps)
    true_again = rep_all[rep_all.status == "designed"].assign(
        hits=lambda d: d.kept_first_seed * d.kept_again).groupby(["cell", "R"])[["hits", "kept_first_seed"]].sum()
    true_again = ((true_again.hits / true_again.kept_first_seed).unstack("R").reindex(columns=[1, 5])
                  .add_prefix("true_kept_again_R"))
    fs = pd.read_csv(out / "false-by-run.csv")
    fs["solid"] = fs.false_kept_direct_per_network + fs.false_kept_interaction_per_network
    fs["cell"] = fs.engine + "|" + fs.n.astype(str) + "|" + fs["shape"]
    solid = fs[fs.run.isin(["r1s0", "r5s0"])].pivot_table(index="cell", columns="run", values="solid")
    solid.columns = ["false_solid_R1", "false_solid_R5"]
    dr = rates.assign(cell=rates.engine + "|" + rates.n.astype(str) + "|" + rates["shape"])
    keep = dr[dr.run.isin(["r1s0", "r5s0"])].groupby(["cell", "run"]).kept.mean().unstack("run")
    keep.columns = ["true_keep_R1", "true_keep_R5"]
    crit = true_again.join(solid).join(keep)
    crit["c1_kept_again_ge_90"] = crit.true_kept_again_R5 >= .9
    crit["c2_false_solid_not_higher"] = crit.false_solid_R5 <= crit.false_solid_R1 + .05
    crit["c3_keep_not_lower_by_05"] = crit.true_keep_R5 >= crit.true_keep_R1 - .05
    crit.round(3).reset_index().to_csv(out / "criteria.csv", index=False)
    print(crit.round(3).to_string())
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("probe")
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--engines", default=",".join(ENGINES))
    b = sub.add_parser("run")
    b.add_argument("--engine", choices=ENGINES, required=True)
    b.add_argument("--n", type=int, choices=SIZES, required=True)
    b.add_argument("--shape", choices=SHAPES, required=True)
    b.add_argument("--reps", required=True)
    b.add_argument("--runs", default=",".join(RUNS))
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("aggregate")
    c.add_argument("--inputs", required=True)
    c.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "probe":
        probe(Path(args.out), args.workers, tuple(args.engines.split(",")))
    elif args.cmd == "run":
        lo, hi = (int(x) for x in args.reps.split(":"))
        runs = tuple(args.runs.split(","))
        if not set(runs) <= set(RUNS):
            raise SystemExit(f"runs must be among {tuple(RUNS)}")
        cond = dict(n=args.n, pool=25, corr="correlated", ptype="normal", shape=args.shape)
        run(Path(args.out), args.engine, cond, lo, hi, runs, args.workers)
    else:
        aggregate(args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
