"""Study ablation: what do the noise bar and recursion each contribute?

Protocol: examples/study_ablation_protocol.md. Reuses study v2's generator, truth
scoring and dataset seeds (``study_v2.py``).

Cells: engine (ols, elastic_net, lightgbm) x study v2's eight core cells (N 1000 /
2000 x pool 13 / 25 x linear / U-shaped, correlated normal predictors), datasets
0-99 (the same datasets as study v2). Each dataset is split 50/50 as in study v2
and its exploration half is explored twice, with max_depth 3:

- ``full``: the shipped rules;
- ``nobar``: the noise bar replaced by zero, so a link passes a fold when its
  held-out gain is above zero. The 80% / 40% fold rule, the hide floor and
  everything else are unchanged.

Two more arms are depth-1 slices of those runs (depth 1 does not depend on later
depths): ``norec`` (noise bar, no recursion) and ``screen`` (neither). For the
first CHECK_REPS datasets, ``full_d1`` and ``nobar_d1`` rerun with max_depth 1
to confirm the slices are exact.

Commands:
  probe --out DIR [--workers W] [--engines E1,E2]
  run --engine E --n N --pool P --shape S --reps LO:HI --out DIR [--workers W]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import study_v2 as sv  # noqa: E402
from open import explore as ex  # noqa: E402
from open.confirm import split_sample  # noqa: E402
from open.explore import ExploreConfig, explore  # noqa: E402

ENGINES = sv.ENGINES
CELLS = sv.CORE
REPS = 100
CHECK_REPS = 5
ARMS = {"full": 3, "nobar": 3, "full_d1": 1, "nobar_d1": 1}   # arm -> max_depth
CELL = ["engine", "n", "pool", "corr", "ptype", "shape"]
KEYS = CELL + ["rep"]
REPORTED = ("full", "nobar", "norec", "screen")


def _no_bar(folds, q):
    """Noise bar of zero: a fold passes when the held-out gain is above zero."""
    folds = folds.copy()
    folds["bar"] = 0.0
    folds["passed"] = folds.value > 0.0
    return folds


def run_task(task):
    engine, cond, rep, arm = task
    dseed = sv.dataset_seed(cond, rep)
    explore_half, _ = split_sample(sv.generate(cond, dseed), .5, dseed)
    base = dict(engine=engine, **cond, rep=rep, seed=dseed, arm=arm)
    original = ex._apply_bars
    if arm.startswith("nobar"):
        ex._apply_bars = _no_bar
    try:
        t = time.perf_counter()
        result = explore(explore_half, "Y", engine=engine, config=ExploreConfig(max_depth=ARMS[arm]))
        seconds = time.perf_counter() - t
    finally:
        ex._apply_bars = original
    return dict(base=base, seconds=seconds, depth_reached=int(result.nodes.depth.max()),
                links=sv.score_open(result, cond))


def _map(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_task, tasks))


def _tasks(engine, cond, lo, hi):
    return [(engine, cond, rep, arm) for arm in ARMS for rep in range(lo, hi)
            if not arm.endswith("_d1") or rep < CHECK_REPS]


def _write(out, records, name):
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([r["links"].assign(**r["base"]) for r in records], ignore_index=True) \
        .to_csv(out / f"{name}-links.csv.gz", index=False)
    pd.DataFrame([dict(r["base"], depth_reached=r["depth_reached"], seconds=r["seconds"]) for r in records]) \
        .to_csv(out / f"{name}-datasets.csv", index=False)


def probe(out, workers, engines=ENGINES):
    # Longest tasks first, so the pool is not left waiting on one long task at the end.
    tasks = [(e, c, 0, a) for a in ("full", "nobar") for e in engines for c in CELLS]
    started = time.perf_counter()
    records = _map(tasks, workers)
    _write(out, records, "probe")
    per = pd.DataFrame([dict(engine=r["base"]["engine"], n=r["base"]["n"], pool=r["base"]["pool"],
                             shape=r["base"]["shape"], arm=r["base"]["arm"],
                             depth_reached=r["depth_reached"], seconds=r["seconds"]) for r in records])
    (out / "probe.json").write_text(json.dumps(dict(workers=workers, wall_seconds=time.perf_counter() - started,
                                                    rows=per.to_dict("records")), indent=2) + "\n")
    print(per.round(1).to_string(index=False))


def run(out, engine, cond, lo, hi, workers):
    records = _map(_tasks(engine, cond, lo, hi), workers)
    _write(out, records, f"{engine}-n{cond['n']}-p{cond['pool']}-{cond['shape']}-r{lo:04d}-{hi:04d}")


# ---------------------------------------------------------------- aggregate

def _read(inputs, table):
    paths = sorted(Path(inputs).rglob(f"*-{table}.csv*"))
    frames = [pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in paths]
    return pd.concat(frames, ignore_index=True) if paths else pd.DataFrame()


def _arms(links):
    """The four reported arms; norec and screen are depth-1 slices of full and nobar."""
    parts = [links[links.arm.isin(["full", "nobar"])]]
    for src, name in (("full", "norec"), ("nobar", "screen")):
        s = links[(links.arm == src) & (links.depth.isna() | (links.depth == 1))]
        parts.append(s.assign(arm=name))
    return pd.concat(parts, ignore_index=True)


def _per_dataset(arms, base):
    """Counts per dataset and arm, zero where a dataset has none."""
    kept, flagged = arms.tier == "kept", arms.tier == "flagged"
    false, true = arms.status == "false", arms.status == "designed"
    deep = arms.depth >= 2
    c = arms.assign(
        false_kept=false & kept, false_kept_d1=false & kept & ~deep, false_kept_d23=false & kept & deep,
        false_kept_interaction=false & kept & (arms.kind == "interaction"),
        false_flagged=false & flagged, true_kept=true & kept,
        true_kept_d1=true & kept & ~deep, true_kept_d23=true & kept & deep)
    cols = [x for x in c.columns if x.startswith(("false_", "true_"))]
    n = c.groupby(KEYS + ["arm"])[cols].sum()
    grid = pd.MultiIndex.from_tuples([(*r, a) for r in base[KEYS].itertuples(index=False) for a in REPORTED],
                                     names=KEYS + ["arm"])
    return n.reindex(grid, fill_value=0).astype(int)


def _contrast(n, a, b, label):
    """Paired difference a - b per cell and measure, with a normal 95% interval."""
    rows = []
    wide = n.unstack("arm")
    for measure in ("false_kept", "false_flagged", "true_kept", "true_kept_d23", "false_kept_d23"):
        d = (wide[(measure, a)] - wide[(measure, b)]).rename("d")
        for cell, g in d.groupby(CELL):
            se = g.std(ddof=1) / np.sqrt(len(g))
            rows.append(dict(zip(CELL, cell), contrast=label, measure=measure, datasets=len(g),
                             mean_diff=g.mean(), lo=g.mean() - 1.96 * se, hi=g.mean() + 1.96 * se))
    return rows


def aggregate(inputs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = _read(inputs, "datasets")
    expected = len(ENGINES) * len(CELLS) * (2 * REPS + 2 * CHECK_REPS)
    counts = dict(tasks=len(datasets), expected=expected,
                  unique=len(datasets.drop_duplicates(KEYS + ["arm"])))
    links = _read(inputs, "links")

    # Depth-1 slices must equal the max_depth 1 reruns.
    k = KEYS + ["target", "source", "kind"]
    mismatch = 0
    for full, d1 in (("full", "full_d1"), ("nobar", "nobar_d1")):
        a = links[(links.arm == full) & (links.depth == 1)][k + ["tier"]]
        b = links[(links.arm == d1) & (links.depth == 1)][k + ["tier"]]
        m = b.merge(a, on=k, how="left", suffixes=("_d1", "_slice"))
        mismatch += int((m.tier_d1 != m.tier_slice).sum())
        counts[f"{d1}_links_checked"] = len(b)
    counts["depth1_slice_mismatch"] = mismatch
    (out / "integrity.json").write_text(json.dumps(counts, indent=2) + "\n")

    base = datasets[datasets.arm == "full"][KEYS].drop_duplicates()
    arms = _arms(links[links.arm.isin(["full", "nobar"])])
    n = _per_dataset(arms, base)

    per_net = n.groupby(CELL + ["arm"]).mean().round(4).reset_index()
    per_net.to_csv(out / "per-network.csv", index=False)

    rows = (_contrast(n, "nobar", "full", "bar_removed_with_recursion")
            + _contrast(n, "full", "norec", "recursion_added_with_bar")
            + _contrast(n, "nobar", "screen", "recursion_added_without_bar")
            + _contrast(n, "norec", "screen", "bar_removed_without_recursion"))
    pd.DataFrame(rows).round(4).to_csv(out / "contrasts.csv", index=False)

    # Per-link keep rate of the designed links (absent / deeper-than-arm rows count as not kept).
    des = arms[arms.status == "designed"].assign(kept=lambda d: d.tier == "kept")
    des.groupby(CELL + ["arm", "target", "key", "kind"]).kept.mean().round(4).reset_index() \
        .to_csv(out / "designed-keep.csv", index=False)

    # Pre-set predictions, per cell.
    w = per_net.pivot_table(index=CELL, columns="arm")
    crit = pd.DataFrame(index=w.index)
    crit["false_kept_nobar_over_full"] = w[("false_kept", "nobar")] / w[("false_kept", "full")].where(
        w[("false_kept", "full")] > 0)
    crit["false_kept_d23_nobar"] = w[("false_kept_d23", "nobar")]
    crit["false_kept_d23_full"] = w[("false_kept_d23", "full")]
    crit["true_kept_d23_full"] = w[("true_kept_d23", "full")]
    crit["true_kept_d23_norec"] = w[("true_kept_d23", "norec")]
    crit["p1_bar_removal_raises_false"] = w[("false_kept", "nobar")] >= 2 * w[("false_kept", "full")]
    crit["p2_recursion_finds_deeper_true"] = crit.true_kept_d23_full > 0
    crit["p3_false_not_higher_with_recursion"] = w[("false_kept", "full")] <= w[("false_kept", "norec")] + .05
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
    b.add_argument("--n", type=int, choices=sv.SIZES, required=True)
    b.add_argument("--pool", type=int, choices=sv.POOLS, required=True)
    b.add_argument("--shape", choices=sv.SHAPES, required=True)
    b.add_argument("--reps", required=True)
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
        cond = dict(n=args.n, pool=args.pool, corr="correlated", ptype="normal", shape=args.shape)
        run(Path(args.out), args.engine, cond, lo, hi, args.workers)
    else:
        aggregate(args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
