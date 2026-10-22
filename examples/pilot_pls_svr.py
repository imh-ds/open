"""Pilot of the experimental PLS and SVR engines on edge-case simulations.

Not a validation study: ten small scenarios, three datasets each, explored with
every engine (OLS, Elastic-Net and LightGBM as references). Each scenario is
built to probe a strength or an expected failure of an engine:

  linear        additive linear effects and a depth-2 chain    all engines should work
  collinear     four near-copies (r ~ .89) of one cause        PLS's home ground; single copies
                                                               add little alone (shared group)
  small_wide    N = 150, 30 candidates                         PLS and Elastic-Net; SVR may struggle
  u_shape       a U-shaped effect                              linear engines (OLS, EN, PLS) should
                                                               miss it; SVR should find it
  interaction   a pure A x B effect, no main effects           linear engines should miss A and B
  step          a threshold (step) effect                      trees ideal; SVR's smooth fit?
  dilution      4 weak effects among 40 candidates, N = 300    expected SVR failure: the kernel
                                                               spreads over many columns
  null          no effects at all                              nothing should be kept
  heavy_tails   t(3) predictors and errors                     robustness (SVR's loss is robust)
  suppression   Y = X1 - X2 with X1, X2 correlated .9          PLS needs a second component

Interactions are off: OPEN's interaction step does not depend on the engine.

  python -I examples/pilot_pls_svr.py run --engines svr --reps 0:3 --out OUT --workers 4
  python -I examples/pilot_pls_svr.py aggregate --runs OUT1 OUT2 ... --out RESULTS
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open.explore import ExploreConfig, explore  # noqa: E402

ENGINES = ("ols", "elastic_net", "lightgbm", "pls", "svr")


def _noise(r, n, k, start=1):
    return {f"N{i}": r.normal(size=n) for i in range(start, start + k)}


def linear(r):
    n = 500
    d = pd.DataFrame({"D": r.normal(size=n), "B": r.normal(size=n), "C": r.normal(size=n), **_noise(r, n, 6)})
    d["A"] = .6 * d.D + .8 * r.normal(size=n)
    d["Y"] = .5 * d.A + .4 * d.B + .3 * d.C + r.normal(size=n)
    return d, [("A", "Y"), ("B", "Y"), ("C", "Y"), ("D", "A")]


def collinear(r):
    n = 500
    latent = r.normal(size=n)
    d = pd.DataFrame({f"X{i}": latent + .35 * r.normal(size=n) for i in range(1, 5)})
    d["B"] = r.normal(size=n)
    d = d.assign(**_noise(r, n, 6))
    d["Y"] = .5 * latent + .4 * d.B + r.normal(size=n)
    return d, [(("X1", "X2", "X3", "X4"), "Y"), ("B", "Y")]


def small_wide(r):
    n = 150
    d = pd.DataFrame({"A": r.normal(size=n), "B": r.normal(size=n), "C": r.normal(size=n), **_noise(r, n, 27)})
    d["Y"] = .5 * d.A + .4 * d.B + .3 * d.C + r.normal(size=n)
    return d, [("A", "Y"), ("B", "Y"), ("C", "Y")]


def u_shape(r):
    n = 500
    d = pd.DataFrame({"A": r.normal(size=n), "B": r.normal(size=n), **_noise(r, n, 8)})
    d["Y"] = .6 * (d.A ** 2 - 1) / np.sqrt(2) + .4 * d.B + r.normal(size=n)
    return d, [("A", "Y"), ("B", "Y")]


def interaction(r):
    n = 600
    d = pd.DataFrame({"A": r.normal(size=n), "B": r.normal(size=n), "C": r.normal(size=n), **_noise(r, n, 7)})
    d["Y"] = .5 * d.A * d.B + .3 * d.C + r.normal(size=n)
    return d, [("A", "Y"), ("B", "Y"), ("C", "Y")]


def step(r):
    n = 500
    d = pd.DataFrame({"A": r.normal(size=n), "B": r.normal(size=n), **_noise(r, n, 8)})
    d["Y"] = .8 * (d.A > .5) + .4 * d.B + r.normal(size=n)
    return d, [("A", "Y"), ("B", "Y")]


def dilution(r):
    n = 300
    d = pd.DataFrame({k: r.normal(size=n) for k in "ABCD"} | _noise(r, n, 36))
    d["Y"] = .25 * (d.A + d.B + d.C + d.D) + r.normal(size=n)
    return d, [(k, "Y") for k in "ABCD"]


def null(r):
    n = 500
    d = pd.DataFrame(_noise(r, n, 10))
    d["Y"] = r.normal(size=n)
    return d, []


def heavy_tails(r):
    n = 500
    t = lambda: r.standard_t(3, size=n) / np.sqrt(3)       # noqa: E731 - unit variance
    d = pd.DataFrame({"A": t(), "B": t(), "C": t(), **{f"N{i}": t() for i in range(1, 8)}})
    d["Y"] = .4 * d.A + .4 * d.B + .3 * d.C + t()
    return d, [("A", "Y"), ("B", "Y"), ("C", "Y")]


def suppression(r):
    n = 500
    common = r.normal(size=n)
    d = pd.DataFrame({"X1": common + .33 * r.normal(size=n), "X2": common + .33 * r.normal(size=n),
                      "B": r.normal(size=n), **_noise(r, n, 7)})
    d["Y"] = .8 * (d.X1 - d.X2) + .3 * d.B + r.normal(size=n)
    return d, [("X1", "Y"), ("X2", "Y"), ("B", "Y")]


SCENARIOS = {f.__name__: f for f in (linear, collinear, small_wide, u_shape, interaction, step, dilution,
                                     null, heavy_tails, suppression)}


def score(result, truth):
    """Truth recovery and noise links from the drawn (kept or flagged) links."""
    links = result.links[result.links.tier.isin(["kept", "flagged"])]
    row = {}
    found_kept = found_any = 0
    for source, target in truth:
        alts = set(source) if isinstance(source, tuple) else {source}
        here = links[links.target.eq(target) & links.members.map(lambda m: bool(alts & set(m))).astype(bool)]
        found_kept += bool(here.tier.eq("kept").any())
        found_any += bool(len(here))
    noisy = links[links.members.map(lambda m: any(str(v).startswith("N") for v in m)).astype(bool)]
    fit = result.fit.set_index("target")
    row.update(true_links=len(truth), true_kept=found_kept, true_kept_or_flagged=found_any,
               noise_kept=int(noisy.tier.eq("kept").sum()), noise_flagged=int(noisy.tier.eq("flagged").sum()),
               r2_y=float(fit.loc["Y", "r2_network"]) if "Y" in fit.index else float("nan"),
               r2_y_pool=float(fit.loc["Y", "r2_pool"]) if "Y" in fit.index else float("nan"),
               kept=";".join(sorted(f"{'+'.join(r.members)}->{r.target}"
                                    for r in links[links.tier.eq("kept")].itertuples())))
    return row


def task(args):
    scenario, engine, rep = args
    data, truth = SCENARIOS[scenario](np.random.default_rng([2026, rep, list(SCENARIOS).index(scenario)]))
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = explore(data, "Y", engine=engine, config=ExploreConfig(max_depth=2, interactions=False, seed=rep))
    return dict(scenario=scenario, engine=engine, rep=rep, n=len(data), candidates=data.shape[1] - 1,
                seconds=round(time.perf_counter() - start, 2), **score(result, truth))


def run(args):
    lo, hi = (int(x) for x in args.reps.split(":"))
    scenarios = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    tasks = [(s, e, rep) for e in args.engines.split(",") for s in scenarios for rep in range(lo, hi)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(task, tasks):
            name = f"{row['engine']}-{row['scenario']}-{row['rep']}.json"
            (out / name).write_text(json.dumps(row), encoding="utf-8")
            print(f"{row['engine']:<12}{row['scenario']:<13}rep {row['rep']}  {row['seconds']:>7.1f}s  "
                  f"true kept {row['true_kept']}/{row['true_links']}  noise kept {row['noise_kept']}", flush=True)


def aggregate(args):
    rows = [json.loads(p.read_text(encoding="utf-8")) for d in args.runs for p in Path(d).rglob("*.json")]
    runs = pd.DataFrame(rows).sort_values(["scenario", "engine", "rep"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    g = runs.groupby(["scenario", "engine"], sort=False)
    summary = pd.DataFrame({
        "datasets": g.rep.count(),
        "true_kept": g.true_kept.sum().astype(str) + "/" + g.true_links.sum().astype(str),
        "true_kept_or_flagged": g.true_kept_or_flagged.sum().astype(str) + "/" + g.true_links.sum().astype(str),
        "noise_kept": g.noise_kept.sum(), "noise_flagged": g.noise_flagged.sum(),
        "r2_y_network": g.r2_y.mean().round(3), "r2_y_pool": g.r2_y_pool.mean().round(3),
        "seconds_mean": g.seconds.mean().round(1),
    }).reset_index()
    order = {e: i for i, e in enumerate(ENGINES)}
    summary = summary.sort_values(["scenario", "engine"], key=lambda c: c.map(order) if c.name == "engine"
                                  else c.map({s: i for i, s in enumerate(SCENARIOS)}))
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--engines", default=",".join(ENGINES))
    p.add_argument("--scenarios", default="")
    p.add_argument("--reps", default="0:3")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=run)
    a = sub.add_parser("aggregate")
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=aggregate)
    parsed = parser.parse_args()
    parsed.func(parsed)
