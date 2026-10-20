"""Study v2: split-and-confirm, larger and correlated pools, depth 3, comparators.

Design: outline 2026-10-20-study-v2-design (critique items 1-6 and 12).

Each simulated dataset has a known generating graph:

    Y <- A (r² .09), B (.04), C (.01), shared S1/S2 (latent L, .04), C x D (.04)
    A <- E (.09)    B <- F (.01)    E <- G (.09), H (.04)
    correlated condition only: A, B <- K (.30 each, so r(A, B) is about .3);
                               P_A, P_E: noisy copies of A and E (r = .5);
                               N1..N12 in 3 blocks sharing a latent (r = .4)
    independent condition: K, P_A, P_E and N1..N12 are independent noise

Pools: 13 candidates (A..K, P_A, P_E) or 25 (plus N1..N12).

**Scoring.** A link X -> T, explored with pool P, is

- *designed*: one of the links above;
- *false*: X is d-separated from T given P without X in the generating graph,
  so X carries no information about T beyond the pool;
- *other*: genuinely predictive but not designed, for example P_A -> A, or a
  reverse link once the intermediate variable has been placed.

A shared group is false when every member is; an interaction is false when
either member is (an interaction is designed only for C x D -> Y).

Per dataset: explore on one half (defaults, max_depth 3), then confirm on the
other half with ``include_flagged=True``. Kept-only adjusted p-values are
recomputed from the same raw p-values. In core cells the OLS job also runs the
comparators (naive HC4 recursion and lasso stability selection) on the
exploration half, and datasets 0-49 are explored again with a second seed.

Commands:
  probe --out DIR [--workers W] [--engines E1,E2]
  run --engine E --n N --pool P --corr C --ptype T --shape S --reps LO:HI --out DIR [--workers W]
  aggregate --inputs DIR --out DIR
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open.confirm import _adjust, confirm, frozen_network, hc4_pvalues, split_sample  # noqa: E402
from open.explore import ExploreConfig, explore, tier  # noqa: E402

ENGINES = ("ols", "elastic_net", "lightgbm")
SIZES = (1000, 2000)
POOLS = (13, 25)
SHAPES = ("linear", "u_shape")
CORRS = ("correlated", "independent")
PTYPES = ("normal", "skewed", "likert")
SEED_BASE = 4_200_000_000
CONFIG = ExploreConfig(max_depth=3)
SEED2_CONFIG = ExploreConfig(max_depth=3, seed=1)
SEED2_REPS = 50
STABILITY_SUBSAMPLES = 100
STABILITY_THRESHOLD = .8
LIKERT_CUTS = (-1.5, -.5, .5, 1.5)
NULLS = tuple(f"N{i}" for i in range(1, 13))

CORE = [dict(n=n, pool=p, corr="correlated", ptype="normal", shape=s)
        for n in SIZES for p in POOLS for s in SHAPES]
VARIANTS = [dict(n=2000, pool=25, corr=c, ptype=t, shape="linear")
            for c, t in (("independent", "normal"), ("correlated", "skewed"), ("correlated", "likert"))]


def is_core(cond):
    return cond["corr"] == "correlated" and cond["ptype"] == "normal"


# ---------------------------------------------------------------- generator

def _std(v):
    return (v - v.mean()) / v.std()


def shape_fn(shape, x):
    if shape == "linear":
        return _std(x)
    if shape == "u_shape":
        return _std(x ** 2)
    raise ValueError(shape)


def designed(cond):
    """(source, target, kind, r²) for the cell's designed links."""
    links = [("A", "Y", "unique", .09), ("B", "Y", "unique", .04), ("C", "Y", "unique", .01),
             ("S1+S2", "Y", "shared", .04), ("C*D", "Y", "interaction", .04),
             ("E", "A", "unique", .09), ("F", "B", "unique", .01),
             ("G", "E", "unique", .09), ("H", "E", "unique", .04)]
    if cond["corr"] == "correlated":
        links += [("K", "A", "unique", .30), ("K", "B", "unique", .30)]
    return links


def graph(cond):
    """Parents of every node, latent nodes (L, Z1-Z3) included."""
    corr = cond["corr"] == "correlated"
    g = {v: [] for v in ("G", "H", "F", "K", "C", "D", "L", "Z1", "Z2", "Z3")}
    g.update(E=["G", "H"], A=["E"] + (["K"] if corr else []), B=["F"] + (["K"] if corr else []),
             S1=["L"], S2=["L"], P_A=["A"] if corr else [], P_E=["E"] if corr else [],
             Y=["A", "B", "C", "D", "L"])
    if cond["pool"] == 25:
        for i, v in enumerate(NULLS):
            g[v] = [f"Z{i % 3 + 1}"] if corr else []
    return g


def candidates(cond):
    base = ["A", "B", "C", "D", "S1", "S2", "E", "F", "G", "H", "K", "P_A", "P_E"]
    return base + (list(NULLS) if cond["pool"] == 25 else [])


def dataset_seed(cond, rep):
    """Engines share datasets: the seed depends on the condition and rep only."""
    text = f"{cond['n']}|{cond['pool']}|{cond['corr']}|{cond['ptype']}|{cond['shape']}|{rep}"
    return SEED_BASE + int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % 90_000_000


def generate(cond, seed):
    n, shape = cond["n"], cond["shape"]
    corr = cond["corr"] == "correlated"
    rng = np.random.default_rng(seed)

    def z():
        if cond["ptype"] == "skewed":
            return _std(rng.lognormal(0, .6, n))
        return rng.normal(size=n)

    def obs(x):   # the observed variable is what enters the next equation
        x = _std(x)
        if cond["ptype"] == "likert":
            x = _std(np.digitize(x, LIKERT_CUTS).astype(float))
        return x

    f = lambda x: shape_fn(shape, x)  # noqa: E731
    G, H, F, C, D, K = (obs(z()) for _ in range(6))
    L = z()
    E = obs(.3 * f(G) + .2 * f(H) + np.sqrt(.87) * z())
    if corr:
        A = obs(.3 * f(E) + np.sqrt(.3) * f(K) + np.sqrt(.61) * z())
        B = obs(.1 * f(F) + np.sqrt(.3) * f(K) + np.sqrt(.69) * z())
        P_A = obs(.5 * A + np.sqrt(.75) * z())
        P_E = obs(.5 * E + np.sqrt(.75) * z())
    else:
        A = obs(.3 * f(E) + np.sqrt(.91) * z())
        B = obs(.1 * f(F) + np.sqrt(.99) * z())
        P_A, P_E = obs(z()), obs(z())
    S1, S2 = obs(L + .3 * z()), obs(L + .3 * z())   # r(S1, S2) about .92
    if shape == "linear":
        inter = _std(C * D)
    else:   # V-shape in C whose sign flips with D: no main effect of C or D
        inter = _std((np.abs(C) - np.abs(C).mean()) * np.where(D > np.median(D), 1.0, -1.0))
    y = .3 * f(A) + .2 * f(B) + .1 * f(C) + .2 * f(L) + .2 * inter + np.sqrt(.78) * z()
    data = dict(Y=y, A=A, B=B, C=C, D=D, S1=S1, S2=S2, E=E, F=F, G=G, H=H, K=K, P_A=P_A, P_E=P_E)
    if cond["pool"] == 25:
        latent = [z() for _ in range(3)]
        for i, v in enumerate(NULLS):
            data[v] = obs(np.sqrt(.4) * latent[i % 3] + np.sqrt(.6) * z()) if corr else obs(z())
    return pd.DataFrame(data)


# ---------------------------------------------------------------- truth from the graph

def d_separated(g, x, t, given):
    """True when x and t are d-separated given the set ``given`` (moralized ancestral graph)."""
    keep, stack = set(), [x, t, *given]
    while stack:
        v = stack.pop()
        if v not in keep:
            keep.add(v)
            stack.extend(g[v])
    adj = {v: set() for v in keep}
    for v in keep:
        ps = g[v]
        for p in ps:
            adj[v].add(p)
            adj[p].add(v)
        for i, p in enumerate(ps):
            for q in ps[i + 1:]:
                adj[p].add(q)
                adj[q].add(p)
    blocked, seen, stack = set(given), {x}, [x]
    while stack:
        v = stack.pop()
        for w in adj[v] - seen:
            if w == t:
                return False
            if w not in blocked:
                seen.add(w)
                stack.append(w)
    return True


def status(g, design, target, kind, members, pool):
    """'designed', 'false' or 'other' for one link (see module docstring)."""
    if (target, kind, tuple(sorted(members))) in design:
        return "designed"
    free = lambda m: d_separated(g, m, target, set(pool) - set(members))  # noqa: E731
    if kind == "interaction":
        return "false" if any(free(m) for m in members) else "other"
    return "false" if all(free(m) for m in members) else "other"


def _key(source):
    return "+".join(sorted(source.replace("*", "+").split("+")))


def _design_keys(cond):
    return {(t, k, tuple(_key(s).split("+"))) for s, t, k, _ in designed(cond)}


def source_class(cond, source):
    if source in ("P_A", "P_E") and cond["corr"] == "correlated":
        return "proxy"
    if source in NULLS or (cond["corr"] == "independent" and source in ("K", "P_A", "P_E")):
        return "null"
    return "structural"


def _pools(result):
    f = result.folds
    return (f[(f.contrast == "unique") & ~f.noise].groupby(["depth", "target"]).source
            .apply(lambda s: tuple(dict.fromkeys(s))).to_dict())


def score_open(result, cond):
    """Every tested link with its status; designed links that were never tested are added as absent rows."""
    g, design, pools = graph(cond), _design_keys(cond), _pools(result)
    rows = []
    for r in result.links.itertuples():
        members = tuple(r.members)
        rows.append(dict(depth=int(r.depth), target=r.target, source=r.source, kind=r.kind,
                         key="+".join(sorted(members)), status=status(g, design, r.target, r.kind, members, pools[(r.depth, r.target)]),
                         source_class=source_class(cond, r.source) if r.kind == "unique" else r.kind,
                         tier=r.tier, fold_tier=r.fold_tier, pass_rate=r.pass_rate, r2_median=r.r2_median))
    out = pd.DataFrame(rows, columns=["depth", "target", "source", "kind", "key", "status", "source_class",
                                      "tier", "fold_tier", "pass_rate", "r2_median"])
    reached = {t for _, t in pools}   # explored as a target (a node at max_depth is not)
    for s, t, k, _ in designed(cond):
        if not ((out.target == t) & (out.key == _key(s)) & (out.kind == k)).any():
            out.loc[len(out)] = dict(depth=np.nan, target=t, source=s, kind=k, key=_key(s), status="designed",
                                     source_class=k, tier="absent" if t in reached else "not_tested",
                                     fold_tier=None, pass_rate=np.nan, r2_median=np.nan)
    return out


def designed_outcomes(scored, cond):
    """One row per designed link: its tier, with study 1's shared-group fallback (best member alone)."""
    order = {"kept": 0, "flagged": 1, "hidden": 2, "dropped": 3, "absent": 4, "not_tested": 5}
    rows = []
    for s, t, k, r2 in designed(cond):
        hit = scored[(scored.target == t) & (scored.key == _key(s)) & (scored.kind == k)]
        best = hit.tier.iloc[0] if len(hit) else "absent"
        if k == "shared":
            alone = scored[(scored.target == t) & scored.source.isin(s.split("+")) & (scored.kind == "unique")]
            for tr in alone.tier:
                best = min(best, tr, key=order.get)
        rows.append(dict(link=f"{s}->{t}", kind=k, r2_true=r2, tier=best))
    return rows


# ---------------------------------------------------------------- comparators

def recurse(data, outcome, select, max_depth=3):
    """OPEN's recursion with another selection rule: rows (depth, target, source, kept)."""
    placed, targets, rows, depth = {outcome}, [outcome], [], 1
    cands = [c for c in data.columns if c != outcome]
    while targets and depth <= max_depth:
        pool = [c for c in cands if c not in placed]
        if not pool:
            break
        new = []
        for t in targets:
            chosen = set(select(data[pool].to_numpy(float), data[t].to_numpy(float), pool, depth, t))
            rows += [dict(depth=depth, target=t, source=s, kept=s in chosen, pool=tuple(pool)) for s in pool]
            new += [s for s in pool if s in chosen]
        targets = list(dict.fromkeys(new))
        placed |= set(targets)
        depth += 1
    return rows


def select_naive(X, y, pool, depth, target):
    """One OLS fit on the whole pool; keep HC4 p < .05."""
    return [s for s, p in zip(pool, hc4_pvalues(X, y)) if p < .05]


def stability_q(p):
    """Largest q with the Meinshausen-Buhlmann bound q² / ((2 pi - 1) p) <= 1 at pi = .8."""
    return max(1, int(np.floor(np.sqrt((2 * STABILITY_THRESHOLD - 1) * p))))


def make_select_stability(seed):
    from sklearn.linear_model import lars_path

    def select(X, y, pool, depth, target):
        n, p = X.shape
        q = stability_q(p)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
            f"{seed}|stability|{depth}|{target}".encode()).digest()[:4], "big"))
        counts = np.zeros(p)
        for _ in range(STABILITY_SUBSAMPLES):
            idx = rng.choice(n, n // 2, replace=False)
            Xs, ys = X[idx], y[idx]
            Xs = (Xs - Xs.mean(0)) / Xs.std(0)
            ys = ys - ys.mean()
            _, _, coefs = lars_path(Xs, ys, method="lasso")
            active = coefs != 0
            sizes = active.sum(0)
            step = int(np.argmax(sizes >= q)) if (sizes >= q).any() else active.shape[1] - 1
            counts += active[:, step]
        return [s for s, c in zip(pool, counts / STABILITY_SUBSAMPLES) if c >= STABILITY_THRESHOLD]
    return select


def score_comparator(rows, cond, method):
    g, design = graph(cond), _design_keys(cond)
    out = pd.DataFrame(rows)
    out["status"] = [status(g, design, r.target, "unique", (r.source,), r.pool) for r in out.itertuples()]
    out["source_class"] = [source_class(cond, s) for s in out.source]
    return out.drop(columns="pool").assign(method=method)


# ---------------------------------------------------------------- one dataset

def run_one(task):
    engine, cond, rep = task
    seed = dataset_seed(cond, rep)
    data = generate(cond, seed)
    explore_half, confirm_half = split_sample(data, .5, seed)
    base = dict(engine=engine, **cond, rep=rep, seed=seed)
    sec = {}

    t = time.perf_counter()
    result = explore(explore_half, "Y", engine=engine, config=CONFIG)
    sec["explore"] = time.perf_counter() - t
    scored = score_open(result, cond)

    t = time.perf_counter()
    conf = confirm(result, confirm_half, include_flagged=True, seed=seed)
    sec["confirm"] = time.perf_counter() - t
    frozen = frozen_network(result, include_flagged=True)
    g, design = graph(cond), _design_keys(cond)
    info = {(r.depth, r.target, r.source, r.kind): (tuple(r.members), r.pool) for r in frozen.itertuples()}
    conf["status"] = [status(g, design, r.target, r.kind, *info[(r.depth, r.target, r.source, r.kind)])
                      for r in conf.itertuples()]
    conf["p_adjusted_kept_only"] = np.nan
    kept = conf.p.notna() & (conf.explore_tier == "kept")
    if kept.any():
        conf.loc[kept, "p_adjusted_kept_only"] = _adjust(conf.loc[kept, "p"].to_numpy(), "holm")

    comparators = pd.DataFrame()
    if engine == "ols" and is_core(cond):
        t = time.perf_counter()
        parts = [score_comparator(recurse(explore_half, "Y", select_naive), cond, "naive"),
                 score_comparator(recurse(explore_half, "Y", make_select_stability(seed)), cond, "stability")]
        sec["comparators"] = time.perf_counter() - t
        comparators = pd.concat(parts, ignore_index=True)

    seed2 = pd.DataFrame()
    if is_core(cond) and rep < SEED2_REPS:
        t = time.perf_counter()
        second = explore(explore_half, "Y", engine=engine, config=SEED2_CONFIG)
        sec["seed2"] = time.perf_counter() - t
        seed2 = score_open(second, cond)

    f = result.folds
    folds1 = f[(f.depth == 1) & (f.contrast == "unique")][["source", "fold", "noise", "value"]]
    return dict(base=base, seconds=sec, depth_reached=int(result.nodes.depth.max()),
                links=scored, designed=pd.DataFrame(designed_outcomes(scored, cond)), confirmation=conf,
                comparators=comparators, seed2=seed2, folds1=folds1)


def _map(tasks, workers):
    if workers <= 1:
        return [run_one(t) for t in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_one, tasks))


TABLES = ("links", "designed", "confirmation", "comparators", "seed2", "folds1")


def _write(out, records, name):
    out.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        frames = [rec[table].assign(**rec["base"]) for rec in records if len(rec[table])]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / f"{name}-{table}.csv.gz", index=False)
    pd.DataFrame([dict(rec["base"], depth_reached=rec["depth_reached"],
                       **{f"seconds_{k}": v for k, v in rec["seconds"].items()}) for rec in records]) \
        .to_csv(out / f"{name}-datasets.csv", index=False)


def cell_name(engine, cond):
    return f"{engine}-n{cond['n']}-p{cond['pool']}-{cond['corr']}-{cond['ptype']}-{cond['shape']}"


def probe(out, workers, engines=ENGINES):
    tasks = [(e, c, 0) for e in engines for c in CORE]
    started = time.perf_counter()
    records = _map(tasks, workers)
    _write(out, records, "probe")
    per = pd.DataFrame([dict(engine=r["base"]["engine"], n=r["base"]["n"], pool=r["base"]["pool"],
                             shape=r["base"]["shape"], depth_reached=r["depth_reached"], **r["seconds"])
                        for r in records])
    summary = dict(workers=workers, wall_seconds=time.perf_counter() - started, rows=per.to_dict("records"))
    (out / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(per.round(1).to_string(index=False))
    return summary


def run(out, engine, cond, lo, hi, workers):
    records = _map([(engine, cond, rep) for rep in range(lo, hi)], workers)
    _write(out, records, f"{cell_name(engine, cond)}-r{lo:04d}-{hi:04d}")


# ---------------------------------------------------------------- aggregate

CELL = ["engine", "n", "pool", "corr", "ptype", "shape"]


def _read(inputs, table):
    paths = sorted(Path(inputs).rglob(f"*-{table}.csv*"))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def aggregate(inputs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = _read(inputs, "datasets")
    n_sets = datasets.groupby(CELL).size().rename("datasets")

    # Exploration: designed links' tiers; false links per network and per tested link.
    des = _read(inputs, "designed")
    tiers = ["kept", "flagged", "hidden", "dropped", "absent", "not_tested"]
    true_rates = (des.assign(**{t: des.tier == t for t in tiers}).groupby(CELL + ["link"])[tiers].mean()
                  .reset_index())
    true_rates.to_csv(out / "exploration-designed.csv", index=False)
    links = _read(inputs, "links")
    false = links[links.status == "false"]
    tested = false.groupby(CELL + ["depth", "kind", "source_class"]).size().rename("false_tested")
    by_tier = (false.assign(kept=false.tier == "kept", flagged=false.tier == "flagged")
               .groupby(CELL + ["depth", "kind", "source_class"])[["kept", "flagged"]].sum())
    fr = by_tier.join(tested).reset_index().merge(n_sets.reset_index(), on=CELL)
    for t in ("kept", "flagged"):
        fr[f"false_{t}_per_network"] = fr[t] / fr.datasets
        fr[f"false_{t}_rate"] = fr[t] / fr.false_tested
    fr.to_csv(out / "exploration-false.csv", index=False)

    # Confirmation.
    conf = _read(inputs, "confirmation")
    tested_c = conf[conf.p.notna()].copy()
    tested_c["confirmed_kept_only"] = tested_c.p_adjusted_kept_only < .05
    tested_c["confirmed_all"] = tested_c.p_adjusted < .05
    fw = (tested_c[tested_c.status == "false"].groupby(CELL + ["rep"])
          [["confirmed_kept_only", "confirmed_all"]].any().groupby(CELL).sum())
    fw = fw.reindex(n_sets.index, fill_value=0).div(n_sets, axis=0).add_prefix("familywise_false_")
    power = (tested_c[(tested_c.status == "designed")].assign(link=lambda d: d.source + "->" + d.target)
             .groupby(CELL + ["link"])[["confirmed_kept_only", "confirmed_all"]].sum().reset_index()
             .merge(n_sets.reset_index(), on=CELL))
    for c in ("confirmed_kept_only", "confirmed_all"):
        power[c] = power[c] / power.datasets
    fw.reset_index().to_csv(out / "confirmation-familywise.csv", index=False)
    power.to_csv(out / "confirmation-power.csv", index=False)
    nulls = tested_c[tested_c.status == "false"]
    calib = nulls.groupby(CELL + ["explore_tier"]).p.agg(
        tested="size", below_05=lambda p: (p < .05).mean(), below_01=lambda p: (p < .01).mean(),
        ks_uniform_p=lambda p: stats.kstest(p, "uniform").pvalue if len(p) >= 5 else np.nan).reset_index()
    calib.to_csv(out / "null-pvalues.csv", index=False)

    # Comparators against OPEN's solid links (direct links only).
    comp = _read(inputs, "comparators")
    if len(comp):
        direct = links[links.kind.isin(["unique", "shared"])]
        direct = direct[(direct["corr"] == "correlated") & (direct.ptype == "normal")]
        opn = direct.assign(method="open_" + direct.engine, kept=direct.tier == "kept")
        keys = ["n", "pool", "shape", "method"]
        cols = keys + ["rep", "depth", "target", "source", "status", "kept"]
        allm = pd.concat([opn[cols], comp[cols]], ignore_index=True)
        reps = allm.groupby(keys).rep.nunique().rename("datasets")
        flags = allm.dropna(subset=["depth"]).assign(false_kept=lambda d: (d.status == "false") & d.kept)
        fk = pd.concat([flags.groupby(keys + ["depth"]).false_kept.sum().reset_index(),
                        flags.groupby(keys).false_kept.sum().reset_index().assign(depth="all")])
        fk = fk.merge(reps.reset_index(), on=keys)
        fk["false_kept_per_network"] = fk.false_kept / fk.datasets
        dk = allm[(allm.status == "designed") & (allm.source != "S1+S2")]
        dk = dk.assign(link=dk.source + "->" + dk.target)
        # The substitute pair counts for any method that keeps the group or either member (study 1 rule).
        sub = allm[(allm.target == "Y") & allm.source.isin(["S1", "S2", "S1+S2"])]
        sub = sub.groupby(keys + ["rep"]).kept.any().reset_index().assign(link="S1+S2->Y")
        dk = pd.concat([dk[keys + ["rep", "link", "kept"]], sub], ignore_index=True)
        dk = (dk.groupby(keys + ["link"]).kept.sum().rename("kept").reset_index()
              .merge(reps.reset_index(), on=keys))
        dk["kept_rate"] = dk.kept / dk.datasets
        fk.to_csv(out / "comparators-false.csv", index=False)
        dk.to_csv(out / "comparators-designed.csv", index=False)

    # Second seed. Depth 1: tier agreement on the same pool, among links not dropped in
    # either run. Whole network: overlap (Jaccard) of the kept links, per dataset.
    s2 = _read(inputs, "seed2")
    if len(s2):
        tested_tiers = ["kept", "flagged", "hidden", "dropped"]
        k = CELL + ["rep", "depth", "target", "source", "kind"]
        first = links.merge(s2[CELL + ["rep"]].drop_duplicates(), on=CELL + ["rep"])
        a, b = (x[x.tier.isin(tested_tiers)] for x in (first, s2))
        d1 = a[a.depth == 1][k + ["tier"]].merge(b[b.depth == 1][k + ["tier"]], on=k, suffixes=("_1", "_2"))
        d1 = d1[(d1.tier_1 != "dropped") | (d1.tier_2 != "dropped")]
        depth1 = d1.assign(same=d1.tier_1 == d1.tier_2).groupby(CELL).agg(
            depth1_links=("same", "size"), depth1_same_tier=("same", "mean"))
        def kept_sets(x):
            x = x[x.tier == "kept"]
            return {key: set(zip(g.depth, g.target, g.source, g.kind)) for key, g in x.groupby(CELL + ["rep"])}
        ka, kb = kept_sets(a), kept_sets(b)
        rows = []
        for key in s2[CELL + ["rep"]].drop_duplicates().itertuples(index=False, name=None):
            sa, sb = ka.get(key, set()), kb.get(key, set())
            rows.append(dict(zip(CELL + ["rep"], key), kept_jaccard=len(sa & sb) / len(sa | sb) if sa | sb else 1.0))
        network = (pd.DataFrame(rows).groupby(CELL).kept_jaccard.agg(["mean", "median"])
                   .add_prefix("kept_jaccard_"))
        depth1.join(network, how="outer").round(3).reset_index() \
            .to_csv(out / "seed-agreement.csv", index=False)

    # 90% noise bar at depth 1 (unique contrast), recomputed from fold values.
    f1 = _read(inputs, "folds1")
    if len(f1):
        rows = []
        for key, d in f1.groupby(CELL + ["rep"]):
            noise = d[d.noise].value.to_numpy()
            real = d[~d.noise]
            for q in (.95, .90):
                bar = np.quantile(noise, q)
                rate = real.assign(p=real.value > bar).groupby("source").p.mean()
                for s, r in rate.items():
                    rows.append(dict(zip(CELL + ["rep"], key), quantile=q, source=s,
                                     fold_tier=tier(r, CONFIG.keep, CONFIG.flag)))
        nb = pd.DataFrame(rows)
        nb["status"] = [
            "designed" if (s, "Y") in {("A", "Y"), ("B", "Y"), ("C", "Y")} else
            ("false" if d_separated(graph(dict(corr=c, pool=p)), s, "Y",
                                    set(candidates(dict(pool=p))) - {s}) else "other")
            for s, c, p in zip(nb.source, nb["corr"], nb.pool)]
        nb.assign(kept=nb.fold_tier == "kept", flagged=nb.fold_tier == "flagged") \
            .groupby(CELL + ["quantile", "status"])[["kept", "flagged"]].mean().reset_index() \
            .to_csv(out / "noise-bar-90.csv", index=False)
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
    b.add_argument("--pool", type=int, choices=POOLS, required=True)
    b.add_argument("--corr", choices=CORRS, required=True)
    b.add_argument("--ptype", choices=PTYPES, required=True)
    b.add_argument("--shape", choices=SHAPES, required=True)
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
        cond = dict(n=args.n, pool=args.pool, corr=args.corr, ptype=args.ptype, shape=args.shape)
        run(Path(args.out), args.engine, cond, lo, hi, args.workers)
    else:
        aggregate(args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
