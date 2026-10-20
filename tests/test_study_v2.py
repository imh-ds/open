"""Study v2: generator, graph-based truth scoring and comparators."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import study_v2 as sv  # noqa: E402

CORR = dict(n=4000, pool=25, corr="correlated", ptype="normal", shape="linear")
INDEP = dict(CORR, corr="independent")


def _status(cond, target, source, kind="unique", pool=None):
    members = tuple(source.replace("*", "+").split("+"))
    pool = sv.candidates(cond) if pool is None else pool
    return sv.status(sv.graph(cond), sv._design_keys(cond), target, kind, members, pool)


def test_d_separation_chain_fork_collider():
    g = {"a": [], "b": ["a"], "c": ["b"], "d": [], "e": ["a", "d"]}
    assert sv.d_separated(g, "a", "c", {"b"})
    assert not sv.d_separated(g, "a", "c", set())
    assert sv.d_separated(g, "a", "d", set())          # collider e not conditioned on
    assert not sv.d_separated(g, "a", "d", {"e"})      # conditioning on the collider opens it


def test_truth_at_depth_1_with_the_full_pool():
    assert _status(CORR, "Y", "A") == "designed"
    assert _status(CORR, "Y", "S1+S2", "shared") == "designed"
    assert _status(CORR, "Y", "C*D", "interaction") == "designed"
    assert _status(CORR, "Y", "D*C", "interaction") == "designed"
    for false in ("P_A", "P_E", "E", "G", "K", "N1"):
        assert _status(CORR, "Y", false) == "false", false
    assert _status(CORR, "Y", "S1") == "other"         # carries the latent's signal
    assert _status(CORR, "Y", "A*N1", "interaction") == "false"


def test_truth_deeper_depends_on_the_pool():
    pool_a = [c for c in sv.candidates(CORR) if c not in ("A", "B", "S1", "S2")]
    assert _status(CORR, "A", "E", pool=pool_a) == "designed"
    assert _status(CORR, "A", "K", pool=pool_a) == "designed"
    assert _status(CORR, "A", "P_A", pool=pool_a) == "other"   # a noisy copy of the target
    assert _status(CORR, "A", "P_E", pool=pool_a) == "false"   # E is in the pool
    assert _status(CORR, "A", "G", pool=pool_a) == "false"
    assert _status(INDEP, "A", "K", pool=pool_a) == "false"
    without_e = [c for c in pool_a if c != "E"]
    assert _status(CORR, "A", "P_E", pool=without_e) == "other"
    blocks = [c for c in sv.candidates(CORR) if c not in ("A", "N1")]
    assert _status(CORR, "N1", "N4", pool=blocks) == "other"  # same latent block
    assert _status(INDEP, "N1", "N4", pool=blocks) == "false"


def test_generator_correlations_and_types():
    d = sv.generate(CORR, 1)
    r = d.corr()
    assert abs(r.loc["A", "B"] - .3) < .06
    assert abs(r.loc["A", "P_A"] - .5) < .05
    assert abs(r.loc["N1", "N4"] - .4) < .06 and abs(r.loc["N1", "N2"]) < .06
    assert r.loc["S1", "S2"] > .88
    i = sv.generate(INDEP, 1).corr()
    assert abs(i.loc["A", "B"]) < .06 and abs(i.loc["A", "P_A"]) < .06
    lik = sv.generate(dict(CORR, ptype="likert"), 2)
    assert all(lik[c].nunique() == 5 for c in lik.columns if c != "Y")
    sk = sv.generate(dict(CORR, ptype="skewed"), 3)
    assert sk.G.skew() > 1
    assert list(sv.generate(dict(CORR, pool=13), 1).columns[1:]) == sv.candidates(dict(CORR, pool=13))


def test_datasets_are_shared_across_engines_and_differ_across_cells():
    assert sv.dataset_seed(CORR, 0) == sv.dataset_seed(dict(CORR), 0)
    assert sv.dataset_seed(CORR, 0) != sv.dataset_seed(INDEP, 0)
    assert sv.dataset_seed(CORR, 0) != sv.dataset_seed(CORR, 1)


def test_comparators_follow_the_recursion():
    assert sv.stability_q(13) == 2 and sv.stability_q(25) == 3
    r = np.random.default_rng(4)
    n = 1000
    x2 = r.normal(size=n)
    x1 = .6 * x2 + r.normal(size=n)
    d = pd.DataFrame({"x1": x1, "x2": x2, "x3": r.normal(size=n)})
    d.insert(0, "y", .6 * x1 + r.normal(size=n))
    for select in (sv.select_naive, sv.make_select_stability(0)):
        rows = pd.DataFrame(sv.recurse(d, "y", select))
        kept = set(rows[rows.kept].apply(lambda x: (x.source, x.target), axis=1))
        assert ("x1", "y") in kept and ("x2", "x1") in kept
        assert not (rows.target == "x1").any() or "x1" not in set(rows[rows.target == "x1"].source)
        assert set(rows[rows.depth == 2].source) <= {"x2", "x3"}


def test_designed_link_into_an_unexplored_target_is_not_tested():
    from open.explore import ExploreResult
    folds = pd.DataFrame(dict(depth=[1, 1], target=["Y", "Y"], contrast="unique", noise=False,
                              source=["A", "B"]))
    links = pd.DataFrame(dict(depth=[1, 1], target=["Y", "Y"], source=["A", "B"], kind="unique",
                              members=[("A",), ("B",)], tier=["kept", "dropped"],
                              fold_tier=["kept", "dropped"], pass_rate=[.9, .1], r2_median=[.05, 0]))
    nodes = pd.DataFrame(dict(variable=["Y", "A"], depth=[0, 1], parents=[(), ("Y",)]))
    result = ExploreResult(nodes, links, pd.DataFrame(), folds, pd.DataFrame(), [], {})
    out = sv.designed_outcomes(sv.score_open(result, dict(CORR, pool=13)), dict(CORR, pool=13))
    tiers = {r["link"]: r["tier"] for r in out}
    assert tiers["A->Y"] == "kept" and tiers["C->Y"] == "absent"
    assert tiers["E->A"] == "not_tested"     # A is a node but was never explored as a target
