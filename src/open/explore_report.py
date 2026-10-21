"""Tables and graph for exploratory-mode results (C3, C6).

``write_explore`` saves the result tables, settings, a text summary and the
network graph. ``plot_explore`` draws the graph as a radial tree:

- the outcome in the centre, each depth on a wider ring, and every target's
  predictors fanning outwards from it;
- circular nodes, with edges drawn centre to centre behind them;
- solid edges for kept links and dashed edges for flagged links, with widths
  scaled by held-out R² lost and labelled with it (a check mark when the link
  was confirmed on new data);
- around each explored target, a ring filled in proportion to its network
  fit: held-out R² from its kept predictors (and, with confirmation, the
  cross-validated R² on the new data, as text);
- shared (substitute) groups as a blue band joining their members, with one
  edge to the target;
- interactions as a fork: solid orange lines from the two predictors meet in a
  small diamond, labelled with the pair ("F·N7"), and one edge runs on to the
  target. Only that last edge is solid (kept) or dashed (flagged), because the
  pairing itself is not uncertain;
- predictors that are only flagged, and never kept, as grey dashed nodes.

Display settings live in ``GraphStyle``: node size, edge thickness, label size,
spacing between nodes (more compact or more spread out), which flagged edges and
nodes to show, edge labels and resolution. Pass a style, or the same settings as
keywords: ``plot_explore(result, "g.png", spacing=.8, node_size=1.2)``.

Relations hidden by ``hide_below`` are left out of the graph but stay in the
tables. Flagged interactions are also left out by default
(``show_flagged_interactions``): in the validation study they were mostly
false. They stay in the tables.

An edge means "the outer variable predicts the inner one beyond that target's
other candidates" (held-out R²). It is not causal, and it is not a partial
correlation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from open.explore import ExploreResult


def write_explore(result: ExploreResult, out_dir, *, plot: bool = True,
                  confirmation: pd.DataFrame | None = None,
                  confirmation_fit: pd.DataFrame | None = None,
                  style: "GraphStyle | None" = None, **settings) -> Path:
    """Save the tables, settings, summary and graph.

    ``style`` and ``settings`` (``GraphStyle`` fields such as ``spacing`` or
    ``show_flagged``) control the graph; see ``plot_explore``.

    With ``confirmation`` (from ``open.confirm.confirm``), also save it as
    ``confirmation.csv`` and mark confirmed links in the graph. With
    ``confirmation_fit`` (from ``open.confirm.confirmation_fit``), save it as
    ``confirmation-fit.csv`` and show it with each target's network fit.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = result.nodes.assign(parents=result.nodes.parents.map(";".join))
    nodes.to_csv(out / "nodes.csv", index=False)
    if len(result.links):
        result.links.assign(members=result.links.members.map(";".join)).to_csv(out / "links.csv", index=False)
    result.pairs.to_csv(out / "pairs.csv", index=False)
    result.folds.to_csv(out / "folds.csv", index=False)
    result.bars.to_csv(out / "noise-bars.csv", index=False)
    result.fit.assign(network=result.fit.network.map(";".join)).to_csv(out / "fit.csv", index=False)
    (out / "settings.json").write_text(json.dumps(dict(result.settings, stops=result.stops), indent=2) + "\n")
    (out / "summary.txt").write_text(result.summary() + "\n", encoding="utf-8")
    if confirmation is not None:
        confirmation.to_csv(out / "confirmation.csv", index=False)
    if confirmation_fit is not None:
        confirmation_fit.assign(network=confirmation_fit.network.map(";".join)) \
            .to_csv(out / "confirmation-fit.csv", index=False)
    if plot:
        plot_explore(result, out / "network.png", confirmation=confirmation,
                     confirmation_fit=confirmation_fit, style=style, **settings)
    return out


def write_split(split, out_dir, *, plot: bool = True, style: "GraphStyle | None" = None, **settings) -> Path:
    """Save an ``open.confirm.explore_and_confirm`` result: exploration, confirmation and split settings."""
    out = write_explore(split.exploration, out_dir, plot=plot, confirmation=split.confirmation,
                        confirmation_fit=split.fit, style=style, **settings)
    (out / "split-settings.json").write_text(json.dumps(split.settings, indent=2) + "\n")
    (out / "summary.txt").write_text(split.summary() + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- graph

NODE_R = 0.42          # node radius (data units)
OUTCOME_R = 0.56
MIN_SEP = 2.9 * NODE_R  # minimum distance between node centres on a ring
RING_STEP = 2.3         # base distance between depth rings

INK, MUTED = "#1f2933", "#6b7785"
KEPT, FLAGGED = "#334155", "#94a3b8"
SHARED, INTERACTION, FIT, CONFIRMED = "#2563eb", "#d9480f", "#2f9e44", "#2f9e44"
DEPTH_FILL = ("#1f2933", "#dbe7f6", "#e3f2e1", "#fbefd5", "#f3e3f1", "#e2f0f1")


@dataclass(frozen=True)
class GraphStyle:
    """Display settings for the network graph. Sizes are multipliers of the defaults.

    - ``node_size``: radius of the circles (and of the fit rings and diamonds).
    - ``edge_width``: thickness of every edge.
    - ``label_size``: every text: node names, edge labels, fit values, legend.
    - ``spacing``: distance between nodes. Below 1 is more compact, above 1 more
      spread out. Nodes never overlap, however small it is.
    - ``show_flagged``: draw flagged (dashed) links and the flagged-only nodes
      they bring in. False shows the kept network only.
    - ``show_flagged_interactions``: also draw flagged interactions (off by
      default: in validation they were mostly false).
    - ``edge_labels``: "kept" labels solid edges with their R², "all" also dashed
      ones, None labels none. The tables hold every value.
    - ``dpi``: resolution of a saved image.
    """
    node_size: float = 1.0
    edge_width: float = 1.0
    label_size: float = 1.0
    spacing: float = 1.0
    show_flagged: bool = True
    show_flagged_interactions: bool = False
    edge_labels: str | None = "kept"
    dpi: int = 170

    def __post_init__(self):
        for name in ("node_size", "edge_width", "label_size", "spacing"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 5:
                raise ValueError(f"{name} must be a number in (0, 5]")
        for name in ("show_flagged", "show_flagged_interactions"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False")
        if self.edge_labels not in ("kept", "all", None):
            raise ValueError('edge_labels must be "kept", "all" or None')
        if type(self.dpi) is not int or not 50 <= self.dpi <= 1200:
            raise ValueError("dpi must be an integer in [50, 1200]")

    def node_r(self):
        return NODE_R * self.node_size

    def min_sep(self):
        """Minimum distance between centres: the spaced default, never less than clear of overlap."""
        return max(MIN_SEP * self.spacing, 2.4 * self.node_r())


def _style(style, settings) -> GraphStyle:
    return replace(style or GraphStyle(), **settings)


def _r2(v):
    return f"{v:.3f}" if abs(v) >= .001 else ("<.001" if v > 0 else f"{v:.3f}")


def _apa(v):
    """R² in APA style: three decimals, no leading zero (.120, −.004); tiny positives as <.001."""
    if 0 < v < .0005:
        return "<.001"
    text = f"{abs(v):.3f}".lstrip("0")
    return ("−" if v < 0 and text != ".000" else "") + text


def _passes(pass_rate, n_folds):
    """Folds passed out of all folds (repeats included), e.g. "9/10" or "41/50"."""
    return f"{int(round(pass_rate * n_folds))}/{n_folds}"


def _width(r2):
    return float(np.clip(1.2 + 30 * max(r2, 0), 1.0, 6.5))


def _drawn(links: pd.DataFrame, show_flagged: bool, show_flagged_interactions: bool) -> pd.DataFrame:
    if not len(links):
        return links
    drawn = links.tier.eq("kept") | (show_flagged & links.tier.eq("flagged"))
    if not show_flagged_interactions:
        drawn &= ~(links.kind.eq("interaction") & links.tier.eq("flagged"))
    return links[drawn]


def _tree(result: ExploreResult, links: pd.DataFrame):
    """Each drawn node's ring (depth), parent and whether it is a flagged-only candidate.

    Network nodes hang from their first parent. A flagged-only candidate hangs
    from the target of its deepest drawn link.
    """
    ring = {r.variable: int(r.depth) for r in result.nodes.itertuples()}
    parent = {r.variable: r.parents[0] for r in result.nodes.itertuples() if r.parents}
    candidate = set()
    for r in links.sort_values("depth").itertuples():
        for m in r.members:
            if m not in ring or (m in candidate and int(r.depth) >= ring[m]):
                ring[m], parent[m] = int(r.depth), r.target
                candidate.add(m)
    return ring, parent, candidate


def _radial(outcome, ring, parent, candidate, links, ring_step=RING_STEP):
    """Radial-tree start: each subtree gets an angular sector in proportion to its leaves."""
    order = {}
    for r in links.itertuples():   # siblings in a group or interaction stay next to each other
        for m in r.members:
            order.setdefault(m, len(order))
    children = {}
    for v, p in parent.items():
        children.setdefault(p, []).append(v)
    for p in children:
        children[p].sort(key=lambda v: (v in candidate, order.get(v, len(order)), v))
    leaves = {}

    def count(v):
        leaves[v] = max(1, sum(count(c) for c in children.get(v, [])))
        return leaves[v]

    count(outcome)
    angle = {outcome: 0.0}

    def place(v, lo, hi):
        kids = children.get(v, [])
        total = sum(leaves[c] for c in kids)
        start = lo
        for c in kids:
            width = (hi - lo) * leaves[c] / total
            angle[c] = start + width / 2
            place(c, start, start + width)
            start += width

    place(outcome, 0.0, 2 * math.pi)
    return {v: (ring[v] * ring_step * math.cos(math.pi / 2 - angle[v]),
                ring[v] * ring_step * math.sin(math.pi / 2 - angle[v])) for v in ring}


def _point_segment(p, a, b):
    """Distance from p to segment ab, and the closest point."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q


def _layout(result: ExploreResult, links: pd.DataFrame, style: GraphStyle | None = None):
    """Node positions: (positions, ring of each node, flagged-only candidates, junctions).

    Each drawn interaction gets a junction (the diamond), laid out as a small node
    of its own: pulled towards its two variables and its target, pushed away from
    everything else, so it settles in open space. ``junctions`` maps
    (depth, target, source) to its position.

    Starts from a radial tree (outcome in the centre, each target's predictors
    fanning outwards) and relaxes it with a spring layout: linked nodes attract
    (kept links more than flagged ones), all nodes repel, the outcome stays in
    the centre. A last pass keeps nodes off edges they are not part of and at
    least ``style.min_sep()`` apart. ``style.spacing`` scales every distance and
    ``style.node_size`` the clearances. Deterministic: no random start.
    """
    style = style or GraphStyle()
    ring_step, k = RING_STEP * style.spacing, 1.9 * style.spacing
    min_sep, clear = style.min_sep(), 1.6 * style.node_r()
    outcome = result.settings.get("outcome") or result.nodes.variable.iloc[0]
    ring, parent, candidate = _tree(result, links)
    start = _radial(outcome, ring, parent, candidate, links, ring_step)
    names = list(start)
    inter = links[links.kind == "interaction"]
    keys = [(int(r.depth), r.target, r.source) for r in inter.itertuples()]
    for key, r in zip(keys, inter.itertuples()):  # start between the pair, a third of the way to the target
        a, b, t = (np.array(start[v]) for v in (*r.members, r.target))
        mid = (a + b) / 2
        spot = mid + (t - mid) / 3
        if np.linalg.norm(spot - t) < .45 * ring_step:   # pair on opposite sides of the target: step aside
            side = np.array([-(b - a)[1], (b - a)[0]])
            spot = t + side / (np.linalg.norm(side) + 1e-12) * ring_step * .7
        start[key] = tuple(spot)
    names = list(start)
    if len(names) == 1:
        return {outcome: (0.0, 0.0)}, ring, candidate, {}
    index = {v: i for i, v in enumerate(names)}
    P = np.array([start[v] for v in names], dtype=float)
    springs = {}
    segments = []
    for r in links.itertuples():
        w = 1.0 if r.tier == "kept" else .7
        if r.kind == "interaction":
            j = index[(int(r.depth), r.target, r.source)]
            for m in r.members:
                springs[tuple(sorted((index[m], j)))] = 1.0
                segments.append((index[m], j))
            springs[tuple(sorted((j, index[r.target])))] = w
            segments.append((j, index[r.target]))
            continue
        for m in r.members:
            key = tuple(sorted((index[m], index[r.target])))
            springs[key] = max(springs.get(key, 0), w)
            segments.append((index[m], index[r.target]))
        for a in range(len(r.members)):
            for b in range(a + 1, len(r.members)):
                key = tuple(sorted((index[r.members[a]], index[r.members[b]])))
                springs[key] = max(springs.get(key, 0), .6 if r.kind == "shared" else .3)
    pin = index[outcome]
    n = len(names)
    temp = .5 * k
    for _ in range(400):
        delta = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(delta, axis=-1) + np.eye(n)
        disp = (delta / dist[..., None] * (k * k / dist)[..., None]).sum(axis=1)   # repulsion
        for (a, b), w in springs.items():                                       # attraction
            d = P[a] - P[b]
            length = max(float(np.linalg.norm(d)), 1e-9)
            f = w * length * length / k * d / length
            disp[a] -= f
            disp[b] += f
        disp[pin] = 0
        step = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-12
        P += disp / step * np.minimum(step, temp)
        P[pin] = 0
        temp = max(temp * .985, .01)
    for _ in range(60):                       # keep nodes off other edges, and apart
        moved = False
        for i in range(n):
            if i == pin:
                continue
            for a, b in segments:
                if i in (a, b):
                    continue
                d, q = _point_segment(P[i], P[a], P[b])
                if d < clear:
                    away = P[i] - q if d > 1e-9 else np.array([-(P[b] - P[a])[1], (P[b] - P[a])[0]])
                    P[i] += away / (np.linalg.norm(away) + 1e-12) * (clear - d + .02)
                    moved = True
            for j in range(n):
                if j != i:
                    d = P[i] - P[j]
                    gap = float(np.linalg.norm(d))
                    if gap < min_sep:
                        P[i] += (d / (gap + 1e-12) if gap > 1e-9 else np.array([1.0, 0.0])) * (min_sep - gap) / 2
                        moved = True
        P[pin] = 0
        if not moved:
            break
    out = {v: (float(P[index[v], 0]), float(P[index[v], 1])) for v in names}
    junctions = {key: out.pop(key) for key in keys}
    return out, ring, candidate, junctions


def plot_explore(result: ExploreResult, path=None, *, confirmation: pd.DataFrame | None = None,
                 confirmation_fit: pd.DataFrame | None = None, style: GraphStyle | None = None, **settings):
    """Draw the network; save it to ``path`` if given, else return the figure.

    ``style`` is a ``GraphStyle``; ``settings`` override its fields, so
    ``plot_explore(result, spacing=.8, node_size=1.2, show_flagged=False)`` works
    without building one. With ``confirmation`` / ``confirmation_fit`` (from
    ``open.confirm``), confirmed links get a check mark and targets their R² on
    the new data.
    """
    st = _style(style, settings)
    ns, ew, fs = st.node_size, st.edge_width, st.label_size
    node_r, outcome_r = st.node_r(), OUTCOME_R * ns
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, RegularPolygon, Wedge

    links = _drawn(result.links, st.show_flagged, st.show_flagged_interactions)
    if not len(links):
        links = pd.DataFrame(columns=["depth", "target", "source", "kind", "members", "tier", "pass_rate",
                                      "r2_median"])
    pos, ring, candidate, junctions = _layout(result, links, st)
    outcome = result.settings.get("outcome") or result.nodes.variable.iloc[0]
    xy = np.array(list(pos.values()) + list(junctions.values()))
    pad = .7 + 1.3 * node_r + .25 * fs
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    span = hi - lo
    scale = float(np.clip(9.5 / max(span), .6, .95))       # inches per unit
    points = scale * 72                                    # points per unit
    fig, ax = plt.subplots(figsize=(max(6.5, span[0] * scale + .6), max(5.5, span[1] * scale + 1.9)))
    ax.set_aspect("equal")

    verdict = {}
    if confirmation is not None:
        for c in confirmation.itertuples():
            verdict[(c.depth, c.target, c.source)] = None if pd.isna(c.p) else bool(c.confirmed)

    def edge(a, b, tier_, r2, color, label=None, lw=None):
        label = label if st.edge_labels == "all" or (st.edge_labels == "kept" and tier_ == "kept") else None
        style = "-" if tier_ == "kept" else (0, (4, 2.5))
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color if tier_ == "kept" else _soft(color), ls=style,
                lw=(lw or _width(r2)) * ew, solid_capstyle="round", zorder=1.5 if tier_ == "kept" else 1)
        if label:
            t = .55                                    # nearer the outer end, where edges are spread out
            x, y = b[0] + t * (a[0] - b[0]), b[1] + t * (a[1] - b[1])
            ax.text(x, y, label, fontsize=8 * fs, ha="center", va="center", color=color, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=.92))

    n_folds = int(result.settings.get("folds", 10)) * int(result.settings.get("repeats", 1))

    def label_for(r):
        text = f"{_apa(r.r2_median)} ({_passes(r.pass_rate, n_folds)})"
        return text + (" ✓" if verdict.get((r.depth, r.target, r.source)) else "")

    for r in links.itertuples():
        target = pos[r.target]
        members = list(r.members)
        if r.kind == "unique":
            edge(pos[r.source], target, r.tier, r.r2_median, KEPT, label_for(r))
        elif r.kind == "shared":
            pts = np.array([pos[m] for m in members])
            for i in range(len(pts) - 1):              # band joining the members
                ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=SHARED, alpha=.18, lw=.8 * node_r * points,
                        solid_capstyle="round", zorder=.5)
            for i, m in enumerate(members):           # one edge per member, labelled once
                edge(pos[m], target, r.tier, r.r2_median, SHARED, label_for(r) if i == 0 else None,
                     lw=_width(r.r2_median) * .75)
        else:
            mark = junctions[(int(r.depth), r.target, r.source)]
            for m in members:                         # the pair: always solid
                edge(pos[m], mark, "kept", 0, INTERACTION, lw=1.6)
            edge(mark, target, r.tier, r.r2_median, INTERACTION, label_for(r))
            fill = INTERACTION if r.tier == "kept" else _soft(INTERACTION, .3)
            ax.add_patch(RegularPolygon(mark, numVertices=4, radius=.2 * ns, orientation=0, fc=fill, ec="white",
                                        lw=1.2, zorder=4))
            # name the pair beside the diamond, on the side away from the target
            away = np.array(mark) - np.array(target)
            away = away / (np.linalg.norm(away) + 1e-12)
            normal = np.array([-away[1], away[0]])
            spot = np.array(mark) + (.14 * ns + .2 * fs) * normal + .12 * ns * away
            ax.text(*spot, "·".join(members), fontsize=7.5 * fs, ha="center", va="center", color=INTERACTION,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.9))

    fit = {r.target: r.r2_network for r in result.fit.itertuples()}
    sizes = {r.target: r.network_size for r in result.fit.itertuples()}
    new = ({r.target: r.r2_network for r in confirmation_fit.itertuples()}
           if confirmation_fit is not None else {})
    for v, (x, y) in pos.items():
        is_outcome = v == outcome
        rad = outcome_r if is_outcome else node_r
        grey = v in candidate
        fill = "#f4f5f7" if grey else DEPTH_FILL[min(ring[v], len(DEPTH_FILL) - 1)]
        ax.add_patch(Circle((x, y), rad, fc=fill, ec=MUTED if grey else INK, lw=.9 if grey else 1.2,
                            ls=(0, (2, 1.5)) if grey else "-", zorder=3))
        if v in fit and sizes.get(v, 0) > 0:           # network-fit ring: share filled = held-out R²
            ax.add_patch(Wedge((x, y), rad + .13 * ns, 0, 360, width=.08 * ns, fc="#e9ecef", ec="none",
                               zorder=3))
            share = float(np.clip(fit[v], 0, 1))
            if share > 0:
                ax.add_patch(Wedge((x, y), rad + .13 * ns, 90 - 360 * share, 90, width=.08 * ns, fc=FIT,
                                   ec="none", zorder=3.5))
            text = f"R² {_apa(fit[v])}" + (f"  new {_apa(new[v])}" if v in new else "")
            ax.text(x, y - rad - .22 * ns, text, fontsize=7.5 * fs, ha="center", va="top", color=FIT, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.85))
        size_ = 10.5 if is_outcome else (9 if len(v) <= 4 else max(6, 9 - .45 * (len(v) - 4)))
        ax.text(x, y, v, fontsize=size_ * fs, ha="center", va="center", zorder=6,
                color="white" if is_outcome else (MUTED if grey else INK),
                fontweight="bold" if is_outcome else "normal")

    s = result.settings
    split = f", {s['folds']} folds × {s['repeats']} repeats" if s.get("repeats", 1) > 1 else ""
    ax.set_title(f"{s['outcome']}  ·  {s['engine']}, N = {s['n']}{split}", fontsize=13 * fs, color=INK, pad=12,
                 loc="center")
    handles = [Line2D([], [], color=KEPT, lw=2.5, label=f"kept (≥ {s['keep']:.0%} of folds)")]
    if st.show_flagged:
        handles.append(Line2D([], [], color=_soft(KEPT), lw=2.5, ls=(0, (4, 2.5)),
                              label=f"flagged (≥ {s['flag']:.0%})"))
    kinds = set(links.kind)                       # list only what is drawn
    if "shared" in kinds:
        handles.append(Line2D([], [], color=SHARED, lw=2.5, label="shared group"))
    if "interaction" in kinds:
        handles.append(Line2D([], [], color=INTERACTION, lw=2.5, marker="D", markersize=6, markeredgecolor="white",
                              label="interaction (◆ joins the pair)"))
    if any(sizes.get(v, 0) > 0 for v in pos):
        handles.append(Line2D([], [], color=FIT, lw=0, marker="o", markersize=9, markerfacecolor="white",
                              markeredgewidth=2.5, label="ring: held-out R² from kept links"))
    if candidate:
        handles.append(Line2D([], [], color=MUTED, lw=0, marker="o", markersize=9, markerfacecolor="#f4f5f7",
                              markeredgecolor=MUTED, label="flagged only"))
    for d in sorted({d for v, d in ring.items() if d > 0 and v not in candidate}):
        handles.append(Line2D([], [], lw=0, marker="o", markersize=9, markeredgecolor=INK,
                              markerfacecolor=DEPTH_FILL[min(d, len(DEPTH_FILL) - 1)], label=f"depth {d}"))
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=3, frameon=False,
                       fontsize=8.5 * fs, handlelength=2.6, columnspacing=1.6)
    note = ("Each edge: the outer variable predicts the inner one beyond that target's other candidates.\n"
            "Label: held-out ΔR² (R² lost; gained, for interactions) and folds passed, e.g. .120 (9/10)" +
            (" (✓ confirmed on new data)" if verdict else "") +
            ". Not causal; not a partial correlation.")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()                              # place the footnote just below the legend
    box = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(.5, box.y0 - .012, note, ha="center", va="top", fontsize=8 * fs, color=MUTED,
             linespacing=1.5)
    if path is not None:
        # a tight box keeps the title, legend and footnote inside the image at any label size
        fig.savefig(path, dpi=st.dpi, facecolor="white", bbox_inches="tight", pad_inches=.3)
        plt.close(fig)
        return Path(path)
    return fig


def _soft(hex_color: str, amount: float = .45) -> str:
    """A lighter tint of a colour, for flagged (dashed) edges."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)
