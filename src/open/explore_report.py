"""Tables and graph for exploratory-mode results (C3, C6).

``write_explore`` saves the result tables, settings, a text summary and the
network graph. ``plot_explore`` draws the graph as a radial tree:

- the outcome in the centre, each depth on a wider ring, and every target's
  predictors fanning outwards from it;
- circular nodes, with edges drawn centre to centre behind them and an
  arrowhead at the rim of the variable being predicted;
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
- predictors that are only flagged, and never kept, as grey dashed nodes;
- covariates (``explore(covariates=...)``) as tan nodes: in every model, tested,
  never targets. Controls (``controls=...``) are not drawn; the footnote lists
  them.

Display settings live in ``GraphStyle``: node size, edge thickness, label size,
spacing between nodes (more compact or more spread out), which flagged edges and
nodes to show, edge labels and resolution. Pass a style, or the same settings as
keywords: ``plot_explore(result, "g.png", spacing=.8, node_size=1.2)``.

Relations hidden by ``hide_below`` are left out of the graph but stay in the
tables. Flagged interactions are also left out by default
(``show_flagged_interactions``): in the validation study they were mostly
false. They stay in the tables.

An arrow means "this variable predicts that one on held-out data, beyond the
target's other candidates". It shows the direction of prediction in the
analysis, not of causation (reversed arrows can be predictive too), and it is
not a partial correlation.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from open.explore import ExploreResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from open.multi_outcome import MultiOutcomeResult


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

    Reusing a directory replaces generated files and removes optional artifacts
    absent from this export. Unrelated files are preserved. Preparation happens
    before publication; ordinary publication failures restore prior files.
    """
    return _write_export(result, out_dir, plot=plot, confirmation=confirmation,
                         confirmation_fit=confirmation_fit, style=style, graph_settings=settings)


def write_split(split, out_dir, *, plot: bool = True, style: "GraphStyle | None" = None, **settings) -> Path:
    """Save exploration, confirmation and split settings with overwrite recovery.

    See ``write_explore`` for directory reuse behavior.
    """
    return _write_export(split.exploration, out_dir, plot=plot, confirmation=split.confirmation,
                         confirmation_fit=split.fit, style=style, graph_settings=settings, split=split)


_EXPORT_FILES = (
    "nodes.csv", "links.csv", "pairs.csv", "folds.csv", "noise-bars.csv", "fit.csv",
    "settings.json", "summary.txt", "confirmation.csv", "confirmation-fit.csv",
    "network.png", "split-settings.json",
)


class _ExportRollbackError(OSError):
    """Publication failed and recovery files must remain available."""


def _publish_export(stage: Path, out: Path, managed_files=_EXPORT_FILES) -> None:
    # Check every managed name before changing any destination, including names
    # that this export removes. Never follow a symlink or remove a directory.
    for name in managed_files:
        target = out / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Export destination must be a regular file: {target}")
    backup = stage / ".previous"
    backup.mkdir()
    for name in managed_files:
        if (out / name).exists():
            shutil.copy2(out / name, backup / name)

    changed = []
    try:
        for name in managed_files:
            source, target = stage / name, out / name
            if source.exists():
                source.replace(target)
                changed.append(name)
            elif target.exists():
                target.unlink()
                changed.append(name)
    except OSError as error:
        failures = []
        for name in reversed(changed):
            try:
                if (backup / name).exists():
                    (backup / name).replace(out / name)
                else:
                    (out / name).unlink(missing_ok=True)
            except OSError as restore_error:
                failures.append(f"{name}: {restore_error}")
        if failures:
            raise _ExportRollbackError(
                f"Export failed and could not restore {', '.join(failures)}. "
                f"Recovery backups retained at {backup}"
            ) from error
        raise


def _write_export(result, out_dir, *, plot, confirmation, confirmation_fit,
                  style, graph_settings, split=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    stage = Path(tempfile.mkdtemp(prefix=".open-export-", dir=root))
    retain_stage = False
    try:
        _prepare_export(result, stage, plot=plot, confirmation=confirmation,
                        confirmation_fit=confirmation_fit, style=style,
                        graph_settings=graph_settings, split=split)
        _publish_export(stage, root)
    except _ExportRollbackError:
        retain_stage = True
        raise
    finally:
        # Only this newly generated directory, directly within the resolved
        # output directory, may be recursively cleaned.
        if not retain_stage and stage.parent == root:
            shutil.rmtree(stage)
    return out


def _prepare_export(result, out, *, plot, confirmation, confirmation_fit,
                    style, graph_settings, split):
    nodes = result.nodes.assign(parents=result.nodes.parents.map(";".join))
    nodes.to_csv(out / "nodes.csv", index=False)
    result.links.assign(members=result.links.members.map(";".join)).to_csv(out / "links.csv", index=False)
    result.pairs.to_csv(out / "pairs.csv", index=False)
    result.folds.to_csv(out / "folds.csv", index=False)
    result.bars.to_csv(out / "noise-bars.csv", index=False)
    result.fit.assign(network=result.fit.network.map(";".join)).to_csv(out / "fit.csv", index=False)
    (out / "settings.json").write_text(json.dumps(dict(result.settings, stops=result.stops), indent=2) + "\n",
                                      encoding="utf-8")
    if split is not None:
        (out / "split-settings.json").write_text(json.dumps(split.settings, indent=2) + "\n", encoding="utf-8")
    (out / "summary.txt").write_text((split if split is not None else result).summary() + "\n", encoding="utf-8")
    if confirmation is not None:
        confirmation.to_csv(out / "confirmation.csv", index=False)
    if confirmation_fit is not None:
        confirmation_fit.assign(network=confirmation_fit.network.map(";".join)) \
            .to_csv(out / "confirmation-fit.csv", index=False)
    if plot:
        plot_explore(result, out / "network.png", confirmation=confirmation,
                     confirmation_fit=confirmation_fit, style=style, **graph_settings)


# ---------------------------------------------------------------- graph

NODE_R = 0.42          # node radius (data units)
OUTCOME_R = 0.56
MIN_SEP = 2.9 * NODE_R  # minimum distance between node centres on a ring
RING_STEP = 2.3         # base distance between depth rings

INK, MUTED = "#1f2933", "#6b7785"
KEPT, FLAGGED = "#334155", "#94a3b8"
COVARIATE_FILL, COVARIATE_EDGE = "#f5e6cf", "#9a6b2f"
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
    - ``label_outline``: colour of the solid outline drawn around label text
      (edge labels, fit values, interaction pairs), any matplotlib colour.
    - ``label_outline_width``: thickness of that outline in points; 0 turns it off.
    - ``dpi``: resolution of a saved image.
    """
    node_size: float = 1.0
    edge_width: float = 1.0
    label_size: float = 1.0
    spacing: float = 1.0
    show_flagged: bool = True
    show_flagged_interactions: bool = False
    edge_labels: str | None = "kept"
    label_outline: str = "white"
    label_outline_width: float = 5.0
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
        from matplotlib.colors import is_color_like
        if not is_color_like(self.label_outline):
            raise ValueError("label_outline must be a matplotlib colour")
        w = self.label_outline_width
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not 0 <= w <= 20:
            raise ValueError("label_outline_width must be a number in [0, 20]")
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


def _free_angle(angles) -> float:
    """The direction (radians) farthest from every edge leaving a node; straight down when there is room."""
    down = -math.pi / 2
    if not angles:
        return down
    def gap(a):
        return min(abs((a - b + math.pi) % (2 * math.pi) - math.pi) for b in angles)
    if gap(down) >= math.pi / 3:
        return down
    options = [down + k * math.pi / 12 for k in range(24)]
    return max(options, key=lambda a: (round(gap(a), 3), -abs(a - down)))


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
    """Draw the radial network; return its figure or save to the requested path.

    GraphStyle fields may also be supplied as keyword settings. Confirmation
    tables mark confirmed links and show fit on the independent data.
    """
    return _plot_explore(result, path, confirmation=confirmation, confirmation_fit=confirmation_fit,
                         style=style, **settings)


def _plot_explore(result: ExploreResult, path=None, *, confirmation: pd.DataFrame | None = None,
                  confirmation_fit: pd.DataFrame | None = None, style: GraphStyle | None = None,
                  axes=None, **settings):
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
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon, RegularPolygon, Wedge

    # a solid outline around label text keeps it readable over edges
    halo = ([patheffects.withStroke(linewidth=st.label_outline_width, foreground=st.label_outline)]
            if st.label_outline_width > 0 else [])
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
    if axes is None:
        fig, ax = plt.subplots(figsize=(max(6.5, span[0] * scale + .6), max(5.5, span[1] * scale + 1.9)))
    else:
        fig, ax = axes.figure, axes
    ax.set_aspect("equal")

    verdict = {}
    if confirmation is not None:
        for c in confirmation.itertuples():
            verdict[(c.depth, c.target, c.source)] = None if pd.isna(c.p) else bool(c.confirmed)

    fit = {r.target: r.r2_network for r in result.fit.itertuples()}
    sizes = {r.target: r.network_size for r in result.fit.itertuples()}
    covariates = set(result.settings.get("covariates", ()))

    def rim(v):
        """Distance from a node's centre to where an arrow into it should stop."""
        rad = outcome_r if v == outcome else node_r
        return rad + (.22 * ns if v in fit and sizes.get(v, 0) > 0 else .04 * ns)

    def edge(a, b, tier_, r2, color, label=None, lw=None, head=None):
        """A line from a to b, behind the nodes; with ``head``, an arrowhead that far short of b."""
        label = label if st.edge_labels == "all" or (st.edge_labels == "kept" and tier_ == "kept") else None
        style = "-" if tier_ == "kept" else (0, (4, 2.5))
        shade = color if tier_ == "kept" else _soft(color)
        width = (lw or _width(r2)) * ew
        ax.plot([a[0], b[0]], [a[1], b[1]], color=shade, ls=style,
                lw=width, solid_capstyle="round", zorder=1.5 if tier_ == "kept" else 1)
        if head is not None:
            u = np.array(b, float) - np.array(a, float)
            length = float(np.linalg.norm(u))
            if length > head + .05:
                u /= length
                n = np.array([-u[1], u[0]])
                size = (.15 + .012 * width) * ns
                tip = np.array(b) - u * head
                ax.add_patch(Polygon([tip, tip - u * size + n * size * .55, tip - u * size - n * size * .55],
                                     closed=True, fc=shade, ec="none", zorder=2))
        if label:
            t = .55                                    # nearer the outer end, where edges are spread out
            x, y = b[0] + t * (a[0] - b[0]), b[1] + t * (a[1] - b[1])
            ax.text(x, y, label, fontsize=8 * fs, ha="center", va="center", color=color, zorder=5,
                    path_effects=halo)

    n_folds = int(result.settings.get("folds", 10)) * int(result.settings.get("repeats", 1))

    def label_for(r):
        text = f"{_apa(r.r2_median)} ({_passes(r.pass_rate, n_folds)})"
        return text + (" ✓" if verdict.get((r.depth, r.target, r.source)) else "")

    for r in links.itertuples():
        target = pos[r.target]
        members = list(r.members)
        if r.kind == "unique":
            edge(pos[r.source], target, r.tier, r.r2_median, KEPT, label_for(r), head=rim(r.target))
        elif r.kind == "shared":
            pts = np.array([pos[m] for m in members])
            for i in range(len(pts) - 1):              # band joining the members
                ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], color=SHARED, alpha=.18, lw=.8 * node_r * points,
                        solid_capstyle="round", zorder=.5)
            for i, m in enumerate(members):           # one edge per member, labelled once
                edge(pos[m], target, r.tier, r.r2_median, SHARED, label_for(r) if i == 0 else None,
                     lw=_width(r.r2_median) * .75, head=rim(r.target))
        else:
            mark = junctions[(int(r.depth), r.target, r.source)]
            for m in members:                         # the pair: always solid
                edge(pos[m], mark, "kept", 0, INTERACTION, lw=1.6)
            edge(mark, target, r.tier, r.r2_median, INTERACTION, label_for(r), head=rim(r.target))
            fill = INTERACTION if r.tier == "kept" else _soft(INTERACTION, .3)
            ax.add_patch(RegularPolygon(mark, numVertices=4, radius=.2 * ns, orientation=0, fc=fill, ec="white",
                                        lw=1.2, zorder=4))
            # name the pair beside the diamond, on the side away from the target
            away = np.array(mark) - np.array(target)
            away = away / (np.linalg.norm(away) + 1e-12)
            normal = np.array([-away[1], away[0]])
            spot = np.array(mark) + (.14 * ns + .2 * fs) * normal + .12 * ns * away
            ax.text(*spot, "·".join(members), fontsize=7.5 * fs, ha="center", va="center", color=INTERACTION,
                    zorder=5, path_effects=halo)

    neighbours = {}                               # where each node's edges leave it, to place its fit label
    for r in links.itertuples():
        far = [junctions[(int(r.depth), r.target, r.source)]] if r.kind == "interaction" else \
            [pos[m] for m in r.members]
        neighbours.setdefault(r.target, []).extend(far)
        for m in r.members:
            neighbours.setdefault(m, []).append(junctions.get((int(r.depth), r.target, r.source), pos[r.target]))
    new = ({r.target: r.r2_network for r in confirmation_fit.itertuples()}
           if confirmation_fit is not None else {})
    for v, (x, y) in pos.items():
        is_outcome = v == outcome
        rad = outcome_r if is_outcome else node_r
        covariate = v in covariates
        grey = v in candidate and not covariate
        fill = (COVARIATE_FILL if covariate else "#f4f5f7" if grey
                else DEPTH_FILL[min(ring[v], len(DEPTH_FILL) - 1)])
        ax.add_patch(Circle((x, y), rad, fc=fill, ec=COVARIATE_EDGE if covariate else MUTED if grey else INK,
                            lw=.9 if grey else 1.2, ls=(0, (2, 1.5)) if grey else "-", zorder=3))
        if v in fit and sizes.get(v, 0) > 0:           # network-fit ring: share filled = held-out R²
            ax.add_patch(Wedge((x, y), rad + .13 * ns, 0, 360, width=.08 * ns, fc="#e9ecef", ec="none",
                               zorder=3))
            share = float(np.clip(fit[v], 0, 1))
            if share > 0:
                ax.add_patch(Wedge((x, y), rad + .13 * ns, 90 - 360 * share, 90, width=.08 * ns, fc=FIT,
                                   ec="none", zorder=3.5))
            text = f"R² {_apa(fit[v])}" + (f"  new {_apa(new[v])}" if v in new else "")
            # on the side of the node with the most room between its edges (below if free)
            angle = _free_angle([math.atan2(q[1] - y, q[0] - x) for q in neighbours.get(v, [])])
            dist = rad + .2 * ns + .1 * fs
            ax.text(x + dist * math.cos(angle), y + dist * math.sin(angle), text, fontsize=7.5 * fs,
                    ha="center" if abs(math.cos(angle)) < .38 else ("left" if math.cos(angle) > 0 else "right"),
                    va="center" if abs(math.sin(angle)) < .38 else ("bottom" if math.sin(angle) > 0 else "top"),
                    color=FIT, zorder=5, path_effects=halo)
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
    if candidate - covariates:
        handles.append(Line2D([], [], color=MUTED, lw=0, marker="o", markersize=9, markerfacecolor="#f4f5f7",
                              markeredgecolor=MUTED, label="flagged only"))
    if covariates & set(pos):
        handles.append(Line2D([], [], lw=0, marker="o", markersize=9, markerfacecolor=COVARIATE_FILL,
                              markeredgecolor=COVARIATE_EDGE, label="covariate (every model; never a target)"))
    for d in sorted({d for v, d in ring.items() if d > 0 and v not in candidate}):
        handles.append(Line2D([], [], lw=0, marker="o", markersize=9, markeredgecolor=INK,
                              markerfacecolor=DEPTH_FILL[min(d, len(DEPTH_FILL) - 1)], label=f"depth {d}"))
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.01), ncol=3, frameon=False,
                       fontsize=8.5 * fs, handlelength=2.6, columnspacing=1.6)
    note = ("Arrows point from a predictor to the variable it predicts on held-out data, beyond that "
            "variable's other candidates.\nThey show the direction of prediction in this analysis, not of "
            "causation, and are not partial correlations.\n"
            "Label: held-out ΔR² (R² lost; gained, for interactions) and folds passed, e.g. .120 (9/10)" +
            (" (✓ confirmed on new data)" if verdict else "") + ".")
    if s.get("controls"):
        note += f"\nAll links adjusted for: {', '.join(s['controls'])}."
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    if axes is not None:
        return fig
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


def plot_outcomes(result: MultiOutcomeResult, layout: Literal["facets", "combined"] = "facets",
                  path: str | Path | None = None, style: GraphStyle | None = None) -> Figure | Path:
    """Plot separate outcome networks as radial facets or an outcome-colored overlay."""
    if layout not in ("facets", "combined"):
        raise ValueError("layout must be 'facets' or 'combined'")
    if not result.by_outcome:
        raise ValueError("at least one outcome network is required")
    st = _style(style, {})
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if layout == "facets":
        count = len(result.by_outcome)
        columns = min(count, 2)
        rows = math.ceil(count / columns)
        panel_style = replace(st, spacing=st.spacing * 1.3)
        widths, heights = [], []
        for split in result.by_outcome.values():
            drawn = _drawn(split.exploration.links, st.show_flagged, st.show_flagged_interactions)
            positions, _, _, junctions = _layout(split.exploration, drawn, panel_style)
            points = np.array(list(positions.values()) + list(junctions.values()))
            span = np.ptp(points, axis=0) + 2 * (.7 + 1.3 * st.node_r() + .25 * st.label_size)
            scale = float(np.clip(9.5 / max(span), .6, .95))
            widths.append(max(7.5, span[0] * scale + 1.5))
            heights.append(max(4.5, span[1] * scale + 2))
        fig, axes = plt.subplots(rows, columns,
                                 figsize=(max(widths) * columns, max(heights) * rows + .7), squeeze=False)
        for ax, split in zip(axes.flat, result.by_outcome.values()):
            _plot_explore(split.exploration, confirmation=split.confirmation,
                          confirmation_fit=split.fit, style=panel_style, axes=ax)
        for ax in list(axes.flat)[count:]:
            fig.delaxes(ax)
        fig.suptitle("Separate predictive outcome networks", fontsize=15 * st.label_size)
        fig.subplots_adjust(top=.91, bottom=.14, hspace=.42, wspace=.2)
        fig.text(.5, .035, "Each panel has its own root and depth ordering. Arrows show prediction, not causation.\n"
                 "Edge labels: held-out ΔR² and folds passed; ✓ marks confirmation. Node rings show held-out R².",
                 ha="center", fontsize=9 * st.label_size, color=MUTED)
    else:
        fig = _plot_outcome_overlay(result, st)
    if path is not None:
        fig.savefig(path, dpi=st.dpi, facecolor="white", bbox_inches="tight", pad_inches=.3)
        plt.close(fig)
        return Path(path)
    return fig


def _plot_outcome_overlay(result, st):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    outcomes = tuple(result.by_outcome)
    cmap = plt.get_cmap("tab10" if len(outcomes) <= 10 else "tab20" if len(outcomes) <= 20 else "hsv")
    colors = {outcome: cmap(i if len(outcomes) <= 20 else i / len(outcomes))
              for i, outcome in enumerate(outcomes)}
    variables = dict.fromkeys(outcomes)
    relations, junctions = [], {}
    for outcome, split in result.by_outcome.items():
        links = _drawn(split.exploration.links, st.show_flagged, st.show_flagged_interactions)
        for index, link in enumerate(links.itertuples()):
            variables.setdefault(link.target, None)
            for member in link.members:
                variables.setdefault(member, None)
            if link.kind == "unique":
                relations.append((link.members[0], link.target, outcome, link.tier, link.r2_median, True))
            else:
                # Relation nodes preserve group/interaction semantics in the overlay.
                key = (outcome, index, link.kind)
                junctions[key] = (link.kind, tuple(link.members), link.target)
                for member in link.members:
                    relations.append((member, key, outcome, "kept", 0, False))
                relations.append((key, link.target, outcome, link.tier, link.r2_median, True))
    names = tuple(variables)
    radius = max(2.5, .42 * len(names)) * st.spacing
    positions = {name: np.array([radius * math.cos(math.pi / 2 + 2 * math.pi * i / len(names)),
                                 radius * math.sin(math.pi / 2 + 2 * math.pi * i / len(names))])
                 for i, name in enumerate(names)}
    for index, (key, (_, members, target)) in enumerate(junctions.items()):
        points = [positions[name] for name in (*members, target)]
        center = np.mean(points, axis=0)
        # Distinct relation nodes stay separate even for the same members/target.
        positions[key] = center + .25 * np.array([math.cos(index * 2.4), math.sin(index * 2.4)])
    fig, ax = plt.subplots(figsize=(max(8, radius * 1.8), max(7, radius * 1.8)))
    groups = {}
    for relation in relations:
        a, b = relation[:2]
        pair = frozenset((a, b))
        groups.setdefault(pair, []).append(relation)
    for group in groups.values():
        canonical = group[0][0]
        for index, (a, b, outcome, tier, r2, head) in enumerate(group):
            curvature = (index - (len(group) - 1) / 2) * .28
            if a != canonical:
                curvature *= -1
            arrow = FancyArrowPatch(
                positions[a], positions[b], arrowstyle="-|>" if head else "-",
                connectionstyle=f"arc3,rad={curvature}", color=colors[outcome],
                linestyle="-" if tier == "kept" else "--", linewidth=_width(r2) * st.edge_width,
                mutation_scale=14 * st.node_size, shrinkA=22 * st.node_size, shrinkB=22 * st.node_size,
                alpha=.9, zorder=1,
            )
            arrow.set_gid(f"{outcome}:{a}->{b}")
            ax.add_patch(arrow)
            if head and (st.edge_labels == "all" or (st.edge_labels == "kept" and tier == "kept")):
                delta = positions[b] - positions[a]
                normal = np.array([-delta[1], delta[0]]) / (np.linalg.norm(delta) + 1e-12)
                center = (positions[a] + positions[b]) / 2 - normal * curvature * np.linalg.norm(delta) / 2
                ax.text(*center, _apa(r2), fontsize=8 * st.label_size, color=colors[outcome],
                        ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    for name in names:
        root = name in colors
        ax.scatter(*positions[name], s=(1100 if root else 850) * st.node_size ** 2,
                   color=colors[name] if root else "#eef2f6", edgecolor=INK, zorder=3)
        ax.text(*positions[name], name, ha="center", va="center", color="white" if root else INK,
                fontsize=9 * st.label_size, fontweight="bold" if root else "normal", zorder=4)
    for key, (kind, members, _) in junctions.items():
        ax.scatter(*positions[key], marker="D" if kind == "interaction" else "s", s=180 * st.node_size ** 2,
                   color=colors[key[0]], zorder=3)
        label = (" × " if kind == "interaction" else " + ").join(members) + f" ({kind})"
        ax.annotate(label, positions[key], xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=8 * st.label_size, color=colors[key[0]])
    handles = [Line2D([], [], color=colors[outcome], lw=2.5, label=outcome) for outcome in outcomes]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.03), ncol=min(4, len(outcomes)),
              frameon=False)
    ax.set_title("Overlay of separate predictive analyses", fontsize=14 * st.label_size, pad=22)
    fig.text(.5, .025, "Colors identify each outcome's analysis; shared variables are drawn once.\n"
             "Solid: kept; dashed: flagged. Labels: held-out ΔR². Arrows show prediction, not causation.",
             ha="center", fontsize=9 * st.label_size, color=MUTED)
    ax.set_aspect("equal")
    ax.margins(.3)
    ax.axis("off")
    fig.subplots_adjust(bottom=.2, top=.9)
    return fig


_MULTI_EXPORT_FILES = ("outcomes-confirmation.csv", "outcomes-fit.csv", "outcomes-settings.json",
                       "outcomes-summary.txt", "outcomes-network.png")


def _check_output_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and
                            getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError(f"Export directory must not be a symlink or reparse point: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Export destination must be a directory: {path}")


def save_multi_outcome_analysis(result: MultiOutcomeResult, out_dir: str | Path, *,
                                 plot: Literal[False, "facets", "combined"] = False,
                                 style: GraphStyle | None = None) -> Path:
    """Save per-outcome analyses and aggregate tables, with an optional batch plot.

    Publication recovers per outcome directory and per aggregate file set.
    A failure can leave earlier outcome directories updated. Unrelated files
    and outcome directories from prior batches are preserved.
    """
    if plot is not False and plot not in ("facets", "combined"):
        raise ValueError("plot must be False, 'facets', or 'combined'")
    if not result.by_outcome:
        raise ValueError("at least one outcome network is required")
    out = Path(out_dir)
    _check_output_directory(out)
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    _check_output_directory(root / "outcomes")
    mapping = {}
    for index, outcome in enumerate(result.by_outcome, 1):
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", outcome).strip("-_")[:32] or "outcome"
        folder = f"outcomes/{index:04d}-{slug}"
        _check_output_directory(root / folder)
        mapping[outcome] = folder
    for name in _MULTI_EXPORT_FILES:
        target = root / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Export destination must be a regular file: {target}")
    stage = Path(tempfile.mkdtemp(prefix=".open-multi-", dir=root))
    retain_stage = False
    try:
        result.confirmation.to_csv(stage / "outcomes-confirmation.csv", index=False)
        fit = result.fit
        if "network" in fit:
            fit = fit.assign(network=fit.network.map(";".join))
        fit.to_csv(stage / "outcomes-fit.csv", index=False)
        (stage / "outcomes-settings.json").write_text(
            json.dumps({**result.settings, "outcome_folders": mapping}, indent=2) + "\n", encoding="utf-8")
        (stage / "outcomes-summary.txt").write_text(result.summary() + "\n", encoding="utf-8")
        if plot:
            plot_outcomes(result, layout=plot, path=stage / "outcomes-network.png", style=style)
        for outcome, split in result.by_outcome.items():
            write_split(split, root / mapping[outcome], plot=False)
        _publish_export(stage, root, managed_files=_MULTI_EXPORT_FILES)
    except _ExportRollbackError:
        retain_stage = True
        raise
    finally:
        if not retain_stage and stage.parent == root:
            shutil.rmtree(stage)
    return out
