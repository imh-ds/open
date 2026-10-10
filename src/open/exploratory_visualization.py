"""Validate and draw observed relations without imposing consensus inclusion."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import textwrap


@dataclass(frozen=True)
class ExploratoryNode:
    name: str
    display_label: str
    depth: int
    count: int
    denominator: int
    probability: float
    depth_probabilities: tuple[tuple[int, float], ...]
    display_only: bool


@dataclass(frozen=True)
class ExploratoryRelation:
    source: str | None
    target: str
    relation: str
    members: tuple[str, ...]
    screened_by: tuple[str, ...]
    count: int
    denominator: int
    probability: float
    eligible_count: int
    eligibility_unknown_count: int
    conditional_probability: float | None
    consensus: bool
    strength_count: int
    raw_median: float | None
    baseline_relative_count: int
    baseline_relative_median: float | None
    strength_scope: str
    gain_relative_count: int
    gain_relative_median: float | None


@dataclass(frozen=True)
class ExploratoryData:
    root: str
    max_depth: int
    completed_runs: int
    pi_min: float
    show_screened: bool
    nodes: tuple[ExploratoryNode, ...]
    edges: tuple[ExploratoryRelation, ...]
    groups: tuple[ExploratoryRelation, ...]
    view: str = "exploratory"


def load_exploratory_payload(payload, legacy):
    """Require coherent counts and finite measurements in the versioned addon."""
    from open.visualization import _integer, _name, _number, _object, _ratio
    file = "analysis_metadata.json"
    field = "exploration"

    def fail(detail):
        raise ValueError(f"{file}: {field}: {detail}")

    def integer(value, minimum=0):
        return _integer(value, file, field, minimum=minimum)

    def number(value, optional=False, probability=False):
        if optional and value is None:
            return None
        return _number(value, file, field, probability=probability)

    def name(value):
        return _name(value, file, field)

    def names(value):
        if not isinstance(value, list):
            fail("expected an array of names")
        result = tuple(name(item) for item in value)
        if len(set(result)) != len(result):
            fail("duplicate member")
        return result

    def rows(key):
        value = payload.get(key)
        if not isinstance(value, list):
            fail(f"{key} must be an array")
        return [_object(row, file, field) for row in value]

    def frequency(row):
        count = integer(row.get("count"))
        denominator = integer(row.get("denominator"), 1)
        probability = number(row.get("probability"), probability=True)
        _ratio(file, field, probability, count, denominator, legacy.completed_runs)
        return count, denominator, probability

    payload = _object(payload, file, field)
    if integer(payload.get("schema_version")) != 1:
        fail("unsupported schema version")
    if integer(payload.get("denominator"), 1) != legacy.completed_runs:
        fail("denominator differs from completed runs")
    name(payload.get("strength_definition"))
    nodes = []
    for row in rows("nodes"):
        node_name = name(row.get("name"))
        depth = integer(row.get("depth"))
        count, denominator, probability = frequency(row)
        display_only = row.get("display_only")
        if type(display_only) is not bool or display_only != (count == 0):
            fail("zero-count nodes must be display-only anchors")
        if depth > legacy.max_depth or (depth == 0) != (node_name == legacy.root):
            fail("invalid node depth")
        raw_depths = row.get("depth_probabilities")
        if not isinstance(raw_depths, list) or any(
                not isinstance(pair, list) or len(pair) != 2 for pair in raw_depths):
            fail("invalid depth probabilities")
        depths = tuple((integer(d), number(p, probability=True)) for d, p in raw_depths)
        if (len({d for d, _ in depths}) != len(depths)
                or any(d > legacy.max_depth or p <= 0 for d, p in depths)
                or not math.isclose(sum(p for _, p in depths), probability, abs_tol=1e-9)):
            fail("depth probabilities disagree with node counts")
        if depths and depth != min(depths, key=lambda pair: (-pair[1], pair[0]))[0]:
            fail("node depth must be its modal observed depth")
        if node_name == legacy.root and (count != denominator or depths != ((0, 1.),)):
            fail("root must appear at depth zero in every completed run")
        nodes.append(ExploratoryNode(node_name, name(row.get("display_label")), depth,
                                    count, denominator, probability, depths, display_only))
    node_names = {node.name for node in nodes}
    if len(node_names) != len(nodes) or legacy.root not in node_names:
        fail("duplicate nodes or missing root")

    def relations(key):
        result = []
        identities = set()
        for row in rows(key):
            group = key == "groups"
            target = name(row.get("target"))
            source = None if group else name(row.get("source"))
            members = names(row.get("members")) if group else ()
            screeners = () if group else names(row.get("screened_by"))
            relation = "SHARED_GROUP" if group else name(row.get("relation"))
            if not group and relation not in (
                    "PROXIMAL_UNIQUE", "SCREENED_BY_PARTNER", "SCREENED_BY_SET"):
                fail("unsupported relation")
            endpoints = set(members) if group else {source}
            identity = (source, target, relation, members)
            if (target not in node_names or not endpoints.issubset(node_names)
                    or target in endpoints or identity in identities
                    or (group and (len(members) < 2 or members != tuple(sorted(members))))):
                fail("invalid or duplicate relation endpoints")
            identities.add(identity)
            count, denominator, probability = frequency(row)
            eligible = integer(row.get("eligible_count"))
            unknown = integer(row.get("eligibility_unknown_count"))
            conditional = number(row.get("conditional_probability"), True, True)
            expected = count / eligible if eligible and not unknown else None
            if (count == 0 or eligible + unknown > denominator or count > eligible + unknown
                    or (expected is None) != (conditional is None)
                    or (expected is not None and not math.isclose(expected, conditional, abs_tol=1e-9))):
                fail("inconsistent eligibility or conditional probability")
            consensus = row.get("consensus")
            if type(consensus) is not bool or consensus != (probability >= legacy.pi_min):
                fail("consensus flag disagrees with frequency")
            strength_count = integer(row.get("strength_count"))
            relative_count = integer(row.get("baseline_relative_count"))
            gain_count = integer(row.get("gain_relative_count"))
            gain = number(row.get("gain_relative_median"), True)
            raw = number(row.get("raw_median"), True)
            relative = number(row.get("baseline_relative_median"), True)
            scope = name(row.get("strength_scope"))
            expected_scope = "selected joint ablations" if group else "all scored eligible local fits"
            if (strength_count > (count if group else eligible + unknown)
                    or relative_count > strength_count
                    or gain_count > strength_count or (gain_count == 0) != (gain is None)
                    or (strength_count == 0) != (raw is None)
                    or (relative_count == 0) != (relative is None) or scope != expected_scope):
                fail("inconsistent strength measurements or scope")
            result.append(ExploratoryRelation(source, target, relation, members, screeners,
                count, denominator, probability, eligible, unknown, conditional, consensus,
                strength_count, raw, relative_count, relative, scope,
                gain_count, gain))
        return tuple(result)

    return ExploratoryData(legacy.root, legacy.max_depth, legacy.completed_runs,
                           legacy.pi_min, legacy.show_screened, tuple(nodes),
                           relations("edges"), relations("groups"))


def plot_exploratory_network(data, *, output_path=None, min_frequency=0.):
    """Draw pooled arrows with fixed strength scale and explicit evidence labels."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyArrowPatch
    except ImportError as exc:
        raise ImportError("Install OPEN's visualization extra: pip install open[visualization]") from exc
    edges = tuple(e for e in data.edges if e.probability >= min_frequency
                  and (data.show_screened or e.relation == "PROXIMAL_UNIQUE"))
    groups = tuple(g for g in data.groups if g.probability >= min_frequency)
    relations = edges + groups
    figure = Figure(figsize=(17, max(7, 2.2 + .63 * len(relations))))
    FigureCanvasAgg(figure)
    axes, details = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.2]})
    figure.subplots_adjust(left=.025, right=.99, top=.9, bottom=.12, wspace=.1)
    axes.axis("off")
    details.axis("off")
    axes.set_aspect("equal")
    axes.set_title("Exploratory predictive relations", fontsize=15, pad=22)
    details.set_title("Selection frequency and predictive strength", fontsize=14, pad=22)
    labels = {node.name: node.display_label for node in data.nodes}
    positions = {data.root: (0., 0.)}
    layers = {}
    for node in data.nodes:
        if node.name != data.root:
            layers.setdefault(node.depth, []).append(node)
    for depth, layer in sorted(layers.items()):
        for index, node in enumerate(sorted(layer, key=lambda n: n.name)):
            angle = depth * 2.4 + 2 * math.pi * index / len(layer)
            positions[node.name] = (depth * math.cos(angle), depth * math.sin(angle))

    def width(relation):
        return .8 + 3.2 * math.sqrt(min(1., max(0., relation.baseline_relative_median or 0.)))

    def opacity(relation):
        return .25 + .75 * relation.probability

    def color(relation):
        if relation.relation == "SHARED_GROUP":
            return "#9b4f16" if relation.consensus else "#bf844c"
        if relation.relation.startswith("SCREENED"):
            return "#656d78"
        return "#174a8b" if relation.consensus else "#5a8cbc"

    pair_counts = {}
    for index, relation in enumerate(relations):
        identifier = f"E{index + 1}"
        target = positions[relation.target]
        group = relation.relation == "SHARED_GROUP"
        if group:
            points = [positions[n] for n in relation.members]
            start = (sum(x for x, _ in points) / len(points),
                     sum(y for _, y in points) / len(points))
            if math.dist(start, target) < .4:
                start = (target[0] + .65, target[1] + .65)
            for point in points:
                axes.plot([point[0], start[0]], [point[1], start[1]], linestyle=":",
                          color=color(relation), alpha=opacity(relation), linewidth=1.)
            axes.text(*start, "joint", fontsize=8, color=color(relation),
                      ha="center", bbox=dict(facecolor="white", edgecolor=color(relation), pad=2))
            source_label = "{" + ", ".join(labels[n] for n in relation.members) + "}"
        else:
            start = positions[relation.source]
            source_label = labels[relation.source]
        # Opposite directions curve on opposite sides of the same chord.
        pair = (relation.source, relation.target)
        curvature = .17 + .13 * pair_counts.get(pair, 0)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        arrow = FancyArrowPatch(start, target, arrowstyle="-|>", mutation_scale=14,
            shrinkA=25 if not group else 12, shrinkB=28, connectionstyle=f"arc3,rad={curvature}",
            linewidth=width(relation), alpha=opacity(relation), color=color(relation),
            linestyle="--" if relation.relation.startswith("SCREENED") else "-", zorder=2)
        axes.add_patch(arrow)
        midpoint = ((start[0] + target[0]) / 2 + curvature * (target[1] - start[1]) / 2,
                    (start[1] + target[1]) / 2 - curvature * (target[0] - start[0]) / 2)
        axes.text(*midpoint, identifier, fontsize=8, ha="center", color=color(relation),
                  bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=5)
        conditional = (f"{relation.count}/{relation.eligible_count} eligible"
                       if relation.conditional_probability is not None else
                       f"eligibility incomplete ({relation.eligibility_unknown_count} unknown)")
        baseline = (f"{relation.baseline_relative_median:+.1%} baseline"
                    if relation.baseline_relative_median is not None else "baseline strength unavailable")
        raw = f"{relation.raw_median:+.4g}" if relation.raw_median is not None else "unavailable"
        kind = ("joint shared" if group else "unique" if relation.relation == "PROXIMAL_UNIQUE"
                else "screened: " + ", ".join(relation.screened_by))
        text = (f"{identifier}  {source_label} → {labels[relation.target]} [{kind}]\n"
                f"{relation.count}/{relation.denominator} overall; {conditional}"
                + ("; consensus" if relation.consensus else "") + "\n"
                f"Median ΔMSE {raw}; {baseline}; n={relation.strength_count}/{relation.baseline_relative_count}"
                + (" (selected joint fits)" if group else " (all scored fits)"))
        details.text(0, 1 - index / max(len(relations), 1), text,
                     transform=details.transAxes, fontsize=9, va="top", linespacing=1.4,
                     color=color(relation))
    if not relations:
        details.text(0, .9, "No observed relations pass this display filter.", fontsize=11)
    for node in data.nodes:
        label = textwrap.fill(node.display_label, 17)
        if node.display_only:
            label += "\n(display-only source)"
        axes.text(*positions[node.name], label,
                  ha="center", va="center", fontsize=10, zorder=6,
                  color="white" if node.name == data.root else "#183153",
                  bbox=dict(boxstyle="round,pad=.45", facecolor="#183153" if node.name == data.root
                            else "#f4f7fb", edgecolor="#183153",
                            linestyle=":" if node.display_only else "-"))
    # Fit the actual anchors; symmetric depth rings waste space when one deep
    # upstream node occupies only one side of the pooled graph.
    axes.set_xlim(min(x for x, _ in positions.values()) - .7,
                  max(x for x, _ in positions.values()) + .7)
    axes.set_ylim(min(y for _, y in positions.values()) - .7,
                  max(y for _, y in positions.values()) + .7)
    figure.text(.035, .045,
        f"Opacity = 0.25 + 0.75 × overall frequency. Dark blue highlights ≥{data.pi_min:.0%} consensus.\n"
        "Width = 0.8 + 3.2 × √(positive median ΔMSE / baseline), capped at 100%; signed scores remain above.\n"
        "Pooled arrows and modal node positions are display-only: not a single fitted network or evidence of pooled paths.\n"
        "Dashed arrows denote screening; dotted connectors join a shared group. n = raw / baseline-relative score counts.\n"
        "Arrows express prediction, not causation. Display-only source anchors were not selected nodes.",
        fontsize=9, color="#39485b")
    if output_path is not None:
        figure.savefig(output_path, dpi=160)
    return figure
