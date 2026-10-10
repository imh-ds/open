"""Read saved OPEN consensus artifacts for optional visualization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class VisualizationNode:
    name: str
    display_label: str
    depth: int
    selection_probability: float
    depth_probabilities: tuple[tuple[int, float], ...]
    measurement_overlap: bool


@dataclass(frozen=True)
class VisualizationEdge:
    source: str
    target: str
    relation: str
    broad_norm: float | None
    unique_norm: float | None
    stability: float
    screened_by: tuple[str, ...]
    evaluated_probability: float
    target_reached_probability: float
    stability_numerator: int
    stability_denominator: int
    evaluation_numerator: int
    evaluation_denominator: int
    target_reached_numerator: int
    target_reached_denominator: int


@dataclass(frozen=True)
class VisualizationGroup:
    group_id: str
    target: str
    members: tuple[str, ...]
    raw_mean: float | None
    raw_median: float | None
    raw_count: int
    normalized_magnitude: float | None
    normalized_median: float | None
    normalized_count: int
    stability: float
    numerator: int
    denominator: int


@dataclass(frozen=True)
class VisualizationPair:
    target: str
    node_a: str
    node_b: str
    probability: float
    numerator: int
    denominator: int


@dataclass(frozen=True)
class VisualizationData:
    root: str
    max_depth: int
    completed_runs: int
    tau_b: float
    pi_min: float
    show_screened: bool
    show_unstable: bool
    distance_mode: str
    edge_width: str
    edge_opacity: str
    nodes: tuple[VisualizationNode, ...]
    edges: tuple[VisualizationEdge, ...]
    groups: tuple[VisualizationGroup, ...]
    measurement_overlap: Mapping[str, bool]
    shared_comembership: tuple[VisualizationPair, ...]
    view: str = "consensus"


def _invalid(filename: str, field: str, detail: str) -> ValueError:
    return ValueError(f"{filename}: {field}: {detail}")


def _object(value: object, filename: str, field: str) -> dict:
    if not isinstance(value, dict):
        raise _invalid(filename, field, "expected an object")
    return value


def _field(row: dict, filename: str, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise _invalid(filename, field, "missing value")
    return value


def _name(value: object, filename: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(filename, field, "expected a nonempty string")
    return value


def _number(value: object, filename: str, field: str, *, optional=False,
            probability=False) -> float | None:
    if value == "" and optional:
        return None
    if isinstance(value, bool):
        raise _invalid(filename, field, "expected a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid(filename, field, "expected a finite number") from exc
    if not math.isfinite(result) or (probability and not 0 <= result <= 1):
        raise _invalid(filename, field, "expected a finite probability" if probability
                       else "expected a finite number")
    return result


def _integer(value: object, filename: str, field: str, *, minimum=0) -> int:
    if isinstance(value, bool):
        raise _invalid(filename, field, "expected an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid(filename, field, "expected an integer") from exc
    if str(result) != str(value) or result < minimum:
        raise _invalid(filename, field, f"expected an integer >= {minimum}")
    return result


def _ratio(filename: str, field: str, probability: float, numerator: int,
           denominator: int, completed: int) -> None:
    if denominator != completed or numerator > denominator:
        raise _invalid(filename, field, "inconsistent numerator or denominator")
    if not math.isclose(probability, numerator / denominator, abs_tol=1e-9):
        raise _invalid(filename, field, "probability disagrees with counts")


def _members(value: str, filename: str, field: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise _invalid(filename, field, "expected a JSON array of names") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) or not item.strip()
                                           for item in parsed):
        raise _invalid(filename, field, "expected a JSON array of names")
    if len(set(parsed)) != len(parsed):
        raise _invalid(filename, field, "duplicate member")
    return tuple(parsed)


def _csv_rows(directory: Path, filename: str, required: tuple[str, ...],
              *, max_depth: int | None = None) -> list[dict]:
    path = directory / filename
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        for field in required:
            if field not in headers:
                raise _invalid(filename, field, "missing required header")
        if max_depth is not None:
            for field in headers:
                if field.startswith("depth_probability_") and field not in required:
                    raise _invalid(filename, field,
                                   f"unexpected depth column for max_depth {max_depth}")
        rows = list(reader)
    for index, row in enumerate(rows, 2):
        if None in row:
            raise _invalid(filename, f"row {index}", "too many columns")
    return rows


def _configuration(payload: dict) -> tuple[str, int, int, float, float, dict]:
    file = "analysis_metadata.json"
    config = _object(payload.get("config"), file, "config")
    analysis = _object(config.get("analysis"), file, "config.analysis")
    thresholds = _object(config.get("thresholds"), file, "config.thresholds")
    display = _object(config.get("visualization"), file, "config.visualization")
    counts = _object(payload.get("batch_counts"), file, "batch_counts")
    root = _name(analysis.get("root"), file, "config.analysis.root")
    max_depth = _integer(analysis.get("max_depth"), file,
                         "config.analysis.max_depth", minimum=1)
    completed = _integer(counts.get("completed_runs"), file,
                         "batch_counts.completed_runs", minimum=1)
    tau_b = _number(thresholds.get("broad_relative"), file,
                    "config.thresholds.broad_relative", probability=True)
    pi_min = _number(thresholds.get("stability"), file,
                     "config.thresholds.stability", probability=True)
    if type(display.get("show_unstable")) is not bool:
        raise _invalid(file, "config.visualization.show_unstable", "expected boolean")
    supported = {"show_screened": True,
                 "distance_mode": "visual_strength", "edge_width": "unique",
                 "edge_opacity": "stability"}
    for field, expected in supported.items():
        if type(display.get(field)) is not type(expected) or display.get(field) != expected:
            raise _invalid(file, f"config.visualization.{field}",
                           f"unsupported value; expected {expected!r}")
    return root, max_depth, completed, tau_b, pi_min, display


def load_visualization_data(output_dir: str | Path, *, view: str | None = None):
    """Load the six P5.1 files without reconstructing analytical objects."""
    if view not in (None, "exploratory", "consensus", "layer_consensus"):
        raise ValueError("view must be exploratory, consensus or layer_consensus")
    directory = Path(output_dir)
    metadata_path = directory / "analysis_metadata.json"
    if metadata_path.is_file():
        header = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if isinstance(header, dict) and header.get("method") == "layer_consensus":
            if view not in (None, "layer_consensus"):
                raise ValueError("view does not match layer_consensus method")
            from open.layer_io import load_layer_bundle
            return load_layer_bundle(directory)
    if view == "layer_consensus":
        raise ValueError("view does not match whole_pipeline method")
    files = ("nodes.csv", "edges.csv", "shared_groups.csv",
             "shared_comembership.csv", "local_targets.csv", "analysis_metadata.json")
    for filename in files:
        if not (directory / filename).is_file():
            raise FileNotFoundError(directory / filename)
    try:
        payload = json.loads((directory / files[-1]).read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise _invalid(files[-1], "JSON", str(exc)) from exc
    payload = _object(payload, files[-1], "root")
    root, max_depth, completed, tau_b, pi_min, display = _configuration(payload)
    overlap = _object(payload.get("measurement_overlap"), files[-1],
                      "measurement_overlap")
    for name, value in overlap.items():
        _name(name, files[-1], "measurement_overlap key")
        if type(value) is not bool:
            raise _invalid(files[-1], f"measurement_overlap.{name}", "expected boolean")

    node_file = "nodes.csv"
    node_fields = ("node", "display_label", "depth", "selection_probability",
                   "selection_numerator", "selection_denominator")
    depth_fields = tuple(f"depth_probability_{depth}" for depth in range(max_depth + 1))
    node_rows = _csv_rows(directory, node_file, node_fields + depth_fields,
                          max_depth=max_depth)
    root_count = sum(row.get("node") == root for row in node_rows)
    if root_count != 1:
        detail = "missing" if root_count == 0 else "duplicate"
        raise _invalid(node_file, "root", f"{detail} configured root {root!r}")
    nodes = []
    node_names = set()
    for row in node_rows:
        name = _name(_field(row, node_file, "node"), node_file, "node")
        if name in node_names:
            raise _invalid(node_file, "node", f"duplicate identity {name!r}")
        node_names.add(name)
        depth = _integer(_field(row, node_file, "depth"), node_file, "depth")
        if depth > max_depth or (name == root and depth != 0) or (name != root and depth == 0):
            raise _invalid(node_file, "depth", f"unexpected depth for {name!r}")
        probability = _number(_field(row, node_file, "selection_probability"),
                              node_file, "selection_probability", probability=True)
        numerator = _integer(_field(row, node_file, "selection_numerator"),
                             node_file, "selection_numerator")
        denominator = _integer(_field(row, node_file, "selection_denominator"),
                               node_file, "selection_denominator", minimum=1)
        _ratio(node_file, "selection_denominator", probability, numerator, denominator,
               completed)
        depths = tuple((index, _number(_field(row, node_file, field), node_file, field,
                                       probability=True)) for index, field in enumerate(depth_fields))
        if not math.isclose(sum(value for _, value in depths), probability, abs_tol=1e-9):
            raise _invalid(node_file, "depth_probability", f"sum differs for {name!r}")
        if name == root and (probability != 1 or depths[0][1] != 1):
            raise _invalid(node_file, "root", "root must be selected at depth zero")
        label = _name(_field(row, node_file, "display_label"), node_file, "display_label")
        if name not in overlap:
            raise _invalid(files[-1], f"measurement_overlap.{name}", "missing node flag")
        nodes.append(VisualizationNode(name, label, depth, probability, depths, overlap[name]))
    if root not in node_names:
        raise _invalid(node_file, "root", f"missing configured root {root!r}")

    edge_file = "edges.csv"
    edge_fields = ("source", "target", "relation", "broad_norm", "unique_norm",
                   "stability", "screened_by", "evaluated_probability",
                   "target_reached_probability", "stability_numerator",
                   "stability_denominator", "evaluated_numerator",
                   "evaluated_denominator", "target_reached_numerator",
                   "target_reached_denominator")
    edges = []
    edge_keys = set()
    for row in _csv_rows(directory, edge_file, edge_fields):
        source = _name(_field(row, edge_file, "source"), edge_file, "source")
        target = _name(_field(row, edge_file, "target"), edge_file, "target")
        relation = _name(_field(row, edge_file, "relation"), edge_file, "relation")
        if relation not in ("PROXIMAL_UNIQUE", "SCREENED_BY_PARTNER", "SCREENED_BY_SET"):
            raise _invalid(edge_file, "relation", f"unsupported {relation!r}")
        key = (source, target, relation)
        if key in edge_keys:
            raise _invalid(edge_file, "identity", f"duplicate {key!r}")
        edge_keys.add(key)
        broad = _number(_field(row, edge_file, "broad_norm"), edge_file,
                        "broad_norm", optional=True)
        unique = _number(_field(row, edge_file, "unique_norm"), edge_file,
                         "unique_norm", optional=True)
        stability = _number(_field(row, edge_file, "stability"), edge_file,
                            "stability", probability=True)
        evaluated = _number(_field(row, edge_file, "evaluated_probability"), edge_file,
                            "evaluated_probability", probability=True)
        reached = _number(_field(row, edge_file, "target_reached_probability"), edge_file,
                          "target_reached_probability", probability=True)
        sn = _integer(_field(row, edge_file, "stability_numerator"), edge_file,
                      "stability_numerator")
        sd = _integer(_field(row, edge_file, "stability_denominator"), edge_file,
                      "stability_denominator", minimum=1)
        en = _integer(_field(row, edge_file, "evaluated_numerator"), edge_file,
                      "evaluated_numerator")
        ed = _integer(_field(row, edge_file, "evaluated_denominator"), edge_file,
                      "evaluated_denominator", minimum=1)
        rn = _integer(_field(row, edge_file, "target_reached_numerator"), edge_file,
                      "target_reached_numerator")
        rd = _integer(_field(row, edge_file, "target_reached_denominator"), edge_file,
                      "target_reached_denominator", minimum=1)
        _ratio(edge_file, "stability_denominator", stability, sn, sd, completed)
        _ratio(edge_file, "evaluated_denominator", evaluated, en, ed, completed)
        _ratio(edge_file, "target_reached_denominator", reached, rn, rd, completed)
        screened = _members(_field(row, edge_file, "screened_by"), edge_file, "screened_by")
        edges.append(VisualizationEdge(source, target, relation, broad, unique, stability,
                                       screened, evaluated, reached, sn, sd, en, ed, rn, rd))

    group_file = "shared_groups.csv"
    group_fields = ("group_id", "target", "members", "group_raw", "group_raw_median",
                    "raw_count", "group_norm", "group_norm_median", "normalized_count",
                    "stability", "stability_numerator", "stability_denominator")
    groups = []
    group_keys = set()
    for row in _csv_rows(directory, group_file, group_fields):
        target = _name(_field(row, group_file, "target"), group_file, "target")
        if target not in node_names:
            raise _invalid(group_file, "target", f"{target!r} is absent from nodes.csv")
        members = _members(_field(row, group_file, "members"), group_file, "members")
        if len(members) < 2:
            raise _invalid(group_file, "members", "expected at least two members")
        for member in members:
            if member not in node_names:
                raise _invalid(group_file, "members", f"{member!r} is absent from nodes.csv")
        group_id = _name(_field(row, group_file, "group_id"), group_file, "group_id")
        try:
            identity = json.loads(group_id)
        except ValueError as exc:
            raise _invalid(group_file, "group_id", "invalid JSON identity") from exc
        if identity != [target, list(members)]:
            raise _invalid(group_file, "group_id", "does not match target and members")
        if group_id in group_keys:
            raise _invalid(group_file, "group_id", f"duplicate identity {group_id!r}")
        group_keys.add(group_id)
        raw = _number(_field(row, group_file, "group_raw"), group_file,
                      "group_raw", optional=True)
        raw_median = _number(_field(row, group_file, "group_raw_median"), group_file,
                             "group_raw_median", optional=True)
        raw_count = _integer(_field(row, group_file, "raw_count"), group_file, "raw_count")
        norm = _number(_field(row, group_file, "group_norm"), group_file,
                       "group_norm", optional=True)
        norm_median = _number(_field(row, group_file, "group_norm_median"), group_file,
                              "group_norm_median", optional=True)
        norm_count = _integer(_field(row, group_file, "normalized_count"), group_file,
                              "normalized_count")
        if (raw_count == 0) != (raw is None and raw_median is None) or (
            norm_count == 0) != (norm is None and norm_median is None):
            raise _invalid(group_file, "raw_count/normalized_count",
                           "counts disagree with available magnitudes")
        stability = _number(_field(row, group_file, "stability"), group_file,
                            "stability", probability=True)
        numerator = _integer(_field(row, group_file, "stability_numerator"), group_file,
                             "stability_numerator")
        denominator = _integer(_field(row, group_file, "stability_denominator"), group_file,
                               "stability_denominator", minimum=1)
        _ratio(group_file, "stability_denominator", stability, numerator, denominator,
               completed)
        if raw_count > numerator:
            raise _invalid(group_file, "raw_count",
                           f"{raw_count} exceeds stability_numerator {numerator}")
        if norm_count > numerator:
            raise _invalid(group_file, "normalized_count",
                           f"{norm_count} exceeds stability_numerator {numerator}")
        groups.append(VisualizationGroup(group_id, target, members, raw, raw_median,
                                         raw_count, norm, norm_median, norm_count,
                                         stability, numerator, denominator))

    pair_file = "shared_comembership.csv"
    pair_fields = ("target", "node_a", "node_b", "shared_probability", "numerator",
                   "denominator")
    pairs = []
    pair_keys = set()
    for row in _csv_rows(directory, pair_file, pair_fields):
        target = _name(_field(row, pair_file, "target"), pair_file, "target")
        a = _name(_field(row, pair_file, "node_a"), pair_file, "node_a")
        b = _name(_field(row, pair_file, "node_b"), pair_file, "node_b")
        if a == b:
            raise _invalid(pair_file, "node_b", "pair members must differ")
        key = (target, *sorted((a, b)))
        if key in pair_keys:
            raise _invalid(pair_file, "identity", f"duplicate {key!r}")
        pair_keys.add(key)
        probability = _number(_field(row, pair_file, "shared_probability"), pair_file,
                              "shared_probability", probability=True)
        numerator = _integer(_field(row, pair_file, "numerator"), pair_file, "numerator")
        denominator = _integer(_field(row, pair_file, "denominator"), pair_file,
                               "denominator", minimum=1)
        _ratio(pair_file, "denominator", probability, numerator, denominator, completed)
        pairs.append(VisualizationPair(target, a, b, probability, numerator, denominator))

    local_file = "local_targets.csv"
    local_rows = _csv_rows(directory, local_file, ("target", "depth"))
    for row in local_rows:
        _name(_field(row, local_file, "target"), local_file, "target")
        depth = _integer(_field(row, local_file, "depth"), local_file, "depth")
        if depth > max_depth:
            raise _invalid(local_file, "depth", "exceeds configured max_depth")
    diagnostics = _object(payload.get("diagnostics"), files[-1], "diagnostics")
    evaluated = _integer(diagnostics.get("n_targets_evaluated"), files[-1],
                         "diagnostics.n_targets_evaluated")
    if evaluated != len(local_rows):
        raise _invalid(files[-1], "diagnostics.n_targets_evaluated",
                       f"{evaluated} differs from {local_file} row count {len(local_rows)}")
    legacy = VisualizationData(root, max_depth, completed, tau_b, pi_min,
                             display["show_screened"], display["show_unstable"],
                             display["distance_mode"], display["edge_width"],
                             display["edge_opacity"], tuple(nodes), tuple(edges),
                             tuple(groups), MappingProxyType(dict(overlap)), tuple(pairs))
    exploratory = view == "exploratory" or (
        view is None and display["show_unstable"] and "exploration" in payload)
    if exploratory:
        if "exploration" not in payload:
            raise ValueError("analysis_metadata.json: exploratory evidence is unavailable in this historical bundle")
        from open.exploratory_visualization import load_exploratory_payload
        return load_exploratory_payload(payload["exploration"], legacy)
    return legacy


def plot_open_network(data, *, output_path: str | Path | None = None,
                      min_frequency: float = 0.):
    """Render saved consensus as a predictive radial display."""
    if (isinstance(min_frequency, bool) or not isinstance(min_frequency, (int, float))
            or not math.isfinite(min_frequency) or not 0 <= min_frequency <= 1):
        raise ValueError("min_frequency must be a finite number in [0, 1]")
    from open.layer_io import LayerBundle
    if isinstance(data, LayerBundle):
        if min_frequency:
            raise ValueError("layer retention is analytical; sensitivity requires reconstruction")
        from open.layer_visualization import plot_layer_network
        return plot_layer_network(data, output_path=output_path)
    if getattr(data, "view", None) == "exploratory":
        from open.exploratory_visualization import plot_exploratory_network
        return plot_exploratory_network(data, output_path=output_path,
                                        min_frequency=min_frequency)
    if min_frequency:
        from dataclasses import replace
        data = replace(data, edges=tuple(e for e in data.edges if e.stability >= min_frequency),
                       groups=tuple(g for g in data.groups if g.stability >= min_frequency))
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyArrowPatch, PathPatch
        from matplotlib.path import Path as MplPath
    except ImportError as exc:
        raise ImportError("Install OPEN's visualization extra: pip install open[visualization]") from exc

    incoming = {node.name: 0.0 for node in data.nodes}
    for edge in data.edges:
        if edge.relation == "PROXIMAL_UNIQUE" and edge.unique_norm is not None:
            incoming[edge.source] = max(incoming[edge.source], edge.unique_norm)
    for group in data.groups:
        strength = max(group.normalized_magnitude or 0.0, 0.0)
        for member in group.members:
            incoming[member] = max(incoming[member], strength)
    largest_incoming = max(incoming.values(), default=0.0)
    by_depth: dict[int, list[VisualizationNode]] = {}
    for node in data.nodes:
        if node.name != data.root:
            by_depth.setdefault(node.depth, []).append(node)

    def radial_positions(rotation_step: int) -> dict[str, tuple[float, float]]:
        placed = {data.root: (0.0, 0.0)}
        for depth, layer in sorted(by_depth.items()):
            phase = depth * (math.pi * (3 - math.sqrt(5))
                             + rotation_step * math.pi / 12)
            for index, node in enumerate(sorted(layer, key=lambda item: item.name)):
                angle = phase + 2 * math.pi * index / len(layer)
                radius = depth - (0.15 * incoming[node.name] / largest_incoming
                                  if largest_incoming > 0 else 0.0)
                placed[node.name] = (radius * math.cos(angle), radius * math.sin(angle))
        return placed

    def group_clearances(placed: dict[str, tuple[float, float]]) -> tuple[float, ...]:
        return tuple(math.dist((sum(placed[name][0] for name in group.members)
                                / len(group.members),
                                sum(placed[name][1] for name in group.members)
                                / len(group.members)), placed[group.target])
                     for group in data.groups)

    # Rotate complete rings relative to one another; retain even spacing.
    best_score = (-1, -1.0)
    for rotation_step in range(24):
        candidate = radial_positions(rotation_step)
        clearances = group_clearances(candidate)
        score = (sum(clearance >= 0.25 for clearance in clearances),
                 sum(min(clearance, 0.25) for clearance in clearances))
        if score > best_score:
            best_positions, best_score = candidate, score
        if score[0] == len(data.groups):
            break
    positions = best_positions

    figure = Figure(figsize=(8, 8))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title("Consensus predictive network (visual distances are display-only)")

    def eligible_screened(edge: VisualizationEdge) -> bool:
        return (data.show_screened and edge.broad_norm is not None
                and edge.broad_norm >= data.tau_b and edge.stability >= data.pi_min)

    missing_sources = sorted({edge.source for edge in data.edges
                              if edge.relation in ("SCREENED_BY_PARTNER", "SCREENED_BY_SET")
                              and eligible_screened(edge) and edge.source not in positions})
    anchor_radius = max((node.depth for node in data.nodes), default=0) + 0.5
    for index, source in enumerate(missing_sources):
        angle = 2 * math.pi * (index + 1) / (len(missing_sources) + 1)
        positions[source] = (anchor_radius * math.cos(angle),
                             anchor_radius * math.sin(angle))

    largest_unique = max((edge.unique_norm for edge in data.edges
                          if edge.relation == "PROXIMAL_UNIQUE" and edge.unique_norm is not None
                          and edge.unique_norm > 0), default=0.0)
    largest_group = max((group.normalized_magnitude for group in data.groups
                         if group.normalized_magnitude is not None
                         and group.normalized_magnitude > 0), default=0.0)

    def width(strength: float | None, largest: float) -> float:
        return 0.8 + 3.2 * max(strength or 0.0, 0.0) / largest if largest > 0 else 0.8

    for edge in sorted(data.edges, key=lambda item: (item.source, item.target, item.relation)):
        screened = edge.relation in ("SCREENED_BY_PARTNER", "SCREENED_BY_SET")
        if screened and not eligible_screened(edge):
            continue
        arrow = FancyArrowPatch(positions[edge.source], positions[edge.target],
                                arrowstyle="-|>", mutation_scale=12,
                                linewidth=0.8 if screened else width(edge.unique_norm, largest_unique),
                                linestyle="--" if screened else "-",
                                color="#9299a5" if screened else "#275dad",
                                alpha=edge.stability, zorder=2)
        axes.add_patch(arrow)

    for group in sorted(data.groups, key=lambda item: item.group_id):
        member_points = [positions[name] for name in group.members]
        centroid = (sum(point[0] for point in member_points) / len(member_points),
                    sum(point[1] for point in member_points) / len(member_points))
        # Trace the outside arc; a chord through opposite members would cross the root.
        circular = sorted((math.atan2(y, x) % (2 * math.pi), (x, y))
                          for x, y in member_points)
        gaps = [circular[index + 1][0] - circular[index][0]
                for index in range(len(circular) - 1)]
        gaps.append(circular[0][0] + 2 * math.pi - circular[-1][0])
        start = (max(range(len(gaps)), key=gaps.__getitem__) + 1) % len(circular)
        ordered = circular[start:] + [(angle + 2 * math.pi, point)
                                      for angle, point in circular[:start]]
        margin = 0.12
        bracket_points = [ordered[0][1]]
        for index, (angle, point) in enumerate(ordered):
            radius = math.hypot(*point) + margin
            if index:
                previous_angle, previous_point = ordered[index - 1]
                previous_radius = math.hypot(*previous_point) + margin
                steps = max(1, math.ceil((angle - previous_angle) / (math.pi / 12)))
                for step in range(1, steps + 1):
                    fraction = step / steps
                    current_angle = previous_angle + fraction * (angle - previous_angle)
                    current_radius = previous_radius + fraction * (radius - previous_radius)
                    bracket_points.append((current_radius * math.cos(current_angle),
                                           current_radius * math.sin(current_angle)))
            else:
                bracket_points.append((radius * math.cos(angle), radius * math.sin(angle)))
        bracket_points.append(ordered[-1][1])
        path = MplPath(bracket_points,
                       [MplPath.MOVETO] + [MplPath.LINETO] * (len(bracket_points) - 1))
        axes.add_patch(PathPatch(path, facecolor="none", edgecolor="#b76728",
                                 linewidth=1.2, alpha=group.stability, zorder=1))
        target_point = positions[group.target]
        arrow_options = dict(arrowstyle="-|>", mutation_scale=12,
                             linewidth=width(group.normalized_magnitude, largest_group),
                             color="#b76728", alpha=group.stability, zorder=2)
        if math.dist(centroid, target_point) < 0.25:
            # A display-only curve keeps a coincident centroid-to-target edge visible.
            curve = MplPath([centroid,
                             (centroid[0] + 0.65, centroid[1] + 0.65),
                             (target_point[0] - 0.65, target_point[1] + 0.65),
                             target_point],
                            [MplPath.MOVETO, MplPath.CURVE4,
                             MplPath.CURVE4, MplPath.CURVE4])
            axes.add_patch(FancyArrowPatch(path=curve, **arrow_options))
        else:
            axes.add_patch(FancyArrowPatch(centroid, target_point, **arrow_options))

    for node in sorted(data.nodes, key=lambda item: item.name):
        x, y = positions[node.name]
        axes.scatter([x], [y], s=160, c="#183153" if node.name == data.root
                     else "#ffffff", edgecolors="#183153", zorder=3)
        axes.text(x, y, node.display_label, ha="center", va="center", fontsize=9,
                  color="#ffffff" if node.name == data.root else "#183153",
                  bbox={"boxstyle": "round,pad=0.3", "facecolor": "#101c2e",
                        "edgecolor": "#101c2e"} if node.name == data.root else None,
                  zorder=4)
    for source in missing_sources:
        x, y = positions[source]
        axes.scatter([x], [y], s=80, marker="x", color="#9299a5", zorder=3)
        axes.text(x, y, f"{source} (display-only screened source)", ha="center",
                  va="bottom", fontsize=8, color="#6c7480", zorder=4)
    axes.autoscale_view()
    axes.margins(0.2)
    if output_path is not None:
        figure.savefig(output_path, format="png" if Path(output_path).suffix.lower() == ".png"
                       else None, dpi=160)
    return figure
