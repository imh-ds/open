"""Unfiltered observed relations, independent of analytical consensus inclusion."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from open.local_target import LocalClass


_SCREENED = {LocalClass.SCREENED_BY_PARTNER, LocalClass.SCREENED_BY_SET}


def _opportunity(local, members):
    if local.input_candidates is not None:
        return set(members).issubset(local.input_candidates)
    # Legacy records can prove presence through measured contrasts/classes;
    # absence from incomplete legacy records cannot prove ineligibility.
    known = set(local.relevance.broad) | set(local.relevance.unique) | set(local.classes)
    return True if set(members).issubset(known) else None


def _strength(values, scope):
    raw = [value for value, _ in values]
    relative = [ratio for _, ratio in values if ratio is not None]
    return {"strength_count": len(raw), "raw_median": median(raw) if raw else None,
            "baseline_relative_count": len(relative),
            "baseline_relative_median": median(relative) if relative else None,
            "strength_scope": scope}


def _frequency(count, total, eligible, unknown, pi_min):
    return {"count": count, "denominator": total, "probability": count / total,
            "eligible_count": eligible, "eligibility_unknown_count": unknown,
            "conditional_probability": count / eligible if eligible and not unknown else None,
            "consensus": count / total >= pi_min}


def build_exploration(result, metadata):
    """Summarize stored runs only; never refit models or change selection.

    Unique strength includes every available score for that source/target,
    including negative and nonselected fits. Shared strength is explicitly
    conditional on retained joint ablations; unavailable scores are not zeros.
    """
    networks = [run.network for run in result.batch.runs if run.completed]
    total = len(networks)
    if not total:
        raise ValueError("exploratory evidence requires completed runs")
    pi_min = result.config.thresholds.stability
    labels = {row["name"]: row.get("display_label") or row["name"]
              for row in metadata.variables}
    depths = defaultdict(Counter)
    edges, groups = Counter(), Counter()
    screeners = defaultdict(set)
    for network in networks:
        for name, depth in network.assigned_depth.items():
            depths[name][depth] += 1
        edges.update((edge.source, edge.target, edge.kind.value) for edge in network.core_edges)
        for target, local in network.local_results.items():
            for source, variable in local.classes.items():
                if variable.kind in _SCREENED:
                    key = (source, target, variable.kind.value)
                    edges[key] += 1
                    screeners[key].update(variable.screened_by_partner or variable.screened_by_set)
        groups.update((relation.target, relation.members) for relation in network.shared_relations)

    edge_rows = []
    for (source, target, relation), count in sorted(edges.items()):
        eligible = unknown = 0
        strengths, gain_ratios = [], []
        for network in networks:
            local = network.local_results.get(target)
            if local is None:
                continue
            opportunity = _opportunity(local, (source,))
            eligible += opportunity is True
            unknown += opportunity is None
            contrast = local.relevance.unique.get(source)
            if contrast is not None:
                baseline = local.relevance.baseline_risk
                strengths.append((contrast.raw, contrast.raw / baseline if baseline > 0 else None))
                if contrast.normalized is not None:
                    gain_ratios.append(contrast.normalized)
        edge_rows.append({"source": source, "target": target, "relation": relation,
                          "screened_by": sorted(screeners[(source, target, relation)]),
                          **_frequency(count, total, eligible, unknown, pi_min),
                          **_strength(strengths, "all scored eligible local fits"),
                          "gain_relative_count": len(gain_ratios),
                          "gain_relative_median": median(gain_ratios) if gain_ratios else None})

    group_rows = []
    for (target, members), count in sorted(groups.items()):
        eligible = unknown = 0
        strengths, gain_ratios = [], []
        for network in networks:
            local = network.local_results.get(target)
            if local is None:
                continue
            opportunity = _opportunity(local, members)
            eligible += opportunity is True
            unknown += opportunity is None
            for relation in network.shared_relations:
                if relation.target == target and relation.members == members:
                    baseline = local.relevance.baseline_risk
                    strengths.append((relation.raw, relation.raw / baseline if baseline > 0 else None))
                    if relation.normalized is not None:
                        gain_ratios.append(relation.normalized)
        group_rows.append({"target": target, "members": list(members),
                           **_frequency(count, total, eligible, unknown, pi_min),
                           **_strength(strengths, "selected joint ablations"),
                           "gain_relative_count": len(gain_ratios),
                           "gain_relative_median": median(gain_ratios) if gain_ratios else None})

    node_rows = []
    for name, counts in sorted(depths.items()):
        node_rows.append({"name": name, "display_label": labels.get(name, name),
                          "depth": min(counts, key=lambda depth: (-counts[depth], depth)),
                          "count": sum(counts.values()), "denominator": total,
                          "probability": sum(counts.values()) / total,
                          "depth_probabilities": [[depth, count / total] for depth, count in sorted(counts.items())],
                          "display_only": False})
    present = {node["name"]: node for node in node_rows}
    for edge in edge_rows:
        source, target = edge["source"], edge["target"]
        if source not in present:
            anchor = {"name": source, "display_label": labels.get(source, source),
                      "depth": min(present[target]["depth"] + 1, result.config.analysis.max_depth),
                      "count": 0, "denominator": total, "probability": 0.,
                      "depth_probabilities": [], "display_only": True}
            present[source] = anchor
            node_rows.append(anchor)
    return {"schema_version": 1, "denominator": total,
            "strength_definition": "median of per-run signed loss reduction / target baseline risk",
            "nodes": sorted(node_rows, key=lambda node: node["name"]),
            "edges": edge_rows, "groups": group_rows}
