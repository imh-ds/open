"""Run reproducible development demonstrations, export results, and plot them.

Usage: python examples/run_synthetic_demo.py --output NEW_DIRECTORY
Requires the project's core dependencies and visualization extra. Existing
outputs are never replaced. These small fixtures do not calibrate the method.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

# Always use this checkout, including when main has an older src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open.config import (AnalysisConfig, OpenConfig, PerformanceCVConfig,
                         StabilityConfig, ThresholdConfig, TuningCVConfig,
                         dump_config)
from open.io import write_open_outputs
from open.metadata import validate_metadata
from open.recurse import build_network_once
from open.stability import fit_stable_open
from open.visualization import load_visualization_data, plot_open_network


def sample_fixtures():
    """Match the fixed seeds and generating equations used in regression review."""
    rng = np.random.default_rng(271828)
    stress = rng.normal(size=80)
    depression = stress + .7 * rng.normal(size=80)
    anxiety = depression + .3 * rng.normal(size=80)
    chain = pd.DataFrame({"Anxiety": anxiety, "Depression": depression, "Stress": stress})
    missing_chain = chain.copy()
    missing_chain.loc[0, "Depression"] = np.nan

    rng = np.random.default_rng(120)
    scale = rng.normal(size=80)
    shared = pd.DataFrame({"Outcome": scale + rng.normal(scale=.15, size=80),
                           "Scale_A": scale, "Scale_B": scale + rng.normal(scale=.01, size=80)})
    return (("chain", chain, "Anxiety", 271828),
            ("chain_missing", missing_chain, "Anxiety", 271828),
            ("shared_predictors", shared, "Outcome", 120))


def _snapshot(network):
    return {
        "depths": dict(network.assigned_depth),
        "edges": [{"source": edge.source, "target": edge.target,
                   "unique_raw": edge.unique_raw, "unique_normalized": edge.unique_normalized}
                  for edge in network.core_edges],
        "shared_relations": [{"target": relation.target, "members": relation.members,
                              "raw": relation.raw, "normalized": relation.normalized}
                             for relation in network.shared_relations],
        "stop_reasons": {target: reason.value for target, reason in network.stop_reasons.items()},
        "targets": {
            target: {
                "baseline_risk": local.relevance.baseline_risk,
                "full_risk": local.relevance.full_risk,
                "relative_gain": local.total_gain_relative,
                "observed_target_rows": local.n_observed_target_rows,
                "missing_target_rows": local.n_missing_target_rows,
                "candidates": {
                    candidate: {"class": variable.kind.value if variable.kind else None,
                                "broad": local.broad[candidate], "unique": local.unique[candidate],
                                "screened_by": variable.screened_by_partner}
                    for candidate, variable in local.classes.items()
                },
            }
            for target, local in network.local_results.items()
        },
    }


def run_demo(output: Path, n_resamples: int = 20):
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for name, frame, root, data_seed in sample_fixtures():
        directory = output / name
        directory.mkdir()
        print(f"Starting {name}: {len(frame)} rows, {n_resamples} whole-pipeline resamples", flush=True)
        config = OpenConfig(
            AnalysisConfig(root, 27, max_depth=3),
            thresholds=ThresholdConfig(unique_relative=.15),
            performance_cv=PerformanceCVConfig(2, 1),
            tuning_cv=TuningCVConfig(2),
            stability=StabilityConfig(fraction=.8, n_resamples=n_resamples),
        )
        metadata = validate_metadata(frame.columns, [
            {"name": column, "role": "root" if column == root else "candidate",
             "type": "continuous", "display_label": column.replace("_", " ")}
            for column in frame
        ], root)
        sample_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        (directory / "sample.csv").write_bytes(sample_bytes)
        dump_config(config, directory / "config.yaml")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            once = build_network_once(frame, root, tuple(frame), config)
            fitted = fit_stable_open(frame, root, tuple(frame), config)
        warning_counts = Counter(warning.category.__name__ for warning in captured)
        warning_messages = Counter((warning.category.__name__, str(warning.message))
                                   for warning in captured)
        (directory / "warnings.json").write_text(json.dumps([
            {"category": category, "message": message, "count": count}
            for (category, message), count in sorted(warning_messages.items())
        ], indent=2), encoding="utf-8")
        write_open_outputs(fitted, directory / "bundle", metadata=metadata)
        # Render the saved bundle, not a separate graph with hand-chosen edges.
        graph = load_visualization_data(directory / "bundle", view="consensus")
        figure = plot_open_network(graph, output_path=directory / "network.png")
        _save_svg(figure, directory / "network.svg")
        single_run = _snapshot(once)
        (directory / "single_run.json").write_text(json.dumps(single_run, indent=2), encoding="utf-8")
        summary = {
            "sample": name, "n_rows": len(frame), "data_seed": data_seed,
            "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "code_revision": fitted.git_revision, "config": asdict(config),
            "completed": fitted.batch.completed_runs, "failed": fitted.batch.failed_runs,
            "denominator": fitted.stability.denominator,
            "elapsed_seconds": fitted.elapsed_seconds, "warnings": dict(warning_counts),
            "single_run": single_run,
            "nodes": [{"name": node.name, "depth": node.depth,
                       "selection_probability": node.selection_probability}
                      for node in fitted.consensus.nodes],
            "edges": [{"source": edge.source, "target": edge.target,
                       "class": edge.relation.value, "probability": edge.frequency.probability,
                       "count": edge.frequency.numerator,
                       "unique_mean": edge.unique.normalized_mean,
                       "broad_mean": edge.broad.normalized_mean}
                      for edge in fitted.consensus.edges],
            "paths": [{"nodes": path.nodes, "count": path.frequency.numerator,
                       "denominator": path.frequency.denominator,
                       "probability": path.frequency.probability}
                      for path in fitted.stability.paths],
            "shared_relations": [{"target": group.target, "members": group.members,
                                  "count": group.frequency.numerator,
                                  "probability": group.frequency.probability}
                                 for group in fitted.consensus.shared_relations],
        }
        summaries.append(summary)
        (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps({key: summary[key] for key in
                          ("sample", "completed", "failed", "nodes", "paths", "shared_relations", "warnings")}),
              flush=True)
    _write_report(output, summaries)
    _plot_overview(output, summaries)
    return summaries


def _save_svg(figure, path):
    """Keep Matplotlib's SVG output free of trailing line whitespace."""
    figure.savefig(path, bbox_inches="tight")
    rendered = path.read_text(encoding="utf-8")
    path.write_bytes(("\n".join(line.rstrip() for line in rendered.splitlines()) + "\n").encode("utf-8"))


def _plot_overview(output, summaries):
    """Compact schematics of the measured consensus, alongside native plots."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import FancyBboxPatch

    figure = Figure(figsize=(15, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(summaries))
    figure.suptitle("OPEN: synthetic recursive analysis", fontsize=18, fontweight="bold")
    for axis, summary in zip(np.atleast_1d(axes), summaries):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        root = summary["config"]["analysis"]["root"]
        depths = {node["name"]: node["depth"] for node in summary["nodes"]}
        depths[root] = 0
        def node(name, x, y):
            axis.text(x, y, name.replace("_", " "), ha="center", va="center", fontsize=10,
                      color="white" if name == root else "#153754",
                      bbox=dict(boxstyle="round,pad=.4", fc="#153754" if name == root else "#e6f0f8",
                                ec="#153754", lw=1.2), zorder=5)
            if name in depths:
                axis.text(x, y-.10, f"Depth {depths[name]}", ha="center", fontsize=9, color="#526473")
        if summary["sample"].startswith("chain"):
            axis.set_title("Complete chain" if summary["sample"] == "chain" else "Chain with one missing value",
                           fontsize=12, pad=20)
            positions = {"Stress": (.16, .51), "Depression": (.50, .51), "Anxiety": (.84, .51)}
            for name, (x, y) in positions.items():
                if name in depths:
                    node(name, x, y)
            for edge in summary["edges"]:
                start, end = positions[edge["source"]], positions[edge["target"]]
                screened = edge["class"].startswith("SCREENED")
                axis.annotate("", xy=end, xytext=start,
                              arrowprops=dict(arrowstyle="-|>", color="#8898a5" if screened else "#2773a5",
                                              lw=1.3 if screened else 2.5, ls="--" if screened else "-",
                                              connectionstyle="arc3,rad=-.45" if screened else "arc3",
                                              shrinkA=30, shrinkB=35), zorder=2)
            axis.text(.50, .78, "Screened at Anxiety" if any(
                edge["class"].startswith("SCREENED") for edge in summary["edges"]
            ) else "", ha="center", color="#617786", fontsize=10)
            path = next((path for path in summary["paths"]
                         if path["nodes"] == ["Stress", "Depression", "Anxiety"]
                         or path["nodes"] == ("Stress", "Depression", "Anxiety")), None)
            axis.text(.50, .23, f"Complete path: {path['count']}/{path['denominator']} runs" if path
                      else "Complete path not observed", ha="center", fontsize=11, color="#153754")
            if summary["sample"] == "chain_missing":
                local = summary["single_run"]["targets"].get("Depression")
                if local:
                    axis.text(.50, .13, f"Depression target: {local['observed_target_rows']} observed, "
                              f"{local['missing_target_rows']} excluded", ha="center", fontsize=10)
            else:
                local = summary["single_run"]["targets"][root]
                axis.text(.50, .13, f"Anxiety MSE: {local['baseline_risk']:.3f} to {local['full_risk']:.3f}",
                          ha="center", fontsize=10)
        else:
            axis.set_title("Redundant shared predictors", fontsize=12, pad=20)
            group = next(iter(summary["shared_relations"]), None)
            if group:
                axis.add_patch(FancyBboxPatch((.08, .58), .84, .21, boxstyle="round,pad=.015",
                                             facecolor="#fff0de", edgecolor="#ba6b25", lw=1.6, zorder=0))
                axis.annotate("", xy=(.50, .34), xytext=(.50, .57),
                              arrowprops=dict(arrowstyle="-|>", color="#ba6b25", lw=2.5))
                axis.text(.50, .14, f"Shared group: {group['count']}/{summary['denominator']} runs",
                          ha="center", fontsize=11, color="#153754")
            node("Scale_A", .29, .70)
            node("Scale_B", .71, .70)
            node(root, .50, .30)
            axis.text(.50, .88, "Both members promoted", ha="center", fontsize=10, color="#915520")
        axis.text(.50, .02, f"{summary['completed']} completed; {summary['failed']} failed",
                  ha="center", fontsize=10, color="#526473")
    figure.text(.50, .015, "Predictive relations; fixed development fixtures and settings; no causal or calibration claim.",
                ha="center", fontsize=10, color="#526473")
    figure.subplots_adjust(top=.79, bottom=.15, left=.025, right=.975, wspace=.22)
    figure.savefig(output / "overview.png", dpi=180, bbox_inches="tight")
    _save_svg(figure, output / "overview.svg")


def _write_report(output, summaries):
    lines = ["# OPEN synthetic sample demonstration", "",
             "Three deterministic 80-row fixtures; 20 development resamples per fixture by default.",
             "Each resample uses 80% of rows without replacement, two outer folds, two inner folds,",
             "maximum depth 3, unique relevance threshold 0.15, and consensus threshold 0.70.",
             "Settings are fixed before running; no thresholds were tuned to these outputs.", "",
             "These fixtures demonstrate software behavior. They do not establish calibrated",
             "scientific performance, real-data validity, or causal direction. Edge probabilities",
             "are empirical frequencies across the completed whole-pipeline runs.", ""]
    lines.extend(["![Overview of the measured consensus graphs](overview.png)", ""])
    for summary in summaries:
        name = summary["sample"]
        lines.extend([f"## {name}", "", f"Completed: {summary['completed']}; failed: {summary['failed']}; "
                      f"structural denominator: {summary['denominator']}.", "",
                      f"Engine revision: `{summary['code_revision']}`.", "",
                      f"![Saved-bundle consensus network]({name}/network.png)", "",
                      "### One full-sample run", "",
                      "| Target | Baseline MSE | Full MSE | Relative gain | Observed / missing target rows |",
                      "|---|---:|---:|---:|---:|"])
        for target, local in summary["single_run"]["targets"].items():
            gain = local["relative_gain"]
            gain_text = f"{gain:.1%}" if gain is not None else "undefined"
            lines.append(f"| {target} | {local['baseline_risk']:.4f} | {local['full_risk']:.4f} | "
                         f"{gain_text} | {local['observed_target_rows']} / {local['missing_target_rows']} |")
        lines.extend(["", "### Whole-pipeline results", "",
                      "| Relation | Count / completed runs | Probability |", "|---|---:|---:|"])
        for edge in summary["edges"]:
            lines.append(f"| {edge['source']} -> {edge['target']} ({edge['class']}) | "
                         f"{edge['count']} / {summary['denominator']} | {edge['probability']:.0%} |")
        for path in summary["paths"]:
            lines.append(f"| {' -> '.join(path['nodes'])} (complete unique-edge path) | "
                         f"{path['count']} / {path['denominator']} | {path['probability']:.0%} |")
        for group in summary["shared_relations"]:
            lines.append(f"| {{{', '.join(group['members'])}}} -> {group['target']} (shared) | "
                         f"{group['count']} / {summary['denominator']} | {group['probability']:.0%} |")
        if name.startswith("chain"):
            lines.extend(["", "Interpretation: the generated fixture has Stress feeding Depression, then Anxiety.",
                          "Inspect the single-run classifications: Stress can predict Anxiety marginally but is",
                          "screened when Depression is available; it remains eligible and is evaluated as a",
                          "predictor of Depression at the next layer. The table reports how often the entire",
                          "unique-edge path appeared across independently rebuilt networks, rather than a",
                          "product of separate edge probabilities."])
            if name == "chain_missing":
                lines.extend(["", "Depression is missing in sample row 0. When Depression becomes a target,",
                              "its risk uses only observed Depression values within the existing folds.",
                              "In resamples that omit row 0, no target rows need exclusion. No target value",
                              "is imputed. This is the formerly blocked recursive case."])
        else:
            lines.extend(["", "Interpretation: the two scales are near copies. Individually removing either",
                          "loses little prediction because the other remains; joint removal loses the",
                          "shared predictive signal. Both members can therefore be promoted together.",
                          "The group is a predictive relation and display object, not a synthetic target."])
        lines.extend(["", f"Captured numerical warnings: `{summary['warnings']}`. Full messages/counts are",
                      f"retained in [{name}/warnings.json]({name}/warnings.json); none were silently discarded.", ""])
    lines.extend(["## Reproduce", "", "Run from the developed checkout with the visualization extra installed:",
                  "", "```powershell", "python examples/run_synthetic_demo.py --output NEW_DIRECTORY --resamples 20",
                  "```", "", "The runner refuses to overwrite an existing directory. Each case saves sample data,",
                  "configuration, a single-run diagnostic, the six-file OPEN bundle, and PNG/SVG plots.",
                  "summary.json records hashes, seeds, code revision, and exact structural denominators.",
                  "Numerical warnings remain recorded; a completed batch is an execution result,",
                  "not a claim that every fit converged. These examples exercise a two-member shared",
                  "group; they do not validate higher-order group discovery or its calibration.", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20)
    arguments = parser.parse_args()
    run_demo(arguments.output, arguments.resamples)
