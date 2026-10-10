"""Recover the original six-composite simulation and audit its recursive runs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.run_synthetic_demo import _save_svg, _snapshot
from open.aggregation import aggregate_stability
from open.config import (AnalysisConfig, OpenConfig, PerformanceCVConfig,
                         StabilityConfig, TuningCVConfig, dump_config)
from open.consensus import build_consensus
from open.io import write_open_outputs
from open.metadata import validate_metadata
from open.stability import (OPENResult, _capture_versions, read_git_revision,
                            run_stability_batch)
from open.stability_types import StabilityBatch
from open.visualization import load_visualization_data, plot_open_network

LABELS = {"wellbeing": "Well-being", "sleep_quality": "Sleep Quality",
          "perceived_stress": "Perceived Stress", "social_support": "Social Support",
          "physical_activity": "Physical Activity", "work_demands": "Work Demands"}


def survey_fixture():
    """The exact seed, draw order, and equations from the earlier demonstration."""
    rng = np.random.default_rng(613)
    n = 180
    support = rng.normal(size=n)
    activity = rng.normal(size=n)
    demands = rng.normal(size=n)
    stress = .62 * demands - .32 * support + rng.normal(scale=.62, size=n)
    sleep = -.56 * stress + .28 * activity + rng.normal(scale=.72, size=n)
    wellbeing = (.42 * sleep + .34 * support + .20 * activity - .28 * stress
                 + rng.normal(scale=.68, size=n))
    return pd.DataFrame({name: np.clip(3 + .62 * values, 1, 5)
                         for name, values in zip(LABELS, (wellbeing, sleep, stress,
                                                         support, activity, demands))})


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_demo(output, n_resamples=20, original_directory=None, max_depth=2, *,
             method="whole_pipeline", sensitivity=False):
    if method not in ("whole_pipeline", "layer_consensus"):
        raise ValueError("unknown analysis method")
    if method == "layer_consensus":
        from examples.run_layer_demo import run_layer_demo
        return run_layer_demo(output, n_resamples=n_resamples, max_depth=max_depth,
                              sensitivity=sensitivity)
    if sensitivity:
        raise ValueError("full-construction sensitivity requires layer_consensus method")
    if n_resamples < 5:
        raise ValueError("at least five runs are needed for the original comparison")
    output.mkdir(parents=True, exist_ok=False)
    frame = survey_fixture()
    data_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (output / "sample.csv").write_bytes(data_bytes)
    config = OpenConfig(AnalysisConfig("wellbeing", 613, max_depth=max_depth),
                        performance_cv=PerformanceCVConfig(2, 1),
                        tuning_cv=TuningCVConfig(2),
                        stability=StabilityConfig(fraction=.8, n_resamples=n_resamples,
                                                  final_resamples=n_resamples))
    dump_config(config, output / "config.yaml")
    metadata = validate_metadata(frame.columns, [
        {"name": name, "display_label": LABELS[name], "type": "continuous",
         "role": "root" if name == "wellbeing" else "candidate"} for name in frame
    ], "wellbeing")
    revision, revision_reason = read_git_revision()
    package, dependencies, learners, availability = _capture_versions(config)
    print(f"Starting recovered survey: 180 rows, six composites, {n_resamples} runs", flush=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        batch = run_stability_batch(frame, "wellbeing", tuple(frame), config)
    if not batch.completed_runs:
        raise RuntimeError("No completed runs; inspect the engine before reporting results")
    stability = aggregate_stability(batch)
    consensus = build_consensus(stability, config.thresholds.stability)
    result = OPENResult(batch, stability, consensus, config, tuple(frame), revision,
                        revision_reason, time.perf_counter() - started, None,
                        "Direct batch API demonstration; memory profiling not enabled",
                        package, dependencies, learners, availability)
    write_open_outputs(result, output / "bundle", metadata=metadata)
    graph = load_visualization_data(output / "bundle")
    figure = plot_open_network(graph, output_path=output / "network.png")
    _save_svg(figure, output / "network.svg")
    consensus_graph = load_visualization_data(output / "bundle", view="consensus")
    figure = plot_open_network(consensus_graph, output_path=output / "consensus_network.png")
    _save_svg(figure, output / "consensus_network.svg")
    payload = json.loads((output / "bundle/analysis_metadata.json").read_text(encoding="utf-8"))
    pd.DataFrame(payload["exploration"]["edges"]).to_csv(output / "edge_evidence.csv", index=False)
    _write_json(output / "warnings.json", [
        {"category": category, "message": message, "count": count}
        for (category, message), count in Counter(
            (item.category.__name__, str(item.message)) for item in captured).items()
    ])

    traces, rows = [], []
    edge_counts, stop_counts = Counter(), Counter()
    for run in batch.runs:
        if not run.completed:
            traces.append({"run_index": run.run_index, "seed": run.seed,
                           "failure_type": run.failure_type, "failure_message": run.failure_message})
            continue
        network = run.network
        trace = {"run_index": run.run_index, "seed": run.seed, **_snapshot(network)}
        for target, local in network.local_results.items():
            depth = network.assigned_depth[target]
            candidates = list(local.input_candidates)
            # Use the engine's actual input set, including low-gain stops.
            if local.broad:
                assert set(candidates) == set(local.broad)
            trace["targets"][target]["input_candidates"] = candidates
            trace["targets"][target]["promoted"] = sorted(local.promoted)
            trace["targets"][target]["unique_scores"] = {
                source: {"raw": score.raw, "gain_relative": score.normalized}
                for source, score in local.relevance.unique.items()}
            rows.append({"run_index": run.run_index, "seed": run.seed, "target": target,
                         "depth": depth, "input_candidates": " | ".join(candidates),
                         "promoted": " | ".join(sorted(local.promoted)),
                         "baseline_mse": local.relevance.baseline_risk,
                         "full_mse": local.relevance.full_risk,
                         "relative_gain": local.total_gain_relative,
                         "stop_reason": (network.stop_reasons[target].value
                                         if target in network.stop_reasons else "")})
        traces.append(trace)
        edge_counts.update((edge.source, edge.target) for edge in network.core_edges)
        stop_counts.update((target, reason.value) for target, reason in network.stop_reasons.items())
    _write_json(output / "recursion_trace.json", traces)
    pd.DataFrame(rows).to_csv(output / "recursive_regressions.csv", index=False)
    _plot_frequencies(output, tuple(frame), edge_counts, batch.completed_runs)

    root_rows = pd.DataFrame(rows).query("target == 'wellbeing'")
    first_five = aggregate_stability(StabilityBatch(5, batch.runs[:5]))
    comparison = {"original_data_seed": 613, "original_rows": 180,
                  "original_max_depth": 2, "rerun_max_depth": max_depth,
                  "original_first_five_run_seeds": [r.seed for r in batch.runs[:5]],
                  "first_five_consensus_edges": [
                      {"source": e.source, "target": e.target, "class": e.relation.value,
                       "count": e.frequency.numerator, "probability": e.frequency.probability}
                      for e in build_consensus(first_five, .7).edges]}
    if original_directory is not None:
        legacy = (original_directory / "synthetic_survey_scores.csv").read_bytes()
        reproduced = frame.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode("utf-8")
        assert legacy.replace(b"\r\n", b"\n") == reproduced
        comparison["reproduced_original_rounded_csv_exactly"] = True
        comparison["original_rounded_csv_sha256"] = hashlib.sha256(legacy).hexdigest()
        old_meta = json.loads((original_directory / "p5.1-output/analysis_metadata.json").read_text())
        comparison["original_engine_revision"] = old_meta["git_revision"]
        assert [(r.seed, list(r.sampled_positions)) for r in batch.runs[:5]] == [
            (r["seed"], r["sampled_positions"]) for r in old_meta["run_ledger"]]
        comparison["first_five_sampled_positions_match_original"] = True
        old_local = pd.read_csv(original_directory / "p5.1-output/local_targets.csv")
        current = pd.DataFrame(rows).query("run_index < 5")
        merged = current.merge(old_local, on=["run_index", "target"], validate="one_to_one")
        assert len(merged) == len(old_local)
        comparison["matched_original_local_regressions"] = len(merged)
        comparison["max_absolute_local_full_mse_change"] = float(
            (merged.full_mse - merged.full_risk).abs().max())
    _write_json(output / "original_comparison.json", comparison)
    summary = {"data_seed": 613, "n_rows": 180, "variables": LABELS,
               "sample_sha256": hashlib.sha256(data_bytes).hexdigest(),
               "engine_revision": revision, "config": asdict(config),
               "completed": batch.completed_runs, "failed": batch.failed_runs,
               "warnings": dict(Counter(item.category.__name__ for item in captured)),
               "median_root_baseline_mse": float(root_rows.baseline_mse.median()),
               "median_root_full_mse": float(root_rows.full_mse.median()),
               "median_root_relative_gain": float(root_rows.relative_gain.median()),
               "evaluated_targets": dict(Counter(row["target"] for row in rows)),
               "stop_counts": [{"target": t, "reason": reason, "count": count}
                               for (t, reason), count in sorted(stop_counts.items())],
               "unique_edges": [{"source": a, "target": b, "count": count,
                                 "probability": count / batch.completed_runs}
                                for (a, b), count in sorted(edge_counts.items())],
               "consensus_edges": [{"source": e.source, "target": e.target,
                                    "class": e.relation.value, "count": e.frequency.numerator,
                                    "probability": e.frequency.probability}
                                   for e in consensus.edges],
               "paths": [{"nodes": path.nodes, "count": path.frequency.numerator,
                          "probability": path.frequency.probability} for path in stability.paths]}
    _write_json(output / "summary.json", summary)
    _write_report(output, summary, traces, comparison)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _write_report(output, summary, traces, comparison):
    completed = summary["completed"]
    depth = summary["config"]["analysis"]["max_depth"]
    example = _plot_run_example(output, traces)
    lines = ["# Six-variable Well-being demonstration", "",
             "Recovered original simulation: 180 independent synthetic rows, seed 613,",
             "continuous composite scores clipped to 1–5. Root: Well-being. Candidates:",
             "Social Support, Sleep Quality, Perceived Stress, Physical Activity, Work Demands.",
             "These are generated values, not observed respondents or validated scales.", "",
             f"Engine: `{summary['engine_revision']}`. Completed: {completed}; failed: {summary['failed']};",
             f"captured warnings: `{summary['warnings']}`. Maximum depth: {depth}.",
             "Two outer folds, two tuning folds, 80% subsamples (144 rows), "
             f"seeds 613–{612 + summary['config']['stability']['n_resamples']}.",
             "Original thresholds retained: unique/broad 0.05, total gain 0.02, group 0.10,",
             "dependence 0.70, consensus 0.70. Only the resample count and, where stated,",
             "depth limit differ from the original five-run example. Display defaults now",
             "show all observed relations; analytical selection settings remain unchanged.", "",
             "## Predicting Well-being", "",
             f"Median baseline held-out MSE: **{summary['median_root_baseline_mse']:.4f}**;",
             f"median full-model MSE: **{summary['median_root_full_mse']:.4f}**;",
             f"median per-run relative gain: **{summary['median_root_relative_gain']:.1%}**.",
             "These are separate medians, not a ratio calculated from the two median MSEs.", "",
             "| Retained consensus edge | Count / complete runs |", "|---|---:|"]
    for edge in summary["consensus_edges"]:
        lines.append(f"| {LABELS[edge['source']]} → {LABELS[edge['target']]} | {edge['count']}/{completed} |")
    lines += ["", "## Exploratory display and consensus", "",
              "![All observed predictive relations](network.png)", "",
              "The exploratory graph includes every observed selected unique or screened-by",
              "relation. Opacity shows selection across all completed runs; dark blue marks",
              "the retained consensus subset. Width uses median per-run signed unique loss",
              "reduction divided by that target's baseline risk, clipped only for drawing.",
              "The evidence panel and [edge evidence table](edge_evidence.csv) preserve signed",
              "strengths, raw/baseline score counts, and selected/complete versus selected/eligible",
              "counts. Strength includes scored nonselections, so a frequently selected edge",
              "need not have a large median contribution. Low-gain eligible fits without scores",
              "count as opportunities but not measurements. Eligibility is descriptive.", "",
              "Pooled directions and modal node positions do not describe a single fitted network.",
              "Some arrows come from mutually incompatible runs; paths must be measured from actual",
              "networks. No new confidence threshold or causal interpretation is implied.", "",
              "![Explicit consensus network](consensus_network.png)", "",
              "The explicit consensus view retains the historical gain-relative width encoding.",
              "Membership follows the configured whole-network threshold; hidden upstream arrows",
              "can still be visible in the exploratory view.", "",
              "## Evidence that the recursive regressions ran", "",
              "| Target | Local regressions evaluated |", "|---|---:|"]
    for target in LABELS:
        lines.append(f"| {LABELS[target]} | {summary['evaluated_targets'].get(target, 0)} |")
    lines += ["", "A selected node is analyzed as a target when it has remaining candidates and is",
              "below the depth limit. All targets in a layer use the same unassigned candidate",
              "set. Variables already assigned to that layer or closer to Well-being are excluded.",
              "Consequently a variable selected at the root cannot also serve as an upstream",
              "predictor of another depth-1 target in that same run. No residuals are passed on.", "",
              "| Upstream unique edge | All complete networks | When target evaluated and source eligible |",
              "|---|---:|---:|"]
    opportunities = []
    for edge in summary["unique_edges"]:
        if edge["target"] == "wellbeing":
            continue
        eligible = sum(edge["target"] in t.get("targets", {}) and edge["source"] in
                       t["targets"][edge["target"]]["input_candidates"] for t in traces)
        opportunities.append({**edge, "eligible_runs": eligible})
        lines.append(f"| {LABELS[edge['source']]} → {LABELS[edge['target']]} | "
                     f"{edge['count']}/{completed} | {edge['count']}/{eligible} |")
    _write_json(output / "edge_opportunities.json", opportunities)
    lines += ["", "The consensus threshold uses **all completed networks**, not the conditional",
              "eligibility denominator. The last column explains opportunity and is descriptive;",
              "it does not replace the configured consensus rule or establish calibrated stability.",
              ("Upstream consensus edges are listed in the retained-edge table above."
               if any(e["target"] != "wellbeing" for e in summary["consensus_edges"])
               else "No upstream edge passes 70% in this run."),
              "Low whole-network frequencies can coexist with consistent local selection when",
              "eligible, because the variable's assigned layer changes between runs.", "",
              "![All observed unique-edge frequencies](edge_frequencies.png)", "",
              f"## One actual recursive network: seed {example['seed']}", "",
              "The first planned run reaching the maximum observed depth is shown for illustration.",
              "This is one network, not the consensus or a claim about stability. All its selected",
              "variables and unique edges come from the saved run snapshot.", "",
              "![Actual single-run cascade](example_cascade.png)", "",
              "## First planned run: seed 613", "",
              "This run is shown by its fixed order, not selected for a desirable graph.", "",
              "| Target | Actual layer-frozen candidate set | Promoted predictors |",
              "|---|---|---|"]
    first = traces[0]
    for target, local in sorted(first.get("targets", {}).items(),
                                key=lambda item: (first["depths"][item[0]], item[0])):
        candidates = ", ".join(LABELS[n] for n in local["input_candidates"])
        promoted = ", ".join(LABELS[n] for n in local["promoted"]) or "None"
        lines.append(f"| {LABELS[target]} | {candidates} | {promoted} |")
    lines += ["", "Full per-run evidence: [recursive regressions](recursive_regressions.csv),",
              "[graph snapshots and stop reasons](recursion_trace.json), and",
              "[whole-path frequencies](summary.json). Paths are counted from complete",
              "networks, not multiplied from separate edge probabilities.", "",
              "## Branch stopping reasons", "", "| Target | Reason | Runs |", "|---|---|---:|"]
    for stop in summary["stop_counts"]:
        lines.append(f"| {LABELS[stop['target']]} | {stop['reason']} | {stop['count']} |")
    lines += ["", "LOW_TOTAL_GAIN means that target's held-out model did not meet the gain floor.",
              "NO_UNASSIGNED means there were no remaining variables to regress on.",
              "MAX_DEPTH marks a selected boundary node, not a fitted regression or an execution error.", "",
              "## Original example and reproduction", "",
              "[Original comparison](original_comparison.json) retains the first five run seeds",
              "and their consensus. When the original artifact directory is supplied, it also",
              "checks the reproduced rounded CSV, identical sampled row positions, and all",
              "19 local regression risks from the original five-run demonstration.", "",
              (f"Matched **{comparison['matched_original_local_regressions']}** original local regressions; "
               f"maximum absolute full-MSE difference **{comparison['max_absolute_local_full_mse_change']:.3g}** "
               "(floating-point roundoff). Original rounded input and sampled rows match."
               if "matched_original_local_regressions" in comparison else
               "Original artifact verification was not requested; first-five results are retained for comparison."),
              "", "```powershell",
              f"python examples/run_survey_demo.py --output NEW_DIRECTORY --resamples {completed} --max-depth {depth}",
              "```", "", "The input is saved at full precision, matching the earlier generator's in-memory",
              "analysis. Its old CSV was rounded to six decimal places. Input hashes, config,",
              "dependencies, run row positions and engine revision are retained in the bundle.",
              "The runner uses existing batch, aggregation and consensus APIs; memory profiling",
              "is disabled, so its runtime is not comparable to a profiled historical run.", "",
              "## Interpretation limits", "",
              "This correlated survey-style simulation exercises the recursive engine; it is",
              "not scientific validation. Selected arrows are predictive and can run opposite",
              "to the generating equations. For example Stress predicting Social Support does",
              "not establish that stress causes support. A zero or omitted edge is not evidence",
              "of independence. The five-variable candidate set, sample size, noise, regularization,",
              "root assignment and thresholds all affect which cascades recur. Higher-order group",
              "discovery and calibrated selection error are not validated by this demonstration.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_run_example(output, traces):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    valid = [t for t in traces if "depths" in t]
    deepest = max(max(t["depths"].values()) for t in valid)
    trace = next(t for t in valid if max(t["depths"].values()) == deepest)
    positions = {}
    for depth in range(deepest + 1):
        names = sorted(n for n, d in trace["depths"].items() if d == depth)
        for name, y in zip(names, np.linspace(.8, -.8, len(names)) if len(names) > 1 else [0]):
            positions[name] = (deepest - depth, y)
    figure = Figure(figsize=(13, 5.5)); FigureCanvasAgg(figure)
    ax = figure.subplots()
    for edge in trace["edges"]:
        ax.annotate("", xy=positions[edge["target"]], xytext=positions[edge["source"]],
                    arrowprops={"arrowstyle": "-|>", "color": "#2878a8", "lw": 2.2,
                                "shrinkA": 40, "shrinkB": 40, "mutation_scale": 17})
    for name, (x, y) in positions.items():
        root = name == "wellbeing"
        ax.text(x, y, LABELS[name].replace(" ", "\n"), ha="center", va="center",
                color="white" if root else "#173b56", fontsize=12,
                bbox={"boxstyle": "round,pad=.5", "facecolor": "#173b56" if root else "#e8f2fa",
                      "edgecolor": "#173b56", "linewidth": 1.5})
    for depth in range(deepest + 1):
        ax.text(deepest - depth, -1.25, f"Depth {depth}", ha="center", color="#546b7b")
    ax.set(xlim=(-.55, deepest + .55), ylim=(-1.4, 1.45))
    ax.set_title(f"Actual recursive network: seed {trace['seed']} · 144 sampled rows\n"
                 "One illustrative run; arrows show selected unique predictive relations", pad=14)
    ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(output / "example_cascade.png", dpi=170)
    _save_svg(figure, output / "example_cascade.svg")
    _write_json(output / "example_selection.json", {"seed": trace["seed"],
                "observed_max_depth": deepest,
                "selection_rule": "First planned completed run reaching maximum observed depth"})
    return trace


def _plot_frequencies(output, names, counts, denominator):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    values = np.array([[counts[(a, b)] / denominator for b in names] for a in names])
    figure = Figure(figsize=(11, 8)); FigureCanvasAgg(figure)
    ax = figure.subplots()
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), [LABELS[n].replace(" ", "\n") for n in names])
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            text = "—" if a == b else f"{counts[(a, b)]}/{denominator}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if values[i, j] > .5 else "#15354b")
    ax.set_xlabel("Target of the local regression")
    ax.set_ylabel("Predictor")
    ax.set_title("Unique predictive edges across complete recursive runs\n"
                 "Including edges below the 70% consensus threshold", pad=20)
    figure.colorbar(image, ax=ax, label="Fraction of completed runs")
    figure.tight_layout()
    figure.savefig(output / "edge_frequencies.png", dpi=170)
    _save_svg(figure, output / "edge_frequencies.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20)
    parser.add_argument("--original-directory", type=Path)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--method", choices=("whole_pipeline", "layer_consensus"), default="whole_pipeline")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    run_demo(args.output, args.resamples, args.original_directory, args.max_depth,
             method=args.method, sensitivity=args.sensitivity)
