"""Outcome provenance in graphs and safe multi-outcome exports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def analysis():
    from open.api import ExploreConfig, explore_and_confirm_outcomes

    rng = np.random.default_rng(42)
    x = rng.normal(size=1200)
    data = pd.DataFrame({"../A": 2 * x + rng.normal(size=1200),
                         "..\\A": -2 * x + rng.normal(size=1200), "X": x})
    return explore_and_confirm_outcomes(
        data, ["../A", "..\\A"], config=ExploreConfig(folds=3, max_depth=1, interactions=False),
    )


@pytest.fixture(scope="module")
def shared_analysis():
    from open.api import ExploreConfig, explore_and_confirm_outcomes

    rng = np.random.default_rng(20261021)
    x = rng.normal(size=(1200, 3))
    x[:, 0] = .8 * x[:, 2] + .6 * x[:, 0]
    data = pd.DataFrame(x, columns=["X1", "X2", "X3"])
    data["Y1"] = 1.2 * x[:, 0] + .8 * x[:, 1] + rng.normal(size=1200)
    data["Y2"] = x[:, 0] + .7 * x[:, 1] + rng.normal(size=1200)
    return explore_and_confirm_outcomes(
        data, ["Y1", "Y2"], predictors=["X1", "X2", "X3"], split_seed=42,
        config=ExploreConfig(folds=3, max_depth=3, interactions=False, seed=42),
    )


def test_exports_reused_assessment_ids_and_outcome_membership(shared_analysis, tmp_path):
    from open.api import save_multi_outcome_analysis

    out = save_multi_outcome_analysis(shared_analysis, tmp_path / "batch")
    settings = json.loads((out / "outcomes-settings.json").read_text(encoding="utf-8"))
    shared_id = shared_analysis.by_outcome["Y1"].exploration.assessment_ids[(2, "X1")]
    assert settings["assessment_outcomes"][shared_id] == ["Y1", "Y2"]
    for outcome, folder in settings["outcome_folders"].items():
        table = pd.read_csv(out / folder / "links.csv")
        assert table.loc[table.target.eq("X1"), "assessment_id"].eq(shared_id).all()
        assert all(outcome in settings["assessment_outcomes"][value] for value in table.assessment_id)


def _x3_x1_labels(figure, result, source="X3"):
    """How many edge labels the link from source to X1 has in a combined figure."""
    from open.explore_report import _apa

    links = next(iter(result.by_outcome.values())).exploration.links
    r2 = links[links.target.eq("X1") & links.source.eq(source)].r2_median.iloc[0]
    return sum(t.get_text().startswith(f"{_apa(r2)} (") for t in figure.axes[0].texts)


def test_overlay_collapses_reused_links_to_one_neutral_line(shared_analysis):
    import matplotlib.pyplot as plt
    from open.api import plot_outcomes
    from open.explore_report import KEPT

    figure = plot_outcomes(shared_analysis, layout="combined")
    axes, = figure.axes
    assert _x3_x1_labels(figure, shared_analysis) == 1
    assert not any("shared ×" in t.get_text() for t in axes.texts)          # the colour says it
    assert sum(line.get_color() == KEPT for line in axes.lines) == 1
    assert "shared by several outcomes (same test)" in {t.get_text() for t in axes.get_legend().texts}
    assert any("Dark grey edges are tested once" in t.get_text() for t in figure.texts)
    plt.close(figure)


@pytest.fixture(scope="module")
def crowded_analysis():
    from open.api import ExploreConfig, explore_and_confirm_outcomes

    rng = np.random.default_rng(20261021)
    x = rng.normal(size=(1200, 10))
    x[:, 0] = .8 * x[:, 2] + .6 * x[:, 0]
    data = pd.DataFrame(x, columns=[f"X{i}" for i in range(1, 11)])
    data["Y1"] = 1.2 * data.X1 + .8 * data.X2 + rng.normal(size=len(data))
    data["Y2"] = data.X1 + .7 * data.X2 + rng.normal(size=len(data))
    rng = np.random.default_rng(7)
    for i in range(3, 7):
        data[f"Y{i}"] = (.9 + .1 * i) * data.X1 + .7 * data.X2 + rng.normal(size=len(data))
    return explore_and_confirm_outcomes(
        data, [f"Y{i}" for i in range(1, 7)], predictors=[f"X{i}" for i in range(1, 11)], split_seed=42,
        config=ExploreConfig(folds=5, max_depth=3, interactions=False, seed=42),
    )


def test_overlay_dodges_labels_without_spreading_nodes(crowded_analysis):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from open.api import GraphStyle, plot_outcomes

    before = plot_outcomes(crowded_analysis, layout="combined", style=GraphStyle(dodge_labels=False))
    after = plot_outcomes(crowded_analysis, layout="combined")
    ax, = after.axes
    after.canvas.draw()
    renderer = after.canvas.get_renderer()
    names = set(crowded_analysis.settings["outcomes"]) | set(crowded_analysis.settings["predictors"])
    labels = [t for t in ax.texts if t.get_text() not in names]
    boxes = [t.get_window_extent(renderer) for t in labels]
    obstacles = [p.get_window_extent(renderer) for p in ax.patches if isinstance(p, (Circle, Wedge))]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert not any(box.overlaps(node) for box in boxes for node in obstacles)
    assert [p.center for p in before.axes[0].patches if isinstance(p, Circle)] == [
        p.center for p in ax.patches if isinstance(p, Circle)]
    np.testing.assert_array_equal(before.get_size_inches(), after.get_size_inches())
    repeat = plot_outcomes(crowded_analysis, layout="combined")
    np.testing.assert_array_equal([t.get_position() for t in ax.texts],
                                  [t.get_position() for t in repeat.axes[0].texts])
    plt.close(repeat)
    plt.close(before)
    plt.close(after)


def test_overlay_rearranges_nodes_locally_instead_of_uniform_scaling(crowded_analysis):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from open.api import GraphStyle, plot_outcomes

    before = plot_outcomes(crowded_analysis, layout="combined",
                           style=GraphStyle(dodge_nodes=False, dodge_labels=False))
    after = plot_outcomes(crowded_analysis, layout="combined", style=GraphStyle(dodge_labels=False))
    p = np.array([node.center for node in before.axes[0].patches if isinstance(node, Circle)])
    q = np.array([node.center for node in after.axes[0].patches if isinstance(node, Circle)])
    assert not np.allclose(p, q)
    a, b = np.triu_indices(len(p), 1)
    ratios = np.linalg.norm(q[a] - q[b], axis=1) / np.linalg.norm(p[a] - p[b], axis=1)
    assert np.std(ratios) > .03
    radii = {node.center: node.radius for node in after.axes[0].patches if isinstance(node, Circle)}
    for ring in after.axes[0].patches:
        if isinstance(ring, Wedge):
            radii[ring.center] = max(radii[ring.center], ring.r)
    envelope = np.array([radii[tuple(center)] for center in q])
    assert np.all(np.linalg.norm(q[a] - q[b], axis=1) >= envelope[a] + envelope[b])
    # Repeated calls choose the same rearrangement.
    repeated = plot_outcomes(crowded_analysis, layout="combined", style=GraphStyle(dodge_labels=False))
    np.testing.assert_array_equal(q, [node.center for node in repeated.axes[0].patches if isinstance(node, Circle)])
    plt.close(before)
    plt.close(after)
    plt.close(repeated)


def test_confirmation_marks_can_be_hidden_without_changing_results(shared_analysis):
    import matplotlib.pyplot as plt
    from open.api import GraphStyle, plot_outcomes

    confirmed = shared_analysis.confirmation.confirmed.copy()
    figure = plot_outcomes(shared_analysis, layout="combined", style=GraphStyle(show_confirmation=False))
    assert not any("✓" in t.get_text() for t in figure.axes[0].texts)
    assert not any("✓" in t.get_text() for t in figure.texts)
    pd.testing.assert_series_equal(confirmed, shared_analysis.confirmation.confirmed)
    plt.close(figure)


@pytest.mark.parametrize("setting", ["show_confirmation", "dodge_labels", "dodge_nodes"])
def test_annotation_style_flags_require_booleans(setting):
    from open.api import GraphStyle

    with pytest.raises(ValueError, match=setting):
        GraphStyle(**{setting: "false"})


def test_annotation_style_preserves_existing_positional_arguments():
    from open.api import GraphStyle

    style = GraphStyle(1, 1, 1, 1, True, False, "kept", "white", 5, 170)
    assert style.edge_labels == "kept" and style.dpi == 170
    assert style.show_confirmation is True and style.dodge_labels is True


@pytest.mark.parametrize("difference", ["identity", "effect", "confirmation"])
def test_overlay_keeps_distinct_assessments_or_verdicts_separate(shared_analysis, difference):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from open.api import plot_outcomes

    result = copy.deepcopy(shared_analysis)
    second = result.by_outcome["Y2"]
    if difference == "identity":
        # The numbers can match exactly; an independent assessment stays separate.
        second.exploration.assessment_ids[(2, "X1")] = "independent-assessment"
    elif difference == "effect":
        links = second.exploration.links.copy()
        kept = links.target.eq("X1") & links.source.eq("X3")
        links.loc[kept, "r2_median"] += 1e-8  # still the same printed label
        second.exploration = replace(second.exploration, links=links)
    else:
        kept = second.confirmation.target.eq("X1") & second.confirmation.source.eq("X3")
        second.confirmation.loc[kept, "confirmed"] = False
    figure = plot_outcomes(result, layout="combined")
    assert _x3_x1_labels(figure, result) == 2
    plt.close(figure)


@pytest.mark.parametrize("kind", ["shared", "interaction"])
def test_overlay_collapses_reused_groups_and_interactions(shared_analysis, kind):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from open.api import plot_outcomes

    result = copy.deepcopy(shared_analysis)
    for split in result.by_outcome.values():
        links = split.exploration.links
        row = links[links.target.eq("X1") & links.source.eq("X3")].iloc[0].to_dict()
        row.update(kind=kind, source="X2+X3", members=("X2", "X3"))
        split.exploration = replace(split.exploration, links=pd.DataFrame([row]))
        # These synthetic relation kinds have no unique-link confirmation test.
        split.confirmation = split.confirmation.iloc[:0]
    figure = plot_outcomes(result, layout="combined")
    axes, = figure.axes
    assert _x3_x1_labels(figure, result, "X2+X3") == 1
    if kind == "interaction":
        assert sum(isinstance(p, RegularPolygon) for p in axes.patches) == 1
    else:
        assert sum(line.get_alpha() == .18 for line in axes.lines) == 1
    plt.close(figure)


def test_facets_keep_independent_outcome_roots(analysis):
    import matplotlib.pyplot as plt
    from open.api import plot_outcomes

    figure = plot_outcomes(analysis)
    assert len(figure.axes) == 2
    for axes, outcome in zip(figure.axes, analysis.by_outcome):
        labels = [text.get_text() for text in axes.texts]
        assert outcome in labels and "X" in labels
        assert set(analysis.by_outcome).intersection(labels) == {outcome}
    plt.close(figure)


def test_combined_overlay_merges_nodes_and_retains_outcome_edges(analysis):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon
    from open.api import plot_outcomes

    figure = plot_outcomes(analysis, layout="combined")
    axes, = figure.axes
    labels = [text.get_text() for text in axes.texts]
    assert labels.count("X") == 1
    assert all(labels.count(outcome) == 1 for outcome in analysis.by_outcome)
    assert "separate predictive" in axes.get_title().lower()
    assert not any(isinstance(p, FancyArrowPatch) for p in axes.patches)
    arrows = [patch for patch in axes.patches if isinstance(patch, Polygon)]
    assert len(arrows) == 2
    assert len({patch.get_facecolor() for patch in arrows}) == 2
    assert len(axes.lines) == 2
    assert {text.get_text() for text in axes.get_legend().texts} >= set(analysis.by_outcome)
    assert all("depth" not in text.get_text().lower() for text in axes.get_legend().texts)
    plt.close(figure)


def test_overlay_keeps_overlapping_and_reversed_relations(analysis):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon, PathPatch
    from open.api import plot_outcomes

    result = copy.deepcopy(analysis)
    for index, split in enumerate(result.by_outcome.values()):
        row = split.exploration.links.iloc[0].to_dict()
        row.update(source="X", target="Z", members=("X",), depth=2)
        rows = [row]
        if index:
            rows.append({**row, "source": "Z", "target": "X", "members": ("Z",)})
        split.exploration = replace(split.exploration, links=pd.DataFrame(rows))
    figure = plot_outcomes(result, layout="combined")
    assert not any(isinstance(p, FancyArrowPatch) for p in figure.axes[0].patches)
    arrows = [p for p in figure.axes[0].patches if isinstance(p, Polygon)]
    assert len(arrows) == 3
    assert len({tuple(p.get_path().vertices.ravel()) for p in arrows}) == 3
    curves = [p for p in figure.axes[0].patches if isinstance(p, PathPatch)]
    assert len(curves) == 3
    assert len({tuple(p.get_path().vertices.ravel()) for p in curves}) == 3
    assert [t.get_text() for t in figure.axes[0].texts].count("Z") == 1
    plt.close(figure)


def test_overlay_many_outcomes_curve_shared_edges_with_separate_triangle_heads(analysis):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon
    from open.api import plot_outcomes

    result = copy.deepcopy(analysis)
    template = next(iter(result.by_outcome.values()))
    result.by_outcome = {f"Y{i}": copy.deepcopy(template) for i in range(6)}
    for split in result.by_outcome.values():
        row = split.exploration.links.iloc[0].to_dict()
        row.update(source="X", target="Z", members=("X",), depth=2)
        split.exploration = replace(split.exploration, links=pd.DataFrame([row]))
    figure = plot_outcomes(result, layout="combined")
    axes, = figure.axes
    curves = [p for p in axes.patches if isinstance(p, PathPatch)]
    assert len(curves) == 6
    assert all(Path.CURVE3 in p.get_path().codes for p in curves)
    assert len({tuple(p.get_path().vertices.ravel()) for p in curves}) == 6
    assert len([p for p in axes.patches if isinstance(p, Polygon)]) == 6
    assert not any(isinstance(p, FancyArrowPatch) for p in axes.patches)
    plt.close(figure)


def test_plot_saves_file_and_rejects_unknown_layout(analysis, tmp_path):
    from open.api import plot_outcomes

    path = tmp_path / "combined.png"
    assert plot_outcomes(analysis, layout="combined", path=path) == path
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="layout"):
        plot_outcomes(analysis, layout="unknown")


def test_combined_uses_network_nodes_fit_rings_and_confirmation_labels(analysis):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from open.api import plot_outcomes

    figure = plot_outcomes(analysis, layout="combined")
    axes, = figure.axes
    # Same data-coordinate nodes and fit rings as the ordinary network graph.
    assert len([p for p in axes.patches if isinstance(p, Circle)]) == 3
    assert any(isinstance(p, Wedge) for p in axes.patches)
    labels = [t.get_text() for t in axes.texts]
    assert any("R²" in t and "new" in t for t in labels)
    assert sum("(3/3)" in t and "✓" in t for t in labels) == 2
    assert all(not t.get_bbox_patch() for t in axes.texts)
    plt.close(figure)


def test_facets_keep_confirmation_marks(analysis):
    import matplotlib.pyplot as plt
    from open.api import plot_outcomes

    figure = plot_outcomes(analysis)
    assert all(any("✓" in t.get_text() for t in ax.texts) for ax in figure.axes)
    plt.close(figure)


def test_overlay_labels_differing_shared_target_fits_without_averaging(analysis):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from open.api import plot_outcomes

    result = copy.deepcopy(analysis)
    for i, split in enumerate(result.by_outcome.values()):
        row = split.exploration.fit.iloc[0].to_dict()
        row.update(target="X", r2_network=.2 + .3 * i, network_size=1)
        split.exploration = replace(split.exploration, fit=pd.concat([
            split.exploration.fit, pd.DataFrame([row])], ignore_index=True))
    figure = plot_outcomes(result, layout="combined")
    labels = [t.get_text() for t in figure.axes[0].texts]
    assert any("../A: R² .200" in t and "..\\A: R² .500" in t for t in labels)
    plt.close(figure)


@pytest.mark.parametrize("kind", ["shared", "interaction"])
def test_overlay_preserves_group_geometry_and_separate_interaction_junctions(analysis, kind):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from open.api import plot_outcomes

    result = copy.deepcopy(analysis)
    for split in result.by_outcome.values():
        row = split.exploration.links.iloc[0].to_dict()
        row.update(kind=kind, source="X+Z", members=("X", "Z"), target="T", depth=2)
        split.exploration = replace(split.exploration, links=pd.DataFrame([row]))
    figure = plot_outcomes(result, layout="combined")
    ax, = figure.axes
    assert [t.get_text() for t in ax.texts].count("X") == 1
    assert [t.get_text() for t in ax.texts].count("Z") == 1
    if kind == "interaction":
        diamonds = [p for p in ax.patches if isinstance(p, RegularPolygon)]
        assert len(diamonds) == 2
        assert not np.allclose(diamonds[0].xy, diamonds[1].xy)
    else:
        assert sum(line.get_alpha() == .18 for line in ax.lines) == 2
    plt.close(figure)


def test_exports_safe_distinct_folders_and_aggregate_tables(analysis, tmp_path):
    from open.api import save_multi_outcome_analysis

    out = tmp_path / "analysis"
    assert save_multi_outcome_analysis(analysis, out) == out
    settings = json.loads((out / "outcomes-settings.json").read_text(encoding="utf-8"))
    mapping = settings["outcome_folders"]
    assert set(mapping) == set(analysis.by_outcome)
    assert len(set(mapping.values())) == 2
    for folder in mapping.values():
        resolved = (out / folder).resolve()
        assert resolved.is_relative_to(out.resolve())
        assert (resolved / "split-settings.json").is_file()
    assert set(pd.read_csv(out / "outcomes-confirmation.csv").root_outcome) == set(mapping)
    assert set(pd.read_csv(out / "outcomes-fit.csv").root_outcome) == set(mapping)
    assert (out / "outcomes-summary.txt").read_text(encoding="utf-8") == analysis.summary() + "\n"
    (out / "notes.txt").write_text("keep", encoding="utf-8")
    save_multi_outcome_analysis(analysis, out, plot="combined")
    assert (out / "outcomes-network.png").is_file()
    save_multi_outcome_analysis(analysis, out, plot=False)
    assert not (out / "outcomes-network.png").exists()
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_aggregate_publication_failure_restores_prior_files(analysis, tmp_path, monkeypatch):
    from open.api import save_multi_outcome_analysis

    out = save_multi_outcome_analysis(analysis, tmp_path / "analysis")
    before = {p.name: p.read_bytes() for p in out.glob("outcomes-*") if p.is_file()}
    original = Path.replace
    failed = False

    def replace_once(source, target):
        nonlocal failed
        if source.name == "outcomes-fit.csv" and source.parent.name.startswith(".open-multi-") and not failed:
            failed = True
            raise OSError("simulated publication failure")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", replace_once)
    with pytest.raises(OSError, match="simulated"):
        save_multi_outcome_analysis(analysis, out)
    assert before == {p.name: p.read_bytes() for p in out.glob("outcomes-*") if p.is_file()}


def test_export_rejects_unsafe_destination_types(analysis, tmp_path):
    from open.api import save_multi_outcome_analysis

    out = tmp_path / "analysis"
    out.mkdir()
    (out / "outcomes-fit.csv").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        save_multi_outcome_analysis(analysis, out)
    assert not (out / "outcomes-confirmation.csv").exists()


def test_export_rejects_symlinked_outcome_directory(analysis, tmp_path):
    from open.api import save_multi_outcome_analysis

    out, external = tmp_path / "analysis", tmp_path / "external"
    out.mkdir()
    external.mkdir()
    try:
        (out / "outcomes").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink|reparse"):
        save_multi_outcome_analysis(analysis, out)
    assert list(external.iterdir()) == []


def test_csv_export_keeps_optional_dependencies_lazy(tmp_path):
    code = textwrap.dedent('''
        import importlib.abc
        import sys
        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'matplotlib', 'lightgbm'}:
                    raise ModuleNotFoundError('optional dependency blocked')
        sys.meta_path.insert(0, BlockOptional())
        import numpy as np
        import pandas as pd
        from open.api import ExploreConfig, explore_and_confirm_outcomes, save_multi_outcome_analysis
        rng = np.random.default_rng(3)
        x = rng.normal(size=1200)
        data = pd.DataFrame({'Y1': x + rng.normal(size=1200), 'Y2': -x + rng.normal(size=1200), 'X': x})
        result = explore_and_confirm_outcomes(data, ['Y1', 'Y2'], config=ExploreConfig(folds=3, max_depth=1, interactions=False))
        save_multi_outcome_analysis(result, sys.argv[1], plot=False)
        assert not {'matplotlib', 'lightgbm'}.intersection(sys.modules)
    ''')
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path / "csv")], cwd=root,
                               env={**os.environ, "PYTHONPATH": str(root / "src")},
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_export_mapping_preserves_outcome_ownership_after_reordering(analysis, tmp_path):
    import copy
    from open.api import save_multi_outcome_analysis

    out = save_multi_outcome_analysis(analysis, tmp_path / "analysis")
    first = json.loads((out / "outcomes-settings.json").read_text(encoding="utf-8"))["outcome_folders"]
    for outcome, folder in first.items():
        (out / folder / "notes.txt").write_text(outcome, encoding="utf-8")
    reordered = copy.deepcopy(analysis)
    reordered.by_outcome = dict(reversed(list(reordered.by_outcome.items())))
    save_multi_outcome_analysis(reordered, out)
    second = json.loads((out / "outcomes-settings.json").read_text(encoding="utf-8"))["outcome_folders"]
    assert second == first
    for outcome, folder in second.items():
        assert (out / folder / "notes.txt").read_text(encoding="utf-8") == outcome
        settings = json.loads((out / folder / "split-settings.json").read_text(encoding="utf-8"))
        assert settings["outcome"] == outcome
