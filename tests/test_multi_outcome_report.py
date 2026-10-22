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
