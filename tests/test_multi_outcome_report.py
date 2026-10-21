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
    from matplotlib.patches import FancyArrowPatch
    from open.api import plot_outcomes

    figure = plot_outcomes(analysis, layout="combined")
    axes, = figure.axes
    labels = [text.get_text() for text in axes.texts]
    assert labels.count("X") == 1
    assert all(labels.count(outcome) == 1 for outcome in analysis.by_outcome)
    assert "separate predictive" in axes.get_title().lower()
    arrows = [patch for patch in axes.patches if isinstance(patch, FancyArrowPatch)]
    assert len(arrows) == 2
    assert len({patch.get_edgecolor() for patch in arrows}) == 2
    assert {text.get_text() for text in axes.get_legend().texts} >= set(analysis.by_outcome)
    assert all("depth" not in text.get_text().lower() for text in axes.get_legend().texts)
    plt.close(figure)


def test_overlay_keeps_overlapping_and_reversed_relations(analysis):
    import copy
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
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
    arrows = [p for p in figure.axes[0].patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == 3
    assert len({tuple(p.get_path().vertices.ravel()) for p in arrows}) == 3
    assert [t.get_text() for t in figure.axes[0].texts].count("Z") == 1
    plt.close(figure)


def test_plot_saves_file_and_rejects_unknown_layout(analysis, tmp_path):
    from open.api import plot_outcomes

    path = tmp_path / "combined.png"
    assert plot_outcomes(analysis, layout="combined", path=path) == path
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="layout"):
        plot_outcomes(analysis, layout="unknown")


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
