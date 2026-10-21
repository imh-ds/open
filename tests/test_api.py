"""Public workflow names preserve the established implementation contracts."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=1200)
    return pd.DataFrame({"Y": 2 * x + rng.normal(size=1200), "X": x, "N": rng.normal(size=1200)})


def test_api_preserves_original_functions_and_types():
    api = importlib.import_module("open.api")
    expected = {
        "explore_network": ("open.explore", "explore"),
        "confirm_network": ("open.confirm", "confirm"),
        "explore_and_confirm": ("open.confirm", "explore_and_confirm"),
        "explore_and_confirm_outcomes": ("open.multi_outcome", "explore_and_confirm_outcomes"),
        "MultiOutcomeResult": ("open.multi_outcome", "MultiOutcomeResult"),
        "plot_outcomes": ("open.explore_report", "plot_outcomes"),
        "save_multi_outcome_analysis": ("open.explore_report", "save_multi_outcome_analysis"),
        "evaluate_confirmation_fit": ("open.confirm", "confirmation_fit"),
        "split_sample": ("open.confirm", "split_sample"),
        "plot_network": ("open.explore_report", "plot_explore"),
        "save_exploration": ("open.explore_report", "write_explore"),
        "save_split_analysis": ("open.explore_report", "write_split"),
        "ExploreConfig": ("open.explore", "ExploreConfig"),
        "ExploreResult": ("open.explore", "ExploreResult"),
        "SplitResult": ("open.confirm", "SplitResult"),
        "GraphStyle": ("open.explore_report", "GraphStyle"),
    }
    assert set(api.__all__) == set(expected)
    for name, (module, original) in expected.items():
        # Identity preserves signatures, exception behavior and return types.
        assert getattr(api, name) is getattr(importlib.import_module(module), original)


def test_exploration_confirmation_and_export_workflow(data, tmp_path):
    from open.api import (ExploreConfig, ExploreResult, confirm_network, evaluate_confirmation_fit,
                          explore_network, save_exploration, split_sample)

    first, independent = split_sample(data, seed=2)
    result = explore_network(first, "Y", config=ExploreConfig(folds=3, max_depth=1, interactions=False))
    assert isinstance(result, ExploreResult)
    confirmation = confirm_network(result, independent)
    fit = evaluate_confirmation_fit(result, independent, folds=3, seed=2)
    assert len(confirmation) and confirmation.confirmed.any()
    assert len(fit) and np.isfinite(fit.r2_network).all()
    out = tmp_path / "exploration"
    assert save_exploration(result, out, plot=False, confirmation=confirmation, confirmation_fit=fit) == out
    assert len(pd.read_csv(out / "confirmation.csv")) == len(confirmation)
    assert len(pd.read_csv(out / "confirmation-fit.csv")) == len(fit)


def test_split_analysis_export_workflow(data, tmp_path):
    from open.api import ExploreConfig, SplitResult, explore_and_confirm, save_split_analysis

    split = explore_and_confirm(data, "Y", config=ExploreConfig(folds=3, max_depth=1, interactions=False), split_seed=2)
    assert isinstance(split, SplitResult)
    out = tmp_path / "split"
    assert save_split_analysis(split, out, plot=False) == out
    assert (out / "split-settings.json").is_file()
    assert (out / "summary.txt").read_text(encoding="utf-8") == split.summary() + "\n"


def test_plot_network_saves_the_expected_path(data, tmp_path):
    pytest.importorskip("matplotlib")
    from open.api import ExploreConfig, GraphStyle, explore_network, plot_network

    result = explore_network(data, "Y", config=ExploreConfig(folds=3, max_depth=1, interactions=False))
    path = tmp_path / "network.png"
    assert plot_network(result, path, style=GraphStyle(label_size=.9)) == path
    assert path.stat().st_size > 0


@pytest.mark.parametrize("order", ["api_first", "modules_first"])
def test_api_keeps_module_imports_and_optional_dependencies_lazy(tmp_path, order):
    code = textwrap.dedent("""
        import importlib.abc
        import pathlib
        import sys
        import types

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'matplotlib', 'lightgbm'}:
                    raise ModuleNotFoundError('Optional dependency deliberately blocked: ' + fullname)

        sys.meta_path.insert(0, BlockOptional())
        if sys.argv[2] == 'modules_first':
            import open.explore
            import open.confirm
        import open.api as api
        import open.explore as exploration_module
        import open.confirm as confirmation_module
        assert isinstance(exploration_module, types.ModuleType)
        assert isinstance(confirmation_module, types.ModuleType)
        assert api.explore_network is exploration_module.explore
        assert api.confirm_network is confirmation_module.confirm
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(3)
        x = rng.normal(size=120)
        result = api.explore_network(pd.DataFrame({'Y': x + rng.normal(size=120), 'X': x}), 'Y',
                                    config=api.ExploreConfig(folds=3, max_depth=1, interactions=False))
        out = api.save_exploration(result, sys.argv[1], plot=False)
        assert isinstance(out, pathlib.Path) and (out / 'links.csv').is_file()
        assert not {'matplotlib', 'lightgbm'}.intersection(sys.modules)
    """)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path / "csv"), order],
                               cwd=root, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
