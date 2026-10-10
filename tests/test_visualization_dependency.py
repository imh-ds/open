from pathlib import Path
import os
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_visualization_extra_is_optional_and_bounded():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert all("matplotlib" not in requirement.lower() for requirement in project["dependencies"])
    assert project["optional-dependencies"]["visualization"] == ["matplotlib>=3.9,<4"]


def test_importing_visualization_does_not_import_matplotlib_or_pyplot():
    script = ("import sys; import open.visualization; "
              "assert 'matplotlib' not in sys.modules; "
              "assert 'matplotlib.pyplot' not in sys.modules")
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def test_plot_missing_extra_names_install_remedy(monkeypatch):
    import builtins
    from open.visualization import plot_open_network

    original_import = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    with pytest.raises(ImportError, match=r"open\[visualization\]"):
        plot_open_network(None)
