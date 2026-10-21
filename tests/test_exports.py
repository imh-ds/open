"""Overwrite and failure recovery contracts for generated export files."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from open.confirm import SplitResult, explore_and_confirm
from open.explore import ExploreConfig
from open import explore_report as report


@pytest.fixture(scope="module")
def analysis():
    r = np.random.default_rng(42)
    x = r.normal(size=1200)
    data = pd.DataFrame({"Y": 2 * x + r.normal(size=1200), "X": x, "N": r.normal(size=1200)})
    return explore_and_confirm(data, "Y", split_seed=2,
                               config=ExploreConfig(folds=3, max_depth=1, interactions=False))


def _snapshot(out):
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("initial", ["split", "confirmed_exploration"])
def test_plain_export_removes_stale_optional_tables_and_preserves_user_files(analysis, tmp_path, initial):
    out = tmp_path / "out"
    if initial == "split":
        report.write_split(analysis, out, plot=False)
    else:
        report.write_explore(analysis.exploration, out, plot=False, confirmation=analysis.confirmation,
                             confirmation_fit=analysis.fit)
    assert (out / "confirmation.csv").exists() and (out / "confirmation-fit.csv").exists()
    (out / "notes.md").write_text("keep these notes", encoding="utf-8")
    user_dir = out / ".open-export-user-files"
    user_dir.mkdir()
    (user_dir / "data.txt").write_text("keep this data", encoding="utf-8")
    returned = report.write_explore(analysis.exploration, out, plot=False)
    assert returned == out
    for name in ("confirmation.csv", "confirmation-fit.csv", "split-settings.json"):
        assert not (out / name).exists()
    assert (out / "notes.md").read_text(encoding="utf-8") == "keep these notes"
    assert (user_dir / "data.txt").read_text(encoding="utf-8") == "keep this data"


def test_unplotted_export_removes_the_previous_graph(analysis, tmp_path):
    out = report.write_explore(analysis.exploration, tmp_path / "out")
    assert (out / "network.png").stat().st_size > 0
    report.write_explore(analysis.exploration, out, plot=False)
    assert not (out / "network.png").exists()


def test_empty_links_overwrite_old_rows_and_keep_the_csv_schema(analysis, tmp_path):
    result = analysis.exploration
    out = report.write_explore(result, tmp_path / "out", plot=False)
    assert len(pd.read_csv(out / "links.csv"))
    empty = replace(result, nodes=result.nodes[result.nodes.depth == 0], links=result.links.iloc[:0],
                    pairs=result.pairs.iloc[:0], folds=result.folds.iloc[:0], bars=result.bars.iloc[:0],
                    fit=result.fit.iloc[:0], stops={"Y": "NOTHING_KEPT"})
    report.write_explore(empty, out, plot=False)
    links = pd.read_csv(out / "links.csv")
    assert links.empty and list(links.columns) == list(result.links.columns)
    assert "No link reached" in (out / "summary.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["style", "render", "split_json", "split_summary"])
def test_staging_failure_preserves_the_previous_export(analysis, tmp_path, monkeypatch, failure):
    out = report.write_split(analysis, tmp_path / "out", plot=False)
    before = _snapshot(out)
    changed = replace(analysis.exploration, settings={**analysis.exploration.settings, "seed": 777})
    if failure == "style":
        with pytest.raises(ValueError, match="node_size"):
            report.write_explore(changed, out, node_size=0)
    elif failure == "render":
        def fail_render(*args, **kwargs):
            raise RuntimeError("simulated render failure")
        monkeypatch.setattr(report, "plot_explore", fail_render)
        with pytest.raises(RuntimeError, match="simulated render failure"):
            report.write_explore(changed, out)
    elif failure == "split_json":
        bad = SplitResult(changed, analysis.confirmation, {**analysis.settings, "bad": {1}}, analysis.fit)
        with pytest.raises(TypeError, match="serializable"):
            report.write_split(bad, out, plot=False)
    else:
        settings = {k: v for k, v in analysis.settings.items() if k != "split_seed"}
        bad = SplitResult(changed, analysis.confirmation, settings, analysis.fit)
        with pytest.raises(KeyError, match="split_seed"):
            report.write_split(bad, out, plot=False)
    assert _snapshot(out) == before


def test_split_export_publishes_its_final_summary_and_settings(analysis, tmp_path):
    out = report.write_explore(analysis.exploration, tmp_path / "out", plot=False)
    assert report.write_split(analysis, out) == out
    assert json.loads((out / "split-settings.json").read_text(encoding="utf-8")) == analysis.settings
    assert (out / "summary.txt").read_text(encoding="utf-8") == analysis.summary() + "\n"
    assert len(pd.read_csv(out / "confirmation.csv")) == len(analysis.confirmation)
    assert (out / "network.png").stat().st_size > 0


def test_publication_failure_rolls_back_replaced_files(analysis, tmp_path, monkeypatch):
    out = report.write_split(analysis, tmp_path / "out", plot=False)
    before = _snapshot(out)
    original = Path.replace

    def fail_publish(path, target):
        if path.name == "summary.txt" and path.parent.name.startswith(".open-export-"):
            raise OSError("simulated publication failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    changed = replace(analysis.exploration, settings={**analysis.exploration.settings, "seed": 777})
    with pytest.raises(OSError, match="simulated publication failure"):
        report.write_explore(changed, out, plot=False)
    assert _snapshot(out) == before


def test_optional_file_removal_failure_rolls_back_the_export(analysis, tmp_path, monkeypatch):
    out = report.write_split(analysis, tmp_path / "out", plot=False)
    before = _snapshot(out)
    original = Path.unlink

    def fail_removal(path, *args, **kwargs):
        if path == out.resolve() / "confirmation-fit.csv":
            raise PermissionError("simulated stale-file removal failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_removal)
    with pytest.raises(PermissionError, match="simulated stale-file removal failure"):
        report.write_explore(analysis.exploration, out, plot=False)
    assert _snapshot(out) == before


def test_failed_rollback_retains_recovery_backups(analysis, tmp_path, monkeypatch):
    out = report.write_split(analysis, tmp_path / "out", plot=False)
    old_settings = (out / "settings.json").read_bytes()
    original = Path.replace

    def fail_publish_and_restore(path, target):
        if path.name == "summary.txt" and path.parent.name.startswith(".open-export-"):
            raise OSError("simulated publication failure")
        if path.name == "settings.json" and path.parent.name == ".previous":
            raise OSError("simulated restoration failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_restore)
    changed = replace(analysis.exploration, settings={**analysis.exploration.settings, "seed": 777})
    with pytest.raises(OSError, match="backup|recover|restore"):
        report.write_explore(changed, out, plot=False)
    recovered = list(out.glob(".open-export-*/.previous/settings.json"))
    assert len(recovered) == 1 and recovered[0].read_bytes() == old_settings


def test_managed_filename_directory_is_preserved_and_rejected(analysis, tmp_path):
    out = report.write_split(analysis, tmp_path / "out", plot=False)
    (out / "network.png").mkdir()
    (out / "network.png" / "user.txt").write_text("keep", encoding="utf-8")
    before = _snapshot(out)
    with pytest.raises(ValueError, match="regular file"):
        report.write_explore(analysis.exploration, out, plot=False)
    assert _snapshot(out) == before
