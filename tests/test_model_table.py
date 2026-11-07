import re

import numpy as np
import pandas as pd
import pytest

from opennet.api import DiscoveryConfig, analyze, model_table
from opennet.model_table import _bounded, _number, _p_text, _stars


@pytest.fixture(scope="module")
def result():
    r = np.random.default_rng(21)
    n = 1500
    d = pd.DataFrame({"support": r.normal(size=n), "autonomy": r.normal(size=n)})
    d["strain"] = -.5 * d.support + r.normal(size=n)
    d["skill"] = .6 * d.autonomy + r.normal(size=n)
    d["wellbeing"] = -.4 * d.strain + .3 * d.skill + r.normal(size=n)
    d.loc[r.choice(n, 40, replace=False), "skill"] = np.nan
    return analyze(d, outcome="wellbeing", config=DiscoveryConfig(folds=5, interactions=False))


def test_number_formatting_follows_digits_and_apa_conventions():
    assert _number(0.5698, 3) == "0.570" and _number(0.5698, 2) == "0.57" and _number(-0.0001, 3) == "0.000"
    assert _bounded(0.228, 3) == ".228" and _bounded(1.0, 3) == "1.000"
    assert _p_text(0.0004, 3) == "<.001" and _p_text(0.0004, 2) == "<.01" and _p_text(0.0312, 3) == ".031"
    assert _p_text(np.nan, 3) == "" and _stars(.0004) == "***" and _stars(.02) == "*" and _stars(.2) == ""


def test_columns_layout_has_separate_bounds_and_a_p_column_without_asterisks(result):
    t = model_table(result)
    assert t.ci == "columns" and t.digits == 3
    assert t.models[-1] == "wellbeing" and set(t.models) == {"skill", "strain", "wellbeing"}
    stats = [s for _, s in t.data.columns]
    assert stats == ["Estimate", "Lower", "Upper", "p"] * len(t.models)
    assert "*" not in t.html.split("<table>")[1].split("</table>")[0]              # p has its own column
    assert "Lower" in t.html and "Upper" in t.html and "CI low" not in t.html
    cell = t.data.loc["Strain", ("Wellbeing", "Estimate")]
    assert re.fullmatch(r"-0\.\d{3}", cell)                                          # three decimals by default
    assert re.fullmatch(r"-0\.\d{3}", t.data.loc["Strain", ("Wellbeing", "Lower")])
    assert t.data.loc["Strain", ("Wellbeing", "p")] == "<.001"
    assert {"Observations", "R² Discovery", "R² Confirmation"} <= set(t.data.index)
    assert t.data.loc["R² Confirmation", ("Wellbeing", "Estimate")].startswith(".")


def test_digits_changes_every_non_whole_number(result):
    two = model_table(result, digits=2)
    assert re.fullmatch(r"-0\.\d{2}", two.data.loc["Strain", ("Wellbeing", "Estimate")])
    assert two.data.loc["Strain", ("Wellbeing", "p")] == "<.01"
    assert re.fullmatch(r"\.\d{2}", two.data.loc["R² Confirmation", ("Wellbeing", "Estimate")])
    for bad in (-1, 11, 2.5, True, "3"):
        with pytest.raises(ValueError, match="digits"):
            model_table(result, digits=bad)


def test_stacked_layout_puts_stars_on_the_estimate_and_the_interval_on_the_next_line(result):
    t = model_table(result, ci="stacked", digits=3)
    assert [s for _, s in t.data.columns] == ["Estimate"] * len(t.models)
    cell = t.data.loc["Strain", ("Wellbeing", "Estimate")]
    assert re.fullmatch(r"-0\.\d{3}\*\*\*\n\[-0\.\d{3}, -0\.\d{3}\]", cell)
    assert "<br>[" in t.html and "p</i> &lt; .001" in t.html                          # key to the asterisks
    assert "Lower" not in t.html


def test_untested_edges_show_the_discovery_coefficient_with_a_dagger_and_no_interval(result):
    t = model_table(result)
    daggers = t.data[t.data.map(lambda v: isinstance(v, str) and v.endswith("†")).any(axis=1)]
    for variable, row in daggers.iterrows():
        for (model, stat), value in row.items():
            if isinstance(value, str) and value.endswith("†"):
                assert row[(model, "Lower")] == "" and row[(model, "Upper")] == "" and row[(model, "p")] == ""
    assert daggers.empty or "Discovery-sample coefficient" in t.html


def test_save_and_validation(result, tmp_path):
    t = model_table(result, ci="stacked", labels={"wellbeing": "Well-Being"}, number=2)
    t.save(tmp_path / "m.html")
    t.save(tmp_path / "m.csv")
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "Well-Being" in page and "Table 2" in page
    assert "[" in (tmp_path / "m.csv").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        t.save(tmp_path / "m.txt")
    with pytest.raises(ValueError, match="ci must"):
        model_table(result, ci="inline")
    assert "Table" not in model_table(result, number=None).html
    assert t._repr_html_() == t.html


def test_a_learner_without_coefficients_is_refused_with_a_pointer():
    pytest.importorskip("lightgbm")
    r = np.random.default_rng(3)
    n = 500
    d = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n)})
    d["y"] = d.a - d.b + r.normal(size=n)
    res = analyze(d, outcome="y", engine="lightgbm", config=DiscoveryConfig(folds=5, interactions=False))
    with pytest.raises(ValueError, match="effects_table"):
        model_table(res.exploration if hasattr(res, "exploration") else res)
