import re
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from opennet.api import DiscoveryConfig, analyze, model_table, effects_table, split_sample
from opennet import explore_report as report
from opennet.confirm import frozen_network
from opennet.model_table import _bounded, _number, _p_text, _stars


def _data(n=1500, seed=21):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"support": r.normal(size=n) * 3 + 10, "autonomy": r.normal(size=n) * .5})
    d["strain"] = -.5 * (d.support - 10) / 3 * 4 + r.normal(size=n) * 4 + 50      # very different scales
    d["skill"] = .6 * d.autonomy / .5 * 2 + r.normal(size=n) * 2 + 7
    d["wellbeing"] = (-.4 * (d.strain - 50) / 4 + .3 * (d.skill - 7) / 2) * 5 + r.normal(size=n) * 5 + 30
    d.loc[r.choice(n, 40, replace=False), "skill"] = np.nan
    return d


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.fixture(scope="module")
def result(data):
    return analyze(data, outcome="wellbeing", config=DiscoveryConfig(folds=5, interactions=False))


def _hc4(X, y):
    """Independent HC4: slopes and standard errors (no intercept row), computed straight from the formulas."""
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    k = p + 1
    inv = np.linalg.inv(A.T @ A)
    b = inv @ A.T @ y
    e = y - A @ b
    h = np.einsum("ij,jk,ik->i", A, inv, A)
    w = e ** 2 / (1 - h) ** np.minimum(4, n * h / k)
    cov = inv @ (A.T * w) @ A @ inv
    return b[1:], np.sqrt(np.diag(cov))[1:], n - k


def _confirmation_rows(result, data, target):
    _, conf = split_sample(data, result.settings["explore_fraction"], result.settings["split_seed"])
    pool = frozen_network(result.exploration).query("target == @target").pool.iloc[0]
    frame = conf[[target, *pool]].dropna()
    return pool, frame[list(pool)].to_numpy(float), frame[target].to_numpy(float)


def test_number_formatting_follows_digits_and_apa_conventions():
    assert _number(0.5698, 3) == "0.570" and _number(0.5698, 2) == "0.57" and _number(-0.0001, 3) == "0.000"
    assert _bounded(0.228, 3) == ".228" and _bounded(1.0, 3) == "1.000"
    assert _p_text(0.0004, 3) == "<.001" and _p_text(0.0004, 2) == "<.01" and _p_text(0.0312, 3) == ".031"
    assert _p_text(np.nan, 3) == "" and _stars(.0004) == "***" and _stars(.02) == "*" and _stars(.2) == ""


def test_standardized_beta_and_its_interval_match_an_independent_calculation(result, data):
    t = model_table(result, digits=6)                                  # default coefficient: standardized
    assert t.coefficient == "standardized" and [s for _, s in t.data.columns][:4] == ["β", "Lower", "Upper", "p"]
    checked = 0
    for target in effects_table(result).target.unique():
        pool, X, y = _confirmation_rows(result, data, target)
        b, se, df = _hc4(X, y)
        z = lambda a: (a - a.mean(0)) / a.std(0, ddof=1)               # standardize BOTH sides, then refit
        beta, beta_se, _ = _hc4(z(X), z(y))
        crit = stats.t.ppf(.975, df)
        for i, predictor in enumerate(pool):
            label = predictor.replace("_", " ").title()
            cell = t.data.loc[label, (target.replace("_", " ").title(), "β")] if label in t.data.index else ""
            if cell == "" or cell.endswith("†"):
                continue
            lo = t.data.loc[label, (target.replace("_", " ").title(), "Lower")]
            hi = t.data.loc[label, (target.replace("_", " ").title(), "Upper")]
            assert float(cell) == pytest.approx(beta[i], abs=1e-6)
            assert float(cell) == pytest.approx(b[i] * X[:, i].std(ddof=1) / y.std(ddof=1), abs=1e-6)
            assert float(lo) == pytest.approx(beta[i] - crit * beta_se[i], abs=1e-6)
            assert float(hi) == pytest.approx(beta[i] + crit * beta_se[i], abs=1e-6)
            checked += 1
    assert checked >= 3


def test_unstandardized_option_shows_b_in_original_units_and_never_mixes_scales(result, data):
    raw = model_table(result, coefficient="unstandardized", digits=6)
    assert raw.coefficient == "unstandardized" and [s for _, s in raw.data.columns][:4] == ["B", "Lower", "Upper", "p"]
    assert model_table(result, coefficient="unstd").coefficient == "unstandardized"
    assert model_table(result, coefficient="std").coefficient == "standardized"
    pool, X, y = _confirmation_rows(result, data, "wellbeing")
    b, se, df = _hc4(X, y)
    for i, predictor in enumerate(pool):
        label = predictor.replace("_", " ").title()
        if label in raw.data.index and raw.data.loc[label, ("Wellbeing", "B")] not in ("",):
            cell = raw.data.loc[label, ("Wellbeing", "B")]
            if not cell.endswith("†"):
                assert float(cell) == pytest.approx(b[i], abs=1e-6)
                assert abs(float(cell)) > 0.05                       # original units (SDs differ), not a beta
    std = model_table(result, digits=6)
    assert std.html.count("<i>B</i>") >= 1 and "β [" not in raw.html
    with pytest.raises(ValueError, match="coefficient"):
        model_table(result, coefficient="partial")


def test_confirmation_table_carries_beta_se_and_the_scaling_it_used(result, data):
    c = result.confirmation
    tested = c[c.p.notna()]
    assert {"beta", "beta_se", "beta_ci_low", "beta_ci_high", "sd_x", "sd_y"} <= set(c.columns)
    assert np.allclose(tested.beta, tested.coef * tested.sd_x / tested.sd_y)
    assert np.allclose(tested.beta_se, tested.se * tested.sd_x / tested.sd_y)
    assert (tested.beta_ci_low < tested.beta).all() and (tested.beta < tested.beta_ci_high).all()
    pool, X, y = _confirmation_rows(result, data, "wellbeing")
    row = tested[(tested.target == "wellbeing")].iloc[0]
    assert row.sd_y == pytest.approx(y.std(ddof=1)) and row.sd_x == pytest.approx(X[:, pool.index(row.source)].std(ddof=1))


def test_discovery_only_cells_are_on_the_same_scale_flagged_in_grey_italics_and_have_no_interval(result, data):
    e = effects_table(result)
    flagged = e[(e.kind == "unique") & e.slope.isna() & e.estimate.notna()]
    assert len(flagged)
    row = flagged.iloc[0]
    sd_y = data[row.target].std()
    assert row.estimate_beta == pytest.approx(row.estimate_std / sd_y, rel=.15)    # training-fold SD ≈ full SD
    assert abs(row.estimate_std) != pytest.approx(abs(row.estimate_beta))          # predictor-only is not beta
    t = model_table(result, digits=6)
    model = row.target.replace("_", " ").title()
    cell = t.data.loc[row.predictor.replace("_", " ").title(), (model, "β")]
    assert cell.endswith("†") and float(cell[:-1]) == pytest.approx(row.estimate_beta, abs=1e-6)
    assert t.data.loc[row.predictor.replace("_", " ").title(), (model, "Lower")] == ""
    assert t.data.loc[row.predictor.replace("_", " ").title(), (model, "p")] == ""
    assert 'class="n disc"' in t.html and ".disc{font-style:italic;color:" in t.html
    raw = model_table(result, coefficient="unstandardized", digits=6)
    raw_cell = raw.data.loc[row.predictor.replace("_", " ").title(), (model, "B")]
    assert float(raw_cell[:-1]) == pytest.approx(row.estimate, abs=1e-6)            # the raw scale, not mixed


def test_a_zero_variance_confirmation_predictor_leaves_blanks_instead_of_a_fabricated_beta(data):
    result = analyze(data, outcome="wellbeing", config=DiscoveryConfig(folds=5, interactions=False))
    _, conf = split_sample(data, result.settings["explore_fraction"], result.settings["split_seed"])
    broken = conf.assign(strain=1.0)                       # constant in the confirmation sample
    from opennet.confirm import confirm
    table = confirm(result.exploration, broken)
    target = table[table.target == "wellbeing"]
    assert target.p.isna().all() and target.beta.isna().all() and target.beta_se.isna().all()
    t = model_table(result.exploration, table, digits=4)
    assert not t.data.map(lambda v: isinstance(v, str) and v.lower() == "nan").any().any()


def test_columns_layout_has_separate_bounds_and_a_p_column_without_asterisks(result):
    t = model_table(result)
    assert t.ci == "columns" and t.digits == 3
    assert t.models[-1] == "wellbeing" and set(t.models) == {"skill", "strain", "wellbeing"}
    assert "*" not in t.html.split("<table>")[1].split("</table>")[0]              # p has its own column
    assert "Lower" in t.html and "Upper" in t.html and "CI low" not in t.html
    cell = t.data.loc["Strain", ("Wellbeing", "β")]
    assert re.fullmatch(r"-0\.\d{3}", cell)                                          # three decimals by default
    assert t.data.loc["Strain", ("Wellbeing", "p")] == "<.001"
    assert {"Observations Discovery", "Observations Confirmation", "R² Discovery",
            "R² Confirmation"} <= set(t.data.index)
    assert t.data.loc["R² Confirmation", ("Wellbeing", "β")].startswith(".")


def test_observations_distinguish_the_discovery_and_confirmation_samples(result, data):
    t = model_table(result)
    explore_part, _ = split_sample(data, result.settings["explore_fraction"], result.settings["split_seed"])
    for target in t.models:
        label = target.replace("_", " ").title()
        pool, X, y = _confirmation_rows(result, data, target)
        assert int(t.data.loc["Observations Confirmation", (label, "β")]) == len(y)
        assert int(t.data.loc["Observations Discovery", (label, "β")]) == int(explore_part[target].notna().sum())
    assert "Observations" not in t.data.index                              # replaced by the two rows
    rows = re.findall(r"<tr[^>]*>.*?</tr>", t.html, flags=re.S)
    assert any("Observations Discovery" in r for r in rows) and any("Observations Confirmation" in r for r in rows)
    assert 'class="last"' not in "".join(rows[:-1])                        # still no rule between the fit rows


def test_observations_collapse_to_one_row_when_the_samples_are_the_same_size(result):
    same = result.confirmation.copy()
    fit = result.exploration.fit.set_index("target").n
    same["n"] = same.target.map(fit)
    t = model_table(result.exploration, same)
    assert "Observations" in t.data.index and "Observations Discovery" not in t.data.index
    only_discovery = model_table(result.exploration)                           # no confirmation: discovery counts
    assert "Observations" in only_discovery.data.index
    assert only_discovery.data.loc["Observations"].iloc[0] == str(int(fit[only_discovery.models[0]]))


def test_digits_changes_every_non_whole_number(result):
    two = model_table(result, digits=2)
    assert re.fullmatch(r"-0\.\d{2}", two.data.loc["Strain", ("Wellbeing", "β")])
    assert two.data.loc["Strain", ("Wellbeing", "p")] == "<.01"
    assert re.fullmatch(r"\.\d{2}", two.data.loc["R² Confirmation", ("Wellbeing", "β")])
    for bad in (-1, 11, 2.5, True, "3"):
        with pytest.raises(ValueError, match="digits"):
            model_table(result, digits=bad)


def test_stacked_layout_puts_stars_on_the_estimate_and_the_interval_on_the_next_line(result):
    t = model_table(result, ci="stacked", digits=3)
    assert [s for _, s in t.data.columns] == ["β (95% CI)"] * len(t.models)            # parentheses by default
    cell = t.data.loc["Strain", ("Wellbeing", "β (95% CI)")]
    assert re.fullmatch(r"-0\.\d{3}\*\*\*\n\(-0\.\d{3}, -0\.\d{3}\)", cell)
    assert "<br>(" in t.html and "p</i> &lt; .001" in t.html and "in parentheses" in t.html
    assert t.html.count("<th class=\"sub\">Lower") == 0
    square = model_table(result, ci="stacked", brackets="square")
    assert [s for _, s in square.data.columns] == ["β [95% CI]"] * len(t.models)
    assert re.fullmatch(r"-0\.\d{3}\*\*\*\n\[-0\.\d{3}, -0\.\d{3}\]",
                        square.data.loc["Strain", ("Wellbeing", "β [95% CI]")])
    assert "<br>[" in square.html and "in square brackets" in square.html
    with pytest.raises(ValueError, match="brackets"):
        model_table(result, ci="stacked", brackets="braces")


def test_stacked_header_has_a_coefficient_row_aligned_with_variable_and_top_aligned_body(result):
    for kind, label in (("standardized", "β (95% CI)"), ("unstandardized", "<i>B</i> (95% CI)")):
        t = model_table(result, ci="stacked", coefficient=kind)
        rows = re.findall(r"<tr[^>]*>.*?</tr>", t.html, flags=re.S)
        numbers, names, second = rows[0], rows[1], rows[2]
        assert 'class="top"' in numbers and "Model 1" in numbers and "Variable" not in numbers
        assert "top" not in re.match(r"<tr[^>]*>", names).group(0) and "Wellbeing" in names
        assert '<th class="var">Variable</th>' in second and second.count(label) == len(t.models)
        assert 'class="sub"' in second and "top" not in re.match(r"<tr[^>]*>", second).group(0)   # no top rule here
        body = [r for r in rows if r.startswith('<tr class="body"')]
        assert len(body) >= 4 and all("Observations" not in r for r in body)
        assert ".opennet-table tr.body td{vertical-align:top}" in t.html
        assert ".opennet-table tr.sub th.var{text-align:left}" in t.html
        assert rows[-1].startswith('<tr class="last"') and "R</i><sup>2</sup> Confirmation" in rows[-1]
        assert 'class="last"' not in "".join(rows[:-1])                        # no rule between the fit rows


def test_untested_edges_carry_a_dagger_and_no_interval_in_both_layouts(result):
    t = model_table(result)
    flagged_cells = [(v, row[(m, "Lower")]) for _, row in t.data.iterrows() for (m, s), v in row.items()
                     if s == "β" and isinstance(v, str) and v.endswith("†")]
    assert flagged_cells and all(lo == "" for _, lo in flagged_cells)
    stacked = model_table(result, ci="stacked")
    cells = [v for v in stacked.data.to_numpy().ravel() if isinstance(v, str) and v.endswith("†")]
    assert cells and all("(" not in v for v in cells)
    assert 'class="s disc"' in stacked.html and "grey" not in stacked.html


def test_export_formats_carry_the_labels_and_the_conventions(result, tmp_path):
    t = model_table(result, ci="stacked", labels={"wellbeing": "Well-Being"}, number=2)
    t.save(tmp_path / "m.html")
    t.save(tmp_path / "m.csv")
    page = (tmp_path / "m.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "Well-Being" in page and "Table 2" in page
    csv = pd.read_csv(tmp_path / "m.csv", header=[0, 1, 2], index_col=0)
    assert ("Model 3", "Well-Being", "β (95% CI)") in csv.columns and "Observations Discovery" in csv.index
    model_table(result, ci="stacked", model_labels=False).save(tmp_path / "plain.csv")
    plain = pd.read_csv(tmp_path / "plain.csv", header=[0, 1], index_col=0)
    assert ("Skill", "β (95% CI)") in plain.columns                             # no number row, no extra header
    assert re.search(r"\(-?0\.\d{3}, -?0\.\d{3}\)", (tmp_path / "m.csv").read_text(encoding="utf-8"))
    text = " ".join(t.conventions)
    assert "fully standardized" in text and "complete rows" in text and "not necessarily absent" in text
    assert "training-fold standard deviation" in text and "adjusted for multiplicity" in text
    assert "fully standardized" in page and "Blank cells" in page
    raw = model_table(result, coefficient="unstandardized")
    assert "original units" in " ".join(raw.conventions) and "fully standardized" not in " ".join(raw.conventions)
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


def test_model_numbers_sit_above_the_outcome_names_with_only_a_top_border_in_both_layouts(result):
    for ci in ("columns", "stacked"):
        t = model_table(result, ci=ci)
        rows = re.findall(r"<tr[^>]*>.*?</tr>", t.html, flags=re.S)
        numbers, names = rows[0], rows[1]
        assert t.model_labels == ("Model 1", "Model 2", "Model 3")
        assert numbers.startswith('<tr class="top">') and numbers.count('class="mnum"') == 3
        assert [m for m in re.findall(r"Model \d", numbers)] == ["Model 1", "Model 2", "Model 3"]
        assert not names.startswith('<tr class="top">')                    # the top rule moved up to the numbers
        assert ".opennet-table th.mnum{text-align:center;font-style:italic;padding-bottom:0}" in t.html
        assert "th.mnum{" in t.html and "mnum{" in t.html.split("th.grp")[0]   # numbers have no rule of their own
        span = 4 if ci == "columns" else 1
        assert numbers.count(f'colspan="{span}"') == 3
        assert t.html.index("Model 1") < t.html.index("Skill") < t.html.index("Variable")


def test_model_labels_can_be_customised_or_turned_off(result):
    custom = model_table(result, model_labels=["Step A", "Step B", "Final"])
    assert custom.model_labels == ("Step A", "Step B", "Final") and "Step B" in custom.html
    by_name = model_table(result, model_labels={"wellbeing": "Full model"})
    assert by_name.model_labels == ("Model 1", "Model 2", "Full model")
    off = model_table(result, model_labels=False)
    rows = re.findall(r"<tr[^>]*>.*?</tr>", off.html, flags=re.S)
    assert off.model_labels == () and "Model 1" not in off.html and rows[0].startswith('<tr class="top">')
    assert "Wellbeing" in rows[0]                                          # the names row carries the top rule again
    for bad in (["only one"], {"nobody": "x"}, "Model 1", 3):
        with pytest.raises(ValueError, match="model_labels"):
            model_table(result, model_labels=bad)


def test_include_matches_the_links_of_the_graph_and_the_confirmation(result):
    e = effects_table(result, include="all")
    unique = e[e.kind == "unique"]
    drawn = model_table(result)
    kept = model_table(result, include="kept")
    confirmed = model_table(result, include="confirmed")

    def cells(t):
        return {(m, v) for v in t.data.index for (m, s), x in t.data.loc[v].items()
                if s == "β" and v not in _FIT_ROWS and x != ""}

    def expected(frame):
        return {(t.replace("_", " ").title(), p.replace("_", " ").title()) for t, p in zip(frame.target, frame.predictor)}

    in_graph_solid = unique[unique.tier == "kept"]
    assert cells(kept) == expected(in_graph_solid)                         # exactly the solid edges
    assert cells(kept) <= cells(drawn) and cells(confirmed) <= cells(kept)
    assert cells(drawn) - cells(kept)                                       # the flagged link is what drops
    flagged = unique[(unique.tier == "flagged")]
    for _, row in flagged.iterrows():
        assert (row.target.replace("_", " ").title(), row.predictor.replace("_", " ").title()) not in cells(kept)
    confirmed_pairs = unique[unique.confirmed.eq(True)]
    assert cells(confirmed) == expected(confirmed_pairs)
    assert "†" not in confirmed.html.split("<table>")[1].split("</table>")[0]   # confirmed cells are all tested
    assert "flagged links are not" in kept.html and "confirmed in the confirmation sample" in confirmed.html
    assert any("Only links kept" in c for c in kept.conventions)


def test_include_is_validated_and_confirmed_needs_a_confirmation(result):
    with pytest.raises(ValueError, match="include"):
        model_table(result, include="all")
    with pytest.raises(ValueError, match="needs a confirmation"):
        model_table(result.exploration, include="confirmed")
    assert model_table(result, include="kept", ci="stacked").ci == "stacked"


_FIT_ROWS = {"Observations", "Observations Discovery", "Observations Confirmation", "R² Discovery", "R² Confirmation"}


def _cells(t):
    """{(model label, predictor label)} for every filled predictor cell of a model table."""
    first = t.data.columns.get_level_values(1)[0]
    return {(m, v) for v in t.data.index if v not in _FIT_ROWS
            for (m, s), x in t.data.loc[v].items() if s == first and x != ""}


def _forge_tiers(result, target, tier):
    """The same result with every kept link of ``target`` demoted to ``tier`` (its estimates are untouched)."""
    links = result.exploration.links.copy()
    links.loc[(links.target == target) & (links.tier == "kept"), "tier"] = tier
    return replace(result.exploration, links=links)


def test_a_model_without_a_retained_predictor_has_no_column_but_stays_a_predictor_row(result):
    forged = _forge_tiers(result, "strain", "flagged")                   # strain keeps nothing: it is only flagged
    kept = model_table(forged, result.confirmation, include="kept")
    assert "strain" not in kept.models
    assert "Strain" not in kept.data.columns.get_level_values(0)
    assert "Strain" in kept.data.index and ("Wellbeing", "Strain") in _cells(kept)   # still a row, for wellbeing
    drawn = model_table(forged, result.confirmation, include="drawn")
    assert "strain" in drawn.models and ("Strain", "Support") in _cells(drawn)        # flagged links bring it back
    assert 'class="n disc"' in drawn.html and "†" in drawn.html                       # as discovery-only † cells
    assert "†" not in kept.html.split("<table>")[1].split("</table>")[0]
    nothing = result.exploration.links.assign(tier="flagged")
    with pytest.raises(ValueError, match='include="kept"'):
        model_table(replace(result.exploration, links=nothing), result.confirmation, include="kept")


def test_predictor_rows_follow_their_first_appearance_with_a_deterministic_tie_break(result):
    t = model_table(result, include="kept")
    outcomes = list(dict.fromkeys(t.data.columns.get_level_values(0)))                # left to right
    stat = t.data.columns[0][1]
    rows = [v for v in t.data.index if v not in _FIT_ROWS]
    first_model = {v: min(i for i, m in enumerate(outcomes) if t.data.loc[v, (m, stat)] != "") for v in rows}
    order = [first_model[v] for v in rows]
    assert order == sorted(order)                                         # never a later model before an earlier one
    for a, b in zip(rows, rows[1:]):                                      # ties: alphabetical by display name
        if first_model[a] == first_model[b]:
            assert a.casefold() <= b.casefold()
    assert rows == ["Autonomy", "Support", "Skill", "Strain"]            # Skill and Strain first appear together
    assert first_model["Skill"] == first_model["Strain"] > first_model["Support"]
    shuffled = replace(result.exploration, links=result.exploration.links.iloc[::-1].reset_index(drop=True))
    again = [v for v in model_table(shuffled, result.confirmation, include="kept").data.index if v not in _FIT_ROWS]
    assert again == rows                                                  # link order in the result does not matter


def test_a_variable_at_the_depth_cap_is_a_predictor_row_never_an_outcome_column(data):
    config = DiscoveryConfig(folds=5, interactions=False, max_depth=1)
    t = model_table(analyze(data, outcome="wellbeing", config=config), include="kept")
    assert list(t.models) == ["wellbeing"]                                # strain and skill were never assessed
    assert {"Strain", "Skill"} <= set(t.data.index)
    assert not ({"Strain", "Skill"} & set(t.data.columns.get_level_values(0)))
    assert "Model 2" not in t.html
    deeper = model_table(analyze(data, outcome="wellbeing",
                                 config=DiscoveryConfig(folds=5, interactions=False)), include="kept")
    assert len(deeper.models) > len(t.models)                             # without a cap the models are assessed


def test_include_kept_matches_the_graphs_retained_edges(result):
    fig = report.plot_explore(result.exploration, show_flagged=False)
    labels = [x.get_text() for x in fig.axes[0].texts if re.search(r"\(\d+/\d+\)", x.get_text())]
    plt.close(fig)
    links = result.exploration.links
    kept = model_table(result, include="kept")
    assert len(labels) == len(_cells(kept)) == int(((links.tier == "kept") & (links.kind == "unique")).sum())
    flagged = links[(links.tier == "flagged") & (links.kind == "unique")]
    assert len(_cells(model_table(result, include="drawn"))) == len(_cells(kept)) + len(flagged)


def test_new_rows_reach_html_and_csv_in_both_layouts(result, tmp_path):
    for ci in ("columns", "stacked"):
        t = model_table(result, ci=ci, include="kept")
        t.save(tmp_path / f"{ci}.csv")
        t.save(tmp_path / f"{ci}.html")
        csv = pd.read_csv(tmp_path / f"{ci}.csv", header=[0, 1, 2], index_col=0)
        assert {"Observations Discovery", "Observations Confirmation"} <= set(csv.index)
        assert list(csv.index[:4]) == ["Autonomy", "Support", "Skill", "Strain"]
        page = (tmp_path / f"{ci}.html").read_text(encoding="utf-8")
        assert "Observations Discovery" in page and "Only links kept in discovery" in page
        assert page.index("Autonomy</td>") < page.index("Support</td>") < page.index("Skill</td>") < page.index("Strain</td>")
