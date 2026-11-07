"""An APA-style table of the stepwise models in a discovered network (similar in spirit to ``tab_model``).

``model_table`` puts one column group on every variable that has predictors, upstream variables first and the
final outcome last, with the predictors the procedure kept as rows. It is built from ``effects_table`` and shows
no new quantity:

* A link the confirmation tested with OLS shows the frozen confirmation slope, its HC4 95% confidence interval and
  its multiplicity-adjusted p-value.
* A link that was not tested (flagged links, blocks, Elastic-Net, LightGBM) shows the discovery-sample coefficient
  when there is one, marked with a dagger, and no interval or p-value. A learner without coefficients has no cell.
* Below the predictors: the number of observations and the target's cross-validated R² on the discovery and the
  confirmation sample, each only when available.

Two layouts, chosen by ``ci``:

* ``"columns"`` (default): Estimate, Lower, Upper and *p* are four columns per model. The p-value has its own
  column, so no asterisks are used.
* ``"stacked"``: one column per model. The estimate carries asterisks for its p-value and the interval sits on the
  next line in brackets: ``0.570***`` over ``[0.490, 0.640]``.

``digits`` (default 3) sets the decimals of every non-whole number: estimates, interval bounds, p-values and R².
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .effects import effects_table

CI_LAYOUTS = ("columns", "stacked")

_STYLE = """<style>
.opennet-table{font-family:'Times New Roman',Times,serif;font-size:15px;color:#000}
.opennet-table .lbl{font-weight:700}.opennet-table .ttl{font-style:italic;margin:2px 0 8px}
.opennet-table table{border-collapse:collapse}.opennet-table th,.opennet-table td{padding:3px 10px;font-weight:400}
.opennet-table tr.top th{border-top:1.5px solid #000}
.opennet-table th.grp{text-align:center;padding-bottom:0}
.opennet-table th.grp span{display:block;border-bottom:1px solid #000;padding-bottom:2px}
.opennet-table tr.sub th{border-bottom:1.5px solid #000;text-align:center}.opennet-table th.var{text-align:left}
.opennet-table .lab{text-align:left;padding-right:28px}
.opennet-table .n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.opennet-table .s{text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap}
.opennet-table .c{text-align:center}
.opennet-table tr.last td{border-bottom:1.5px solid #000}
.opennet-table .note{font-size:13px;margin-top:8px;max-width:820px}
</style>"""


def _title(name: str) -> str:
    return str(name).replace("_", " ").title()


def _number(value, digits: int) -> str:
    if value is None or pd.isna(value):
        return ""
    text = f"{float(value):.{digits}f}"
    return text.lstrip("-") if float(text) == 0 else text         # no "-0.000"


def _bounded(value, digits: int) -> str:
    """A number that cannot exceed 1 in size (p, R²), written without its leading zero as APA does."""
    text = _number(value, digits)
    return text[1:] if text.startswith("0") else text


def _p_text(p, digits: int) -> str:
    if p is None or pd.isna(p):
        return ""
    floor = 10.0 ** -digits
    return f"<{_bounded(floor, digits)}" if p < floor else _bounded(p, digits)


def _stars(p) -> str:
    return "" if p is None or pd.isna(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


@dataclass
class ModelTable:
    """A model table: ``data`` (plain strings, ready for CSV), ``html`` (a fragment) and ``save``.

    ``data`` has one row per predictor and fit statistic and one column per (model, statistic). In the
    "stacked" layout an estimate and its interval are in one cell separated by a line break.
    """
    data: pd.DataFrame
    html: str
    models: tuple
    digits: int
    ci: str

    def _repr_html_(self) -> str:
        return self.html

    def to_html(self) -> str:
        """A complete HTML page."""
        return ('<!doctype html><meta charset="utf-8"><body style="margin:28px">' + self.html + "</body>")

    def save(self, path) -> None:
        """Write the table: ``.html`` as a page, ``.csv`` as the data (the stacked layout keeps its line breaks)."""
        path = str(path)
        if path.lower().endswith(".csv"):
            self.data.to_csv(path)
        elif path.lower().endswith((".html", ".htm")):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_html())
        else:
            raise ValueError("path must end in .html or .csv")


def model_table(result, *, digits: int = 3, ci: str = "columns", labels: dict | None = None,
                title: str | None = None, number: int | str | None = 1) -> ModelTable:
    """Build the model table of a discovered network. See the module documentation.

    ``result`` is a split result (confirmation estimates are used) or a discovery result (discovery coefficients
    only). ``digits`` is the number of decimals for estimates, interval bounds, p-values and R² (default 3).
    ``ci`` is ``"columns"`` or ``"stacked"``. ``labels`` maps variable names to display names (default: the
    name with underscores as spaces, title-cased). ``title`` defaults to the final outcome; ``number`` is the
    table number ("Table 1"), or None to leave the label out.
    """
    if ci not in CI_LAYOUTS:
        raise ValueError(f"ci must be one of {CI_LAYOUTS}")
    if isinstance(digits, bool) or not isinstance(digits, (int, np.integer)) or not 0 <= digits <= 10:
        raise ValueError("digits must be an integer from 0 to 10")
    if getattr(result, "by_outcome", None) is not None:
        raise ValueError("pass one outcome's result, e.g. result.by_outcome[name]")
    labels = dict(labels or {})
    name = lambda v: labels.get(v, _title(v))

    effects = effects_table(result)
    effects = effects[effects.kind == "unique"]
    if not len(effects):
        raise ValueError("the result has no links to tabulate")
    if effects.estimate.isna().all() and effects.slope.isna().all():
        raise ValueError("no signed estimates: this learner has no coefficients. Use effects_table for the "
                         "measures it does have (delta_r2, direction).")

    depth = effects.groupby("target").depth.min()
    models = sorted(depth.index, key=lambda m: (-int(depth[m]), m))
    predictors = list(dict.fromkeys(effects.sort_values(["depth", "predictor"], ascending=[False, True],
                                                        kind="stable").predictor))
    cells = {}
    for row in effects.itertuples():
        tested = pd.notna(row.slope)
        est, lo, hi, p = ((row.slope, row.slope_ci_low, row.slope_ci_high, row.p_adjusted) if tested
                          else (row.estimate, np.nan, np.nan, np.nan))
        if pd.notna(est):
            cells[(row.target, row.predictor)] = (est, lo, hi, p, tested)

    split_fit = getattr(result, "fit", None)
    settings = (result.exploration.settings if hasattr(result, "exploration") else result.settings)
    observed = {}
    if split_fit is not None and "n" in getattr(split_fit, "columns", ()):
        observed = {r.target: int(r.n) for r in split_fit.itertuples()}
    n_total = settings.get("n")
    first = effects.drop_duplicates("target").set_index("target")
    fit_rows = [("Observations", {m: str(observed.get(m, n_total)) if observed.get(m, n_total) is not None else ""
                                  for m in models}, "Observations")]
    for label, column, key in (("<i>R</i><sup>2</sup> Discovery", "target_r2_discovery", "R² Discovery"),
                               ("<i>R</i><sup>2</sup> Confirmation", "target_r2_confirmation", "R² Confirmation")):
        if first[column].notna().any():
            fit_rows.append((label, {m: _bounded(first[column][m], digits) for m in models}, key))

    stats = ("Estimate", "Lower", "Upper", "p") if ci == "columns" else ("Estimate",)
    columns = pd.MultiIndex.from_tuples([(name(m), s) for m in models for s in stats])
    index, data_rows, body = [], [], ""
    for pred in predictors:
        data_row, tds = [], ""
        for m in models:
            c = cells.get((m, pred))
            if c is None:
                data_row += [""] * len(stats)
                tds += "<td></td>" * (4 if ci == "columns" else 1)
                continue
            est, lo, hi, p, tested = c
            dagger = "" if tested else "†"
            if ci == "columns":
                values = [_number(est, digits) + dagger, _number(lo, digits), _number(hi, digits),
                          _p_text(p, digits)]
                data_row += values
                esc = [_html.escape(v) for v in values]
                esc[0] = _html.escape(_number(est, digits)) + ("" if tested else "<sup>†</sup>")
                tds += "".join(f'<td class="n">{v}</td>' for v in esc)
            else:
                head = _number(est, digits) + _stars(p) + dagger
                interval = (f"[{_number(lo, digits)}, {_number(hi, digits)}]" if tested and pd.notna(lo) else "")
                data_row.append(head + ("\n" + interval if interval else ""))
                shown = _html.escape(_number(est, digits) + _stars(p)) + ("" if tested else "<sup>†</sup>")
                tds += f'<td class="s">{shown}' + (f"<br>{_html.escape(interval)}" if interval else "") + "</td>"
        index.append(name(pred))
        data_rows.append(data_row)
        body += f'<tr><td class="lab">{_html.escape(name(pred))}</td>{tds}</tr>'
    span = len(stats)
    foot = ""
    for k, (label, values, key) in enumerate(fit_rows):
        index.append(key)
        data_rows.append([values[m] if j == 0 else "" for m in models for j in range(span)])
        cls = ' class="last"' if k == len(fit_rows) - 1 else ""
        foot += f'<tr{cls}><td class="lab">{label}</td>' + "".join(
            f'<td class="c" colspan="{span}">{_html.escape(values[m])}</td>' for m in models) + "</tr>"
    data = pd.DataFrame(data_rows, index=pd.Index(index, name="Variable"), columns=columns)

    head1 = "".join(f'<th colspan="{span}" class="grp"><span>{_html.escape(name(m))}</span></th>' for m in models)
    sub = ("".join('<th class="sub">Estimate</th><th class="sub">Lower</th><th class="sub">Upper</th>'
                   '<th class="sub"><i>p</i></th>' for _ in models) if ci == "columns" else
           "".join("<th></th>" for _ in models))
    header = (f'<tr class="top"><th></th>{head1}</tr><tr class="sub"><th class="var">Variable</th>{sub}</tr>'
              if ci == "columns" else
              f'<tr class="top sub"><th class="var">Variable</th>'
              + "".join(f'<th>{_html.escape(name(m))}</th>' for m in models) + "</tr>")
    outcome = title or name(models[-1])
    note = ["<i>Note.</i> Cells are ordinary least squares slopes on the confirmation sample with HC4 95% "
            "confidence intervals" + (" (Lower, Upper)" if ci == "columns" else " in brackets")
            + "; <i>p</i> values are adjusted for multiplicity.",
            "Blank cells mean the variable is not in that model."]
    if any(label.endswith("Discovery") for label, _, _ in fit_rows):
        note.append("<i>R</i><sup>2</sup> Discovery is the held-out fit in the discovery sample.")
    if any(not c[4] for c in cells.values()):
        note.append("<sup>†</sup> Discovery-sample coefficient, not confirmed, so no interval or <i>p</i>.")
    if ci == "stacked":
        note.append("* <i>p</i> &lt; .05. ** <i>p</i> &lt; .01. *** <i>p</i> &lt; .001.")
    label_html = (f'<div class="lbl">Table {_html.escape(str(number))}</div>' if number is not None else "")
    fragment = (f'{_STYLE}<div class="opennet-table">{label_html}<div class="ttl">Stepwise Models Leading to '
                f'{_html.escape(outcome)}</div><table>{header}{body}{foot}</table>'
                f'<div class="note">{" ".join(note)}</div></div>')
    return ModelTable(data=data, html=fragment, models=tuple(models), digits=int(digits), ci=ci)
