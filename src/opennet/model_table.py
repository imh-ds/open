"""An APA-style table of the stepwise models in a discovered network (similar in spirit to ``tab_model``).

``model_table`` puts one column group on every variable that has predictors, upstream variables first and the
final outcome last, with the predictors the procedure kept as rows. It is built from ``effects_table`` and shows
no new quantity:

* A link the confirmation tested with OLS shows the frozen confirmation slope, its HC4 95% confidence interval and
  its multiplicity-adjusted p-value.
* A link that was not tested (flagged links, blocks, Elastic-Net, LightGBM) shows the discovery-sample coefficient
  when there is one, marked with a dagger and set in grey italics, and no interval or p-value. A learner without
  coefficients has no cell. Discovery and confirmation values are on the same scale within a table.
* ``coefficient="standardized"`` (default) shows the fully standardized beta, B x SD(predictor) / SD(outcome);
  ``"unstandardized"`` shows B in the variables' own units. Predictor-only scaling is never labelled beta. Where an
  SD is zero the cell is left blank. The conventions (sample, scaling, intervals) are in ``ModelTable.conventions``
  and in the table note.
* Below the predictors: the number of observations and the target's cross-validated R² on the discovery and the
  confirmation sample, each only when available.

Two layouts, chosen by ``ci``:

* ``"columns"`` (default): Estimate, Lower, Upper and *p* are four columns per model. The p-value has its own
  column, so no asterisks are used.
* ``"stacked"``: one column per model. The estimate carries asterisks for its p-value and the interval sits on the
  next line in parentheses (or square brackets with ``brackets="square"``): ``0.570***`` over
  ``(0.490, 0.640)``, under a ``β (95% CI)`` column header.

``digits`` (default 3) sets the decimals of every non-whole number: estimates, interval bounds, p-values and R².
"""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .effects import effects_table

CI_LAYOUTS = ("columns", "stacked")
BRACKETS = {"parentheses": ("(", ")"), "square": ("[", "]")}

_STYLE = """<style>
.opennet-table{font-family:'Times New Roman',Times,serif;font-size:15px;color:#000}
.opennet-table .lbl{font-weight:700}.opennet-table .ttl{font-style:italic;margin:2px 0 8px}
.opennet-table table{border-collapse:collapse}.opennet-table th,.opennet-table td{padding:3px 10px;font-weight:400}
.opennet-table tr.top th{border-top:1.5px solid #000}
.opennet-table th.mnum{text-align:center;font-style:italic;padding-bottom:0}
.opennet-table th.grp{text-align:center;padding-bottom:0}
.opennet-table th.grp span{display:block;border-bottom:1px solid #000;padding-bottom:2px}
.opennet-table tr.sub th{border-bottom:1.5px solid #000;text-align:center}
.opennet-table tr.sub th.var{text-align:left}
.opennet-table tr.body td{vertical-align:top}
.opennet-table .disc{font-style:italic;color:#6b6b6b}
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

    ``data`` has one row per predictor and fit statistic and one column per (model, statistic); ``model_labels``
    are the numbers set above the models ("Model 1", ...), written as an extra header row in a CSV. In the
    "stacked" layout an estimate and its interval are in one cell separated by a line break.
    """
    data: pd.DataFrame
    html: str
    models: tuple
    digits: int
    ci: str
    coefficient: str = "standardized"
    conventions: tuple = ()
    model_labels: tuple = ()

    def _repr_html_(self) -> str:
        return self.html

    def to_html(self) -> str:
        """A complete HTML page."""
        return ('<!doctype html><meta charset="utf-8"><body style="margin:28px">' + self.html + "</body>")

    def save(self, path) -> None:
        """Write the table: ``.html`` as a page, ``.csv`` as the data (the stacked layout keeps its line breaks)."""
        path = str(path)
        if path.lower().endswith(".csv"):
            data = self.data
            if self.model_labels:                    # the model numbers become the top header row of the file
                data = data.copy()
                data.columns = pd.MultiIndex.from_tuples(
                    [(self.model_labels[i // (len(data.columns) // len(self.models))], a, b)
                     for i, (a, b) in enumerate(data.columns)])
            data.to_csv(path)
        elif path.lower().endswith((".html", ".htm")):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_html())
        else:
            raise ValueError("path must end in .html or .csv")


_COEFFICIENTS = {"standardized": "standardized", "std": "standardized", "beta": "standardized",
                 "unstandardized": "unstandardized", "unstd": "unstandardized", "raw": "unstandardized",
                 "b": "unstandardized"}


def model_table(result, confirmation: pd.DataFrame | None = None, *, coefficient: str = "standardized",
                digits: int = 3, ci: str = "columns", brackets: str = "parentheses",
                labels: dict | None = None, model_labels=True, title: str | None = None,
                number: int | str | None = 1) -> ModelTable:
    """Build the model table of a discovered network. See the module documentation.

    ``result`` is a split result (confirmation estimates are used) or a discovery result (discovery coefficients
    only); ``confirmation`` supplies a confirmation table for a discovery result, as in ``effects_table``.
    ``coefficient`` is ``"standardized"`` (default; fully standardized beta, aliases "std" and "beta") or
    ``"unstandardized"`` (B in original units; aliases "unstd" and "raw"). ``digits`` is the number of decimals for estimates, interval bounds, p-values and R² (default 3).
    ``ci`` is ``"columns"`` or ``"stacked"``. ``brackets`` is ``"parentheses"`` (default) or ``"square"`` and
    sets what encloses the interval in the stacked layout and its header. ``labels`` maps variable names to display names (default: the
    name with underscores as spaces, title-cased). ``model_labels`` sets the row above the outcome names: True
    (default) numbers the models left to right ("Model 1", "Model 2", ...), False or None leaves the row out, and a
    list of strings (one per model, left to right) or a dict from outcome name to label sets your own. ``title`` defaults to the final outcome; ``number`` is the
    table number ("Table 1"), or None to leave the label out.
    """
    if ci not in CI_LAYOUTS:
        raise ValueError(f"ci must be one of {CI_LAYOUTS}")
    if brackets not in BRACKETS:
        raise ValueError(f"brackets must be one of {tuple(BRACKETS)}")
    left, right = BRACKETS[brackets]
    if coefficient not in _COEFFICIENTS:
        raise ValueError('coefficient must be "standardized" or "unstandardized"')
    coefficient = _COEFFICIENTS[coefficient]
    standardized = coefficient == "standardized"
    sym_text, sym_html = ("β", "β") if standardized else ("B", "<i>B</i>")
    if isinstance(digits, bool) or not isinstance(digits, (int, np.integer)) or not 0 <= digits <= 10:
        raise ValueError("digits must be an integer from 0 to 10")
    if getattr(result, "by_outcome", None) is not None:
        raise ValueError("pass one outcome's result, e.g. result.by_outcome[name]")
    labels = dict(labels or {})
    name = lambda v: labels.get(v, _title(v))

    confirmation = getattr(result, "confirmation", None) if confirmation is None else confirmation
    effects = effects_table(result, confirmation)
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
        if tested:                                   # confirmation: one scale throughout, never a fallback
            est, lo, hi = ((row.beta, row.beta_ci_low, row.beta_ci_high) if standardized
                           else (row.slope, row.slope_ci_low, row.slope_ci_high))
            p = row.p_adjusted
        else:                                        # discovery only: no interval, no p
            est, lo, hi, p = (row.estimate_beta if standardized else row.estimate), np.nan, np.nan, np.nan
        if pd.notna(est):
            cells[(row.target, row.predictor)] = (est, lo, hi, p, tested)

    settings = (result.exploration.settings if hasattr(result, "exploration") else result.settings)
    observed = {}
    if confirmation is not None and "n" in getattr(confirmation, "columns", ()):
        observed = {r.target: int(r.n) for r in confirmation.drop_duplicates("target").itertuples()}
    n_total = settings.get("n")
    first = effects.drop_duplicates("target").set_index("target")
    fit_rows = [("Observations", {m: str(observed.get(m, n_total)) if observed.get(m, n_total) is not None else ""
                                  for m in models}, "Observations")]
    for label, column, key in (("<i>R</i><sup>2</sup> Discovery", "target_r2_discovery", "R² Discovery"),
                               ("<i>R</i><sup>2</sup> Confirmation", "target_r2_confirmation", "R² Confirmation")):
        if first[column].notna().any():
            fit_rows.append((label, {m: _bounded(first[column][m], digits) for m in models}, key))

    stats = ((sym_text, "Lower", "Upper", "p") if ci == "columns" else (f"{sym_text} {left}95% CI{right}",))
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
                tds += "".join(f'<td class="n{" disc" if (not tested and k == 0) else ""}">{v}</td>'
                               for k, v in enumerate(esc))
            else:
                head = _number(est, digits) + _stars(p) + dagger
                interval = (f"{left}{_number(lo, digits)}, {_number(hi, digits)}{right}" if tested and pd.notna(lo) else "")
                data_row.append(head + ("\n" + interval if interval else ""))
                shown = _html.escape(_number(est, digits) + _stars(p)) + ("" if tested else "<sup>†</sup>")
                tds += (f'<td class="s{"" if tested else " disc"}">{shown}'
                        + (f"<br>{_html.escape(interval)}" if interval else "") + "</td>")
        index.append(name(pred))
        data_rows.append(data_row)
        body += f'<tr class="body"><td class="lab">{_html.escape(name(pred))}</td>{tds}</tr>'
    span = len(stats)
    foot = ""
    for k, (label, values, key) in enumerate(fit_rows):
        index.append(key)
        data_rows.append([values[m] if j == 0 else "" for m in models for j in range(span)])
        cls = ' class="last"' if k == len(fit_rows) - 1 else ""
        foot += f'<tr{cls}><td class="lab">{label}</td>' + "".join(
            f'<td class="c" colspan="{span}">{_html.escape(values[m])}</td>' for m in models) + "</tr>"
    data = pd.DataFrame(data_rows, index=pd.Index(index, name="Variable"), columns=columns)

    if model_labels is True:
        numbers = tuple(f"Model {k + 1}" for k in range(len(models)))
    elif model_labels is False or model_labels is None:
        numbers = ()
    elif isinstance(model_labels, dict):
        unknown = set(model_labels) - set(models)
        if unknown:
            raise ValueError(f"model_labels names unknown models: {sorted(unknown)}")
        numbers = tuple(str(model_labels.get(m, f"Model {k + 1}")) for k, m in enumerate(models))
    elif isinstance(model_labels, (list, tuple)) and not isinstance(model_labels, str):
        if len(model_labels) != len(models):
            raise ValueError(f"model_labels needs {len(models)} labels (one per model, left to right)")
        numbers = tuple(str(x) for x in model_labels)
    else:
        raise ValueError("model_labels must be True, False, a list of labels or a dict")
    head0 = ("".join(f'<th colspan="{span}" class="mnum">{_html.escape(x)}</th>' for x in numbers)
             if numbers else "")
    head1 = "".join(f'<th colspan="{span}" class="grp"><span>{_html.escape(name(m))}</span></th>' for m in models)
    if ci == "columns":
        sub = "".join(f'<th class="sub">{sym_html}</th><th class="sub">Lower</th><th class="sub">Upper</th>'
                      '<th class="sub"><i>p</i></th>' for _ in models)
    else:
        sub = "".join(f'<th class="sub">{sym_html} {left}95% CI{right}</th>' for _ in models)
    names_tag = "<tr>" if numbers else '<tr class="top">'
    names_row = f'{names_tag}<th></th>{head1}</tr>'
    number_row = f'<tr class="top"><th></th>{head0}</tr>' if numbers else ""
    header = (f'{number_row}{names_row}<tr class="sub"><th class="var">Variable</th>{sub}</tr>')

    outcome = title or name(models[-1])
    if standardized:
        scale = ("β is the fully standardized coefficient, <i>B</i> × <i>SD</i>(predictor) / <i>SD</i>(outcome), with "
                 "sample standard deviations (<i>n</i> − 1) from the same complete rows as the slope; its "
                 "interval is scaled the same way and its <i>p</i> value is that of <i>B</i>.")
    else:
        scale = "<i>B</i> is the unstandardized slope, in the original units of the variables."
    note = ["<i>Note.</i> Cells are ordinary least squares slopes on the confirmation sample (complete cases for "
            "each model's variables; <i>n</i> is shown per model) with HC4 95% confidence intervals"
            + (" (Lower, Upper)" if ci == "columns" else
               (" in parentheses" if brackets == "parentheses" else " in square brackets"))
            + "; <i>p</i> values are adjusted for multiplicity.", scale,
            "Blank cells are variables not kept as a predictor in that model; they were not necessarily absent "
            "from the pool the procedure assessed."]
    if any(label.endswith("Discovery") for label, _, _ in fit_rows):
        note.append("<i>R</i><sup>2</sup> Discovery is the held-out fit in the discovery sample.")
    if any(not c[4] for c in cells.values()):
        penalized = bool((effects.direction_model == "elastic_net").any())
        extra = (" scaled by the training-fold standard deviation of the outcome" if standardized else "")
        note.append('<span class="disc"><sup>†</sup> Discovery-sample coefficient (median over training folds'
                    + ("; penalized" if penalized else "") + ")" + extra
                    + ", not confirmed, so no interval or <i>p</i>.</span>")
    if ci == "stacked":
        note.append("* <i>p</i> &lt; .05. ** <i>p</i> &lt; .01. *** <i>p</i> &lt; .001.")
    conventions = tuple(re.sub(r"<[^>]+>", "", t).replace("&lt;", "<").removeprefix("Note. ") for t in note)
    label_html = (f'<div class="lbl">Table {_html.escape(str(number))}</div>' if number is not None else "")
    fragment = (f'{_STYLE}<div class="opennet-table">{label_html}<div class="ttl">Stepwise Models Leading to '
                f'{_html.escape(outcome)}</div><table>{header}{body}{foot}</table>'
                f'<div class="note">{" ".join(note)}</div></div>')
    return ModelTable(data=data, html=fragment, models=tuple(models), digits=int(digits), ci=ci,
                      coefficient=coefficient, conventions=conventions, model_labels=numbers)
