"""Fixed-denominator matched-chain recovery gate; original null rules stay frozen."""
import copy
from pathlib import Path
import qualify_tree_recursive_validation as base

SCHEMA = "open.tree_recursive_signal.v1"
ENGINES = base.ENGINES


def summarize_cell(expected_cases, audited_cases):
    audited = copy.deepcopy(audited_cases)
    for row in audited:
        if type(row.get("exact_network")) is not bool or (row.get("exact_network") and row.get("false_network")):
            row.update(status="audit_failed", fit_cost_unknown=True)
    cell = base.summarize_cell(expected_cases, audited)
    count = sum(not row["unknown"] and row.get("exact_network") is True for row in cell["cases"])
    cell["exact_recovery"] = count
    cell["recovery_interval"] = base.clopper_pearson(count, cell["planned"])
    cell["passed"] = cell["passed"] and cell["recovery_interval"]["lower"] >= .80
    return cell


def qualify_study(wave80, wave200):
    result = base.qualify_study(wave80, wave200)
    result["schema"] = SCHEMA
    return result


def render_markdown(report):
    lines = ["# Recursive matched-chain validation", "",
             "Truth: X1 -> X2 -> Y, at depths 2 -> 1 -> 0. Z1/Z2 are unrelated.",
             "All cases use shipped settings; this is linear-chain evidence only.", "",
             "|Engine|N|Complete / 55|Unknown|Exact network+depth|Extra findings|Recovery 95% lower|Error 95% upper|Gate|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cell in report["cells"]:
        lines.append(f"|{cell['engine']}|{cell['sample_size']}|{cell['complete']}|{cell['unknown']}|"
                     f"{cell['exact_recovery']}|{cell['incorrect']}|{cell['recovery_interval']['lower']:.6f}|"
                     f"{cell['incorrect_interval']['upper']:.6f}|{'PASS' if cell['passed'] else 'STOP'}|")
    lines += ["", "Missing/incomplete/audit-failed cases remain unknown and block a pass.",
              "A pass applies only to this linear Gaussian chain, sample size, and shipped setup.",
              "These per-cell confidence bounds do not provide family-wise error control."]
    return "\n".join(lines) + "\n"


def write_report(report, json_path, markdown_path):
    import json
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(markdown_path).write_text(render_markdown(report), encoding="utf-8")
