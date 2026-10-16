"""Fixed-denominator gate and report for the tree-engine null study."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from scipy.stats import beta


SCHEMA = "open.tree_recursive_null.v1"
ENGINES = ("random_forest", "gbm", "lightgbm")
PLANNED_PER_CELL = 55
FALSE_NETWORK_UPPER_LIMIT = 0.10


def clopper_pearson(successes, trials=PLANNED_PER_CELL, confidence=0.95):
    """Return the two-sided exact binomial interval without rounded gate logic."""
    if type(successes) is not int or type(trials) is not int:
        raise ValueError("success and trial counts must be integers")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial count")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return {"lower": lower, "upper": upper, "confidence": confidence}


def _required_identity(row):
    fields = ("case_id", "engine", "sample_size", "repetition", "manifest_sha256",
              "source_sha", "study_id", "attempt")
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError("case identity/provenance missing: " + ", ".join(missing))
    return fields


def _validate_expected(expected_cases):
    if len(expected_cases) != PLANNED_PER_CELL:
        raise ValueError("every null cell must retain 55 planned case identities")
    if any(not isinstance(row, Mapping) for row in expected_cases):
        raise ValueError("expected case must be a mapping")
    ids = [row.get("case_id") for row in expected_cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        raise ValueError("invalid expected case identity")
    if len(set(ids)) != PLANNED_PER_CELL:
        raise ValueError("duplicate expected case identity")
    engine_values = {row.get("engine") for row in expected_cases}
    sample_values = {row.get("sample_size") for row in expected_cases}
    study_values = {row.get("study_id") for row in expected_cases}
    manifest_values = {row.get("manifest_sha256") for row in expected_cases}
    source_values = {row.get("source_sha") for row in expected_cases}
    if len(engine_values) != 1 or next(iter(engine_values)) not in ENGINES:
        raise ValueError("expected cell must contain one supported engine")
    if len(sample_values) != 1 or next(iter(sample_values)) not in (80, 200):
        raise ValueError("expected cell must contain one supported sample size")
    if len(study_values) != 1 or not next(iter(study_values)):
        raise ValueError("expected cell must bind one study identity")
    if len(manifest_values) != 1 or not next(iter(manifest_values)):
        raise ValueError("expected cell must bind one manifest digest")
    if len(source_values) != 1 or not next(iter(source_values)):
        raise ValueError("expected cell must bind one source SHA")
    repetitions = [row.get("repetition") for row in expected_cases]
    if any(type(value) is not int for value in repetitions) or set(repetitions) != set(range(1, 56)):
        raise ValueError("expected repetitions must be exactly 1 through 55")
    if any(type(row.get("attempt")) is not int or row["attempt"] != 1 for row in expected_cases):
        raise ValueError("only first attempts belong in the frozen denominator")
    if re.fullmatch(r"[0-9a-f]{64}", next(iter(manifest_values))) is None:
        raise ValueError("expected manifest identity must be a SHA256 digest")
    if re.fullmatch(r"[0-9a-f]{40}", next(iter(source_values))) is None:
        raise ValueError("expected source identity must be a full Git SHA")
    for row in expected_cases:
        _required_identity(row)
    return ids


def summarize_cell(expected_cases: Sequence[Mapping[str, object]],
                   audited_cases: Sequence[Mapping[str, object]]):
    """Summarize one engine/N cell; absent and bad cases remain unknown."""
    expected_ids = _validate_expected(expected_cases)
    expected_by_id = {row["case_id"]: row for row in expected_cases}
    found = {}
    for row in audited_cases:
        if not isinstance(row, Mapping):
            raise ValueError("audit result must be a mapping")
        _required_identity(row)
        case_id = row["case_id"]
        if case_id not in expected_by_id:
            raise ValueError("foreign case identity")
        if case_id in found:
            raise ValueError("duplicate case audit identity")
        expected = expected_by_id[case_id]
        for field in _required_identity(expected):
            if row[field] != expected[field]:
                raise ValueError(f"case {field} differs from frozen manifest")
        found[case_id] = dict(row)

    rows, incorrect = [], 0
    for case_id in expected_ids:
        row = found.get(case_id)
        if row is None:
            row = {**expected_by_id[case_id], "status": "missing", "fit_cost_unknown": True,
                   "false_network": None, "archive_sha256": None}
        is_known = (row.get("status") == "complete" and row.get("fit_cost_unknown") is False
                    and type(row.get("false_network")) is bool
                    and isinstance(row.get("archive_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", row["archive_sha256"]) is not None)
        if not is_known:
            row["unknown"] = True
        else:
            row["unknown"] = False
            incorrect += row["false_network"]
        rows.append(row)

    complete = sum(not row["unknown"] for row in rows)
    unknown = PLANNED_PER_CELL - complete
    interval = clopper_pearson(incorrect, PLANNED_PER_CELL)
    possible_interval = clopper_pearson(incorrect + unknown, PLANNED_PER_CELL)
    passed = unknown == 0 and interval["upper"] <= FALSE_NETWORK_UPPER_LIMIT
    return {
        "engine": expected_cases[0]["engine"],
        "sample_size": expected_cases[0]["sample_size"],
        "study_id": expected_cases[0]["study_id"],
        "manifest_sha256": expected_cases[0]["manifest_sha256"],
        "source_sha": expected_cases[0]["source_sha"],
        "planned": PLANNED_PER_CELL,
        "complete": complete,
        "unknown": unknown,
        "incorrect": incorrect,
        "incorrect_interval": interval,
        "possible_interval_if_unknowns_are_false": possible_interval,
        "passed": passed,
        "cases": rows,
    }


def qualify_study(wave80: Mapping[str, object], wave200: Mapping[str, object]):
    """Combine N waves only when their run and source provenance is coherent."""
    waves = (wave80, wave200)
    if any(not isinstance(wave, Mapping) for wave in waves):
        raise ValueError("wave report must be a mapping")
    if (wave80.get("sample_size"), wave200.get("sample_size")) != (80, 200):
        raise ValueError("source waves must be N=80 and N=200")
    for wave in waves:
        if wave.get("attempt") != 1 or not wave.get("transport_verified"):
            raise ValueError("wave attempt or transport verification failed")
        if (not wave.get("source_sha") or not wave.get("manifest_sha256")
                or not wave.get("run_id") or not wave.get("wave_id")):
            raise ValueError("wave provenance is incomplete")
        if (type(wave.get("attempt")) is not int
                or re.fullmatch(r"[0-9a-f]{40}", wave["source_sha"]) is None
                or re.fullmatch(r"[0-9a-f]{64}", wave["manifest_sha256"]) is None
                or not str(wave["run_id"]).isdecimal()):
            raise ValueError("wave provenance identity is malformed")
        cells = wave.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise ValueError("wave must contain cell summaries")
        if len(cells) != len(ENGINES) or {cell.get("engine") for cell in cells} != set(ENGINES):
            raise ValueError("wave must contain exactly one cell for each engine")
        if any(cell.get("sample_size") != wave["sample_size"] for cell in cells):
            raise ValueError("cell sample size differs from its wave")
        if any(cell.get("study_id") != wave.get("study_id") for cell in cells):
            raise ValueError("cell study identity differs from its wave")
        if any(cell.get("manifest_sha256") != wave["manifest_sha256"]
               or cell.get("source_sha") != wave["source_sha"] for cell in cells):
            raise ValueError("cell manifest/source identity differs from its wave")
    if wave80.get("source_sha") != wave200.get("source_sha"):
        raise ValueError("wave source SHA differs")
    if wave80.get("study_id") != wave200.get("study_id"):
        raise ValueError("wave study identity differs")
    if wave80.get("run_id") != wave200.get("run_id"):
        raise ValueError("sample-size waves must come from the same workflow run")
    if (wave80.get("wave_id"), wave200.get("wave_id")) != ("n80", "n200"):
        raise ValueError("sample-size wave identities differ")

    all_cells = list(wave80["cells"]) + list(wave200["cells"])
    eligibility = {
        engine: all(cell["passed"] for cell in all_cells if cell["engine"] == engine)
        for engine in ENGINES
    }
    return {
        "schema": SCHEMA,
        "study_id": wave80["study_id"],
        "source_sha": wave80["source_sha"],
        "transport_verified": True,
        "cells": sorted(all_cells, key=lambda cell: (cell["engine"], cell["sample_size"])),
        "engine_eligibility": eligibility,
    }


def render_markdown(report):
    lines = [
        "# Tree-engine recursive null qualification",
        "",
        f"Study `{report['study_id']}` used source `{report['source_sha']}`.",
        "An engine is eligible for a separately approved signal study only if both sample-size cells pass.",
        "",
        "|Engine|N|Complete / 55|Unknown|False networks|Exact 95% upper bound|Gate|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in report["cells"]:
        interval = cell["incorrect_interval"]["upper"]
        lines.append(
            f"|{cell['engine']}|{cell['sample_size']}|{cell['complete']} / 55|{cell['unknown']}|"
            f"{cell['incorrect']}|{interval:.6f}|{'PASS' if cell['passed'] else 'STOP'}|"
        )
    lines.extend(["", "## Engine eligibility", ""])
    for engine, eligible in report["engine_eligibility"].items():
        lines.append(f"- `{engine}`: {'eligible for separate signal review' if eligible else 'stopped'}")
    lines.extend(["", "Unknown cases remain in the denominator and prevent a cell from passing."])
    return "\n".join(lines) + "\n"


def write_report(report, json_path: Path, markdown_path: Path):
    """Write deterministic report files from a completed, verified study summary."""
    import json

    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
