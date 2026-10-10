import re
from pathlib import Path


def test_report_template_covers_spec_50_and_predictive_language():
    template_path = Path(__file__).resolve().parents[1] / "docs" / "open-report-template.md"
    template = template_path.read_text(encoding="utf-8").lower()

    per_target_fields = (
        "target name",
        "depth",
        "candidate set",
        "baseline risk",
        "full-model risk",
        "total relative gain",
        "learner",
        "broad relevance",
        "unique relevance",
        "tested group relevance",
        "classification",
        "stability probability",
        "target-evaluation count",
    )
    network_fields = (
        "root",
        "max depth",
        "node selection probabilities",
        "depth distributions",
        "edge-class probabilities",
        "shared co-membership probabilities",
        "screening-partner probabilities",
        "thresholds",
        "learner configuration",
        "resampling procedure",
        "sensitivity analysis",
    )
    artifact_names = (
        "nodes.csv",
        "edges.csv",
        "shared_groups.csv",
        "shared_comembership.csv",
        "local_targets.csv",
        "analysis_metadata.json",
    )

    for field in (*per_target_fields, *network_fields, *artifact_names):
        assert field.lower() in template, f"missing report field or artifact reference: {field}"

    headings = {line for line in template.splitlines() if line.startswith("#")}
    assert "## network results" in headings
    assert "## local target results" in headings
    destinations = re.findall(r"\[[^\]]+\]\(([^)\n]+)\)", template)
    guide_destinations = [d for d in destinations if not d.startswith('<')]
    assert 'procedure-evaluation.md' in guide_destinations
    for destination in guide_destinations:
        assert (template_path.parent/destination).is_file(), f"broken documentation link: {destination}"
    artifact_destinations = [d for d in destinations if d.startswith('<')]
    assert len(artifact_destinations) == len(artifact_names)
    linked_artifacts = set()
    for destination in artifact_destinations:
        match = re.fullmatch(r"<p5\.1_output_dir/([^<>]+)>", destination)
        assert match, f"malformed artifact link destination: {destination}"
        linked_artifacts.add(match.group(1))
    assert linked_artifacts == set(artifact_names)

    assert "predictive contribution" in template
    assert "do not establish causality" in template
    assert "display-only" in template
