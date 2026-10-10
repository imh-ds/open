"""Pure consensus for fixed target/candidate comparisons within one layer."""
from collections import Counter, defaultdict
from itertools import combinations
from statistics import median

from open.config import _unit_interval
from open.layer_types import (CandidateEvidence, LocalAssessment, RelationEvidence,
                             SelectionCount, StrengthSummary, TargetEvidence, _integer)


def selected_joint_relations(local):
    """One scored joint identity per assessment, preferring component provenance."""
    joint = {}
    for pair in local.retained_pairs:
        joint[tuple(sorted(pair.members))] = (pair.pair.raw, pair.pair.normalized, pair)
    for component in local.shared_components:
        if component.validated:
            joint[tuple(sorted(component.members))] = (component.raw, component.normalized, component)
    return joint


def summarize_strength(values, scope):
    """Summarize signed scores without converting missing measurements to zeros."""
    raw, baseline, gain = [], [], []
    for value, baseline_risk, normalized in values:
        if value is None:
            continue
        raw.append(value)
        if baseline_risk > 0:
            baseline.append(value / baseline_risk)
        if normalized is not None:
            gain.append(normalized)
    def mid(items):
        return median(items) if items else None
    return StrengthSummary(len(raw), mid(raw), len(baseline), mid(baseline),
                           len(gain), mid(gain), scope)


def aggregate_target_assessments(assessments: tuple[LocalAssessment, ...], *,
                                 planned_count: int, pi_min: float) -> TargetEvidence:
    """Count all completed opportunities; failures withhold final frequencies."""
    _integer(planned_count, 1)
    _unit_interval("pi_min", pi_min)
    if (not isinstance(assessments, tuple) or len(assessments) != planned_count
            or any(not isinstance(a, LocalAssessment) for a in assessments)
            or sorted(a.sample.run_index for a in assessments) != list(range(planned_count))):
        raise ValueError("expected each planned assessment exactly once")
    assessments = tuple(sorted(assessments, key=lambda a: a.sample.run_index))
    first = assessments[0]
    if any((a.depth, a.target, a.candidates) != (first.depth, first.target, first.candidates)
           for a in assessments):
        raise ValueError("cannot aggregate different target/candidate comparisons")
    if any(a.sample.seed != first.sample.seed + a.sample.run_index for a in assessments):
        raise ValueError("assessment seed identities are inconsistent")
    completed = tuple(a.local for a in assessments if a.status == "completed")
    count = lambda selected: SelectionCount(selected, planned_count, len(completed))
    classes, promotions = defaultdict(Counter), Counter()
    broad, unique, joint_scores = defaultdict(list), defaultdict(list), defaultdict(list)
    relations, co_membership = Counter(), Counter()
    for local in completed:
        if (set(local.relevance.broad) != set(first.candidates)
                or set(local.relevance.unique) != set(first.candidates)
                or (not local.stop and set(local.classes) != set(first.candidates))):
            raise ValueError("local evidence must cover the frozen candidate set")
        baseline = local.relevance.baseline_risk
        for source in first.candidates:
            variable = local.classes.get(source)
            kind = ("LOW_TOTAL_GAIN" if local.stop else
                    variable.kind.value if variable.kind is not None else "UNCLASSIFIED")
            classes[source][kind] += 1
            promotions[source] += source in local.promoted
            b, u = local.relevance.broad[source], local.relevance.unique[source]
            broad[source].append((b.raw, baseline, b.normalized))
            unique[source].append((u.raw, baseline, u.normalized))
            if kind == "PROXIMAL_UNIQUE":
                relations[(source, kind, (), ())] += 1
            elif kind in ("SCREENED_BY_PARTNER", "SCREENED_BY_SET", "SCREENED_GENERAL"):
                screened = (variable.screened_by_partner if kind == "SCREENED_BY_PARTNER"
                            else variable.screened_by_set if kind == "SCREENED_BY_SET" else ())
                relations[(source, kind, (), tuple(sorted(screened)))] += 1
        pairs = set()
        for members, (raw, normalized, _) in selected_joint_relations(local).items():
            if not set(members) <= set(first.candidates):
                raise ValueError("joint relation references foreign candidates")
            relations[(None, "SHARED_GROUP", members, ())] += 1
            joint_scores[members].append((raw, baseline, normalized))
            pairs.update(combinations(members, 2))
        co_membership.update(pairs)
    scope = "all scored completed local fits"
    candidates = tuple(CandidateEvidence(source, first.target,
        {kind: count(n) for kind, n in sorted(classes[source].items())}, count(promotions[source]),
        summarize_strength(broad[source], scope), summarize_strength(unique[source], scope))
        for source in sorted(first.candidates))
    strengths = {candidate.source: candidate.unique for candidate in candidates}
    evidence = []
    for (source, kind, members, screened), numerator in sorted(relations.items(),
            key=lambda item: (item[0][1], item[0][0] or "", item[0][2], item[0][3])):
        frequency = count(numerator)
        qualifies = (kind in ("PROXIMAL_UNIQUE", "SHARED_GROUP") and numerator > 0
                     and frequency.probability is not None and frequency.probability >= pi_min)
        strength = (summarize_strength(joint_scores[members], "selected joint ablations")
                    if kind == "SHARED_GROUP" else strengths[source])
        evidence.append(RelationEvidence(source, first.target, kind, members, screened,
                                         frequency, strength, qualifies))
    return TargetEvidence(first.depth, first.target, first.candidates, assessments, candidates,
                          tuple(evidence), {pair: count(n) for pair, n in sorted(co_membership.items())})
