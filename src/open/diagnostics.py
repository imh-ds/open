"""Build portable export metadata from a completed OPEN result."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import hashlib
import json
from typing import Any

from open.metadata import MetadataValidation
from open.stability import OPENResult
from open.exploration import build_exploration


def _json_value(value: Any) -> Any:
    """Convert stored tuples and enums to ordinary JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def hash_variable_metadata(metadata: MetadataValidation) -> str:
    """Hash ordered normalized variable rows using canonical UTF-8 JSON."""
    if not isinstance(metadata, MetadataValidation):
        raise TypeError("metadata must be a MetadataValidation")
    encoded = json.dumps(
        _json_value(metadata.variables), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_analysis_metadata(
    result: OPENResult, metadata: MetadataValidation,
) -> dict[str, object]:
    """Serialize retained fit provenance and exact evaluation denominators."""
    if not isinstance(result, OPENResult):
        raise TypeError("result must be an OPENResult")
    if not isinstance(metadata, MetadataValidation):
        raise TypeError("metadata must be a MetadataValidation")
    if metadata.root != result.config.analysis.root:
        raise ValueError("metadata root does not match the fitted analysis root")
    root_included = (
        result.variables.count(metadata.root) == 1
        and tuple(name for name in result.variables if name != metadata.root)
        == metadata.candidates
    )
    if result.variables != metadata.candidates and not root_included:
        raise ValueError("metadata candidates do not match the fitted variables")

    availability = dict(result.version_availability or {})
    package_version = result.package_version
    if package_version is None:
        availability.setdefault("package_version", "package version was not captured at fit time")
    dependency_versions = dict(result.dependency_versions or {})
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML"):
        if dependency_versions.get(name) is None:
            dependency_versions[name] = None
            availability.setdefault(f"dependency_versions.{name}",
                                    f"{name} version was not captured at fit time")
    learner_versions = dict(result.learner_versions or {})
    for learner in result.config.learner.library:
        if learner_versions.get(learner) is None:
            learner_versions[learner] = None
            availability.setdefault(f"learner_versions.{learner}",
                                    f"{learner} version was not captured at fit time")
    if result.git_revision is None:
        availability["git_revision"] = result.git_revision_unavailable_reason or "Git revision was not captured"
    if result.peak_traced_memory_bytes is None:
        availability["peak_traced_memory_bytes"] = (
            result.peak_memory_unavailable_reason or "traced memory was not captured"
        )

    ledger = [
        {
            "run_index": run.run_index,
            "seed": run.seed,
            "sampled_positions": list(run.sampled_positions),
            "completed": run.completed,
            "failure_type": run.failure_type,
            "failure_message": run.failure_message,
            "target_stop_reasons": ({target: reason.value
                                     for target, reason in run.network.stop_reasons.items()}
                                    if run.network is not None else {}),
            "stop_reason": (run.network.stop_reason.value if run.network is not None
                            and run.network.stop_reason is not None else None),
        }
        for run in result.batch.runs
    ]
    omissions = [
        {
            "structure": item.structure,
            "identity": _json_value(item.identity),
            "frequency": asdict(item.frequency),
            "reason": item.reason,
        }
        for item in result.consensus.omissions
    ]

    locals_ = [local for run in result.batch.runs if run.completed
               for local in run.network.local_results.values()]
    outer_fits = sum(local.outer_model_fit_count for local in locals_)
    tuning_fits = sum(local.tuning_model_fit_count for local in locals_)
    diagnostics = {
        "n_targets_evaluated": len(locals_),
        "n_pair_ablations": sum(
            pair.evaluated_fold_count for local in locals_
            if local.pair_ablations is not None for pair in local.pair_ablations.pairs
        ),
        "n_group_ablations": sum(len(local.shared_components) for local in locals_),
        "n_group_ablation_folds": sum(
            len(component.ablation.folds) for local in locals_
            for component in local.shared_components
        ),
        "n_outer_model_fits": outer_fits,
        "n_tuning_model_fits": tuning_fits,
        "n_model_fits": outer_fits + tuning_fits,
    }
    config = _json_value(asdict(result.config))
    return {
        "package_version": package_version,
        "git_revision": result.git_revision,
        "git_revision_unavailable_reason": result.git_revision_unavailable_reason,
        "config": config,
        "seed": result.config.analysis.random_seed,
        "learner_versions": learner_versions,
        "dependency_versions": dependency_versions,
        "preprocessing": config["preprocessing"],
        "loss": result.config.outcome.loss,
        "thresholds": config["thresholds"],
        "resampling_design": {
            "stability": config["stability"],
            "performance_cv": config["performance_cv"],
            "tuning_cv": config["tuning_cv"],
        },
        "variable_metadata_sha256": hash_variable_metadata(metadata),
        "measurement_overlap": {
            item["name"]: bool(item.get("measurement_overlap", False))
            for item in metadata.variables
        },
        "batch_counts": {
            "requested_runs": result.batch.requested_runs,
            "attempted_runs": result.batch.attempted_runs,
            "completed_runs": result.batch.completed_runs,
            "failed_runs": result.batch.failed_runs,
        },
        "run_ledger": ledger,
        "consensus_omissions": omissions,
        "exploration": build_exploration(result, metadata),
        "diagnostics": diagnostics,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_traced_memory_bytes": result.peak_traced_memory_bytes,
        "peak_memory_unavailable_reason": result.peak_memory_unavailable_reason,
        "metadata_availability": availability,
    }
