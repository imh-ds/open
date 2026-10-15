"""Test contexts must preserve real provenance guards and shared state."""
from pathlib import Path
import sys

import pytest

from study_fixture_helpers import isolate_pair_calibration, isolate_prospective

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


def test_temporary_calibration_rejects_changed_source_and_restores_originals(tmp_path, monkeypatch):
    import pair_confidence_study as old
    path, manifests, cached = old.THRESHOLDS, old.MANIFESTS, old.identity
    before, cache_info = path.read_bytes(), cached.cache_info()
    with monkeypatch.context() as scoped:
        isolate_pair_calibration(old, tmp_path, scoped)
        old.validate_rule()
        assert old.THRESHOLDS.is_relative_to(tmp_path)
        scoped.setattr(old, 'sources', lambda: {'changed': 'source'})
        with pytest.raises(ValueError, match='provenance'):
            old.validate_rule()
    assert old.THRESHOLDS == path and old.MANIFESTS is manifests
    assert old.identity is cached and cached.cache_info() == cache_info
    assert path.read_bytes() == before


def test_prospective_fixture_keeps_signal_selection_gate_closed(tmp_path, monkeypatch):
    import prospective_interaction_study as study
    isolate_prospective(study, tmp_path, monkeypatch)
    study.freeze()
    with pytest.raises(ValueError, match='null audit'):
        study.selection()
