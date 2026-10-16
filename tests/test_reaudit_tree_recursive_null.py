"""No-fit replay is isolated so its fitting guards cannot affect other tests."""
from pathlib import Path
import subprocess
import sys


def test_saved_evidence_replay_blocks_fits_and_rebuilds_fixed_denominator_gates(tmp_path):
    root = Path(__file__).parents[1]
    code = '''
import sys, copy
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "src"), str(root / "examples")]
import reaudit_tree_recursive_null as replay
from sklearn.ensemble import RandomForestRegressor
try:
    RandomForestRegressor().fit([[0], [1]], [0, 1])
except RuntimeError as exc:
    assert "fitting is prohibited" in str(exc)
else:
    raise AssertionError("audit allowed model fitting")
original = replay.load(replay.ORIGINAL / "study-report.json")
manifest = replay.load(replay.ORIGINAL / "study-manifest.json")
before = copy.deepcopy(original)
rf = [dict(row, status="complete", fit_cost_unknown=False, false_network=False,
           archive_sha256="a" * 64)
      for cell in original["cells"] if cell["engine"] == "random_forest"
      for row in cell["cases"]]
report = replay.corrected_report(original, manifest, rf)
assert original == before
assert all(report["engine_eligibility"].values())
assert all(c["complete"] == 55 and c["unknown"] == 0 for c in report["cells"])
missing = replay.corrected_report(original, manifest, rf[:-1])
assert missing["engine_eligibility"]["random_forest"] is False
assert sum(c["unknown"] for c in missing["cells"]) == 1
changed = copy.deepcopy(original)
cell = next(c for c in changed["cells"] if c["engine"] == "gbm" and c["sample_size"] == 80)
for row in cell["cases"][:2]:
    row["edges"] = [["X1", "Y"]]
    row["false_network"] = False  # old endpoint must be independently reclassified
report = replay.corrected_report(changed, manifest, rf)
assert report["engine_eligibility"]["gbm"] is False
assert next(c for c in report["cells"] if c["engine"] == "gbm" and c["sample_size"] == 80)["incorrect"] == 2
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(root)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
