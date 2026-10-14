import json,pathlib,sys,datetime
sys.path.insert(0,"examples")
import realistic_interaction_study as study
p=pathlib.Path("examples/results/2026-10-13-realistic-interaction-recovery")
q=p/"recovered"
a=json.loads((q/"audit.json").read_text());proof=json.loads((q/"recovery.json").read_text());o=json.loads((p/"original-audit.json").read_text());index=json.loads((p/"original-raw-index.json").read_text())
study.freeze();study.thresholds()
assert a["planned"]==a["complete"]==10800 and not a["missing"] and not a["audit_errors"]
assert not a["failed_tasks"] and not a["unknown_attempt_cost_ids"] and a["warning_count"]==0
assert a["manifest"]==proof["frozen_manifest"]==study.identity()
for key in ("attempt_costs","known_attempt_cost","checkpoint_fits"):
 assert a[key]==o[key],key
assert proof["thresholds"]==study.io.fingerprint(study.thresholds())
assert proof["new_fits"]==proof["new_refits"]==0
assert proof["status"]=="complete" and proof["relative_tolerance"]==0 and proof["absolute_tolerance"]==1e-12
assert proof["recovery_source"]=="26c708a6d944f1ddd78680ba9b88641c60c4d958" and proof["original_run"]==38482368226
assert proof["statistics"]["comparisons"]==43200 and proof["statistics"]["max_absolute_difference"]<=1e-12
rows={r["id"]:r for r in a["rows"]};expected={c["id"] for c in study.d.stage_tasks("development")}
assert len(rows)==10800 and set(rows)==expected
for r in o["rows"]:assert rows[r["id"]]==r,r["id"]
recovered=set(rows)-{r["id"] for r in o["rows"]};assert recovered==set(o["missing_ids"]) and len(recovered)==480
for c in study.d.stage_tasks("development"):
 for z in rows[c["id"]]["pairs"]:
  assert z["threshold"]==study.thresholds()["thresholds"][c["cell"]]
  assert z["selected"]==bool(z["eligible"] and z["score"]>z["threshold"])
raw={(r["sha256"],r["bytes"]) for r in a["merge"]["archives"]}
assert raw=={(r["sha256"],r["bytes"]) for r in o["merge"]["archives"]}=={(r["sha256"],r["bytes"]) for r in index["archives"]} and len(raw)==135
strata=json.loads((q/"strata.json").read_text());assert len(strata)==135
assert all(z["complete"]==z["planned"]==80 and z["status"]=="complete" for z in strata)
run=json.loads((p/"recovery-completed-run.json").read_text());assert run["conclusion"]=="success" and run["head_sha"]==proof["recovery_source"] and run["event"]=="workflow_dispatch" and run["run_attempt"]==1
j=json.loads((p/"recovery-completed-jobs.json").read_text());assert len(j["jobs"])==1
job=j["jobs"][0];assert job["conclusion"]=="success"
duration=(datetime.datetime.fromisoformat(job["completed_at"].replace("Z","+00:00"))-datetime.datetime.fromisoformat(job["started_at"].replace("Z","+00:00"))).total_seconds()
assert duration<=2700 and duration+781<=3600
v=dict(status="passed",run=38491674970,source=proof["recovery_source"],complete=10800,newly_recovered=480,previously_accepted_rows_exactly_unchanged=len(o["rows"]),groups_complete=135,original_archive_hashes_preserved=135,max_absolute_difference=proof["statistics"]["max_absolute_difference"],new_fits=0,new_refits=0,original_aggregate_job_seconds=781,recovery_job_seconds=duration,cumulative_aggregate_job_seconds=duration+781,approved_aggregate_cap_seconds=3600,standalone=a["standalone"],scope="Original failure retained; all original identities, saved decisions, result hashes and original raw archive hashes preserved.")
(p/"recovery-verification.json").write_text(json.dumps(v,indent=2)+"\n")
print(json.dumps(v))
