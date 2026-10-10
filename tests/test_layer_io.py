from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

import open.layer_io as module
import open.layer_stability as engine
from open.io import write_open_outputs
from open.metadata import validate_metadata
from layer_helpers import make_config, make_local, install_scripted_analyzer


def metadata(names=("y","a","x")):
    return validate_metadata(names,[{"name":n,"display_label":"睡眠" if n=="a" else n,
        "type":"continuous","role":"root" if n=="y" else "candidate"} for n in names],"y")


def result(monkeypatch, *, fail=False, shared=False):
    frame = pd.DataFrame({n:range(20) for n in ("y","a","x")})
    if fail:
        monkeypatch.setattr(engine,"analyze_target",lambda *a,**k: (_ for _ in ()).throw(RuntimeError("example")))
    elif shared:
        monkeypatch.setattr(engine,"analyze_target",lambda d,t,c,cfg,s: make_local(t,c,shared=(("a","x"),)))
    else:
        install_scripted_analyzer(monkeypatch,selections={("y",b):("a",) for b in range(10)})
    return engine.fit_layer_consensus(frame,"y",tuple(frame),make_config())


def saved(tmp_path,monkeypatch,**kwargs):
    fit = result(monkeypatch,**kwargs)
    path=tmp_path/"bundle"
    module.write_layer_outputs(fit,path,metadata=metadata())
    return fit,path


def rows(path):
    with path.open(encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def test_round_trip_audits_counts_scores_layers_unicode_and_deterministic_bytes(tmp_path,monkeypatch):
    fit,path=saved(tmp_path,monkeypatch)
    bundle=module.load_layer_bundle(path)
    assert bundle.metadata["method"]=="layer_consensus"
    assert bundle.metadata["planned_resamples_per_target"]==10
    assert len(bundle.assessments)==20
    assert bundle.layers[0]["new_nodes"] == ("a",)
    assert rows(path/"nodes.csv")[1]["display_label"]=="睡眠"
    assert rows(path/"edges.csv")[0]["selection_frequency"]=="1.0"
    assert len(list(path.iterdir()))==7
    with pytest.raises(TypeError):
        bundle.metadata["method"]="wrong"
    second=tmp_path/"second"
    write_open_outputs(fit,second,metadata=metadata())
    assert {p.name:p.read_bytes() for p in path.iterdir()}=={p.name:p.read_bytes() for p in second.iterdir()}


def test_all_failed_root_remains_exportable_without_finalized_edges(tmp_path,monkeypatch):
    _,path=saved(tmp_path,monkeypatch,fail=True)
    bundle=module.load_layer_bundle(path)
    assert bundle.metadata["status"]=="INCOMPLETE_LAYER"
    assert len(bundle.assessments)==10
    assert rows(path/"edges.csv")==rows(path/"shared_groups.csv")==[]
    assert len(rows(path/"local_targets.csv"))==10


def test_joint_relations_preserve_actual_joint_scores_and_members(tmp_path,monkeypatch):
    _,path=saved(tmp_path,monkeypatch,shared=True)
    bundle=module.load_layer_bundle(path)
    assert rows(path/"edges.csv")==[]
    assert json.loads(rows(path/"shared_groups.csv")[0]["members"])==["a","x"]
    assert bundle.assessments[0]["local"]["shared_components"][0]["raw"]==.3


@pytest.mark.parametrize("corruption",["frequency","missing_record","candidates","depth","measurement",
                                     "foreign_node","schema","nan","path","split_seed","duplicate_key"])
def test_corrupted_evidence_is_rejected(tmp_path,monkeypatch,corruption):
    _,path=saved(tmp_path,monkeypatch)
    filename="analysis_metadata.json"
    value=json.loads((path/filename).read_text(encoding="utf-8"))
    if corruption=="frequency":
        filename="layer_evidence.json"; value=json.loads((path/filename).read_text(encoding="utf-8"))
        value["layers"][0]["evidence"][0]["relations"][0]["frequency"]["selected"]=9
    elif corruption=="missing_record":
        filename="assessment_ledger.json"; value=json.loads((path/filename).read_text(encoding="utf-8")); value["assessments"].pop()
    elif corruption=="candidates":
        filename="assessment_ledger.json"; value=json.loads((path/filename).read_text(encoding="utf-8")); value["assessments"][0]["candidates"]=["x"]
    elif corruption=="depth": value["assigned_depth"]["a"]=2
    elif corruption=="measurement":
        filename="layer_evidence.json"; value=json.loads((path/filename).read_text(encoding="utf-8"))
        value["layers"][0]["evidence"][0]["relations"][0]["strength"]["baseline_relative_count"]=0
    elif corruption=="foreign_node": value["assigned_depth"]["foreign"]=1
    elif corruption=="schema": value["schema_version"]=999
    elif corruption=="nan": value["provenance"]["elapsed_seconds"]=float("nan")
    elif corruption=="path": value["path_frequency_available"]=True
    elif corruption=="split_seed": value["plans"][0]["split_plan"]["random_seed"]+=1
    elif corruption=="duplicate_key":
        (path/filename).write_text('{"schema":1,"schema":2}',encoding="utf-8")
        with pytest.raises(ValueError): module.load_layer_bundle(path)
        return
    (path/filename).write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(ValueError): module.load_layer_bundle(path)


def test_csv_disagreement_is_rejected(tmp_path,monkeypatch):
    _,path=saved(tmp_path,monkeypatch)
    text=(path/"edges.csv").read_text(encoding="utf-8")
    (path/"edges.csv").write_text(text.replace("a,y,0","x,y,0"),encoding="utf-8")
    with pytest.raises(ValueError): module.load_layer_bundle(path)


def test_cannot_drop_the_next_required_layer(monkeypatch):
    fit = result(monkeypatch)
    with pytest.raises(ValueError, match="unfinished"):
        replace(fit, layers=fit.layers[:-1])


@pytest.mark.parametrize('stops', [{}, {'y':'NO_UNASSIGNED'}, {'a':'INCOMPLETE_LAYER'}])
def test_stop_reasons_must_match_the_actual_recursive_stop(monkeypatch, stops):
    fit = result(monkeypatch)
    with pytest.raises(ValueError, match='stop'):
        replace(fit, stop_reasons=stops)


@pytest.mark.parametrize('corruption', ['training_row','unknown_fold','duplicate_fold',
                                      'predictors','component_risk','pair_risk'])
def test_joint_ablation_provenance_cannot_contradict_the_saved_comparison(tmp_path, corruption):
    source = Path(__file__).resolve().parents[1]/'examples/results/2026-10-08-layer-fixtures/shared_predictors/bundle'
    path = tmp_path/'bundle'
    shutil.copytree(source,path)
    ledger = json.loads((path/'assessment_ledger.json').read_text(encoding='utf-8'))
    local = ledger['assessments'][0]['local']
    component = local['shared_components'][0]
    risk = component['ablation']
    if corruption == 'training_row':
        meta=json.loads((path/'analysis_metadata.json').read_text(encoding='utf-8'))
        fold=risk['folds'][0]
        split=next(s for s in meta['plans'][0]['split_plan']['splits']
                   if (s['repeat'],s['fold']) == (fold['repeat_index'],fold['fold_index']))
        fold['observations'][0]['row_position']=split['train_indices'][0]
    elif corruption == 'unknown_fold': risk['folds'][0]['fold_index']=999
    elif corruption == 'duplicate_fold':
        fold=risk['folds'][0]
        rows=fold['observations']
        duplicate=dict(fold,observations=rows[len(rows)//2:])
        fold['observations']=rows[:len(rows)//2]
        for f in (fold,duplicate): f['risk']=sum(r['loss'] for r in f['observations'])/len(f['observations'])
        risk['folds'].append(duplicate)
    elif corruption == 'predictors':
        risk['predictors']=list(component['members'])
        for fold in risk['folds']: fold['predictors']=list(component['members'])
    else:
        if corruption == 'pair_risk': risk=local['pair_ablations']['pairs'][0]['ablation']
        for fold in risk['folds']:
            for obs in fold['observations']: obs.update(predicted=obs['observed'],loss=0.)
            fold['risk']=0.
        risk['aggregate_risk']=0.
    (path/'assessment_ledger.json').write_text(json.dumps(ledger),encoding='utf-8')
    with pytest.raises(ValueError): module.load_layer_bundle(path)


def test_existing_destination_race_cleanup_and_metadata_mismatch(tmp_path,monkeypatch):
    fit,path=saved(tmp_path,monkeypatch)
    before={p.name:p.read_bytes() for p in path.iterdir()}
    with pytest.raises(FileExistsError): module.write_layer_outputs(fit,path,metadata=metadata())
    assert before=={p.name:p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError): module.write_layer_outputs(fit,tmp_path/"wrong",metadata=metadata(("y","x")))
    def collision(stage,destination):
        destination.mkdir(); (destination/"sentinel").write_text("keep")
        raise FileExistsError(destination)
    monkeypatch.setattr(module,"_publish_no_replace",collision)
    with pytest.raises(FileExistsError): module.write_layer_outputs(fit,tmp_path/"race",metadata=metadata())
    assert (tmp_path/"race"/"sentinel").read_text()=="keep"
    assert not list(tmp_path.glob(".race.*"))
    monkeypatch.setattr(module,"_json_write",lambda *a,**k: (_ for _ in ()).throw(ValueError("serialization")))
    with pytest.raises(ValueError): module.write_layer_outputs(fit,tmp_path/"broken",metadata=metadata())
    assert not (tmp_path/"broken").exists() and not list(tmp_path.glob(".broken.*"))
