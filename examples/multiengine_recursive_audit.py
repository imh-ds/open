"""Versioned multi-engine no-fit auditor, adapted from the frozen EN auditor.

The historical auditor is unchanged. This copy retains its row/loss/recursion
checks and explicitly validates engine-specific grids and fit accounting.
"""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from open.layer_artifacts import decode, portable, reconstruct_payloads
from open.layer_types import LocalAssessment
from open.risk import RiskResult
from open.learners.elastic_net import ALPHA_GRID, L1_RATIO_GRID
from open.learners.trees import GRIDS
from open.layer_stability import make_layer_sample_plans

import recursive_preview_design as design
from recursive_preview_runner import case_context, write_json


class Store:
    """Read case evidence directly from a directory or ZIP; never extract archives."""
    def __init__(self, location, case_id=None):
        self.location = Path(location)
        self.zip = zipfile.ZipFile(self.location) if self.location.suffix == '.zip' else None
        self.prefix = f'cases/{case_id}/' if self.zip else ''
        if self.zip:
            names = self.zip.namelist()
            if len(names) != len(set(names)):
                self.zip.close()
                raise ValueError('duplicate archive members')
            self.names = sorted(n[len(self.prefix):] for n in names
                                if n.startswith(self.prefix) and not n.endswith('/'))
        else:
            self.names = sorted(p.relative_to(self.location).as_posix()
                                for p in self.location.rglob('*') if p.is_file())
        if any('..' in Path(n).parts or Path(n).is_absolute() for n in self.names):
            raise ValueError('invalid evidence path')

    def read(self, name):
        if name not in self.names:
            raise ValueError(f'missing evidence: {name}')
        return self.zip.read(self.prefix+name) if self.zip else (self.location/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def compressed(self, name):
        return json.loads(gzip.decompress(self.read(name)))

    def close(self):
        if self.zip:
            self.zip.close()


def same(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('saved arithmetic differs from independent replay')


def audit_pair_group_anchors(pairs, components, baseline, full):
    """Anchor every pair/group contrast to the original full/baseline held-out rows."""
    def losses(risk, selected):
        return [row['loss'] for fold in risk['folds']
                if (fold['repeat_index'],fold['fold_index']) in selected
                for row in fold['observations']]
    params = {(f['repeat_index'],f['fold_index']):f['fitted_parameters'] for f in full['folds']}
    def reused(ablation):
        if ablation['predictors']:
            for fold in ablation['folds']:
                if fold['fitted_parameters'] != params[fold['repeat_index'],fold['fold_index']]:
                    raise ValueError('pair/group did not reuse full-model parameters')
    def normalized(actual, raw, gain):
        expected = raw/gain if gain > 0 else None
        if expected is None:
            if actual is not None:
                raise ValueError('undefined pair/group normalization must remain unavailable')
        else:
            same(actual, expected)
    for pair in pairs:
        selected = {tuple(k) for k in pair['selected_folds']}
        b = float(np.mean(losses(baseline, selected)))
        f = float(np.mean(losses(full, selected)))
        a = pair['ablation']['aggregate_risk']
        for actual, expected in ((pair['baseline_risk'],b), (pair['full_risk'],f),
                (pair['ablated_risk'],a), (pair['total_gain'],b-f), (pair['raw'],a-f)):
            same(actual, expected)
        normalized(pair['normalized'],a-f,b-f)
        reused(pair['ablation'])
    for component in components:
        if component['ablation'] is not None:
            raw = component['ablation']['aggregate_risk']-full['aggregate_risk']
            same(component['raw'],raw)
            normalized(component['normalized'],raw,baseline['aggregate_risk']-full['aggregate_risk'])
            reused(component['ablation'])


def grid_for(engine):
    if engine=='elastic_net':
        return dict(alpha=ALPHA_GRID,l1_ratio=L1_RATIO_GRID)
    if engine not in GRIDS:
        raise ValueError('unsupported audit engine')
    return GRIDS[engine]


def valid_parameters(params,engine):
    grid=grid_for(engine)
    return (isinstance(params,dict) and set(params)==set(grid)
        and all(type(v) in (int,float) and v in grid[k] for k,v in params.items())
        and all(type(params[k]) is int for k in ('min_samples_leaf','max_depth','num_leaves') if k in params))


def expected_tuning(risk,config,tuned):
    size=math.prod(len(v) for v in grid_for(config.learner.library[0]).values())
    return len(risk['folds'])*size*config.tuning_cv.folds if tuned else 0


def verify_tuning(risk,config,tuned):
    expected=expected_tuning(risk,config,tuned)
    if risk['tuning_model_fit_count']!=expected:
        raise ValueError('saved tuning count differs from frozen engine grid/folds')
    return expected


def audit_risk(payload, data, plan, target, candidates, predictors, *, all_folds=True, engine="elastic_net"):
    risk = decode(RiskResult, payload)
    split_id = hashlib.sha256(plan.split_plan.to_json(indent=None).encode()).hexdigest()
    if (risk.target, risk.candidate_set, risk.predictors, risk.loss, risk.split_seed,
        risk.split_plan_id) != (target, candidates, predictors, 'mse', plan.seed, split_id):
        raise ValueError('risk data/split/predictor identity mismatch')
    if risk.learner != (engine if predictors else None):
        raise ValueError('risk engine differs from reference learner/baseline')
    splits = {(s.repeat, s.fold):s for s in plan.split_plan.splits}
    found, losses = set(), []
    for fold in risk.folds:
        key = (fold.repeat_index, fold.fold_index)
        if key in found or key not in splits or fold.predictors != predictors:
            raise ValueError('invalid risk fold identity')
        found.add(key)
        split = splits[key]
        positions = [r.row_position for r in fold.observations]
        if len(positions) != len(set(positions)) or set(positions) != set(split.test_indices):
            raise ValueError('risk held-out row coverage mismatch')
        if not predictors:
            if fold.fitted_parameters is not None:
                raise ValueError('baseline cannot have fitted parameters')
        elif not valid_parameters(fold.fitted_parameters,engine):
            raise ValueError('parameters outside frozen engine grid')
        per_fold = []
        baseline = float(np.mean(data[target].iloc[list(split.train_indices)].to_numpy()))
        for row in fold.observations:
            same(row.observed, float(data[target].iloc[row.row_position]))
            same(row.loss, (row.observed-row.predicted)**2)
            if not predictors:
                same(row.predicted, baseline)
            per_fold.append(row.loss)
        same(fold.risk, float(np.mean(per_fold)))
        losses.extend(per_fold)
    if not found or (all_folds and found != set(splits)):
        raise ValueError('risk fold coverage mismatch')
    expected_fits = len(found) if predictors else 0
    if risk.outer_model_fit_count != expected_fits or risk.tuning_model_fit_count < 0:
        raise ValueError('risk fit count mismatch')
    if not predictors and risk.tuning_model_fit_count != 0:
        raise ValueError('baseline cannot have tuning fits')
    same(risk.aggregate_risk, float(np.mean(losses)))
    return len(losses)


def audit_checkpoint(payload, data, assessment, context, config):
    signature = payload.pop('digest')
    if design.digest(payload) != signature or payload['context'] != context:
        raise ValueError('checkpoint context/digest mismatch')
    if design.digest(payload['assessment']) != design.digest(assessment):
        raise ValueError('checkpoint differs from network assessment')
    local, p = assessment.local, len(assessment.candidates)
    if assessment.status != 'completed' or local is None:
        raise ValueError('completed network contains failed assessment')
    recorded = payload['risks']
    main = [r['risk'] for r in recorded if r['origin'] == 'analysis']
    if len(main) != 2+2*p:
        raise ValueError('missing baseline/full/broad/LOCO observation evidence')
    names = assessment.candidates
    predictors = [(), names, *((n,) for n in names),
                  *(tuple(v for v in names if v != n) for n in names)]
    observations = sum(audit_risk(risk, data, assessment.sample, assessment.target, names, inputs, engine=config.learner.library[0])
                       for risk, inputs in zip(main, predictors))
    r0, rf = main[0]['aggregate_risk'], main[1]['aggregate_risk']
    same(local.relevance.baseline_risk, r0)
    same(local.relevance.full_risk, rf)
    gain = r0-rf
    same(local.total_gain, gain)
    params = {(f['repeat_index'],f['fold_index']):f['fitted_parameters'] for f in main[1]['folds']}
    for i, name in enumerate(names):
        same(local.relevance.broad[name].raw, r0-main[2+i]['aggregate_risk'])
        same(local.relevance.unique[name].raw, main[2+p+i]['aggregate_risk']-rf)
        for fold in main[2+p+i]['folds']:
            if main[2+p+i]['predictors'] and fold['fitted_parameters'] != params[fold['repeat_index'],fold['fold_index']]:
                raise ValueError('LOCO did not reuse full model parameters')
    for i,risk in enumerate(main):
        tuned = i == 1 or 2 <= i < 2+p
        verify_tuning(risk,config,tuned)
    expected_group = ([] if local.pair_ablations is None else
                      [portable(pair.ablation) for pair in local.pair_ablations.pairs if pair.ablation])
    expected_group += [portable(c.ablation) for c in local.shared_components if c.ablation]
    groups = [r['risk'] for r in recorded if r['origin'] == 'grouping']
    if sorted(map(design.digest, expected_group)) != sorted(map(design.digest, groups)):
        raise ValueError('group risk capture differs from local evidence')
    for risk in groups:
        observations += audit_risk(risk, data, assessment.sample, assessment.target, names,
                                   tuple(risk['predictors']), all_folds=False, engine=config.learner.library[0])
        verify_tuning(risk,config,False)
    audit_pair_group_anchors([] if local.pair_ablations is None else portable(local.pair_ablations.pairs),
                             portable(local.shared_components), main[0], main[1])
    if len(recorded) != len(main)+len(groups):
        raise ValueError('foreign risk capture origin')
    for field in ('outer_model_fit_count','tuning_model_fit_count'):
        if sum(r['risk'][field] for r in recorded) != getattr(local, field):
            raise ValueError('captured risk fits differ from native local total')
    if not payload['worker_os_peak_bytes'] or payload['seconds'] < 0:
        raise ValueError('missing worker resource evidence')
    return dict(observation_checks=observations, outer_fits=local.outer_model_fit_count,
                tuning_fits=local.tuning_model_fit_count,
                worker_os_peak_bytes=payload['worker_os_peak_bytes'])


def audit_case(spec, store, manifest, *, data=None, config=None):
    data = design.generate(spec) if data is None else data
    config = design.settings(spec) if config is None else config
    receipt = store.json('receipt.json')
    context = case_context(spec, data, config, manifest)
    if (receipt['spec'] != spec or receipt['context'] != context
        or receipt['manifest'] != design.digest(manifest) or receipt['config'] != portable(config)):
        raise ValueError('case/source/settings identity mismatch')
    names = [n for n in store.names if n != 'receipt.json']
    actual = {n:hashlib.sha256(store.read(n)).hexdigest() for n in names}
    sealed = 'files' in receipt
    if not sealed and (receipt['status'] != 'STARTED' or not receipt['fit_cost_unknown']):
        raise ValueError('unsealed receipt must remain an unknown started case')
    if sealed and actual != receipt['files']:
        raise ValueError('case evidence inventory/digest mismatch')
    if receipt['runtime'] != manifest['runtime'] or receipt['python'] != '3.11.9':
        raise ValueError('case runtime differs from frozen versions')
    if 'site-packages' not in Path(receipt['module']).parts:
        raise ValueError('case did not use installed package')
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(store.read('data.csv')),
                                 float_precision='round_trip'), data, check_exact=True)
    if store.json('config.json') != portable(config):
        raise ValueError('saved configuration differs')
    row = dict(spec=spec, status=receipt['status'], seconds=receipt.get('seconds'),
               receipt_sha256=hashlib.sha256(store.read('receipt.json')).hexdigest(),
               fit_cost_unknown=receipt['fit_cost_unknown'], structure=None)
    starts = [n for n in names if n.endswith('.started.json')]
    finishes = [n for n in names if n.endswith('.finished.json')]
    row['started_assessments'], row['finished_assessments'] = len(starts), len(finishes)
    if receipt['status'] != 'complete':
        row['unknown_attempts'] = len(starts)-len(finishes)
        row['receipt_inventory_sealed'] = sealed
        return row
    payload = store.compressed('network.json.gz')
    network, _ = reconstruct_payloads(*(payload[n] for n in (
        'analysis_metadata.json', 'layer_evidence.json', 'assessment_ledger.json')))
    if network.config != config or network.variables != tuple(data) or network.status != 'COMPLETE':
        raise ValueError('network/settings/completion mismatch')
    if network.plans != make_layer_sample_plans(data, config):
        raise ValueError('network samples/splits differ from frozen plan')
    edges, groups, checks, warnings = [], [], [], 0
    expected_checkpoints = set()
    for layer in network.layers:
        for target in layer.evidence:
            for relation in target.relations:
                if relation.qualifies:
                    if relation.members:
                        groups.append((relation.target, relation.members))
                    else:
                        edges.append((relation.source, relation.target))
            for assessment in target.assessments:
                relative = f'checkpoints/d{layer.depth}-{target.target}/{assessment.sample.run_index:03}.json.gz'
                expected_checkpoints.add(relative)
                local_context = design.digest(dict(case=context, depth=layer.depth,
                    target=target.target, candidates=target.candidates, config=config))
                checkpoint = store.compressed(relative)
                sampled = data.iloc[list(assessment.sample.sampled_positions)]
                checks.append(audit_checkpoint(checkpoint, sampled, assessment, local_context, config))
                base = relative[:-3]
                start, finish = store.json(base+'.started.json'), store.json(base+'.finished.json')
                for attempt in (start, finish):
                    if (attempt['context'], attempt['depth'], attempt['target'],
                        attempt['run_index'], attempt['seed']) != (local_context, layer.depth,
                        target.target, assessment.sample.run_index, assessment.sample.seed):
                        raise ValueError('attempt identity mismatch')
                if start['status'] != 'STARTED' or finish['status'] != 'completed' or finish['fit_cost_unknown']:
                    raise ValueError('incomplete attempt cost evidence')
                warnings += len(assessment.warnings)
    if expected_checkpoints != {n for n in names if n.startswith('checkpoints/') and n.endswith('.json.gz')}:
        raise ValueError('checkpoint coverage differs from finalized layers')
    if len(starts) != len(checks) or len(finishes) != len(checks) or receipt['fit_cost_unknown']:
        raise ValueError('attempt coverage/known fit cost mismatch')
    row.update(structure=design.structure(spec, edges, groups, network.assigned_depth)
               if 'index' in spec else None, warnings=warnings, assessments=len(checks),
               observation_checks=sum(c['observation_checks'] for c in checks),
               outer_fits=sum(c['outer_fits'] for c in checks),
               tuning_fits=sum(c['tuning_fits'] for c in checks),
               worker_os_peak_bytes=max((c['worker_os_peak_bytes'] for c in checks), default=0),
               assigned_depth=portable(network.assigned_depth), edges=portable(edges), groups=portable(groups))
    return row
