"""Independent no-fit replay of recorded rows, risks, recursion and study endpoints."""
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


def audit_risk(payload, data, plan, target, candidates, predictors, *, all_folds=True):
    risk = decode(RiskResult, payload)
    split_id = hashlib.sha256(plan.split_plan.to_json(indent=None).encode()).hexdigest()
    if (risk.target, risk.candidate_set, risk.predictors, risk.loss, risk.split_seed,
        risk.split_plan_id) != (target, candidates, predictors, 'mse', plan.seed, split_id):
        raise ValueError('risk data/split/predictor identity mismatch')
    if risk.learner != ('elastic_net' if predictors else None):
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
        elif (fold.fitted_parameters is None or set(fold.fitted_parameters) != {'alpha','l1_ratio'}
              or fold.fitted_parameters['alpha'] not in ALPHA_GRID
              or fold.fitted_parameters['l1_ratio'] not in L1_RATIO_GRID):
            raise ValueError('parameters outside frozen elastic-net grid')
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
    observations = sum(audit_risk(risk, data, assessment.sample, assessment.target, names, inputs)
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
        expected_tuning = len(risk['folds'])*25*config.tuning_cv.folds if tuned else 0
        if risk['tuning_model_fit_count'] != expected_tuning:
            raise ValueError('saved tuning count differs from frozen grid/folds')
    expected_group = ([] if local.pair_ablations is None else
                      [portable(pair.ablation) for pair in local.pair_ablations.pairs if pair.ablation])
    expected_group += [portable(c.ablation) for c in local.shared_components if c.ablation]
    groups = [r['risk'] for r in recorded if r['origin'] == 'grouping']
    if sorted(map(design.digest, expected_group)) != sorted(map(design.digest, groups)):
        raise ValueError('group risk capture differs from local evidence')
    for risk in groups:
        observations += audit_risk(risk, data, assessment.sample, assessment.target, names,
                                   tuple(risk['predictors']), all_folds=False)
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
    if actual != receipt['files']:
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


def summarize(rows, stage, manifest_digest):
    planned = [s for s in design.specs() if (s['scenario'] == 'noise') == (stage == 'noise')]
    by_id = {}
    for row in rows:
        name = row['spec']['id']
        if name in by_id:
            raise ValueError('duplicate case row')
        if row['spec'] not in planned:
            raise ValueError('foreign case identity')
        by_id[name] = row
    cases = [by_id.get(s['id'], dict(spec=s, status='missing', structure=None)) for s in planned]
    cells = {}
    for scenario in ('noise',) if stage == 'noise' else ('chain', 'correlated'):
        for n in (80,200):
            cell = [r for r in cases if (r['spec']['scenario'],r['spec']['n']) == (scenario,n)]
            summary = design.summarize_cell(cell, signal=stage != 'noise')
            complete = [r for r in cell if r['status'] == 'complete']
            summary['depth_not_recovered_count'] = sum(
                not all(r['structure'].get('depth_matches',{}).values()) for r in complete)
            cells[f'{scenario}-n{n}'] = summary
    return dict(stage=stage, manifest=manifest_digest, planned=len(cases),
                complete=sum(r['status'] == 'complete' for r in cases),
                audit_errors=sum(r['status'] == 'audit_failed' for r in cases),
                cases=cases, cells=cells, qualified=all(c['qualified'] for c in cells.values()),
                audit_model_fits=0, interval_scope='marginal per cell; not general error control')


def validate_noise_gate(report, manifest_digest):
    if report.get('stage') != 'noise' or report.get('manifest') != manifest_digest:
        raise ValueError('noise gate source/stage mismatch')
    recomputed = summarize(report['cases'], 'noise', manifest_digest)
    if not recomputed['qualified'] or recomputed != report:
        raise ValueError('noise gate incomplete, failed or edited')
    return True


def audit_task(args):
    spec, location, manifest = args
    store = Store(location, spec['id'])
    try:
        return audit_case(spec, store, manifest)
    except Exception as exc:
        return dict(spec=spec, status='audit_failed', structure=None,
                    failure=type(exc).__name__+': '+str(exc)[:2000])
    finally:
        store.close()


def aggregate(directory, manifest, stage, output):
    inventory = {}
    for archive in sorted(Path(directory).rglob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('duplicate archive members')
            for name in z.namelist():
                if name.startswith('cases/') and name.endswith('/receipt.json'):
                    case_id = name.split('/')[1]
                    case_files = {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()
                                  if n.startswith(f'cases/{case_id}/') and not n.endswith('/')}
                    signature = design.digest(case_files)
                    if case_id in inventory and inventory[case_id][0] != signature:
                        raise ValueError('conflicting case archives')
                    inventory[case_id] = (signature, str(archive))
    planned = [s for s in design.specs() if (s['scenario'] == 'noise') == (stage == 'noise')]
    allowed = {s['id'] for s in planned}
    if set(inventory)-{s['id'] for s in design.specs()}:
        raise ValueError('foreign case archive')
    tasks = [(s,inventory[s['id']][1],manifest) for s in planned if s['id'] in inventory]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(audit_task, tasks))
    report = summarize(rows, stage, design.digest(manifest))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'audit.json', report)
    lines = ['# Recursive reference validation', '',
        f"{report['complete']}/{report['planned']} independent datasets completed and passed integrity checks.",
        f"Stage {stage}: {'met' if report['qualified'] else 'did not meet'} the prespecified reference criteria.", '',
        '|Condition|Complete|Wrong-finding datasets|Exact networks|Qualified|', '|---|---:|---:|---:|---|']
    for label,c in report['cells'].items():
        lines.append(f"|{label}|{c['complete']}/80|{c['incorrect']['count']}|{c['recovery']['count']}|{c['qualified']}|")
    lines += ['', 'Wrong findings mean extra/incorrect retained relations; missed links and wrong depths remain visible in the raw case rows.',
        'These are narrow default-elastic-net simulations. Interactions, other engines, causal directions and a universal minimum sample size are not certified.',
        'Intervals are marginal per cell. Incomplete cases remain unknown; audit uses zero new model fits.']
    (output/'interpretation.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels, values = list(report['cells']), list(report['cells'].values())
    fig, axes = plt.subplots(1,3, figsize=(14,4))
    for ax, metric, title in zip(axes, ('incorrect','recovery'), ('Any wrong finding','Exact network recovered')):
        for i,c in enumerate(values):
            bound = c[metric]
            if not c['unknown']:
                rate = bound['count']/80
                ax.errorbar(i, rate, yerr=[[rate-bound['lower']],[bound['upper']-rate]], fmt='o', capsize=4)
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(-.03,1.03), title=title, ylabel='Proportion of independent datasets')
        ax.tick_params(axis='x', rotation=35)
    ax = axes[2]
    for i,c in enumerate(values):
        if not c['unknown'] and stage != 'noise':
            bound = design.interval(c['depth_not_recovered_count'])
            rate = bound['count']/80
            ax.errorbar(i,rate,yerr=[[rate-bound['lower']],[bound['upper']-rate]],fmt='o',capsize=4)
    ax.set(xticks=range(len(labels)),xticklabels=labels,ylim=(-.03,1.03),
           title='Expected depths not recovered',ylabel='Proportion of independent datasets')
    if stage == 'noise':
        ax.text(.5,.5,'No non-root depths expected',ha='center',transform=ax.transAxes)
    ax.tick_params(axis='x',rotation=35)
    fig.tight_layout()
    fig.savefig(output/'reference-rates.png', dpi=160)
    plt.close(fig)
    return report
