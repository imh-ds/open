"""Render the fixed study's audited summaries; no fitting or policy selection."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = dict(noise='#666666', ordinary='#2874a6', redundant='#239b56', interaction='#b9770e')


def findings(cases, audit):
    if audit['completed_cases'] != audit['planned_cases']:
        return ['The study is incomplete; a final recommendation is withheld.', '']
    signal = [c for c in cases if c['scenario'] != 'noise']
    noise_checks = [c for c in cases if c['scenario'] in ('noise', 'ordinary')]
    misses = [c for c in signal if not c['root_recovered']]
    large = [c for c in misses if c['n'] == 400]
    false = sum(c['false_input'] for c in noise_checks)
    lines = [f"Required root inputs were retained in {len(signal)-len(misses)}/{len(signal)} signal datasets.",
             f"A noise input was retained in {false}/{len(noise_checks)} noise/ordinary datasets.", '']
    if len(large) >= 2:
        lines += [f"The frozen decision rule is triggered: useful inputs were lost in {len(large)} datasets at n=400.",
                  'Next: design a separate selection safeguard and evaluate it on fresh development',
                  'and confirmation datasets before changing the production rule.']
    elif large:
        lines += ['One n=400 dataset lost a required input. This does not satisfy the repeated',
                  'large-sample failure criterion; investigate that case and obtain fresh confirmation',
                  'before accepting a policy change.']
    elif misses:
        lines += ['Required-input misses occurred only at smaller sample sizes. Next: examine local',
                  'training sizes and weak prediction first. This does not establish a safe minimum n.']
    else:
        lines += ['No required input was lost in these strong-signal simulations. The results do not',
                  'justify a selection-rule patch. Next: improve training-size, weak-fit and',
                  'post-selection prediction-loss diagnostics for cases like the earlier small pilot.']
    if false:
        lines += ['The observed noise retention also needs follow-up; correct signal recovery alone',
                  'does not establish reliable network structure.']
    for c in misses:
        lines += ['', f"In `{c['id']}`, a was promoted {c['a_promoted']}/{c['planned']} times and",
                  f"b {c['b_promoted']}/{c['planned']} times; {c['low_gain']} assessments stopped for low total gain.",
                  f"Local model fits used {c['min_fit_n']}–{c['max_fit_n']} people; tuning fits used {c['min_tuning_n']}–{c['max_tuning_n']}.",
                  f"The full-data model's independent-test skill was {c['full_skill']:.3f}, versus {c['retained_skill']:.3f} after selection.",
                  'An empty stable network therefore did not mean there was no useful predictive',
                  'relationship. It meant the relationships failed the unchanged stability requirement',
                  'under these resampled training conditions. A larger number of assessments does not',
                  'automatically make each assessment learn the relationship better.']
    return lines+['', 'These are four independent datasets per scenario/sample size and only two inputs.',
                  'The study does not establish general reliability, a safe sample-size cutoff,',
                  'or recovery of a complete recursive network.', '']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base = args.output
    summary = json.loads((base/'summary.json').read_text())
    audit = json.loads((base/'audit.json').read_text())
    cells, cases = summary['cells'], summary['cases']
    sizes = sorted({c['n'] for c in cells})
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    for scenario, offset in (('ordinary', -.12), ('redundant', 0), ('interaction', .12)):
        rows = sorted((c for c in cells if c['scenario'] == scenario), key=lambda c:c['n'])
        for i, c in enumerate(rows):
            if 'root_recovered' not in c:
                continue
            count = c['root_recovered']['count']
            y = count/c['planned']
            low, high = c['root_recovered']['wilson95']
            axes[0].errorbar(i+offset, y, yerr=np.array([[y-low], [high-y]]),
                fmt='o', capsize=4, color=COLORS[scenario], label=scenario if i == 0 else None)
    axes[0].set(title='Recovery of required root inputs: independent datasets', ylabel='Fraction of datasets')
    axes[0].legend(loc='lower right')
    for scenario, offset in (('noise', -.08), ('ordinary', .08)):
        rows = sorted((c for c in cells if c['scenario'] == scenario), key=lambda c:c['n'])
        for i, c in enumerate(rows):
            if 'false_input' not in c:
                continue
            y = c['false_input']['count']/c['planned']
            low, high = c['false_input']['wilson95']
            axes[1].errorbar(i+offset, y, yerr=np.array([[y-low], [high-y]]),
                fmt='o', capsize=4, color=COLORS[scenario], label=scenario if i == 0 else None)
    axes[1].set(title='Retention of a noise input', ylabel='Fraction of datasets')
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc='upper right')
    for ax in axes[:2]:
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set_xlabel('Discovery participants (before subsampling/CV)')
        ax.set_ylim(-.06, 1.06)
        ax.grid(axis='y', alpha=.2)
    for si, scenario in enumerate(COLORS):
        for c in cases:
            if c['scenario'] == scenario and c['status'] == 'complete':
                jitter = (c['repetition']-1.5)*.035+(si-1.5)*.12
                axes[2].scatter(sizes.index(c['n'])+jitter, c['skill_loss'], color=COLORS[scenario],
                    label=scenario if c['n'] == sizes[0] and c['repetition'] == 0 else None)
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].axhline(.05, color='gray', linestyle='--', label='.05 descriptive flag')
    axes[2].set_xticks(range(len(sizes)), sizes)
    axes[2].set(title='Prediction lost after selecting inputs: each dataset',
                ylabel='All-input skill minus retained-input skill', xlabel='Discovery participants')
    axes[2].legend(loc='best', ncol=2, fontsize=9)
    axes[2].grid(axis='y', alpha=.2)
    fig.suptitle('GBM root selection with normal research settings\n4 independent datasets per scenario/sample size; bars show Wilson 95% intervals', fontsize=12)
    fig.savefig(base/'reliability.png', dpi=170)
    plt.close(fig)
    lines = ['# Root-selection reliability investigation', '',
        'Fixed GBM; five local folds, two repeats, three tuning folds, 100 stability',
        'assessments per dataset, 80% subsampling, unchanged 70% retention.',
        'Independent test participants are generated after each discovery dataset;',
        'they do not select inputs, engines, thresholds or settings.', '',
        f"Completed {audit['completed_cases']}/{audit['planned_cases']} datasets and {audit['completed_assessments']}/{audit['planned_assessments']} local assessments.",
        f"Audited {audit['independent_test_predictions']} independent test participants; {audit['warnings']} recorded warnings.", '']
    lines += findings(cases, audit)
    lines += [
        '![Recovery, noise retention and prediction loss](reliability.png)', '',
        '## Results by scenario and sample size', '',
        '| Scenario | Discovery n | Completed/planned | Root recovered | Noise input retained | Shared group retained | Skill loss >.05 | Mean full skill | Mean retained skill | Mean loss (MC SE) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        def count(label):
            return f"{c[label]['count']}/{c[label]['planned']}" if label in c else 'withheld'
        def number(label):
            return f"{c[label]:.3f}" if label in c else 'withheld'
        loss = (f"{c['mean_skill_loss']:.3f} ({c['mcse_skill_loss']:.3f})" if c.get('mcse_skill_loss') is not None else 'withheld')
        shared = count('shared_retained') if c['scenario'] == 'redundant' else '—'
        false = count('false_input') if c['scenario'] in ('noise', 'ordinary') else '—'
        lines.append(f"| {c['scenario']} | {c['n']} | {c['completed']}/{c['planned']} | {count('root_recovered')} | {false} | {shared} | {count('loss_over_point05')} | {number('mean_full_skill')} | {number('mean_retained_skill')} | {loss} |")
    lines += ['', 'Root recovered means a retained for ordinary, both a and b retained for',
        'interaction, at least one retained for redundancy, and neither retained for',
        'noise. Redundant-group preservation is a separate classification result.',
        'A retained noise input is a false input in the known simulated conditional',
        'mean; it is not evidence about causal arrows. Skill loss is signed: a',
        'negative loss means the retained model predicted better on that test set.',
        '.05 is a predeclared descriptive flag, not an acceptable-gap threshold.', '',
        '## Failures and local training sizes', '',
        '| Discovery n | Local subsample n | Full-model fit n | Tuning-model fit n |',
        '|---:|---:|---:|---:|']
    for n in sizes:
        rows = [c for c in cases if c['n'] == n and c['min_fit_n'] is not None]
        if rows:
            lines.append(f"| {n} | {int(np.floor(.8*n))} | {min(c['min_fit_n'] for c in rows)}–{max(c['max_fit_n'] for c in rows)} | {min(c['min_tuning_n'] for c in rows)}–{max(c['max_tuning_n'] for c in rows)} |")
    lines += ['', 'Each case records its local low-gain stop count, conditional contribution',
        'and promotion/class counts. Low predictive gain is a valid nonselection,',
        'distinct from an execution failure. Incomplete cases preserve their planned',
        'denominators and withhold test scores and cell-level performance estimates.', '',
        '## Limits and decision boundary', '',
        'This is a targeted development study of one engine and one root comparison.',
        'It does not validate deeper-layer placement, whole-network recovery,',
        'same-depth links, other engines, weak effects, correlated interaction inputs,',
        'non-Gaussian predictors, measurement error, missingness or real survey data.',
        'The chosen signals are strong and the redundancy nearly exact.', '',
        'There are four independent datasets per cell. Even 4/4 recovery has a Wilson',
        '95% interval of approximately 51%–100%; 0/4 noise retention has an interval of',
        'approximately 0%–49%. Therefore no observed misses does not prove reliability',
        'and no observed false links does not establish a low false-positive rate.',
        'The 100 overlapping assessments inside a dataset are not 100 independent',
        'simulation replications. Monte Carlo SE reflects variability across the',
        'four datasets, not uncertainty of an individual edge.', '',
        'The fixed protocol calls for a safeguard design if useful inputs disappear',
        'at n=400 in multiple datasets. Loss concentrated at smaller sizes supports',
        'training-size and weak-fit diagnostics first. Neither result calibrates a',
        'minimum sample size or accepts a new production policy. Examine shared-class',
        'fragmentation and noise retention alongside interaction recovery.', '',
        '## Reproduction and audit', '', '```powershell',
        'python examples/run_selection_reliability.py --workers 8 --output NEW_DIRECTORY',
        'python examples/audit_selection_reliability.py NEW_DIRECTORY',
        'python examples/build_selection_reliability_report.py NEW_DIRECTORY', '```', '',
        'The runner refuses an existing destination unless `--resume` is explicit.',
        'Resume verifies the frozen configuration and dataset identities, validates',
        'production-source fingerprints, and reuses atomic per-assessment checkpoints',
        'under `.superpowers/selection-checkpoints`. Initial and resumed runner',
        'fingerprints remain separate in the manifest. The initial interrupted run',
        'saved only progress counts and had to replay those samples; its counters',
        'are not accepted statistical evidence. Elapsed resumed time excludes that',
        'lost computation and is not a pristine runtime benchmark.',
        'This study then moved to a dedicated GitHub Actions branch. The synthetic',
        'transfer preserved 1,946 assessments and 12 completed datasets. Hosted jobs',
        'restored the exact source bytes, used pinned numerical dependencies, and',
        'computed remaining assessments with four workers per dataset. Per-case',
        '`actions-execution.json` files record reuse, platform, versions and GitHub',
        'run/commit provenance. See [Actions workflow documentation](../../selection_actions.md).',
        '`--smoke` uses deliberately tiny',
        'settings for plumbing checks and is excluded from this study. The smoke',
        'check compares singleton orchestration with the production batch assessor.',
        'An initial smoke-only attribute-name error was corrected before study',
        'dispatch; no study dataset was discarded or changed.', '',
        'See [fixed protocol](protocol.md), [manifest](manifest.json),',
        '[audit](audit.json), [case table](cases.csv), [summary](summary.json),',
        'and [execution record](execution.json). Each case contains discovery/test',
        'data, predictions, and compressed local evidence. Seeds and recorded hashes',
        'allow exact sample/split reconstruction. The audit recomputes scores, local',
        'gain/normalization and relation/class/retention counts, but does not refit',
        'every local model or independently inspect every tuning decision.', '',
        'Simulation reporting follows [Morris, White & Crowther (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/).', '']
    (base/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
