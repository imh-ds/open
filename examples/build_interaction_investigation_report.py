"""Render the diagnostic evidence without refitting any models."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/results/2026-10-10-interaction-investigation'


def main():
    summary = json.loads((BASE/'summary.json').read_text())
    subsets = [[s for s in summary['summaries'] if s['budget'] == 10 and
                s['condition'].startswith(prefix)] for prefix in ('original', 'fresh')]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, rows, title in zip(axes, subsets,
        ('Original training partitions: more training rows per local fit',
         'Fresh development datasets: sample size still matters')):
        x = np.arange(len(rows))
        for offset, v, color in ((-.18, 'a', '#2874a6'), (.18, 'b', '#b9770e')):
            bars = ax.bar(x+offset, [r['selected'][v]/10 for r in rows], .34, label=v, color=color)
            ax.bar_label(bars, labels=[f"{r['selected'][v]}/10" for r in rows], fontsize=9)
        ax.axhline(.7, color='black', linestyle='--', linewidth=1, label='70% retention')
        labels = [r['condition'].replace('original-fold-', 'Partition ').replace('-k', '\nlocal folds: ')
                  .replace('fresh-interaction-', 'Interaction\nseed ').replace('fresh-noise-', 'Noise\nseed ')
                  .replace('-n', '\nn=') for r in rows]
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, 1.16)
        ax.set_ylabel('Individual selection frequency')
        ax.set_title(title)
        ax.legend(loc='upper right', ncol=3, fontsize=9)
    fig.suptitle('Interaction diagnosis: fixed engine and unchanged thresholds\nDevelopment evidence; no new outer-test evaluation', fontsize=13)
    fig.savefig(BASE/'selection.png', dpi=170)
    plt.close(fig)
    lines = ['# Interaction-selection investigation', '',
        'The local rule can recognize interaction inputs. In this example, small local',
        'training sets and coarse three-assessment consensus caused lost inputs.',
        'No production selection rule, engine, tuning grid or threshold was changed.', '',
        '![Selection counts](selection.png)', '',
        '| Training dataset | Local folds | Assessments | a selected | b selected | Retained | Low-gain stops |',
        '|---|---:|---:|---:|---:|---|---:|']
    for s in summary['summaries']:
        k = int(s['condition'][-1]) if s['condition'].startswith('original') else 5
        lines.append(f"| {s['condition']} | {k} | {s['budget']} | {s['selected']['a']}/{s['budget']} | {s['selected']['b']}/{s['budget']} | {', '.join(s['retained']) or 'none'} | {s['low_gain']} |")
    lines += ['', '## What happened', '',
        'Nine original root assessments were reproduced against the saved evidence.',
        'Three stopped because the full model failed the 2% predictive-gain gate.',
        'Of the other six, five retained both inputs and one retained only b.',
        'The original consensus therefore retained neither input in two partitions',
        'and only b in the third. Two out of three is 66.7%, below 70%.', '',
        'This was not a requirement that inputs predict well alone. For example, the',
        'first original assessment had negative standalone contributions for both',
        'inputs but positive conditional contributions and retained both.',
        'Conditional ablation removes one input while the other remains; it can',
        'detect a jointly useful input even when its standalone model is poor.', '',
        'The original outer-training partitions contained 53/54 people. Subsampling',
        'left 42/43. Two local folds trained each full model on only 21/22 people;',
        'two-fold tuning inside that trained candidates on roughly 10/11.',
        'Five local folds trained on 33–35 people and tuned on roughly 16–18.',
        'The CV change alters both training size and validation partitions, so these',
        'comparisons cannot isolate training size as the sole causal explanation.', '',
        'At ten assessments, two local folds retained both in two of three original',
        'partitions. Five local folds retained both in all three, at 10/10, 8/10,',
        'and 9/10. More repetitions alone did not resolve the second partition.', '',
        '## Viable next step', '',
        'Use the existing research defaults (five local performance folds, two',
        'repeats; the default assessment budget is 100), with settings fixed before',
        'outer evaluation. This investigation used one repeat and ten assessments;',
        'it has not calibrated those defaults. Do not treat a two-fold/three-run',
        'speed check as a dependable structural analysis. More assessments estimate',
        'repeatability more precisely; they do not make weak individual fits stronger.', '',
        'Continue comparing retained-input prediction with the all-candidate model.',
        'If selection loses substantial prediction, report a possible loss of useful',
        'inputs; do not call absent arrows proof of no relationship. No universal',
        'size of an acceptable performance gap has been calibrated.', '',
        'Before changing selection policy, run a separate multi-seed development',
        'study with noise, additive, redundancy, interaction and chain cases, then',
        'freeze independent confirmation cases. Measure missed inputs, false links,',
        'depth/class errors and prediction loss, along with actual per-fit sample',
        'sizes and low-gain stop rates. Stop short of accepting a new policy if it',
        'rescues interactions by routinely admitting noise or disturbing redundancy.', '',
        'If adequate-data validation still loses useful combinations, a separate',
        'prediction-preserving retention step is a candidate design: compare the',
        'consensus input set against richer sets using training-only validation.',
        'It would require its own tuning/validation layer and honest distinction',
        'between stable network links and extra prediction inputs. It should not',
        'silently promote unstable inputs into the recursive network. No such step',
        'or interaction-group label is implemented or validated here.', '',
        '## Scope and reproducibility', '',
        '110 local assessments; nine exact original replays; eleven conditions.',
        'Fresh cases are small development probes, not scientific calibration.',
        'Only original outer-training rows are used for refitting. No new held-out',
        'score or engine/threshold choice is made from the old test outcomes.',
        'Prefixes and overlapping subsamples are dependent; counts are not confidence',
        'probabilities. Original unfavorable results remain unchanged.', '',
        'The new diagnostic script initially used an incorrect ledger field name,',
        'then omitted the original outer-fold seed offset. Assertions stopped both',
        'attempts. The corrected run verifies all nine replays; completed matching',
        'conditions were reused only after input/configuration checks. These were',
        'diagnostic-script errors, not discovered production-engine defects.', '',
        'Reproduce from the repository root:', '', '```powershell',
        'python examples/investigate_interaction_selection.py',
        'python examples/audit_interaction_investigation.py',
        'python examples/build_interaction_investigation_report.py', '```', '',
        'The runner refuses an existing destination unless `--resume` is explicit.',
        'See [protocol](protocol.md), [summary](summary.json), [audit](audit.json),',
        'and [assessment table](assessments.csv); per-condition JSON retains local',
        'risk arithmetic, classifications and realized samples/splits.', '',
        'General CV background: [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).', '']
    (BASE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
