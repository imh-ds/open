"""Explicitly exploratory pair junctions and supported local-contrast plots."""
import html
from pathlib import Path

import numpy as np


def _label(value):
    return html.escape(str(value),quote=True).replace('\n',' ').replace('\r',' ')


def save_interaction_report(comparison,directory,*,plots=False,max_displayed=5):
    """Write a separate view; never replace predictive edges with main-effect claims."""
    if type(plots) is not bool or type(max_displayed) is not int or max_displayed < 0:
        raise ValueError('plots must be boolean and max_displayed nonnegative integer')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    lines=['# Exploratory fitted-model interaction patterns','',
        'A junction is a candidate joint pattern in fitted predictions, not a validated population interaction. '
        'It does not establish causal moderation. Ordinary predictive arrows do not certify separate main effects.','',
        'Shortlisting is computational triage. Scores describe local mixed prediction contrasts, not variance '
        'explained, coefficients, p-values or full second-order ALE. They can include higher-order interactions. '
        'Absence of a junction is not evidence of no interaction.','']
    for engine in comparison['engines']:
        summary=engine.get('interactions')
        if summary is None:
            continue
        lines += [f"## {_label(engine['engine'])}",'',
            f"Planned folds: {summary['planned_folds']}; evaluated prediction rows: {summary['prediction_rows']}; "
            f"additional model fits: {summary['additional_model_fits']}.",'',
            '|Pair|Supported screens / planned folds|Shortlisted folds|Supported details|Screen RMS (outcome SD)|Status|',
            '|---|---:|---:|---:|---:|---|']
        for pair in summary['pairs']:
            label=' + '.join(_label(n).replace('|','&#124;') for n in pair['members'])
            score='unavailable' if pair['mean_screen_rms'] is None else f"{pair['mean_screen_rms']:.4g}"
            lines.append(f"|{label}|{pair['supported_screen_folds']}/{pair['planned_folds']}|"
                f"{pair['shortlisted_folds']}|{pair['supported_detail_folds']}|{score}|{pair['status']}|")
        candidates=[p for p in summary['pairs'] if p['mean_screen_rms'] is not None
                    and p['shortlisted_folds'] and p['supported_detail_folds']]
        candidates.sort(key=lambda p:(-p['mean_screen_rms'],p['members']))
        displayed=candidates[:max_displayed]
        lines += ['',f'Displayed {len(displayed)} of {len(candidates)} supported detailed candidates; '
                  'rank is not statistical acceptance.','']
        if not displayed:
            lines += ['No supported detailed candidates are displayed. Unsupported or unexamined pairs '
                      'remain unavailable, not zero.','']
            continue
        names=sorted({n for p in displayed for n in p['members']})
        ids={name:f'n{i}' for i,name in enumerate(names)}
        lines += ['```mermaid','flowchart LR',f'  target["{_label(comparison["target"])}"]']
        for name in names:
            lines.append(f'  {ids[name]}["{_label(name)}"]')
        for i,pair in enumerate(displayed):
            lines.append(f'  joint{i}{{"Candidate joint pattern"}}')
            for name in pair['members']:
                lines.append(f'  {ids[name]} --> joint{i}')
            lines.append(f'  joint{i} --> target')
        lines += ['```','']
        if not plots:
            continue
        import matplotlib.pyplot as plt
        for i,pair in enumerate(displayed):
            # Fixed explanatory choice: first supported detailed fold in saved order.
            chosen=next(f for f in summary['fold_screens'] if any(
                p['members']==pair['members'] and p['detail']['status']=='available' for p in f['pairs']))
            detail=next(p['detail'] for p in chosen['pairs'] if p['members']==pair['members'])
            shape=tuple(detail['realized_bins'])
            mean=np.zeros(shape)
            rms=np.zeros(shape)
            counts=np.zeros(shape,dtype=int)
            for cell in detail['cells']:
                index=cell['i'],cell['j']
                mean[index]=cell['standardized_mean']
                rms[index]=cell['standardized_rms']
                counts[index]=cell['count']
            fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
            fig.suptitle('Exploratory local joint contrasts: '+' + '.join(pair['members'])+
                f"\n{engine['engine']}; repeat {chosen['repeat']}, fold {chosen['fold']}")
            for ax,values,title in zip(axes,[mean,rms],['Signed cell mean','Within-cell RMS magnitude']):
                bound=max(float(np.max(np.abs(values))),1e-12)
                image=ax.imshow(values.T,origin='lower',aspect='auto',cmap='coolwarm' if ax is axes[0] else 'viridis',
                    vmin=-bound if ax is axes[0] else 0,vmax=bound)
                ax.set_title(title+' (outcome SD)')
                ax.set_xlabel(pair['members'][0]+' quantile bin')
                ax.set_ylabel(pair['members'][1]+' quantile bin')
                ax.set_xticks(range(shape[0]),range(1,shape[0]+1))
                ax.set_yticks(range(shape[1]),range(1,shape[1]+1))
                for a in range(shape[0]):
                    for b in range(shape[1]):
                        ax.text(a,b,f'n={counts[a,b]}',ha='center',va='center',fontsize=8,
                                bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
                fig.colorbar(image,ax=ax)
            name=f"interaction-{engine['engine']}-{i}.png"
            fig.savefig(directory/name,dpi=150)
            plt.close(fig)
            lines += [f"![Local joint contrasts]({name})",'',
                'This shows the first supported detailed fold in saved order, not an aggregate population '
                'surface. Cell counts refer to sampled reference observations. Signed means may cancel '
                'when effects vary with other variables; RMS retains their magnitude. Bin corners may '
                'leave the observed manifold. Other folds and boundaries remain in comparison.json.','']
    (directory/'INTERACTIONS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
