"""One reference graph for finalized layers, with auditable conditional evidence."""
import math
from pathlib import Path
import textwrap

CAPTION = ('Layer-wise consensus reference network. Frequencies describe repeated '
           'fixed-candidate assessments conditional on earlier finalized layers; '
           'arrows are predictive, not causal.')


def relation_width(value):
    return .8 + 3.2 * math.sqrt(min(1., max(0., value or 0.)))


def evidence_lines(bundle, *, finalized=True):
    lines = []
    def number(value):
        return 'unavailable' if value is None else f'{value:.4g}'
    for layer in bundle.layers:
        if layer['finalized'] != finalized:
            continue
        for target in layer['evidence']:
            completed = sum(a['status'] == 'completed' for a in bundle.assessments
                            if a['depth'] == layer['depth'] and a['target'] == target['target'])
            budget = bundle.metadata['planned_resamples_per_target']
            lines.append(f"{target['target']}: candidates={', '.join(target['candidates'])}; "
                         f"{completed}/{budget} completed")
            for relation in target['relations']:
                if relation['relation'] not in ('PROXIMAL_UNIQUE', 'SHARED_GROUP'):
                    continue
                count, strength = relation['frequency'], relation['strength']
                source = relation['source'] or '{' + ', '.join(relation['members']) + '}'
                if finalized:
                    tag = 'retained' if relation['qualifies'] else 'below threshold'
                    selection = f"{count['selected']}/{count['planned']}"
                else:
                    tag = 'provisional; layer unfinalized'
                    selection = (f"{count['selected']}/{count['completed']} completed selections; "
                                 f"planned {count['planned']}")
                lines.append(f"  {source} -> {target['target']}: {selection} "
                    f"({tag}); raw={number(strength['raw_median'])} [n={strength['raw_count']}]; "
                    f"baseline-relative={number(strength['baseline_relative_median'])} "
                    f"[n={strength['baseline_relative_count']}]; gain-relative={number(strength['gain_relative_median'])} "
                    f"[n={strength['gain_relative_count']}]")
    return lines


def plot_layer_network(bundle, *, output_path=None):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.text import Text
    except ImportError as exc:
        raise ImportError("Install OPEN's visualization extra: pip install open[visualization]") from exc
    depths = bundle.metadata['assigned_depth']
    labels = {v['name']: v.get('display_label') or v['name'] for v in bundle.metadata['variable_metadata']}
    positions = {}
    for depth in sorted(set(depths.values())):
        names = sorted(n for n, d in depths.items() if d == depth)
        for index, name in enumerate(names):
            positions[name] = (-depth, (len(names) - 1)/2 - index)
    lines = evidence_lines(bundle)
    partial = evidence_lines(bundle, finalized=False)
    wrapped = [part for line in lines for part in textwrap.wrap(line, width=135)]
    if partial:
        wrapped += ['Unfinalized current-layer evidence (excluded from the reference graph):']
        wrapped += [part for line in partial for part in textwrap.wrap(line, width=135)]
    fig = Figure(figsize=(13, 6 + .19 * len(wrapped)))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 1, height_ratios=[5, max(1, .19 * len(wrapped))])
    ax, panel = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    def label_radius(label, fontsize=10):
        # Measure in points so arrow clearance remains valid at exported DPI.
        text = Text(text=label, fontsize=fontsize)
        text.set_figure(fig)
        bounds = text.get_window_extent(fig.canvas.get_renderer())
        return math.hypot(bounds.width * 36 / fig.dpi + 6,
                          bounds.height * 36 / fig.dpi + 6)
    node_radii = {name: label_radius(textwrap.fill(labels[name], 18)) for name in positions}
    root = bundle.metadata['root']
    unavailable = False
    group_index = 0
    for layer in bundle.layers:
        if not layer['finalized']:
            continue
        for target in layer['evidence']:
            for relation in target['relations']:
                if not relation['qualifies']:
                    continue
                end = positions[target['target']]
                strength = relation['strength']['baseline_relative_median']
                unavailable |= strength is None or strength < 0
                width = relation_width(strength)
                freq = relation['frequency']
                alpha = .25 + .75 * freq['selected']/freq['planned']
                if relation['relation'] == 'SHARED_GROUP':
                    members = [positions[n] for n in relation['members']]
                    start = (end[0] - .45, sum(p[1] for p in members)/len(members) + .08*group_index)
                    group_index += 1
                    for point in members:
                        ax.plot([point[0], start[0]], [point[1], start[1]], color='#76579a', alpha=alpha)
                    ax.text(*start, 'Joint', ha='center', fontsize=9,
                            bbox=dict(boxstyle='round', fc='#eee7f5', ec='#76579a'))
                    color = '#76579a'
                    start_radius = label_radius('Joint', fontsize=9)
                elif relation['relation'] == 'PROXIMAL_UNIQUE':
                    start, color = positions[relation['source']], '#277fa5'
                    start_radius = node_radii[relation['source']]
                else:
                    continue
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                    linewidth=width, alpha=alpha, color=color, shrinkA=start_radius,
                    shrinkB=node_radii[target['target']]))
                ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.10,
                        f"{freq['selected']}/{freq['planned']}", fontsize=9, ha='center',
                        bbox=dict(fc='white', ec='none', alpha=.85))
    for name, point in positions.items():
        ax.text(*point, textwrap.fill(labels[name], 18), ha='center', va='center', fontsize=10,
                color='white' if name == root else '#24475b',
                bbox=dict(boxstyle='round,pad=.4', fc='#173f59' if name == root else '#ecf4fa', ec='#42657b'))
    ys = [p[1] for p in positions.values()]
    for depth in sorted(set(depths.values())):
        ax.text(-depth, min(ys)-.65, f'Depth {depth}', ha='center', color='#617483')
    ax.set_xlim(-max(depths.values())-.6, .6)
    ax.set_ylim(min(ys)-.9, max(ys)+.7)
    ax.axis('off')
    notice = (' — INCOMPLETE: finalized prefix only' if bundle.metadata['status'] != 'COMPLETE' else '')
    ax.set_title('Layer-wise consensus reference network' + notice, pad=18)
    panel.axis('off')
    panel.text(0, 1, '\n'.join(wrapped) or 'No local comparisons required.', va='top', fontsize=8,
               transform=panel.transAxes)
    note = '\nMinimum width denotes unavailable or negative strength.' if unavailable else ''
    fig.text(.03, .015, textwrap.fill(CAPTION, 155) + note, fontsize=9)
    fig.subplots_adjust(left=.05, right=.97, bottom=.10, top=.94, hspace=.20)
    if output_path is not None:
        fig.savefig(Path(output_path), dpi=160)
    return fig
