import pytest

from layer_helpers import save_root_only_bundle
from test_layer_io import saved
from open.layer_io import load_layer_bundle
from open.visualization import load_visualization_data, plot_open_network


def test_layer_bundle_refuses_a_legacy_consensus_view(tmp_path):
    path = tmp_path / 'root'
    save_root_only_bundle(path)
    with pytest.raises(ValueError, match='method'):
        load_visualization_data(path, view='consensus')
    assert load_visualization_data(path).metadata['method'] == 'layer_consensus'
    with pytest.raises(ValueError, match='reconstruction'):
        plot_open_network(load_layer_bundle(path), min_frequency=.8)


def test_unique_fixed_depth_count_width_and_evidence(tmp_path, monkeypatch):
    _, path = saved(tmp_path, monkeypatch)
    fig = plot_open_network(load_visualization_data(path))
    ax = fig.axes[0]
    assert len(ax.patches) == 1
    assert ax.patches[0].get_linewidth() == pytest.approx(.8 + 3.2 * .2**.5)
    texts = ' '.join(t.get_text() for a in fig.axes for t in a.texts) + ' '.join(t.get_text() for t in fig.texts)
    assert '10/10' in texts and 'predictive, not causal' in texts
    assert 'Depth 0' in texts and 'Depth 1' in texts
    assert 'raw=' in texts and 'candidates=' in texts


def test_one_joint_connector_and_no_unique_member_arrows(tmp_path, monkeypatch):
    _, path = saved(tmp_path, monkeypatch, shared=True)
    fig = plot_open_network(load_visualization_data(path))
    assert len(fig.axes[0].patches) == 1  # joint arrow only; member spokes are lines
    assert any('Joint' in text.get_text() for text in fig.axes[0].texts)


def test_incomplete_root_has_visible_notice_and_partial_table(tmp_path, monkeypatch):
    _, path = saved(tmp_path, monkeypatch, fail=True)
    fig = plot_open_network(load_visualization_data(path))
    texts = ' '.join(t.get_text() for a in fig.axes for t in a.texts) + fig.axes[0].get_title()
    assert 'INCOMPLETE' in texts and 'Unfinalized' in texts and '0/10 completed' in texts
    assert not fig.axes[0].patches


def test_unavailable_and_negative_strength_keep_minimum_width():
    from open.layer_visualization import relation_width
    assert relation_width(None) == relation_width(-.5) == .8
    assert relation_width(4) == 4


def test_partial_selection_reports_completed_denominator_without_rejection(tmp_path, monkeypatch):
    import open.layer_stability as engine
    from layer_helpers import make_config, make_local
    from test_layer_io import metadata
    from open.layer_io import write_layer_outputs
    import pandas as pd
    def analyze(d,t,c,cfg,s):
        if cfg.analysis.random_seed == 21: raise RuntimeError('one failed fit')
        return make_local(t,c,unique=('a',))
    monkeypatch.setattr(engine,'analyze_target',analyze)
    frame=pd.DataFrame({n:range(20) for n in ('y','a','x')})
    fit=engine.fit_layer_consensus(frame,'y',tuple(frame),make_config(),n_resamples=2)
    path=tmp_path/'partial'
    write_layer_outputs(fit,path,metadata=metadata())
    from open.layer_visualization import evidence_lines
    text=' '.join(evidence_lines(load_layer_bundle(path),finalized=False))
    assert '1/1 completed selections; planned 2' in text
    assert 'below threshold' not in text and '(retained)' not in text


def test_complete_sibling_in_unfinalized_layer_is_not_called_retained(tmp_path, monkeypatch):
    import open.layer_stability as engine
    from layer_helpers import make_config, make_local
    from test_layer_io import metadata
    from open.layer_io import write_layer_outputs
    from open.layer_visualization import evidence_lines
    import pandas as pd
    def analyze(d,t,c,cfg,s):
        if t == 'a': raise RuntimeError('failed sibling')
        return make_local(t,c,unique=('a','x') if t=='y' else ('c',))
    monkeypatch.setattr(engine,'analyze_target',analyze)
    frame=pd.DataFrame({n:range(20) for n in ('y','a','c','x')})
    fit=engine.fit_layer_consensus(frame,'y',tuple(frame),make_config(),n_resamples=2)
    path=tmp_path/'siblings';write_layer_outputs(fit,path,metadata=metadata(tuple(frame)))
    text=' '.join(evidence_lines(load_layer_bundle(path),finalized=False))
    assert '2/2 completed selections; planned 2' in text
    assert '(retained)' not in text and 'unfinalized' in text


def test_long_target_label_does_not_cover_the_arrowhead(tmp_path,monkeypatch):
    from test_layer_io import result
    from open.metadata import validate_metadata
    from open.layer_io import write_layer_outputs
    fit=result(monkeypatch)
    metadata=validate_metadata(fit.variables,[{'name':n,'type':'continuous',
        'role':'root' if n=='y' else 'candidate',
        'display_label':'Perceived Stress' if n=='y' else n} for n in fit.variables],'y')
    path=tmp_path/'long_label';write_layer_outputs(fit,path,metadata=metadata)
    fig=plot_open_network(load_layer_bundle(path));fig.canvas.draw()
    ax=fig.axes[0]
    target=next(t for t in ax.texts if t.get_text()=='Perceived Stress')
    assert ax.patches[0].get_extents().x1 < target.get_bbox_patch().get_extents().x0
