from dataclasses import replace
import pandas as pd
import pytest

from layer_helpers import make_config, install_scripted_analyzer
from open.layer_sensitivity import fit_layer_sensitivity


def test_sensitivity_reconstructs_later_candidate_sets(monkeypatch):
    selections = {('y', b): tuple(n for n, limit in [('a',7),('c',6)] if b < limit) for b in range(10)}
    calls = install_scripted_analyzer(monkeypatch, selections=selections)
    frame = pd.DataFrame({n: range(20) for n in ('y','a','c','x')})
    config = make_config()
    results = fit_layer_sensitivity(frame, 'y', tuple(frame), config)
    assert ('a',('x',),20,.6) in calls
    assert ('a',('c','x'),20,.7) in calls
    assert not any(t == 'a' and p == .8 for t,c,s,p in calls)
    assert results[-1].assigned_depth == {'y':0}
    assert results[0].plans == results[1].plans == results[2].plans
    assert config.thresholds.stability == .7


@pytest.mark.parametrize('thresholds',[(True,), (float('nan'),), (-.1,), (1.1,)])
def test_bad_thresholds(thresholds):
    with pytest.raises(ValueError):
        fit_layer_sensitivity(pd.DataFrame({'y':range(20)}),'y',('y',),make_config(),thresholds=thresholds)


def test_duplicates_reference_and_incomplete(monkeypatch):
    import open.layer_stability as engine
    monkeypatch.setattr(engine,'analyze_target',lambda *a: (_ for _ in ()).throw(RuntimeError('test')))
    frame = pd.DataFrame({'y':range(20),'a':range(20)})
    results = fit_layer_sensitivity(frame,'y',tuple(frame),make_config(),thresholds=(.6,.6),n_resamples=2)
    assert [r.config.thresholds.stability for r in results] == [.6,.7]
    assert all(r.status == 'INCOMPLETE_LAYER' and r.effective_resamples == 2 for r in results)
