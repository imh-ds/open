import numpy as np
import pandas as pd


def test_candidate_graph_and_plots_are_explicitly_exploratory(tmp_path):
    from open.interactions import PairScreenConfig,screen_pair_interactions,aggregate_pair_screens
    from open.interaction_visualization import save_interaction_report
    class Product:
        def predict(self,X): return X.iloc[:,0]*X.iloc[:,1]
    X=pd.DataFrame(np.random.default_rng(7).normal(size=(400,2)),columns=['a','b'])
    screen=screen_pair_interactions(Product(),X,('a','b'),outcome_sd=1,
                                    config=PairScreenConfig(max_details=1))
    screen.update(repeat=0,fold=0)
    summary=aggregate_pair_screens([screen])
    summary['fold_screens']=[screen]
    comparison=dict(target='y',engines=[dict(engine='known_function',interactions=summary)])
    save_interaction_report(comparison,tmp_path,plots=True)
    report=(tmp_path/'INTERACTIONS.md').read_text()
    assert 'Candidate joint pattern' in report
    assert 'not a validated population interaction' in report
    assert '```mermaid' in report
    assert (tmp_path/'interaction-known_function-0.png').exists()


def test_unsupported_pairs_are_not_drawn_as_zero_effects(tmp_path):
    from open.interactions import PairScreenConfig,screen_pair_interactions,aggregate_pair_screens
    from open.interaction_visualization import save_interaction_report
    class Product:
        def predict(self,X): return X.a*X.b
    X=pd.DataFrame({'a':np.arange(100),'b':np.arange(100)})
    screen=screen_pair_interactions(Product(),X,('a','b'),outcome_sd=1,config=PairScreenConfig())
    summary=aggregate_pair_screens([screen])
    summary['fold_screens']=[screen]
    comparison=dict(target='y',engines=[dict(engine='known_function',interactions=summary)])
    save_interaction_report(comparison,tmp_path)
    report=(tmp_path/'INTERACTIONS.md').read_text()
    assert 'No supported detailed candidates' in report
    assert 'incomplete_support' in report
    assert '```mermaid' not in report
