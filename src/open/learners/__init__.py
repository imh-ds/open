"""Fixed-engine factory. Optional engines are imported only when requested."""
from open.learners.elastic_net import ElasticNetPredictor


def make_predictor(config):
    engine = config.learner.library[0]
    if engine == "elastic_net":
        return ElasticNetPredictor(config)
    from open.learners.trees import TreePredictor
    return TreePredictor(engine, config.preprocessing, config.analysis.random_seed)
