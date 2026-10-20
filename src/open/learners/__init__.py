"""Engine factory for the validated engines: OLS, Elastic-Net and LightGBM."""
from open.learners.elastic_net import ElasticNetPredictor

ENGINES = ("ols", "elastic_net", "lightgbm")


def make_predictor(engine: str, random_seed: int = 0, preprocessing=None):
    """Return an unfitted predictor. LightGBM is imported only when requested."""
    if engine == "elastic_net":
        return ElasticNetPredictor(preprocessing)
    if engine == "ols":
        from open.learners.ols import OLSPredictor
        return OLSPredictor(preprocessing)
    if engine == "lightgbm":
        from open.learners.lightgbm import LightGBMPredictor
        return LightGBMPredictor(preprocessing, random_seed)
    raise ValueError(f"unknown engine {engine!r}; use one of {ENGINES}")
