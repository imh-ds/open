"""Engine factory: the validated engines (OLS, Elastic-Net, LightGBM) the experimental SVR, and PLS kept as a backend engine."""
from open.learners.elastic_net import ElasticNetPredictor

ENGINES = ("ols", "elastic_net", "lightgbm")
# Not yet through the validation study; explore() warns when one is used.
EXPERIMENTAL_ENGINES = ("svr",)
# Available by name but not offered; see docs/engines.md.
BACKEND_ENGINES = ("pls",)


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
    if engine == "pls":
        from open.learners.pls import PLSPredictor
        return PLSPredictor(preprocessing)
    if engine == "svr":
        from open.learners.svr import SVRPredictor
        return SVRPredictor(preprocessing)
    raise ValueError(f"unknown engine {engine!r}; use one of {ENGINES + EXPERIMENTAL_ENGINES}")
