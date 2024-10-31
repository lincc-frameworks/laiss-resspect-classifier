import numpy as np

from resspect.feature_extractors import LightCurve

class LaissFeatureExtractor(LightCurve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, band:str = None) -> np.ndarray:
        pass

    def fit_all(self) -> np.ndarray:
        pass
