import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

class GloveVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dim = 300

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        nlp = spacy.load("en_core_web_lg")
        return np.array([nlp(text).vector for text in X])
