from sklearn.base import BaseEstimator, TransformerMixin


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def fit(self, X, y=None):
        self.means = X[self.num_features].mean(axis=0)
        self.stds = X[self.num_features].std(axis=0) + 1e-10
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.num_features] -= self.means
        X_[self.num_features] /= self.stds
        return X_
