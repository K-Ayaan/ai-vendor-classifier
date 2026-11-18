"""Committees: bagging and random forests (Breiman 1996, 2001; Bishop 14.2; Murphy 16.2.5).

Bagging trains M models on bootstrap resamples and averages them,
    y_COM(x) = (1/M) sum_m y_m(x).
Bishop's argument: with errors e_m(x) = y_m(x) - h(x), E_COM = E[((1/M) sum e_m)^2]. If the e_m
had zero mean and were uncorrelated, E_COM = E_AV / M; in practice they are correlated, so the
gain is smaller, but E_COM <= E_AV always (Jensen / Cauchy-Schwarz).
A random forest also draws a random feature subset at every split to de-correlate the trees.

A bootstrap resample is passed to the tree as integer sample weights (the count of each row),
which is how scikit-learn implements it too. Rows not drawn (about e^-1 = 36.8%) are
"out-of-bag" and give an honest validation prediction for free.
"""
import numpy as np

from .tree import CART


class Forest:
    """task: "regression" or "classification" (binary, y in {0, 1}).
    max_features=None -> bagged trees; a fraction or int -> random forest."""

    def __init__(self, task="regression", n_trees=200, max_features=None, min_samples_leaf=1,
                 max_depth=None, bootstrap=True, seed=0):
        self.task, self.B, self.max_features = task, n_trees, max_features
        self.min_leaf, self.max_depth, self.bootstrap, self.seed = (min_samples_leaf, max_depth,
                                                                     bootstrap, seed)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        oob_sum, oob_cnt = np.zeros(n), np.zeros(n)
        for b in range(self.B):
            counts = (np.bincount(rng.integers(0, n, n), minlength=n) if self.bootstrap
                      else np.ones(n, int))
            tree = CART(self.task, max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
                        max_features=self.max_features,
                        seed=int(rng.integers(2 ** 31))).fit(X, y, sample_weight=counts)
            self.trees.append(tree)
            out = counts == 0
            if out.any():
                oob_sum[out] += self._one(tree, X[out])
                oob_cnt[out] += 1
        with np.errstate(invalid="ignore"):
            self.oob_prediction_ = oob_sum / oob_cnt               # nan if never out of bag
        self.oob_coverage_ = float(np.mean(oob_cnt > 0))
        return self

    def _one(self, tree, X):
        if self.task == "regression":
            return tree.predict(X)
        P = tree.predict_proba(X)
        return P[:, list(tree.classes_).index(1)] if 1 in tree.classes_ else np.zeros(len(X))

    def member_predictions(self, X):
        return np.array([self._one(t, X) for t in self.trees])

    def predict(self, X):
        return self.member_predictions(X).mean(0)

    def feature_importances(self):
        return np.mean([t.feature_importances() for t in self.trees], 0)
