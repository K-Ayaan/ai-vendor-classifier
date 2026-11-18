"""Boosting: AdaBoost (Bishop 14.3) and gradient boosting (Friedman 2001; Murphy 16.4).

Both are adaptive basis-function models f(x) = sum_m beta_m phi(x; v_m) fitted by forward
stagewise additive modelling: at round m the earlier terms are frozen and only the new basis
function and its weight are chosen.

AdaBoost, t in {-1, +1}, weak learner y_m(x) in {-1, +1} (a decision stump here):
    eps_m   = sum_n w_n I(y_m(x_n) != t_n) / sum_n w_n          (weighted error)
    alpha_m = ln((1 - eps_m) / eps_m)
    w_n    <- w_n exp(alpha_m I(y_m(x_n) != t_n))
    f_M(x)  = (1/2) sum_m alpha_m y_m(x);   predict sign(f_M(x))
This is exactly the stagewise minimisation of the exponential error E = sum_n exp(-t_n f(x_n)),
and with normalised weights E_M / N = prod_m 2 sqrt(eps_m (1 - eps_m)), which bounds the
training error. Since the minimiser of E is f = (1/2) ln(p / (1 - p)), P(t = 1 | x) = sigma(2 f).

Gradient boosting: each round fits a regression tree to the negative gradient of the loss,
    squared error  r = y - F(x)               leaf value = mean residual
    logistic loss  r = y - sigma(F(x))        leaf value = one Newton step sum r / sum p(1 - p)
and adds nu * tree (nu = shrinkage / learning rate).
"""
import numpy as np

from .tree import CART, EPS_THRESHOLD


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


# ---------------------------------------------------------------- decision stump
def fit_stump(X, t, w, binary=None):
    """Stump minimising the weighted 0-1 error directly: y(x) = s if x_j > c else -s.
    Returns (j, c, s, weighted_error). Constant classifiers are allowed (j = -1)."""
    X = np.asarray(X, float)
    n, p = X.shape
    pos, neg = w * (t > 0), w * (t < 0)
    Wp, Wn = pos.sum(), neg.sum()
    best = (-1, np.inf, 1 if Wp >= Wn else -1, min(Wp, Wn))   # constant: predict majority
    binary = np.all((X == 0) | (X == 1), axis=0) if binary is None else binary
    jb = np.where(binary)[0]
    if len(jb):
        Xb = X[:, jb]
        p1, q1 = Xb.T @ pos, Xb.T @ neg              # weight of +/- among x = 1
        err_plus = (Wp - p1) + q1                    # s = +1: predict +1 iff x = 1
        err = np.minimum(err_plus, Wp + Wn - err_plus)
        k = int(np.argmin(err))
        if err[k] < best[3] - 1e-15:
            s = 1 if err_plus[k] <= Wp + Wn - err_plus[k] else -1
            best = (int(jb[k]), 0.5, s, float(err[k]))
    js = np.where(~binary)[0]
    if len(js):
        Xs_raw = X[:, js]
        order = np.argsort(Xs_raw, axis=0, kind="stable")
        Xs = np.take_along_axis(Xs_raw, order, 0)
        cp = np.cumsum(pos[order], 0)[:-1]           # + weight on the left (x <= c)
        cn = np.cumsum(neg[order], 0)[:-1]
        err_plus = cp + (Wn - cn)                    # s = +1: left -> -1, right -> +1
        err = np.minimum(err_plus, Wp + Wn - err_plus)
        err = np.where(Xs[1:] > Xs[:-1] + EPS_THRESHOLD, err, np.inf)
        i, c = np.unravel_index(int(np.argmin(err)), err.shape)
        if err[i, c] < best[3] - 1e-15:
            s = 1 if err_plus[i, c] <= Wp + Wn - err_plus[i, c] else -1
            best = (int(js[c]), float(Xs[i, c] / 2 + Xs[i + 1, c] / 2), s, float(err[i, c]))
    return best


def stump_predict(X, j, c, s):
    if j < 0:
        return np.full(len(X), float(s))
    return np.where(np.asarray(X)[:, j] > c, s, -s).astype(float)


class AdaBoost:
    def __init__(self, n_rounds=200, eps_floor=1e-10):
        self.M, self.eps_floor = n_rounds, eps_floor

    def fit(self, X, y01):
        X = np.asarray(X, float)
        t = np.where(np.asarray(y01) > 0, 1.0, -1.0)
        n = len(t)
        binary = np.all((X == 0) | (X == 1), axis=0)
        w = np.full(n, 1.0 / n)
        self.stumps, self.alphas, self.eps = [], [], []
        F = np.zeros(n)
        self.exp_loss, self.train_error, self.bound = [], [], []
        bound = 1.0
        for _ in range(self.M):
            j, c, s, err = fit_stump(X, t, w, binary)
            eps = float(np.clip(err / w.sum(), self.eps_floor, 1 - self.eps_floor))
            if eps >= 0.5:
                break
            a = np.log((1 - eps) / eps)
            h = stump_predict(X, j, c, s)
            miss = h != t
            w = w * np.exp(a * miss)
            w /= w.sum()
            self.stumps.append((j, c, s)); self.alphas.append(a); self.eps.append(eps)
            F += 0.5 * a * h
            bound *= 2 * np.sqrt(eps * (1 - eps))
            self.exp_loss.append(float(np.mean(np.exp(-t * F))))
            self.train_error.append(float(np.mean(np.sign(F) != t)))
            self.bound.append(float(bound))
        self.alphas = np.array(self.alphas)
        return self

    def staged_f(self, X):
        """f_m(x) = (1/2) sum_{l <= m} alpha_l y_l(x) for every m: array (M, n)."""
        H = np.array([stump_predict(X, *st) for st in self.stumps])
        return np.cumsum(0.5 * self.alphas[:, None] * H, 0)

    def predict_proba(self, X, m=None):
        S = self.staged_f(X)
        return sigmoid(2 * S[(m or len(S)) - 1])


# ---------------------------------------------------------------- gradient boosting
class GradientBoosting:
    """loss = "squared" (regression) or "logistic" (y in {0, 1})."""

    def __init__(self, loss="squared", n_rounds=200, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, max_features=None, seed=0):
        self.loss, self.M, self.nu = loss, n_rounds, learning_rate
        self.max_depth, self.min_leaf, self.subsample = max_depth, min_samples_leaf, subsample
        self.max_features, self.seed = max_features, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        if self.loss == "squared":
            self.init_ = float(y.mean())
        else:
            pbar = y.mean()
            self.init_ = float(np.log(pbar / (1 - pbar)))
        F = np.full(n, self.init_)
        self.trees, self.train_loss = [], []
        for m in range(self.M):
            if self.loss == "squared":
                r = y - F
            else:
                p = sigmoid(F)
                r = y - p
            rows = (np.sort(rng.choice(n, int(round(self.subsample * n)), replace=False))
                    if self.subsample < 1 else np.arange(n))
            tree = CART("regression", max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
                        max_features=self.max_features, seed=self.seed + m).fit(X[rows], r[rows])
            if self.loss == "logistic":                       # Newton step in every leaf
                leaf = tree.apply(X[rows])
                num = np.bincount(leaf, r[rows], tree.n_nodes)
                den = np.bincount(leaf, (p * (1 - p))[rows], tree.n_nodes)
                tree.value_[:, 0] = np.where(den > 1e-150, num / np.maximum(den, 1e-150), 0.0)
            F = F + self.nu * tree.predict(X)
            self.trees.append(tree)
            self.train_loss.append(self._loss(y, F))
        return self

    def _loss(self, y, F):
        if self.loss == "squared":
            return float(np.mean((y - F) ** 2))
        return float(np.mean(np.logaddexp(0.0, F) - y * F))

    def staged_raw(self, X):
        X = np.asarray(X, float)
        P = np.array([t.predict(X) for t in self.trees])
        return self.init_ + self.nu * np.cumsum(P, 0)

    def predict(self, X, m=None):
        F = self.staged_raw(X)[(m or self.M) - 1]
        return F if self.loss == "squared" else sigmoid(F)
