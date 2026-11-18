"""Bayesian model averaging (Bishop 14.1; Hoeting et al. 1999; Murphy 16.6.3).

With models M_1..M_L and prior p(M) (uniform here), the posterior predictive is
    p(y | x, D) = sum_l p(y | x, M_l, D) p(M_l | D),   p(M_l | D) propto p(D | M_l) p(M_l).
The weights come from the marginal likelihood (evidence) p(D | M). BMA assumes ONE model
generated all the data and we are unsure which; as N grows p(M | D) concentrates on one model.
That is why BMA is not a model combination like a mixture or a committee (Bishop 14.1).

Linear models: every subset of the inputs (always with an intercept). Two evidences:
  * Zellner g-prior, beta_M | sigma^2 ~ N(0, g sigma^2 (X_c^T X_c)^-1), p(alpha, sigma^2) flat.
    Closed form, relative to the intercept-only model (Liang et al. 2008):
        ln BF(M : M_0) = ((n - 1 - p_M) / 2) ln(1 + g) - ((n - 1) / 2) ln(1 + g (1 - R_M^2)).
    Posterior mean beta = g / (1 + g) * beta_OLS (shrinkage); predictive is Student-t with n - 1
    degrees of freedom and scale^2 = s_M^2 (1 + 1/n + g/(1+g) x_c^T (X_c^T X_c)^-1 x_c),
    s_M^2 = (y_c^T y_c)(1 - g/(1+g) R_M^2) / (n - 1).    g = n is the unit-information prior.
  * BIC: ln p(D | M) ~ -BIC / 2 with BIC = n ln(RSS / n) + k ln n.
Logistic models: BIC with the MLE fitted by IRLS (models chosen so the MLE exists).
"""
from itertools import combinations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t


def all_subsets(p):
    return [s for k in range(p + 1) for s in combinations(range(p), k)]


def _weights(logev):
    logev = np.asarray(logev, float)
    return np.exp(logev - logsumexp(logev))


class LinearBMA:
    def __init__(self, g=None, evidence="gprior"):
        self.g, self.evidence = g, evidence

    def fit(self, X, y, names=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        self.n, self.p = n, p
        self.names = names or [f"x{j}" for j in range(p)]
        g = float(n if self.g is None else self.g)
        self.g_ = g
        self.xm, self.ym = X.mean(0), y.mean()
        Xc, yc = X - self.xm, y - self.ym
        tss = float(yc @ yc)
        self.models = []
        for s in all_subsets(p):
            s = list(s)
            k = len(s)
            if k:
                A = Xc[:, s]
                XtX_inv = np.linalg.pinv(A.T @ A)
                b = XtX_inv @ A.T @ yc
                rss = float(np.sum((yc - A @ b) ** 2))
            else:
                XtX_inv, b, rss = np.zeros((0, 0)), np.zeros(0), tss
            r2 = 1 - rss / tss
            log_bf = 0.5 * (n - 1 - k) * np.log1p(g) - 0.5 * (n - 1) * np.log1p(g * (1 - r2))
            bic = n * np.log(rss / n) + (k + 2) * np.log(n)        # k slopes + intercept + sigma
            shrink = g / (1 + g)
            self.models.append({"subset": s, "k": k, "r2": r2, "rss": rss, "log_bf": log_bf,
                                "bic": bic, "beta_ols": b, "beta_post": shrink * b,
                                "XtX_inv": XtX_inv,
                                "s2": tss * (1 - shrink * r2) / (n - 1), "s2_ols": rss / n})
        le = ([m["log_bf"] for m in self.models] if self.evidence == "gprior"
              else [-0.5 * m["bic"] for m in self.models])
        self.post = _weights(le)
        for m, w in zip(self.models, self.post):
            m["post"] = float(w)
        self.inclusion = np.array([sum(m["post"] for m in self.models if j in m["subset"])
                                   for j in range(p)])
        self.order = np.argsort(-self.post)
        return self

    def model_predictions(self, X):
        """Per-model predictive mean and scale^2: arrays (L, n)."""
        Xc = np.asarray(X, float) - self.xm
        mus, scales = [], []
        shrink = self.g_ / (1 + self.g_)
        for m in self.models:
            s = m["subset"]
            if self.evidence == "gprior":
                mu = self.ym + (Xc[:, s] @ m["beta_post"] if s else 0.0)
                lev = (np.einsum("ij,jk,ik->i", Xc[:, s], m["XtX_inv"], Xc[:, s]) if s
                       else np.zeros(len(Xc)))
                sc2 = m["s2"] * (1 + 1 / self.n + shrink * lev)
            else:
                mu = self.ym + (Xc[:, s] @ m["beta_ols"] if s else 0.0)
                sc2 = np.full(len(Xc), m["s2_ols"])
            mus.append(np.broadcast_to(mu, (len(Xc),)))
            scales.append(sc2)
        return np.array(mus), np.array(scales)

    def predict(self, X, top=None):
        """BMA mean. top=k averages only the k most probable models (weights renormalised)."""
        mus, _ = self.model_predictions(X)
        w = self.post.copy()
        if top is not None:
            mask = np.zeros_like(w); mask[self.order[:top]] = 1; w = w * mask / (w * mask).sum()
        return w @ mus

    def predictive_interval(self, X, level=0.95, grid=None):
        """Central interval of the BMA mixture of Student-t predictives, by bisection."""
        mus, sc2 = self.model_predictions(X)
        sd = np.sqrt(sc2)
        dof = self.n - 1
        w = self.post[:, None]

        def cdf(v):
            return np.sum(w * student_t.cdf((v - mus) / sd, dof), 0)

        out = []
        for q in ((1 - level) / 2, (1 + level) / 2):
            lo, hi = (mus - 10 * sd).min(0), (mus + 10 * sd).max(0)
            for _ in range(60):
                mid = (lo + hi) / 2
                below = cdf(mid) < q
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            out.append((lo + hi) / 2)
        return out[0], out[1]


class LogisticBMA:
    """BMA over logistic models on every subset of the given columns, BIC evidence."""

    def __init__(self, irls):
        self.irls = irls     # function (X, y) -> beta (with an intercept column first)

    def fit(self, X, y, names=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        self.names = names or [f"x{j}" for j in range(p)]
        ones = np.ones((n, 1))
        self.models = []
        for s in all_subsets(p):
            s = list(s)
            A = np.hstack([ones, X[:, s]])
            beta = self.irls(A, y)
            z = A @ beta
            ll = float(np.sum(y * z - np.logaddexp(0, z)))
            self.models.append({"subset": s, "k": len(s), "beta": beta, "loglik": ll,
                                "bic": -2 * ll + (len(s) + 1) * np.log(n)})
        self.post = _weights([-0.5 * m["bic"] for m in self.models])
        for m, w in zip(self.models, self.post):
            m["post"] = float(w)
        self.inclusion = np.array([sum(m["post"] for m in self.models if j in m["subset"])
                                   for j in range(p)])
        self.order = np.argsort(-self.post)
        return self

    def model_probs(self, X):
        X = np.asarray(X, float)
        ones = np.ones((len(X), 1))
        P = []
        for m in self.models:
            z = np.hstack([ones, X[:, m["subset"]]]) @ m["beta"]
            P.append(np.exp(-np.logaddexp(0, -z)))
        return np.array(P)

    def predict_proba(self, X):
        return self.post @ self.model_probs(X)
