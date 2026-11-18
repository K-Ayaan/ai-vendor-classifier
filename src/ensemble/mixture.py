"""Conditional mixture models (Bishop 14.5): combining models where each data point is explained
by ONE component, but we do not know which. Unlike BMA, different points can use different
components, so this IS a model combination.

Mixture of linear regressions (Bishop 14.5.1):
    p(t | phi) = sum_k pi_k N(t | w_k^T phi, beta_k^-1)
EM:  E-step  gamma_nk = pi_k N(t_n | w_k^T phi_n, beta_k^-1) / sum_j (...)
     M-step  pi_k = mean_n gamma_nk;  w_k = (Phi^T R_k Phi)^-1 Phi^T R_k t   (weighted LS);
             1/beta_k = sum_n gamma_nk (t_n - w_k^T phi_n)^2 / sum_n gamma_nk.
A variance floor stops a component collapsing onto points with identical t (here, the many
pairs scored exactly 0), where the likelihood would go to infinity.

Mixture of experts (Bishop 14.5.3): the mixing weights depend on the input, pi_k(x). With K = 2
and a logistic gate pi_1(x) = sigma(v^T x), the M-step for the gate is a logistic regression
with soft targets gamma_n1, fitted by Newton/IRLS (still concave for fractional targets).
"""
import numpy as np
from scipy.special import logsumexp

VAR_FLOOR = 1.0   # score units: scores are integers, so a residual s.d. < 1 point is not real


def _norm_logpdf(t, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (t - mu) ** 2 / var)


def _wls(Phi, t, r, ridge=1e-8):
    A = (Phi * r[:, None]).T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(A, (Phi * r[:, None]).T @ t)


def mix_linreg(Phi, t, K, n_init=10, seed=0, max_iter=500, tol=1e-8, var_floor=VAR_FLOOR):
    """EM for a K-component mixture of linear regressions; best of n_init random starts."""
    Phi, t = np.asarray(Phi, float), np.asarray(t, float)
    n, d = Phi.shape
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        R = rng.dirichlet(np.ones(K), size=n)        # random soft assignment to start
        hist = []
        for _ in range(max_iter):
            Nk = R.sum(0) + 1e-12
            pi = Nk / n
            W = np.array([_wls(Phi, t, R[:, k]) for k in range(K)])
            mu = Phi @ W.T
            var = np.maximum((R * (t[:, None] - mu) ** 2).sum(0) / Nk, var_floor)
            logp = np.log(pi) + _norm_logpdf(t[:, None], mu, var)
            ll = float(logsumexp(logp, 1).sum())
            R = np.exp(logp - logsumexp(logp, 1, keepdims=True))
            hist.append(ll)
            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
                break
        if best is None or ll > best["loglik"]:
            best = {"pi": pi, "W": W, "var": var, "loglik": ll, "history": hist, "resp": R}
    best["n_params"] = K * d + K + (K - 1)
    best["bic"] = -2 * best["loglik"] + best["n_params"] * np.log(n)
    return best


def mix_linreg_predict(model, Phi):
    """Mean of the mixture (pi_k do not depend on x, so it is one blended line)."""
    return np.asarray(Phi, float) @ model["W"].T @ model["pi"]


def _gate_newton(G, r, v, l2=1e-2, iters=25):
    """Logistic regression with soft targets r in [0, 1]; small L2 (not on the intercept)."""
    P = np.eye(G.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        p = np.exp(-np.logaddexp(0, -G @ v))
        grad = G.T @ (r - p) - l2 * P @ v
        H = (G * (p * (1 - p))[:, None]).T @ G + l2 * P
        step = np.linalg.solve(H + 1e-10 * np.eye(len(v)), grad)
        v = v + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return v


def mixture_of_experts(Phi, G, t, n_init=10, seed=0, max_iter=300, tol=1e-8,
                       var_floor=VAR_FLOOR):
    """Two linear experts, logistic gate on inputs G (first column = 1)."""
    Phi, G, t = np.asarray(Phi, float), np.asarray(G, float), np.asarray(t, float)
    n = len(t)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        r1 = rng.random(n)
        R = np.column_stack([r1, 1 - r1])
        v = np.zeros(G.shape[1])
        hist = []
        for _ in range(max_iter):
            W = np.array([_wls(Phi, t, R[:, k]) for k in range(2)])
            mu = Phi @ W.T
            var = np.maximum((R * (t[:, None] - mu) ** 2).sum(0) / (R.sum(0) + 1e-12), var_floor)
            v = _gate_newton(G, R[:, 0], v)
            g1 = np.exp(-np.logaddexp(0, -G @ v))
            logpi = np.log(np.column_stack([g1, 1 - g1]) + 1e-300)
            logp = logpi + _norm_logpdf(t[:, None], mu, var)
            ll = float(logsumexp(logp, 1).sum())
            R = np.exp(logp - logsumexp(logp, 1, keepdims=True))
            hist.append(ll)
            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
                break
        if best is None or ll > best["loglik"]:
            best = {"W": W, "var": var, "v": v, "loglik": ll, "history": hist, "resp": R}
    best["n_params"] = 2 * Phi.shape[1] + 2 + G.shape[1]
    best["bic"] = -2 * best["loglik"] + best["n_params"] * np.log(n)
    return best


def moe_predict(model, Phi, G):
    g1 = np.exp(-np.logaddexp(0, -np.asarray(G, float) @ model["v"]))
    mu = np.asarray(Phi, float) @ model["W"].T
    return g1 * mu[:, 0] + (1 - g1) * mu[:, 1], g1
