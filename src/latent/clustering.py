"""K-means and Gaussian mixtures fitted by EM, implemented from the update equations.

All arrays are float64. X has shape (n, d). Responsibilities R have shape (n, K).
"""
import numpy as np
from scipy.special import logsumexp

COV_REG = 1e-6  # added to covariance diagonals so no component collapses onto one point


# ---------------------------------------------------------------- k-means
def kmeans_pp_init(X, K, rng):
    """k-means++: first centre uniform, each next centre with prob. proportional to D(x)^2."""
    n = len(X)
    centres = [X[rng.integers(n)]]
    d2 = np.sum((X - centres[0]) ** 2, axis=1)
    for _ in range(1, K):
        centres.append(X[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, np.sum((X - centres[-1]) ** 2, axis=1))
    return np.array(centres)


def sq_dists(X, C):
    """(n, K) squared Euclidean distances, ||x||^2 - 2 x.c + ||c||^2, clipped at 0."""
    return np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0.0)


def kmeans(X, K, n_init=10, max_iter=300, tol=1e-10, seed=42):
    """Lloyd's algorithm with k-means++ starts; keeps the restart with the lowest distortion J.

    J = sum_n ||x_n - mu_{z_n}||^2 never increases: the assignment step minimises J over z with
    mu fixed, the update step minimises J over mu with z fixed (mu = cluster mean).
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = kmeans_pp_init(X, K, rng)
        history = []
        for _ in range(max_iter):
            z = sq_dists(X, C).argmin(1)
            history.append(float(sq_dists(X, C).min(1).sum()))
            newC = np.array([X[z == k].mean(0) if np.any(z == k) else C[k] for k in range(K)])
            shift = float(np.max(np.abs(newC - C)))
            C = newC
            if shift < tol:
                break
        D = sq_dists(X, C)
        z = D.argmin(1)
        J = float(D.min(1).sum())
        if best is None or J < best["inertia"]:
            best = {"labels": z, "centres": C, "inertia": J, "history": history,
                    "n_iter": len(history)}
    return best


# ---------------------------------------------------------------- Gaussian mixture
def mvn_logpdf(X, mu, cov):
    """log N(x | mu, cov) for every row, through a Cholesky factor (no explicit inverse)."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (sol ** 2).sum(0))


def log_joint(X, pi, mus, covs):
    """(n, K) matrix of ln pi_k + ln N(x_n | mu_k, Sigma_k) = ln p(x_n, z_n = k)."""
    return np.column_stack([np.log(pi[k]) + mvn_logpdf(X, mus[k], covs[k])
                            for k in range(len(pi))])


def gmm_loglik(X, pi, mus, covs):
    """ln p(X | theta) = sum_n ln sum_k pi_k N(x_n | mu_k, Sigma_k)  (log-sum-exp, stable)."""
    return float(logsumexp(log_joint(X, pi, mus, covs), axis=1).sum())


def e_step(X, pi, mus, covs):
    """Responsibilities r_nk = p(z_n = k | x_n, theta) and the log-likelihood."""
    lj = log_joint(X, pi, mus, covs)
    lse = logsumexp(lj, axis=1, keepdims=True)
    return np.exp(lj - lse), float(lse.sum())


def m_step(X, R, reg=COV_REG):
    """Closed-form maximisers of the expected complete-data log-likelihood Q(theta, theta_old)."""
    n, d = X.shape
    Nk = R.sum(0)
    pi = Nk / n
    mus = (R.T @ X) / Nk[:, None]
    covs = []
    for k in range(R.shape[1]):
        D = X - mus[k]
        covs.append((R[:, k, None] * D).T @ D / Nk[k] + reg * np.eye(d))
    return pi, mus, np.array(covs)


def elbo(X, R, pi, mus, covs):
    """L(q, theta) = sum_nk r_nk ln p(x_n, z_n=k | theta) - sum_nk r_nk ln r_nk.

    ln p(X | theta) = L(q, theta) + KL(q || p(Z | X, theta)), with KL >= 0, so L is a lower
    bound that becomes tight (KL = 0) right after an E-step.
    """
    lj = log_joint(X, pi, mus, covs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(R > 0, R * np.log(R), 0.0)
    return float((R * lj).sum() - ent.sum())


def gmm_em(X, K, max_iter=500, tol=1e-8, seed=42, init="kmeans", reg=COV_REG, track_bound=False):
    """EM for a full-covariance Gaussian mixture.

    Returns parameters, responsibilities and the per-iteration log-likelihood (which must be
    non-decreasing). With track_bound=True, also records for each iteration the lower bound
    L(q_old, theta_new) evaluated after the M-step (before the next E-step) and the KL gap
    ln p(X | theta_new) - L(q_old, theta_new) >= 0.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if init == "kmeans":
        z = kmeans(X, K, n_init=3, seed=seed)["labels"]
    else:
        z = rng.integers(K, size=n)
    R = np.eye(K)[z]
    pi, mus, covs = m_step(X, R, reg)
    ll_hist, bound_hist, kl_hist = [], [], []
    ll_old = -np.inf
    for _ in range(max_iter):
        R, ll = e_step(X, pi, mus, covs)          # E-step: KL -> 0, bound = ln p
        ll_hist.append(ll)
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
        pi, mus, covs = m_step(X, R, reg)          # M-step: raise the bound with q fixed
        if track_bound:
            b = elbo(X, R, pi, mus, covs)
            bound_hist.append(b)
            kl_hist.append(gmm_loglik(X, pi, mus, covs) - b)
    out = {"pi": pi, "mus": mus, "covs": covs, "R": R, "labels": R.argmax(1),
           "loglik": ll_hist[-1], "history": ll_hist, "n_iter": len(ll_hist),
           "n_params": n_params(K, d)}
    if track_bound:
        out["bound_after_m"] = bound_hist
        out["kl_after_m"] = kl_hist
    return out


def n_params(K, d):
    """Free parameters of a full-covariance GMM: (K-1) weights + K*d means + K*d(d+1)/2 covs."""
    return (K - 1) + K * d + K * d * (d + 1) // 2


def bic(loglik, k_params, n):
    return float(-2 * loglik + k_params * np.log(n))


def aic(loglik, k_params):
    return float(-2 * loglik + 2 * k_params)
