"""PCA from the covariance eigendecomposition, and the probabilistic-PCA likelihood used to
choose the number of latent dimensions."""
import numpy as np


def pca_eig(X):
    """Centre X, eigendecompose S = X_c^T X_c / n, sort eigenvalues descending.

    Returns mean, eigenvalues (variances along each principal axis) and eigenvectors (columns).
    The sign of each eigenvector is fixed so its largest-|.| entry is positive (determinism).
    """
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / len(X)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    signs = np.sign(vecs[np.abs(vecs).argmax(0), np.arange(vecs.shape[1])])
    return mu, vals, vecs * signs


def ppca_params(vals, vecs, q):
    """Tipping & Bishop closed-form ML solution for PPCA with q latent dimensions.

    sigma^2 = mean of the discarded eigenvalues; W = U_q (Lambda_q - sigma^2 I)^{1/2}.
    """
    d = len(vals)
    sigma2 = float(vals[q:].mean()) if q < d else 1e-12
    W = vecs[:, :q] * np.sqrt(np.maximum(vals[:q] - sigma2, 0.0))
    return W, max(sigma2, 1e-12)


def ppca_loglik(X, mu, W, sigma2):
    """Average log-likelihood of rows of X under x ~ N(mu, C), C = W W^T + sigma^2 I."""
    d = X.shape[1]
    C = W @ W.T + sigma2 * np.eye(d)
    L = np.linalg.cholesky(C)
    sol = np.linalg.solve(L, (X - mu).T)
    ll = -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (sol ** 2).sum(0))
    return float(ll.mean())
