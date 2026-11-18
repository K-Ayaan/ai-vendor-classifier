"""Checks of the PBL-2 implementations against brute force and scikit-learn."""
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from latent import clustering as cl
from latent import hmm, lds
from latent.pca import pca_eig, ppca_loglik, ppca_params


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [6, 0], [0, 6]])
    return np.vstack([rng.normal(c, [1.0, 0.6], size=(150, 2)) for c in centres])


@pytest.fixture
def small_hmm():
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
    B = np.array([[0.5, 0.4, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
    return pi, A, B


def test_kmeans_matches_sklearn_inertia(blobs):
    ours = cl.kmeans(blobs, 3, n_init=10, seed=1)
    sk = KMeans(3, n_init=10, random_state=1).fit(blobs)
    assert ours["inertia"] == pytest.approx(sk.inertia_, rel=1e-6)
    assert all(np.diff(ours["history"]) <= 1e-9)


def test_gmm_em_monotone_and_matches_sklearn(blobs):
    ours = cl.gmm_em(blobs, 3, track_bound=True)
    assert all(np.diff(ours["history"]) >= -1e-8)
    assert min(ours["kl_after_m"]) >= -1e-8
    sk = GaussianMixture(3, covariance_type="full", reg_covar=cl.COV_REG, tol=1e-10,
                         max_iter=1000, n_init=3, random_state=0).fit(blobs)
    assert ours["loglik"] == pytest.approx(sk.score(blobs) * len(blobs), rel=1e-5)


def test_elbo_tight_after_e_step(blobs):
    g = cl.gmm_em(blobs, 3)
    R, ll = cl.e_step(blobs, g["pi"], g["mus"], g["covs"])
    assert cl.elbo(blobs, R, g["pi"], g["mus"], g["covs"]) == pytest.approx(ll, rel=1e-10)


def test_pca_eig_matches_sklearn(blobs):
    _, vals, vecs = pca_eig(blobs)
    sk = PCA().fit(blobs)
    n = len(blobs)
    assert np.allclose(vals, sk.explained_variance_ * (n - 1) / n)
    assert np.allclose(np.abs(vecs.T), np.abs(sk.components_))


def test_ppca_loglik_matches_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6))
    mu, vals, vecs = pca_eig(X)
    # sklearn's PCA uses the (n-1) covariance; rescale our eigenvalues to match it
    n = len(X)
    W, s2 = ppca_params(vals * n / (n - 1), vecs, 2)
    assert ppca_loglik(X, mu, W, s2) == pytest.approx(PCA(2).fit(X).score(X), rel=1e-8)


def test_forward_backward_and_viterbi_match_brute_force(small_hmm):
    pi, A, B = small_hmm
    obs = np.array([0, 1, 3, 2, 2, 0, 1])
    px, marg, best = hmm.brute_force(obs, pi, A, B)
    gamma, xi_sum, ll = hmm.posteriors(obs, pi, A, B)
    assert ll == pytest.approx(np.log(px), rel=1e-12)
    assert np.allclose(gamma, marg)
    assert np.allclose(gamma.sum(1), 1)
    assert xi_sum.sum() == pytest.approx(len(obs) - 1)
    path, _ = hmm.viterbi(obs, pi, A, B)
    assert np.array_equal(path, best)


def test_scaling_avoids_underflow(small_hmm):
    pi, A, B = small_hmm
    obs = np.random.default_rng(0).choice([0, 1, 2], size=3000)
    assert hmm.forward_unscaled(obs, pi, A, B) == 0.0
    assert np.isfinite(hmm.loglik(obs, pi, A, B))


def test_baum_welch_never_decreases(small_hmm):
    pi, A, B = small_hmm
    rng = np.random.default_rng(1)
    seqs = [rng.choice(4, size=rng.integers(4, 9)) for _ in range(60)]
    *_, hist = hmm.baum_welch(seqs, *hmm.random_params(3, 4, rng), pseudo=0.0, max_iter=50)
    assert all(np.diff(hist) >= -1e-8)


def test_supervised_mle_counts():
    pi, A, B = hmm.supervised_mle([np.array([0, 1, 1])], [np.array([0, 1, 1])], 2, 2)
    assert np.allclose(pi, [1, 0])
    assert np.allclose(A, [[0, 1], [0, 1]])
    assert np.allclose(B, [[1, 0], [0, 1]])


def test_kalman_loglik_equals_joint_gaussian():
    """For a short series, the filter's innovations log-likelihood must equal the log-density of
    the stacked observations under their exact joint Gaussian."""
    F, Q, H = lds.constant_velocity(1.0, 0.05)
    R = 0.3 * np.eye(2)
    m0, P0 = np.zeros(4), np.eye(4)
    rng = np.random.default_rng(2)
    xs = rng.normal(size=(5, 2))
    kf = lds.kalman_filter(xs, F, H, Q, R, m0, P0)
    T = len(xs)
    # build the joint covariance of x_1..x_T
    Phi = [np.linalg.matrix_power(F, t) for t in range(T)]
    Cz = {}
    for s in range(T):
        for t in range(T):
            P = Phi[s] @ P0 @ Phi[t].T
            for u in range(1, min(s, t) + 1):
                P = P + Phi[s - u] @ Q @ Phi[t - u].T
            Cz[s, t] = P
    C = np.block([[H @ Cz[s, t] @ H.T + (R if s == t else 0) for t in range(T)]
                  for s in range(T)])
    from scipy.stats import multivariate_normal
    assert kf["loglik"] == pytest.approx(
        multivariate_normal(np.zeros(2 * T), C).logpdf(xs.ravel()), rel=1e-9)
    ms, _ = lds.rts_smoother(F, kf)
    assert np.allclose(ms[-1], kf["m"][-1])
