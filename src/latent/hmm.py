"""Discrete-emission hidden Markov model, implemented from the recursions.

Parameters: pi (K,) initial-state probabilities, A (K, K) with A[i, j] = p(z_t = j | z_{t-1} = i),
B (K, M) with B[k, m] = p(x_t = m | z_t = k). A sequence is a 1-D int array of symbols in 0..M-1.

Scaling (Bishop 13.2.4): alpha_hat_t = alpha_t / p(x_1..x_t) and c_t = p(x_t | x_1..x_{t-1}),
so alpha_hat stays O(1) and ln p(X) = sum_t ln c_t, with no underflow for long sequences.
"""
import itertools

import numpy as np


# ---------------------------------------------------------------- inference
def forward_unscaled(obs, pi, A, B):
    """Plain alpha recursion. Underflows to 0 for long sequences; kept only to show why
    scaling is needed."""
    alpha = pi * B[:, obs[0]]
    for x in obs[1:]:
        alpha = (alpha @ A) * B[:, x]
    return float(alpha.sum())


def forward(obs, pi, A, B):
    """Scaled forward pass. Returns alpha_hat (T, K) and scaling factors c (T,)."""
    T, K = len(obs), len(pi)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * B[:, obs[0]]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def backward(obs, A, B, c):
    """Scaled backward pass using the forward scaling factors. Returns beta_hat (T, K)."""
    T, K = len(obs), A.shape[0]
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
    return beta


def loglik(obs, pi, A, B):
    return float(np.log(forward(obs, pi, A, B)[1]).sum())


def posteriors(obs, pi, A, B):
    """Sum-product on the chain. gamma[t, k] = p(z_t = k | X);
    xi_sum[i, j] = sum_t p(z_{t-1} = i, z_t = j | X); and ln p(X)."""
    alpha, c = forward(obs, pi, A, B)
    beta = backward(obs, A, B, c)
    gamma = alpha * beta
    xi_sum = np.zeros_like(A)
    for t in range(1, len(obs)):
        xi_sum += (alpha[t - 1][:, None] * A * (B[:, obs[t]] * beta[t])[None, :]) / c[t]
    return gamma, xi_sum, float(np.log(c).sum())


def viterbi(obs, pi, A, B):
    """Max-product (in logs): most probable state path and its joint log-probability."""
    with np.errstate(divide="ignore"):
        lpi, lA, lB = np.log(pi), np.log(A), np.log(B)
    T, K = len(obs), len(pi)
    omega = lpi + lB[:, obs[0]]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        scores = omega[:, None] + lA
        back[t] = scores.argmax(0)
        omega = scores.max(0) + lB[:, obs[t]]
    path = np.empty(T, dtype=int)
    path[-1] = int(omega.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(omega.max())


def brute_force(obs, pi, A, B):
    """Enumerate all K^T paths. Returns p(X), the marginals p(z_t | X) and the best path.
    Exponential cost: only for checking forward-backward and Viterbi on short sequences."""
    T, K = len(obs), len(pi)
    px, marg, best, best_p = 0.0, np.zeros((T, K)), None, -1.0
    for path in itertools.product(range(K), repeat=T):
        p = pi[path[0]] * B[path[0], obs[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
        px += p
        marg[np.arange(T), path] += p
        if p > best_p:
            best, best_p = np.array(path), p
    return px, marg / px, best


# ---------------------------------------------------------------- learning
def supervised_mle(seqs, labels, K, M, pseudo=0.0):
    """Maximum likelihood when the states are observed: normalised counts
    (plus an optional add-`pseudo` smoothing, i.e. a Dirichlet MAP estimate)."""
    pi = np.full(K, pseudo)
    A = np.full((K, K), pseudo)
    B = np.full((K, M), pseudo)
    for x, z in zip(seqs, labels):
        pi[z[0]] += 1
        for t in range(len(x)):
            B[z[t], x[t]] += 1
            if t:
                A[z[t - 1], z[t]] += 1
    return normalise(pi), normalise(A), normalise(B)


def normalise(P):
    P = np.asarray(P, dtype=float)
    s = P.sum(-1, keepdims=True)
    return np.divide(P, s, out=np.full_like(P, 1.0 / P.shape[-1]), where=s > 0)


def baum_welch(seqs, pi, A, B, max_iter=200, tol=1e-7, pseudo=1e-3):
    """EM for the HMM over many independent sequences. E-step: gamma, xi from the scaled
    forward-backward; M-step: expected counts normalised (with a tiny pseudo-count so no
    probability becomes exactly 0). Returns parameters and the total log-likelihood history,
    which is non-decreasing up to the pseudo-count."""
    pi, A, B = pi.copy(), A.copy(), B.copy()
    K, M = B.shape
    hist = []
    for _ in range(max_iter):
        pi_n, A_n, B_n = np.full(K, pseudo), np.full((K, K), pseudo), np.full((K, M), pseudo)
        total = 0.0
        for x in seqs:
            gamma, xi_sum, ll = posteriors(x, pi, A, B)
            total += ll
            pi_n += gamma[0]
            A_n += xi_sum
            np.add.at(B_n.T, x, gamma)
        hist.append(total)
        pi, A, B = normalise(pi_n), normalise(A_n), normalise(B_n)
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-1]):
            break
    return pi, A, B, hist


def random_params(K, M, rng):
    return (normalise(rng.random(K) + 0.5), normalise(rng.random((K, K)) + 0.5),
            normalise(rng.random((K, M)) + 0.5))
