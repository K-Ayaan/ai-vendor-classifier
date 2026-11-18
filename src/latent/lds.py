"""Linear dynamical system (linear-Gaussian state-space model).

    z_t = F z_{t-1} + w_t,  w_t ~ N(0, Q)      (hidden state)
    x_t = H z_t     + v_t,  v_t ~ N(0, R)      (observation)

It is the continuous-state analogue of the HMM: the Kalman filter is the forward pass
(alpha), the RTS smoother is the backward pass (gamma), and the predictive densities of the
observations give the log-likelihood used to fit parameters by maximum likelihood.
"""
import numpy as np


def constant_velocity(dt, q):
    """2-D constant-velocity model, state [x, y, vx, vy]; q = acceleration noise intensity."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    Q = q * np.kron(G, np.eye(2))   # kron(G, I2) orders the state as [x, y, vx, vy]
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    return F, Q, H


def kalman_filter(xs, F, H, Q, R, m0, P0):
    """Forward recursion. Returns filtered means/covs, predicted means/covs and
    ln p(x_1..x_T) = sum_t ln N(x_t | H m_pred, H P_pred H^T + R)."""
    T, dz, dx = len(xs), len(m0), xs.shape[1]
    m_f, P_f = np.empty((T, dz)), np.empty((T, dz, dz))
    m_p, P_p = np.empty((T, dz)), np.empty((T, dz, dz))
    m, P, ll = m0, P0, 0.0
    for t in range(T):
        if t:
            m, P = F @ m, F @ P @ F.T + Q
        m_p[t], P_p[t] = m, P
        S = H @ P @ H.T + R
        innov = xs[t] - H @ m
        L = np.linalg.cholesky(S)
        sol = np.linalg.solve(L, innov)
        ll += -0.5 * (dx * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + sol @ sol)
        Kg = np.linalg.solve(S, H @ P).T          # Kalman gain P H^T S^{-1}
        m = m + Kg @ innov
        P = (np.eye(dz) - Kg @ H) @ P
        m_f[t], P_f[t] = m, P
    return {"m": m_f, "P": P_f, "m_pred": m_p, "P_pred": P_p, "loglik": float(ll)}


def rts_smoother(F, filt):
    """Rauch-Tung-Striebel backward recursion: p(z_t | x_1..x_T)."""
    m, P = filt["m"].copy(), filt["P"].copy()
    for t in range(len(m) - 2, -1, -1):
        Pp = filt["P_pred"][t + 1]
        J = np.linalg.solve(Pp, F @ filt["P"][t]).T   # P_t F^T Pp^{-1}
        m[t] = filt["m"][t] + J @ (m[t + 1] - filt["m_pred"][t + 1])
        P[t] = filt["P"][t] + J @ (P[t + 1] - Pp) @ J.T
    return m, P
