"""Generalised additive model (Hastie & Tibshirani 1990; Murphy 16.3.1).

    y = alpha + sum_j f_j(x_j) + eps,   each f_j centred: sum_n f_j(x_nj) = 0.

Smooth terms are P-splines (Eilers & Marx 1996): f_j(x) = B_j(x) beta_j with a cubic B-spline
basis on equally spaced knots and a second-difference roughness penalty lam * ||D2 beta_j||^2.
Binary inputs enter as linear terms. A "by" term f(x) * z lets the curve differ when z = 1
(a varying-coefficient term: the one interaction an additive model can be given explicitly).

Fitting the penalised least-squares problem
    min ||y - alpha - sum_j B_j beta_j||^2 + sum_j lam_j beta_j^T D_j^T D_j beta_j
two ways:
  * direct: one ridge-type solve on the stacked design (gives the hat matrix, so the effective
    degrees of freedom edf = tr(H) and GCV = n RSS / (n - edf)^2);
  * backfitting: cycle f_j <- S_j (y - alpha - sum_{k != j} f_k) with the term's own penalised
    smoother S_j. This is block Gauss-Seidel on the same normal equations, so it converges to
    the direct solution (checked in tests/test_ensemble.py).
"""
import numpy as np
from scipy.interpolate import BSpline

RIDGE = 1e-8   # tiny ridge on every block: makes the centred, penalised blocks strictly convex


class Term:
    def __init__(self, col, kind="spline", n_segments=12, degree=3, by=None, name=None):
        assert kind in ("spline", "linear")
        self.col, self.kind, self.nseg, self.deg, self.by = col, kind, n_segments, degree, by
        self.name = name or (f"s(x{col})" if kind == "spline" else f"x{col}")

    def setup(self, X):
        x = X[:, self.col]
        if self.kind == "spline":
            self.lo, self.hi = float(x.min()), float(x.max())
            h = (self.hi - self.lo) / self.nseg
            self.knots = self.lo + h * np.arange(-self.deg, self.nseg + self.deg + 1)
            k = self.nseg + self.deg
            D = np.diff(np.eye(k), 2, axis=0)
            self.P = D.T @ D
        else:
            self.P = np.zeros((1, 1))
        self.mean = self.raw(X).mean(0)
        return self

    def raw(self, X):
        x = X[:, self.col].astype(float)
        if self.kind == "spline":
            # constant extrapolation outside train; clip to the knot range itself, because
            # lo + h * nseg can round to just below hi
            x = np.clip(x, self.knots[self.deg], self.knots[-self.deg - 1])
            B = BSpline.design_matrix(x, self.knots, self.deg, extrapolate=False).toarray()
        else:
            B = x[:, None]
        if self.by is not None:
            B = B * X[:, [self.by]]
        return B

    def basis(self, X):
        return self.raw(X) - self.mean

    @property
    def size(self):
        return self.P.shape[0]


class GAM:
    def __init__(self, terms, lam=1.0):
        self.terms, self.lam = terms, lam

    def _blocks(self, X):
        return [t.basis(X) for t in self.terms]

    def _penalty(self):
        sizes = [t.size for t in self.terms]
        P = np.zeros((sum(sizes), sum(sizes)))
        o = 0
        for t, s in zip(self.terms, sizes):
            lam = self.lam if t.kind == "spline" else 0.0
            P[o:o + s, o:o + s] = lam * t.P + RIDGE * np.eye(s)
            o += s
        return P

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        for t in self.terms:
            t.setup(X)
        self.alpha = float(y.mean())
        B = np.hstack(self._blocks(X))
        A = B.T @ B + self._penalty()
        self.coef = np.linalg.solve(A, B.T @ (y - self.alpha))
        H_diag_blocks = np.linalg.solve(A, B.T)                 # (p, n): coef = this @ y_c
        self.edf = 1.0 + float(np.trace(B @ H_diag_blocks))     # +1 for the intercept
        o, self.term_edf = 0, []
        for t in self.terms:
            s = t.size
            self.term_edf.append(float(np.trace(B[:, o:o + s] @ H_diag_blocks[o:o + s])))
            o += s
        yhat = self.predict(X)
        n = len(y)
        self.rss = float(np.sum((y - yhat) ** 2))
        self.gcv = n * self.rss / (n - self.edf) ** 2
        return self

    def split_coef(self):
        out, o = [], 0
        for t in self.terms:
            out.append(self.coef[o:o + t.size]); o += t.size
        return out

    def predict(self, X):
        return self.alpha + np.hstack(self._blocks(np.asarray(X, float))) @ self.coef

    def partial(self, X, j):
        """The fitted centred function f_j at the rows of X."""
        return self.terms[j].basis(np.asarray(X, float)) @ self.split_coef()[j]

    def backfit(self, X, y, tol=1e-10, max_iter=500):
        """Backfitting with each term's own penalised smoother. Returns per-sweep change."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        for t in self.terms:
            t.setup(X)
        self.alpha = float(y.mean())
        Bs = self._blocks(X)
        smooth = []
        for t, B in zip(self.terms, Bs):
            lam = self.lam if t.kind == "spline" else 0.0
            smooth.append(np.linalg.solve(B.T @ B + lam * t.P + RIDGE * np.eye(t.size), B.T))
        coefs = [np.zeros(t.size) for t in self.terms]
        f = [np.zeros(len(y)) for _ in self.terms]
        history = []
        for _ in range(max_iter):
            change = 0.0
            for j, B in enumerate(Bs):
                partial = y - self.alpha - (sum(f) - f[j])
                coefs[j] = smooth[j] @ partial
                new = B @ coefs[j]
                change = max(change, float(np.max(np.abs(new - f[j]))))
                f[j] = new
            history.append(change)
            if change < tol:
                break
        self.coef = np.concatenate(coefs)
        return history
