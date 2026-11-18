"""CART: classification and regression trees (Breiman et al. 1984; Bishop 14.4; Murphy 16.2).

Greedy recursive binary splitting. At each node every candidate split "x_j <= s" is scored by the
drop in total impurity

    regression      I(node) = sum_i w_i (y_i - ybar)^2                (weighted squared error)
    classification  I(node) = W - sum_k W_k^2 / W   = W * Gini        (W_k = class weight)

and the best one is taken. Two search paths give identical answers:
  * sort path, any column: sort once, cumulative sums give the left/right statistics at every
    threshold (thresholds are midpoints between consecutive distinct values, as in scikit-learn);
  * binary path, 0/1 columns: left = {x = 0}, right = {x = 1}, so all statistics are matrix
    products X^T w, X^T (w y), ...  This is what makes trees on 441 word indicators fast.

Cost-complexity pruning (weakest link): with R(t) = I(t) / W_total, the subtree cost is
R_alpha(T) = sum_leaves R(t) + alpha |leaves|. Collapsing node t is worth it once
alpha >= g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1); repeatedly collapsing the node with the
smallest g(t) gives the nested sequence of optimal subtrees and their alphas.
"""
import numpy as np

EPS_THRESHOLD = 1e-7  # two values closer than this are treated as equal (as scikit-learn)


def _totals_reg(w, wy, wyy):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w > 0, wyy - wy ** 2 / w, 0.0)


def _totals_gini(W, Wk):
    """W: (...,), Wk: (..., K). Total Gini impurity W * (1 - sum p_k^2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(W > 0, W - np.sum(Wk ** 2, axis=-1) / W, 0.0)


class CART:
    """task: "regression" or "classification". max_features: None (all), int, or float fraction;
    a fresh random subset is drawn at every node (random forests)."""

    def __init__(self, task="regression", max_depth=None, min_samples_leaf=1,
                 min_samples_split=2, max_features=None, seed=0):
        assert task in ("regression", "classification")
        self.task, self.max_depth = task, max_depth if max_depth is not None else 10 ** 6
        self.min_leaf, self.min_split = int(min_samples_leaf), int(min_samples_split)
        self.max_features, self.seed = max_features, seed

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        n, p = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        keep = w > 0
        X, y, w = X[keep], np.asarray(y)[keep], w[keep]
        self.n_features_ = p
        self.binary_ = np.all((X == 0) | (X == 1), axis=0)
        if self.task == "classification":
            self.classes_, yi = np.unique(y, return_inverse=True)
            self._Y = np.eye(len(self.classes_))[yi] * w[:, None]       # class weights per row
        else:
            y = y.astype(float)
            self._wy, self._wyy = w * y, w * y * y
        self._w, self._X = w, X
        m = p if self.max_features is None else (
            int(self.max_features) if isinstance(self.max_features, (int, np.integer))
            else max(1, int(round(self.max_features * p))))
        self._m, self._rng = min(m, p), np.random.default_rng(self.seed)

        feat, thr, left, right, value, nn, wn, imp, depth = ([] for _ in range(9))
        stack = [(np.arange(len(w)), 0, -1, False)]
        while stack:
            idx, d, parent, is_right = stack.pop()
            node = len(feat)
            if parent >= 0:
                (right if is_right else left)[parent] = node
            W = w[idx].sum()
            if self.task == "classification":
                Wk = self._Y[idx].sum(0)
                total, val = float(_totals_gini(W, Wk)), Wk / W
            else:
                sy = self._wy[idx].sum()
                total, val = float(_totals_reg(W, sy, self._wyy[idx].sum())), np.array([sy / W])
            feat.append(-1); thr.append(np.nan); left.append(-1); right.append(-1)
            value.append(val); nn.append(len(idx)); wn.append(W); imp.append(total / W)
            depth.append(d)
            if (d >= self.max_depth or len(idx) < max(self.min_split, 2 * self.min_leaf)
                    or total <= 1e-12 * max(W, 1.0)):
                continue
            best = self._best_split(idx, total)
            if best is None:
                continue
            j, s = best
            feat[node], thr[node] = j, s
            go_left = X[idx, j] <= s
            stack.append((idx[~go_left], d + 1, node, True))
            stack.append((idx[go_left], d + 1, node, False))   # left popped first: preorder
        self.feature_, self.threshold_ = np.array(feat), np.array(thr)
        self.left_, self.right_ = np.array(left), np.array(right)
        self.value_, self.n_node_samples_ = np.array(value), np.array(nn)
        self.weighted_n_, self.impurity_, self.depth_ = np.array(wn), np.array(imp), np.array(depth)
        self.collapsed_ = np.zeros(len(feat), bool)
        del self._X, self._w
        self._Y = self._wy = self._wyy = None
        return self

    def _best_split(self, idx, parent_total):
        X, w = self._X, self._w
        p = X.shape[1]
        cand = (np.arange(p) if self._m >= p
                else np.sort(self._rng.choice(p, self._m, replace=False)))
        best_gain, best = 0.0, None
        bmask = self.binary_[cand]
        n = len(idx)
        wi = w[idx]
        if self.task == "classification":
            Yi = self._Y[idx]
            Wk_tot = Yi.sum(0)
        else:
            sy_tot, sq_tot = self._wy[idx].sum(), self._wyy[idx].sum()
        W_tot = wi.sum()

        # binary path: right child = {x = 1}
        jb = cand[bmask]
        if len(jb):
            Xb = X[np.ix_(idx, jb)]
            n1 = Xb.sum(0)
            W1 = Xb.T @ wi
            if self.task == "classification":
                A1 = Xb.T @ Yi
                child = _totals_gini(W1, A1) + _totals_gini(W_tot - W1, Wk_tot - A1)
            else:
                s1, q1 = Xb.T @ self._wy[idx], Xb.T @ self._wyy[idx]
                child = _totals_reg(W1, s1, q1) + _totals_reg(W_tot - W1, sy_tot - s1, sq_tot - q1)
            gain = parent_total - child
            ok = (n1 >= self.min_leaf) & (n - n1 >= self.min_leaf)
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best = gain[k], (int(jb[k]), 0.5)

        # sort path
        js = cand[~bmask]
        if len(js) and n >= 2:
            Xs_raw = X[np.ix_(idx, js)]
            order = np.argsort(Xs_raw, axis=0, kind="stable")
            Xs = np.take_along_axis(Xs_raw, order, 0)
            cw = np.cumsum(wi[order], 0)[:-1]
            if self.task == "classification":
                cA = np.cumsum(Yi[order], 0)[:-1]                       # (n-1, q, K)
                child = (_totals_gini(cw, cA) + _totals_gini(W_tot - cw, Wk_tot - cA))
            else:
                cs = np.cumsum(self._wy[idx][order], 0)[:-1]
                cq = np.cumsum(self._wyy[idx][order], 0)[:-1]
                child = _totals_reg(cw, cs, cq) + _totals_reg(W_tot - cw, sy_tot - cs, sq_tot - cq)
            nl = np.arange(1, n)[:, None]
            ok = ((Xs[1:] > Xs[:-1] + EPS_THRESHOLD) & (nl >= self.min_leaf)
                  & (n - nl >= self.min_leaf))
            gain = np.where(ok, parent_total - child, -np.inf)
            flat = int(np.argmax(gain))
            i, c = np.unravel_index(flat, gain.shape)
            if gain[i, c] > best_gain:
                lo, hi = Xs[i, c], Xs[i + 1, c]
                s = lo / 2.0 + hi / 2.0
                if s >= hi:
                    s = lo
                best_gain, best = gain[i, c], (int(js[c]), float(s))
        return best

    # ------------------------------------------------------------------ predict
    def apply(self, X, collapsed=None):
        X = np.asarray(X, float)
        col = self.collapsed_ if collapsed is None else collapsed
        node = np.zeros(len(X), int)
        rows = np.arange(len(X))
        while True:
            f = self.feature_[node]
            active = (f >= 0) & ~col[node]
            if not active.any():
                return node
            a = rows[active]
            go_left = X[a, f[active]] <= self.threshold_[node[active]]
            node[a] = np.where(go_left, self.left_[node[active]], self.right_[node[active]])

    def predict(self, X, alpha=None):
        v = self.value_[self.apply(X, self._mask_for(alpha))]
        if self.task == "regression":
            return v[:, 0]
        return self.classes_[np.argmax(v, 1)]

    def predict_proba(self, X, alpha=None):
        assert self.task == "classification"
        return self.value_[self.apply(X, self._mask_for(alpha))]

    # ------------------------------------------------------------------ structure
    @property
    def n_nodes(self):
        return len(self.feature_)

    def n_leaves(self, collapsed=None):
        col = self.collapsed_ if collapsed is None else collapsed
        return int(np.sum(self._active(col) & ((self.feature_ < 0) | col)))

    def _active(self, col):
        act = np.zeros(self.n_nodes, bool)
        act[0] = True
        for t in range(self.n_nodes):          # parents come before children (preorder ids)
            if act[t] and self.feature_[t] >= 0 and not col[t]:
                act[self.left_[t]] = act[self.right_[t]] = True
        return act

    def feature_importances(self):
        """Mean decrease in impurity, normalised to sum to 1 (as scikit-learn)."""
        imp = np.zeros(self.n_features_)
        tot = self.weighted_n_ * self.impurity_
        for t in np.where(self.feature_ >= 0)[0]:
            imp[self.feature_[t]] += tot[t] - tot[self.left_[t]] - tot[self.right_[t]]
        return imp / imp.sum() if imp.sum() > 0 else imp

    # ------------------------------------------------------------------ pruning
    def pruning_path(self):
        """Weakest-link pruning. Returns alphas (increasing, starting at 0), total leaf
        impurity R(T_alpha) and the collapsed-node mask of every subtree in the sequence."""
        R = self.weighted_n_ * self.impurity_ / self.weighted_n_[0]
        internal = self.feature_ >= 0
        col = np.zeros(self.n_nodes, bool)
        alphas, impurities, masks = [0.0], [], [col.copy()]
        while True:
            RT, L = R.copy(), np.ones(self.n_nodes)
            for t in range(self.n_nodes - 1, -1, -1):    # children have larger ids
                if internal[t] and not col[t]:
                    RT[t] = RT[self.left_[t]] + RT[self.right_[t]]
                    L[t] = L[self.left_[t]] + L[self.right_[t]]
            if len(impurities) < len(alphas):
                impurities.append(float(RT[0]))
            cand = internal & ~col & self._active(col)
            if not cand.any():
                break
            g = np.where(cand, (R - RT) / np.maximum(L - 1, 1), np.inf)
            t = int(np.argmin(g))
            col[t] = True
            alphas.append(float(max(g[t], alphas[-1])))
            masks.append(col.copy())
        self._path = (np.array(alphas), np.array(impurities), masks)
        return self._path

    def _mask_for(self, alpha):
        if alpha is None:
            return self.collapsed_
        if not hasattr(self, "_path"):
            self.pruning_path()
        alphas, _, masks = self._path
        return masks[int(np.searchsorted(alphas, alpha + 1e-15, side="right") - 1)]

    def prune(self, alpha):
        self.collapsed_ = self._mask_for(alpha).copy()
        return self

    def describe(self, names, max_depth=3, collapsed=None, fmt="{:.2f}"):
        """Readable if/else text of the top of the tree."""
        col = self.collapsed_ if collapsed is None else collapsed
        lines = []

        def rec(t, d):
            pad = "    " * d
            leaf = self.feature_[t] < 0 or col[t]
            v = self.value_[t]
            val = fmt.format(v[0]) if self.task == "regression" else f"P(1)={v[-1]:.3f}"
            if leaf or d >= max_depth:
                lines.append(f"{pad}-> {val}  (n={self.n_node_samples_[t]})")
                return
            nm, s = names[self.feature_[t]], self.threshold_[t]
            cond = f"{nm} = 0" if self.binary_[self.feature_[t]] else f"{nm} <= {s:.3f}"
            lines.append(f"{pad}if {cond}:   (n={self.n_node_samples_[t]}, mean {val})")
            rec(self.left_[t], d + 1)
            lines.append(f"{pad}else:")
            rec(self.right_[t], d + 1)

        rec(0, 0)
        return "\n".join(lines)
