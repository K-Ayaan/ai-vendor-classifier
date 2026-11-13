"""Step 4 - logistic regression fitted by maximum likelihood.

Usage: python src/logistic_regression_classification.py
Reads data/processed/vendors.csv and writes:
  reports/classification_metrics.json
  data/annotation/label_audit_sample.csv   (200 test rows, blank true_is_pharmacy)
  data/processed/label_audit_key.csv       (weak label + model scores for those rows)

Model:  z = X beta,  p = sigmoid(z),  y_i ~ Bernoulli(p_i)
        l(beta)  = sum_i [ y_i z_i - log(1 + e^{z_i}) ]
        grad     = X^T (y - p)
        Hessian  = -X^T W X,  W = diag(p_i (1 - p_i))
The score equations X^T (y - p(beta)) = 0 are non-linear in beta, so there is no
closed form; we maximise l(beta) numerically with Newton-Raphson (= IRLS).
"""
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching.features import (MIN_DF, TOP_K_PINCODES, FeatureBuilder,  # noqa: E402,F401
                               independent_columns)  # shared with the Phase 2 app
from preprocessing import norm_title  # noqa: E402

SEED = 42
VENDORS = Path("data/processed/vendors.csv")
OUT_METRICS = Path("reports/classification_metrics.json")
OUT_AUDIT = Path("data/annotation/label_audit_sample.csv")
OUT_AUDIT_KEY = Path("data/processed/label_audit_key.csv")
TARGET_CATEGORY = "pharmacy"
TEST_SIZE = 0.2
C_GRID = np.logspace(-3, 3, 13)
AUDIT_N = 200
SMALL_MAX_FEATURES = 10


# ---------------------------------------------------------------- data & features
def load_vendors():
    v = pd.read_csv(VENDORS, encoding="utf-8", dtype={"pincode": str},
                    keep_default_na=False, na_values={"locality": [""]})
    v["y"] = v.categories.str.split(";").map(lambda c: TARGET_CATEGORY in c).astype(int)
    return v


def split(v):
    """Stratified 80/20 split. One row per vendor, so no vendor can be in both sets."""
    train, test = train_test_split(v, test_size=TEST_SIZE, stratify=v.y, random_state=SEED)
    assert not set(train.vendor_id) & set(test.vendor_id), "vendor in both train and test"
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------- likelihood & IRLS
def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def log_likelihood(X, y, beta):
    """l(beta) = sum y z - log(1 + e^z), computed stably with logaddexp."""
    z = X @ beta
    return float(np.sum(y * z - np.logaddexp(0.0, z)))


def irls(X, y, l2=0.0, max_iter=100, tol=1e-10):
    """Newton-Raphson / IRLS for (optionally L2-penalised) logistic regression.

    Maximises l(beta) - (l2 / 2) * ||beta_without_intercept||^2.
    Newton step: beta_new = beta + (X^T W X + l2*P)^{-1} (X^T (y - p) - l2*P beta),
    which is the same as a weighted least-squares solve on the working response (IRLS).
    Step-halving keeps every step an ascent step. Returns beta, history, converged.
    """
    n, k = X.shape
    P = np.eye(k)
    P[0, 0] = 0.0  # never penalise the intercept
    beta = np.zeros(k)

    def objective(b):
        return log_likelihood(X, y, b) - 0.5 * l2 * b @ P @ b

    obj = objective(beta)
    history = []
    converged = False
    for it in range(1, max_iter + 1):
        p = sigmoid(X @ beta)
        w = p * (1 - p)
        grad = X.T @ (y - p) - l2 * P @ beta
        H = (X * w[:, None]).T @ X + l2 * P          # negative Hessian
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        t = 1.0
        while True:
            new_beta = beta + t * step
            new_obj = objective(new_beta)
            if new_obj >= obj - 1e-12 or t < 1e-8:
                break
            t /= 2
        history.append({"iter": it, "objective": new_obj, "gain": new_obj - obj,
                        "beta_norm": float(np.linalg.norm(new_beta)),
                        "max_abs_beta": float(np.max(np.abs(new_beta))),
                        "max_step": float(np.max(np.abs(new_beta - beta))),
                        "cond_H": float(np.linalg.cond(H))})
        done = abs(new_obj - obj) < tol and np.max(np.abs(new_beta - beta)) < 1e-6
        beta, obj = new_beta, new_obj
        if done:
            converged = True
            break
    return beta, history, converged


def detect_separation(X, y, names):
    """Single-feature quasi-complete separation: a binary column that is 1 only for rows
    of one class. Then l(beta) keeps rising as that beta_j -> +/- infinity, so no finite
    maximiser exists (the MLE does not exist)."""
    out = []
    for j, name in enumerate(names):
        if name == "intercept":
            continue
        on = X[:, j] == 1
        if on.any() and (y[on].min() == y[on].max()):
            out.append({"feature": name, "n_rows": int(on.sum()), "only_class": int(y[on][0])})
    return out


def separation_lp(X, y, tol=1e-7):
    """Joint (multi-feature) separation check by linear programming.

    With s_i = 2 y_i - 1, the data are (quasi-)separated iff some b != 0 has
    s_i x_i^T b >= 0 for every row with strict inequality for at least one row; then
    l(beta + t b) is non-decreasing in t and the MLE does not exist (Albert & Anderson,
    1984). We maximise sum_i s_i x_i^T b subject to s_i x_i^T b >= 0 and -1 <= b_j <= 1.
    Optimum 0 -> no separating direction (with full column rank, the MLE exists);
    optimum > 0 -> separated."""
    s = 2 * y - 1
    A = s[:, None] * X
    res = linprog(c=-A.sum(axis=0), A_ub=-A, b_ub=np.zeros(len(y)),
                  bounds=[(-1, 1)] * X.shape[1], method="highs")
    if res.status != 0:
        raise RuntimeError(f"separation LP failed: {res.message}")
    return {"separated": bool(-res.fun > tol), "lp_optimum": float(-res.fun)}


def token_indicators(df, tokens):
    toks = df.title.map(lambda t: set(norm_title(t).split()))
    return np.column_stack([toks.map(lambda s, t=t: t in s).to_numpy() for t in tokens]
                           ).astype(float)


def select_mle_tokens(train, max_features=SMALL_MAX_FEATURES):
    """Small model where the MLE exists. Walk title tokens in order of TRAINING-set
    document frequency (ties alphabetical) and keep a token only if the design
    [1, kept tokens] stays full column rank and jointly non-separated on train
    (separation_lp). Test labels are never used."""
    y = train.y.to_numpy()
    df = Counter(t for title in train.title for t in set(norm_title(title).split()))
    order = sorted(df, key=lambda t: (-df[t], t))
    chosen, skipped = [], []
    for tok in order:
        if len(chosen) == max_features:
            break
        X = np.column_stack([np.ones(len(train)), token_indicators(train, chosen + [tok])])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            skipped.append({"token": tok, "train_df": df[tok], "reason": "rank"})
        elif separation_lp(X, y)["separated"]:
            skipped.append({"token": tok, "train_df": df[tok], "reason": "separation"})
        else:
            chosen.append(tok)
    return chosen, skipped, {t: df[t] for t in chosen}


def fit_small_mle(train, test):
    """Unpenalised MLE on the small token set: IRLS vs sklearn penalty=None."""
    tokens, skipped, dfs = select_mle_tokens(train)
    X_tr = np.column_stack([np.ones(len(train)), token_indicators(train, tokens)])
    X_te = np.column_stack([np.ones(len(test)), token_indicators(test, tokens)])
    y_tr = train.y.to_numpy()
    sep = separation_lp(X_tr, y_tr)
    beta_irls, hist, conv = irls(X_tr, y_tr, l2=0.0, max_iter=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sk = LogisticRegression(penalty=None, solver="newton-cholesky", tol=1e-12,
                                max_iter=1000).fit(X_tr[:, 1:], y_tr)
    beta_sk = sk_beta(sk)
    ll_irls, ll_sk = log_likelihood(X_tr, y_tr, beta_irls), log_likelihood(X_tr, y_tr, beta_sk)
    names = ["intercept"] + [f"tok={t}" for t in tokens]
    return {
        "tokens": tokens, "train_doc_freq": dfs, "skipped_before_10": skipped,
        "n_features_excl_intercept": len(tokens),
        "rank_X_train": int(np.linalg.matrix_rank(X_tr)), "n_columns": X_tr.shape[1],
        "separation_lp": sep, "single_feature_separators": detect_separation(X_tr, y_tr, names),
        "irls_converged": conv, "irls_iterations": len(hist),
        "sklearn_n_iter": int(sk.n_iter_[0]),
        "sklearn_warnings": sorted({type(w.message).__name__ for w in caught}),
        "coefficients": dict(zip(names, map(float, beta_irls))),
        "coefficients_sklearn": dict(zip(names, map(float, beta_sk))),
        "max_abs_coef_diff_irls_vs_sklearn": float(np.max(np.abs(beta_irls - beta_sk))),
        "loglik_irls": ll_irls, "loglik_sklearn": ll_sk,
        "abs_loglik_diff": abs(ll_irls - ll_sk),
        "mle_exists": bool(conv and not sep["separated"]
                           and np.linalg.matrix_rank(X_tr) == X_tr.shape[1]),
        "p_test": sigmoid(X_te @ beta_irls),
    }


# ---------------------------------------------------------------- evaluation
def evaluate(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])}


def majority_baseline(y_train, y_test):
    majority = int(np.mean(y_train) >= 0.5)
    pred = np.full(len(y_test), majority)
    return {"predicts_class": majority, "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": 0.5, "pr_auc_equals_prevalence": float(np.mean(y_test))}


def sk_logreg(C, **kw):
    """sklearn model on X without its ones column. With lbfgs the intercept is not
    penalised, matching irls() which never penalises beta_0."""
    return LogisticRegression(C=C, fit_intercept=True, solver="lbfgs", **kw)


def sk_beta(model):
    return np.r_[model.intercept_, model.coef_.ravel()]


def choose_c(X, y):
    """5-fold stratified CV on the TRAINING split only; pick C with the lowest log-loss."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for c in C_GRID:
        losses = []
        for tr, va in folds.split(X, y):
            m = sk_logreg(c, max_iter=5000).fit(X[tr][:, 1:], y[tr])
            losses.append(log_loss(y[va], m.predict_proba(X[va][:, 1:])[:, 1], labels=[0, 1]))
        scores.append(float(np.mean(losses)))
    return float(C_GRID[int(np.argmin(scores))]), dict(zip(map(float, C_GRID), scores))


def round_floats(obj, sig=12):
    """Round to 12 significant digits (keeps tiny values and log-likelihood digits)."""
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, sig) for v in obj]
    return obj


# ---------------------------------------------------------------- main
def run():
    """Fit everything and return a results dict (used by the script and the notebook)."""
    v = load_vendors()
    train, test = split(v)
    fb = FeatureBuilder().fit(train)
    X_tr, X_te = fb.transform(train), fb.transform(test)
    y_tr, y_te = train.y.to_numpy(), test.y.to_numpy()

    # 1) unpenalised MLE: hand-written IRLS and sklearn (C=inf means no penalty)
    beta_irls, hist, conv_irls = irls(X_tr, y_tr, l2=0.0, max_iter=100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sk = sk_logreg(np.inf, max_iter=10000, tol=1e-10).fit(X_tr[:, 1:], y_tr)
    beta_sk = sk_beta(sk)
    conv_sk = bool(sk.n_iter_[0] < sk.max_iter)
    sk_warnings = sorted({type(w.message).__name__ for w in caught})

    separated = detect_separation(X_tr, y_tr, fb.names)
    lp_full = separation_lp(X_tr, y_tr)
    full_rank = int(np.linalg.matrix_rank(X_tr)) == X_tr.shape[1]
    mle_exists = not separated and not lp_full["separated"] and conv_irls and full_rank

    # 2) L2-penalised model, C chosen by CV on train. sklearn minimises
    #    0.5||w||^2 + C * sum(logloss)  <=>  maximises l(beta) - (1/(2C)) ||w||^2,
    #    so hand-written IRLS with l2 = 1/C must reproduce its coefficients.
    best_c, cv_curve = choose_c(X_tr, y_tr)
    l2_sk = sk_logreg(best_c, max_iter=10000, tol=1e-10).fit(X_tr[:, 1:], y_tr)
    beta_l2_irls, _, conv_l2_irls = irls(X_tr, y_tr, l2=1.0 / best_c, max_iter=200)

    p_unpen = sigmoid(X_te @ beta_irls)
    p_l2 = l2_sk.predict_proba(X_te[:, 1:])[:, 1]

    coef = pd.DataFrame({"feature": fb.names, "beta_irls_unpenalised": beta_irls,
                         "beta_sklearn_unpenalised": beta_sk,
                         "beta_sklearn_l2": sk_beta(l2_sk), "beta_irls_l2": beta_l2_irls})
    return {"train": train, "test": test, "fb": fb, "X_tr": X_tr, "X_te": X_te,
            "y_tr": y_tr, "y_te": y_te, "history": hist, "coef": coef,
            "conv_irls": conv_irls, "conv_sk": conv_sk, "sk_n_iter": int(sk.n_iter_[0]),
            "sk_warnings": sk_warnings, "conv_l2_irls": conv_l2_irls,
            "separated": separated, "lp_full": lp_full, "full_rank": full_rank,
            "mle_exists": mle_exists, "best_c": best_c,
            "small": fit_small_mle(train, test),
            "cv_curve": cv_curve, "p_unpen": p_unpen, "p_l2": p_l2,
            "ll_irls": log_likelihood(X_tr, y_tr, beta_irls),
            "ll_sk": log_likelihood(X_tr, y_tr, beta_sk),
            "ll_l2": log_likelihood(X_tr, y_tr, sk_beta(l2_sk)),
            "ll_null": log_likelihood(X_tr, y_tr, np.r_[np.log(y_tr.mean() / (1 - y_tr.mean())),
                                                       np.zeros(X_tr.shape[1] - 1)])}


def write_label_audit(r):
    if OUT_AUDIT.exists():
        print(f"{OUT_AUDIT} exists - not overwriting (may contain human labels)")
        key = pd.read_csv(OUT_AUDIT_KEY, encoding="utf-8")
        return {"rows": len(key), "weak_positive": int((key.weak_is_pharmacy == 1).sum()),
                "weak_negative": int((key.weak_is_pharmacy == 0).sum())}
    test = r["test"].assign(p_l2=r["p_l2"])
    rng = np.random.default_rng(SEED)
    # up to half weak positives (all of them if fewer), rest weak negatives, so label
    # noise can be measured in both classes; the positive class is oversampled vs test.
    n_pos = min(AUDIT_N // 2, int((test.y == 1).sum()))
    parts = []
    for label, n in ((1, n_pos), (0, AUDIT_N - n_pos)):
        pool = test[test.y == label]
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))])
    s = pd.concat(parts)
    s = s.iloc[rng.permutation(len(s))]
    # blind sheet: no weak label, no categories, no model score
    sheet = s[["vendor_id", "title", "address"]].assign(true_is_pharmacy="", notes="")
    key = s[["vendor_id", "y", "categories", "p_l2"]].rename(columns={"y": "weak_is_pharmacy"})
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig", lineterminator="\n")
    key.to_csv(OUT_AUDIT_KEY, index=False, encoding="utf-8", lineterminator="\n")
    return {"rows": len(sheet), "weak_positive": int((key.weak_is_pharmacy == 1).sum()),
            "weak_negative": int((key.weak_is_pharmacy == 0).sum())}


def main():
    r = run()
    coef, hist = r["coef"], r["history"]
    l2_diff = float(np.max(np.abs(coef.beta_irls_l2 - coef.beta_sklearn_l2)))
    m_unpen = evaluate(r["y_te"], r["p_unpen"])
    m_l2 = evaluate(r["y_te"], r["p_l2"])
    small = r["small"]
    m_small = evaluate(r["y_te"], small["p_test"])
    base = majority_baseline(r["y_tr"], r["y_te"])
    audit = write_label_audit(r)

    big = coef.reindex(coef.beta_irls_unpenalised.abs().sort_values(ascending=False).index)
    metrics = {
        "target": f"is_{TARGET_CATEGORY}",
        "split": {"method": "stratified train_test_split", "test_size": TEST_SIZE,
                  "seed": SEED, "n_train": len(r["y_tr"]), "n_test": len(r["y_te"]),
                  "positives_train": int(r["y_tr"].sum()), "positives_test": int(r["y_te"].sum()),
                  "prevalence_train": float(r["y_tr"].mean()),
                  "prevalence_test": float(r["y_te"].mean())},
        "features": {"n_total_incl_intercept": len(r["fb"].names),
                     "n_title_tokens": sum(n.startswith("tok=") for n in r["fb"].names),
                     "n_pincode_indicators": len(r["fb"].pincodes),
                     "min_df": MIN_DF, "top_k_pincodes": TOP_K_PINCODES,
                     "dropped_linearly_dependent_columns": r["fb"].dropped_dependent,
                     "rank_X_train": int(np.linalg.matrix_rank(r["X_tr"])),
                     "full_column_rank": r["full_rank"]},
        "unpenalised_mle": {
            "mle_exists": r["mle_exists"],
            "irls_converged": r["conv_irls"], "irls_iterations": len(hist),
            "sklearn_converged": r["conv_sk"], "sklearn_iterations": r["sk_n_iter"],
            "sklearn_warnings": r["sk_warnings"],
            "loglik_irls": r["ll_irls"], "loglik_sklearn": r["ll_sk"],
            "loglik_null_intercept_only": r["ll_null"],
            "max_abs_coef_diff_irls_vs_sklearn": float(np.max(np.abs(
                coef.beta_irls_unpenalised - coef.beta_sklearn_unpenalised))),
            "max_abs_beta_irls": float(coef.beta_irls_unpenalised.abs().max()),
            "n_separating_features": len(r["separated"]),
            "separating_features": r["separated"],
            "joint_separation_lp": r["lp_full"],
            "irls_trace_last5": hist[-5:],
            "test_metrics_NOT_A_VALID_MLE": m_unpen},
        "small_mle_model": {
            "label": "unpenalised MLE, small token set (MLE exists)",
            **{k: v for k, v in small.items() if k != "p_test"},
            "test_metrics": m_small},
        "l2_model": {"label": "penalised likelihood (MAP), not MLE",
                     "C": r["best_c"], "lambda_equiv": 1.0 / r["best_c"],
                     "cv_mean_logloss_by_C": r["cv_curve"], "loglik_train": r["ll_l2"],
                     "max_abs_coef_diff_sklearn_vs_hand_irls": l2_diff,
                     "hand_irls_converged": r["conv_l2_irls"],
                     "test_metrics": m_l2},
        "majority_baseline": base,
        "label_audit_sample": audit,
    }
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(round_floats(metrics), indent=2, ensure_ascii=False),
                           encoding="utf-8")
    coef.to_csv("reports/classification_coefficients.csv", index=False, encoding="utf-8",
                lineterminator="\n")

    s = metrics["split"]
    print(f"train {s['n_train']} ({s['positives_train']} pos, {s['prevalence_train']:.4f}) | "
          f"test {s['n_test']} ({s['positives_test']} pos, {s['prevalence_test']:.4f})")
    print(f"features: {metrics['features']['n_total_incl_intercept']} "
          f"(tokens {metrics['features']['n_title_tokens']}, pincodes "
          f"{metrics['features']['n_pincode_indicators']}), rank {metrics['features']['rank_X_train']}, "
          f"full rank {r['full_rank']}, dropped dependent columns: {r['fb'].dropped_dependent}")
    u = metrics["unpenalised_mle"]
    print(f"IRLS converged={u['irls_converged']} iters={u['irls_iterations']} | sklearn "
          f"converged={u['sklearn_converged']} iters={u['sklearn_iterations']}")
    print(f"loglik IRLS={r['ll_irls']:.6f} sklearn={r['ll_sk']:.6f} null={r['ll_null']:.6f}")
    print(f"max|beta| IRLS={u['max_abs_beta_irls']:.3f} | max coef diff IRLS vs sklearn="
          f"{u['max_abs_coef_diff_irls_vs_sklearn']:.4f}")
    print("IRLS trace (iter, gain, ||beta||, cond H):",
          [(h["iter"], f"{h['gain']:.2e}", round(h["beta_norm"], 2), f"{h['cond_H']:.1e}")
           for h in hist[:3] + hist[-3:]])
    print(f"separating features: {len(r['separated'])} -> "
          f"{[d['feature'] for d in r['separated'][:12]]}")
    print(f"joint separation LP (full model): {r['lp_full']}")
    print(f"MLE exists: {r['mle_exists']}")
    print(f"SMALL MLE MODEL tokens={small['tokens']} rank={small['rank_X_train']}/"
          f"{small['n_columns']} LP={small['separation_lp']} single-feature separators="
          f"{len(small['single_feature_separators'])} skipped={small['skipped_before_10']}")
    print(f"  IRLS converged={small['irls_converged']} iters={small['irls_iterations']} | "
          f"sklearn(penalty=None) iters={small['sklearn_n_iter']} warnings="
          f"{small['sklearn_warnings']}")
    print(f"  max|beta_IRLS - beta_sklearn|={small['max_abs_coef_diff_irls_vs_sklearn']:.3e} | "
          f"loglik IRLS={small['loglik_irls']:.10f} sklearn={small['loglik_sklearn']:.10f} "
          f"|diff|={small['abs_loglik_diff']:.3e} | MLE exists={small['mle_exists']}")
    print("  coefficients:", {k: round(v, 4) for k, v in small["coefficients"].items()})
    print("largest |beta| unpenalised:", big.head(6)[["feature", "beta_irls_unpenalised",
                                                     "beta_sklearn_unpenalised"]]
          .round(2).values.tolist())
    print(f"L2: C={r['best_c']:.4g}, loglik={r['ll_l2']:.4f}, hand IRLS vs sklearn max diff "
          f"{l2_diff:.2e}")
    for name, m in [("unpenalised (not a valid MLE)", m_unpen), ("small MLE", m_small),
                    ("L2 (penalised likelihood, MAP)", m_l2)]:
        print(f"TEST {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()
                                           if isinstance(v, float)) + f", cm={m['confusion_matrix']}")
    print(f"majority baseline: {base}")
    print(f"label audit: {audit}")
    print(f"wrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
