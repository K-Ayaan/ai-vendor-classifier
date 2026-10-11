"""Extended Unit-II models on the same data, splits and seed as Phase 1.

Usage: python src/extended_models.py   ->  reports/extended_models_metrics.json

Regression (target: AI-judged expected_score, not ground truth; split by requirement):
  OLS (= Gaussian MLE, Phase 1), ridge, robust (Huber), Bayesian linear regression,
  kernel ridge (polynomial and RBF kernels).
Classification (target: weak is_pharmacy labels; stratified split):
  Fisher LDA (discriminant function), Bernoulli naive Bayes (generative),
  Bayesian logistic regression via the Laplace approximation, linear and RBF SVMs,
  next to Phase 1's L2 logistic (MAP) and small-model MLE.
All hyperparameters are chosen by cross-validation on the training split only.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402
import logistic_regression_classification as L  # noqa: E402

SEED = 42
OUT = Path("reports/extended_models_metrics.json")
RIDGE_ALPHAS = np.logspace(-3, 4, 29)


# ------------------------------------------------------------------ regression
def reg_metrics(y, yhat):
    rss = float(np.sum((y - yhat) ** 2))
    return {"RMSE": float(np.sqrt(rss / len(y))), "MAE": float(mean_absolute_error(y, yhat)),
            "R2": float(1 - rss / np.sum((y - y.mean()) ** 2))}


def bayesian_posterior(X, y, alpha_noise, lam_w):
    """Conjugate posterior for centred data: S_N = (lam I + a XᵀX)^-1, m_N = a S_N Xᵀy."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    S = np.linalg.inv(lam_w * np.eye(X.shape[1]) + alpha_noise * Xc.T @ Xc)
    m = alpha_noise * S @ Xc.T @ yc
    return m, S, ym - xm @ m


def run_regression():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = lr.run(lr.load_annotations())
    X_tr, X_te = r["X_tr"][:, 1:], r["X_te"][:, 1:]          # drop the ones column
    y_tr, y_te = r["y_tr"], r["y_te"]
    groups = r["train"].requirement_id.to_numpy()
    cv = GroupKFold(n_splits=4)                              # whole requirements per fold
    out, preds = {}, {}

    ols = LinearRegression().fit(X_tr, y_tr)
    preds["OLS (Gaussian MLE)"] = ols

    ridge = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                         {"ridge__alpha": RIDGE_ALPHAS}, cv=cv,
                         scoring="neg_mean_squared_error").fit(X_tr, y_tr, groups=groups)
    preds["Ridge (L2, MAP)"] = ridge.best_estimator_

    huber = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=5000).fit(X_tr, y_tr)
    preds["Robust (Huber)"] = huber

    bayes = BayesianRidge(fit_intercept=True).fit(X_tr, y_tr)
    preds["Bayesian linear"] = bayes
    m, S, b0 = bayesian_posterior(X_tr, y_tr, bayes.alpha_, bayes.lambda_)
    mu_te, sd_te = bayes.predict(X_te, return_std=True)

    kpoly = GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel="poly", degree=2)),
                         {"kernelridge__alpha": np.logspace(-2, 3, 11),
                          "kernelridge__coef0": [1.0]},
                         cv=cv, scoring="neg_mean_squared_error").fit(X_tr, y_tr, groups=groups)
    preds["Kernel ridge (poly deg 2)"] = kpoly.best_estimator_
    krbf = GridSearchCV(make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                        {"kernelridge__alpha": np.logspace(-2, 2, 9),
                         "kernelridge__gamma": np.logspace(-3, 0, 7)},
                        cv=cv, scoring="neg_mean_squared_error").fit(X_tr, y_tr, groups=groups)
    preds["Kernel ridge (RBF)"] = krbf.best_estimator_

    for name, model in preds.items():
        out[name] = {"train": reg_metrics(y_tr, model.predict(X_tr)),
                     "test": reg_metrics(y_te, model.predict(X_te))}

    feats = lr.FEATURES
    z = np.abs((y_te - mu_te) / sd_te)
    details = {
        "features": feats,
        "ols_coef": dict(zip(feats, map(float, ols.coef_))), "ols_intercept": float(ols.intercept_),
        "huber_coef": dict(zip(feats, map(float, huber.coef_))),
        "huber_intercept": float(huber.intercept_), "huber_scale": float(huber.scale_),
        "huber_n_outliers_train": int(huber.outliers_.sum()), "huber_epsilon": 1.35,
        "ridge_alpha": float(ridge.best_params_["ridge__alpha"]),
        "bayes_noise_precision_alpha": float(bayes.alpha_),
        "bayes_noise_variance_1_over_alpha": float(1 / bayes.alpha_),
        "bayes_weight_precision_lambda": float(bayes.lambda_),
        "bayes_coef": dict(zip(feats, map(float, bayes.coef_))),
        "bayes_coef_sd": dict(zip(feats, map(float, np.sqrt(np.diag(S))))),
        "bayes_closed_form_vs_sklearn_max_abs_diff": float(np.max(np.abs(m - bayes.coef_))),
        "bayes_test_mean_predictive_sd": float(sd_te.mean()),
        "bayes_test_95pct_interval_coverage": float(np.mean(z <= 1.96)),
        "kpoly_alpha": float(kpoly.best_params_["kernelridge__alpha"]),
        "krbf_alpha": float(krbf.best_params_["kernelridge__alpha"]),
        "krbf_gamma": float(krbf.best_params_["kernelridge__gamma"]),
    }
    return {"metrics": out, "details": details, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr,
            "y_te": y_te, "models": preds, "bayes_mu_te": mu_te, "bayes_sd_te": sd_te}


def ridge_path(X_tr, y_tr):
    """Standardised ridge coefficients over a grid of lambda (for the shrinkage graph)."""
    Xs = StandardScaler().fit_transform(X_tr)
    return RIDGE_ALPHAS, np.array([Ridge(alpha=a).fit(Xs, y_tr).coef_ for a in RIDGE_ALPHAS])


# ------------------------------------------------------------------ classification
def clf_metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
    return {"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score))}


def laplace_bayesian_logistic(X_tr, y_tr, X_te, lam):
    """Prior beta_j ~ N(0, 1/lam) (intercept flat). MAP by penalised IRLS, then Laplace:
    posterior ~ N(beta_MAP, S) with S = (XᵀWX + lam P)^-1. Predictive via the probit
    approximation p = sigmoid(mu / sqrt(1 + pi s2 / 8))."""
    beta, _, conv = L.irls(X_tr, y_tr, l2=lam, max_iter=200)
    p = L.sigmoid(X_tr @ beta)
    P = np.eye(X_tr.shape[1]); P[0, 0] = 0.0
    S = np.linalg.inv((X_tr * (p * (1 - p))[:, None]).T @ X_tr + lam * P)
    mu = X_te @ beta
    s2 = np.einsum("ij,jk,ik->i", X_te, S, X_te)
    p_pred = L.sigmoid(mu / np.sqrt(1 + np.pi * s2 / 8))
    return {"beta": beta, "S": S, "converged": conv, "p_map": L.sigmoid(mu),
            "p_pred": p_pred, "s2": s2}


def run_classification():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = L.run()
    X_tr, X_te, y_tr, y_te = c["X_tr"], c["X_te"], c["y_tr"], c["y_te"]
    Xn_tr, Xn_te = X_tr[:, 1:], X_te[:, 1:]                   # sklearn adds its own intercept
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    res, scores, extra = {}, {}, {}

    scores["Logistic L2 (MAP), Phase 1"] = (c["p_l2"], 0.5)
    scores["Logistic small-model MLE, Phase 1"] = (c["small"]["p_test"], 0.5)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xn_tr, y_tr)
    scores["Fisher LDA (discriminant)"] = (lda.predict_proba(Xn_te)[:, 1], 0.5)

    nb = BernoulliNB(alpha=1.0).fit(Xn_tr, y_tr)
    scores["Bernoulli naive Bayes (generative)"] = (nb.predict_proba(Xn_te)[:, 1], 0.5)

    lam = 1.0 / c["best_c"]
    bl = laplace_bayesian_logistic(X_tr, y_tr, X_te, lam)
    scores["Bayesian logistic (Laplace)"] = (bl["p_pred"], 0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svl = GridSearchCV(SVC(kernel="linear"), {"C": [0.01, 0.1, 1, 10]}, cv=folds,
                           scoring="f1").fit(Xn_tr, y_tr)
        svr = GridSearchCV(SVC(kernel="rbf", gamma="scale"), {"C": [0.1, 1, 10, 100]}, cv=folds,
                           scoring="f1").fit(Xn_tr, y_tr)
    scores["SVM linear"] = (svl.decision_function(Xn_te), 0.0)
    scores["SVM RBF kernel"] = (svr.decision_function(Xn_te), 0.0)

    for name, (sc, thr) in scores.items():
        res[name] = clf_metrics(y_te, sc, thr)

    log_loss = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-15, 1)) +  # noqa: E731
                                           (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))))
    extra = {
        "laplace_lambda": lam, "laplace_irls_converged": bl["converged"],
        "laplace_test_logloss_map_plugin": log_loss(y_te, bl["p_map"]),
        "laplace_test_logloss_predictive": log_loss(y_te, bl["p_pred"]),
        "laplace_mean_abs_shift_map_to_predictive": float(np.mean(np.abs(bl["p_pred"] - bl["p_map"]))),
        "laplace_max_abs_shift_map_to_predictive": float(np.max(np.abs(bl["p_pred"] - bl["p_map"]))),
        "lda_shrinkage": "auto (Ledoit-Wolf)", "nb_alpha": 1.0,
        "svm_linear_C": float(svl.best_params_["C"]), "svm_rbf_C": float(svr.best_params_["C"]),
        "svm_linear_n_support": int(svl.best_estimator_.n_support_.sum()),
        "svm_rbf_n_support": int(svr.best_estimator_.n_support_.sum()),
        "n_train": int(len(y_tr)),
    }
    return {"metrics": res, "details": extra, "scores": scores, "y_te": y_te, "laplace": bl,
            "c": c}


def main():
    reg = run_regression()
    clf = run_classification()
    out = {"note": "Regression target is the AI-judged expected_score (not ground truth); "
                   "classification target is the weak is_pharmacy label.",
           "regression": {"metrics": reg["metrics"], "details": reg["details"]},
           "classification": {"metrics": clf["metrics"], "details": clf["details"]}}
    OUT.write_text(json.dumps(L.round_floats(out), indent=2), encoding="utf-8")
    print("REGRESSION (test)")
    for k, v in reg["metrics"].items():
        print(f"  {k:28s} RMSE {v['test']['RMSE']:.4f}  MAE {v['test']['MAE']:.4f}  R2 {v['test']['R2']:.4f}"
              f"   | train R2 {v['train']['R2']:.4f}")
    print("  details:", {k: v for k, v in reg["details"].items() if not isinstance(v, (dict, list))})
    print("CLASSIFICATION (test)")
    for k, v in clf["metrics"].items():
        print(f"  {k:36s} acc {v['accuracy']:.4f} P {v['precision']:.4f} R {v['recall']:.4f} "
              f"F1 {v['f1']:.4f} ROC {v['roc_auc']:.4f} PR {v['pr_auc']:.4f} cm {v['confusion_matrix']}")
    print("  details:", clf["details"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
