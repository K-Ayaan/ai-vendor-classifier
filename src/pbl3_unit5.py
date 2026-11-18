"""PBL-3 Unit V: combining models, on the Phase 1 data and splits.

Usage: python src/pbl3_unit5.py          (about 6-8 minutes)
Reads data/processed/ and data/annotation/ (read-only). Writes reports/pbl3_unit5_metrics.json,
reports/pbl3_predictions.npz (test predictions, for the comparison script),
pbl-3/03_unit_topics_implementation/figures/U5_*.png and .../unit5_numbers.md.

Match score (regression, 360 train / 90 test pairs, split by requirement; every hyperparameter
chosen by grouped 4-fold CV on train, whole requirements per fold):
  R1 CART with cost-complexity pruning            R5 Generalised additive model (P-splines)
  R2 Bagging and random forests (committees)      R6 Bayesian model averaging (64 linear models)
  R3 Bias-variance by bootstrap replicates         R7 Mixture of regressions, mixture of experts
  R4 Gradient boosting (adaptive basis functions) R8 Stacking (ensemble learning)
Pharmacy (classification, 3,689 / 923 vendors, 441 binary columns; stratified 5-fold CV):
  C1 CART  C2 Bagging / random forest  C3 AdaBoost  C4 Gradient boosting  C5 Logistic BMA
  C6 Stacking and soft voting
All own implementations (src/ensemble/); scikit-learn only as a cross-check.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import logsumexp
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import logistic_regression_classification as L
from ensemble import bma, gam, mixture
from ensemble.boosting import AdaBoost, GradientBoosting
from ensemble.committee import Forest
from ensemble.tree import CART
from pbl3_style import (AQUA, BLUE, FEATURES, INK2, LIGHT, MUTED, ORANGE, PALETTE, PBL3, PINK,
                        PURPLE, SEED, YELLOW, classification_data, clf_metrics, f4, md_table,
                        plt, reg_metrics, regression_data, save, write_json)

OUT = PBL3 / "03_unit_topics_implementation"
FIG = OUT / "figures"
METRICS = Path("reports/pbl3_unit5_metrics.json")
PRED = Path("reports/pbl3_predictions.npz")
FI = {f: i for i, f in enumerate(FEATURES)}
PRED_R, PRED_C = {}, {}       # test predictions by model name
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def gfolds(r):
    return list(GroupKFold(n_splits=4).split(r["Xn_tr"], r["y_tr"], r["groups"]))


def sfolds(c):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(c["Xn_tr"], c["y_tr"]))


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def metric_row(name, m, extra=""):
    return {"model": name, "test RMSE": f4(m["RMSE"]), "test MAE": f4(m["MAE"]),
            "test R²": f4(m["R2"]), "note": extra}


def clf_row(name, m, extra=""):
    return {"model": name, "F1": f4(m["f1"]), "precision": f4(m["precision"]),
            "recall": f4(m["recall"]), "ROC-AUC": f4(m["roc_auc"]), "PR-AUC": f4(m["pr_auc"]),
            "log loss": f4(m["log_loss"]), "FP/FN": f"{m['fp']}/{m['fn']}", "note": extra}


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ======================================================================== R1 CART
def r1_cart(r, md, m):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    full = CART("regression").fit(X, y)
    alphas, imps, masks = full.pruning_path()
    leaves = [full.n_leaves(mk) for mk in masks]
    cand = np.r_[0.0, np.sqrt(alphas[1:-1] * alphas[2:])] if len(alphas) > 2 else alphas
    cand = np.unique(np.r_[cand, alphas[-1]])
    folds = gfolds(r)
    fold_trees = [CART("regression").fit(X[tr], y[tr]) for tr, _ in folds]
    cv = np.array([[mse(t.predict(X[va], alpha=a), y[va]) for (_, va), t in zip(folds, fold_trees)]
                   for a in cand])
    cv_mean, cv_se = cv.mean(1), cv.std(1, ddof=1) / np.sqrt(cv.shape[1])
    k = int(np.argmin(cv_mean))
    k1 = int(np.max(np.where(cv_mean <= cv_mean[k] + cv_se[k])[0]))       # 1-SE: simplest
    a_best, a_1se = float(cand[k]), float(cand[k1])
    full.pruning_path()
    leaves_best = full.n_leaves(full._mask_for(a_best))
    leaves_1se = full.n_leaves(full._mask_for(a_1se))
    res = {"full_tree": {"leaves": full.n_leaves(), "nodes": full.n_nodes,
                         "depth": int(full.depth_.max()),
                         "train": reg_metrics(y, full.predict(X)),
                         "test": reg_metrics(yte, full.predict(Xte))},
           "pruning_path_steps": len(alphas), "alpha_best": a_best, "alpha_1se": a_1se,
           "leaves_best": leaves_best, "leaves_1se": leaves_1se,
           "pruned_best": {"train": reg_metrics(y, full.predict(X, alpha=a_best)),
                           "test": reg_metrics(yte, full.predict(Xte, alpha=a_best))},
           "pruned_1se": {"train": reg_metrics(y, full.predict(X, alpha=a_1se)),
                          "test": reg_metrics(yte, full.predict(Xte, alpha=a_1se))}}
    sk = DecisionTreeRegressor(random_state=0, ccp_alpha=a_best).fit(X, y)
    res["sklearn_max_abs_diff_test"] = float(np.max(np.abs(sk.predict(Xte) - full.predict(Xte, alpha=a_best))))
    res["importance"] = dict(zip(FEATURES, full.feature_importances()))
    PRED_R["CART (pruned, CV)"] = full.predict(Xte, alpha=a_best)
    PRED_R["CART (full depth)"] = full.predict(Xte)
    m["R1_cart"] = res
    small = CART("regression", max_depth=3, min_samples_leaf=5).fit(X, y)
    text = small.describe(FEATURES, max_depth=3)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    lv_cand = [full.n_leaves(full._mask_for(a)) for a in cand]
    ax[0].plot(leaves, [imp * len(y) / len(y) for imp in imps], color=MUTED, lw=1.5,
               label="training error R(T) (MSE)")
    ax[0].errorbar(lv_cand, cv_mean, cv_se, color=BLUE, fmt="o-", ms=3, lw=1.2, capsize=2,
                   label="grouped 4-fold CV MSE")
    ax[0].axvline(leaves_best, color=ORANGE, ls="--", label=f"min CV: {leaves_best} leaves")
    ax[0].axvline(leaves_1se, color=AQUA, ls=":", label=f"1-SE rule: {leaves_1se} leaves")
    ax[0].set_xscale("log"); ax[0].set_xlabel("leaves (log)"); ax[0].set_ylabel("MSE")
    ax[0].set_title("U5_01a  Cost-complexity pruning path"); ax[0].legend(fontsize=7.5)
    ax[1].axis("off")
    ax[1].text(0, 1, "Depth-3 tree (min leaf 5), training pairs\n\n" + text, va="top",
               family="monospace", fontsize=6.6)
    ax[1].set_title("U5_01b  What the tree learned")
    save(fig, FIG, "U5_01_cart_pruning")
    md += ["## R1 CART regression tree (own), cost-complexity pruning", "",
           md_table(pd.DataFrame([
               metric_row("full-depth tree", res["full_tree"]["test"],
                          f"{res['full_tree']['leaves']} leaves, depth {res['full_tree']['depth']}, "
                          f"train R² {f4(res['full_tree']['train']['R2'])}"),
               metric_row("pruned, min CV", res["pruned_best"]["test"],
                          f"α = {a_best:.4f}, {leaves_best} leaves, train R² {f4(res['pruned_best']['train']['R2'])}"),
               metric_row("pruned, 1-SE rule", res["pruned_1se"]["test"],
                          f"α = {a_1se:.4f}, {leaves_1se} leaves")])), "",
           f"- Pruning path: {len(alphas)} nested subtrees. Own pruned tree vs scikit-learn "
           f"(same α): max |Δprediction| {res['sklearn_max_abs_diff_test']:.2e}.",
           "- Impurity importance: " + ", ".join(f"{k} {v:.3f}" for k, v in res["importance"].items()),
           "", "Depth-3 tree (for reading, min leaf 5):", "", "```", text, "```", ""]
    return a_best


# ======================================================================== R2 committees
def r2_forests(r, md, m):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    folds = gfolds(r)
    grid = []
    for mf in [1, 2, 3, 4, 6]:
        for leaf in [1, 5]:
            cvm = np.mean([mse(Forest("regression", 150, max_features=mf, min_samples_leaf=leaf,
                                      seed=SEED).fit(X[tr], y[tr]).predict(X[va]), y[va])
                           for tr, va in folds])
            grid.append({"max_features": mf, "min_leaf": leaf, "cv_mse": float(cvm)})
    g = pd.DataFrame(grid)
    best = g.loc[g.cv_mse.idxmin()]
    bag_best = g[g.max_features == 6].sort_values("cv_mse").iloc[0]
    rf = Forest("regression", 500, max_features=int(best.max_features),
                min_samples_leaf=int(best.min_leaf), seed=SEED).fit(X, y)
    bag = Forest("regression", 500, max_features=None, min_samples_leaf=int(bag_best.min_leaf),
                 seed=SEED).fit(X, y)
    res = {"grid": grid, "rf_max_features": int(best.max_features), "rf_min_leaf": int(best.min_leaf),
           "bag_min_leaf": int(bag_best.min_leaf)}
    for name, f in [("Random forest", rf), ("Bagged trees", bag)]:
        M = f.member_predictions(Xte)
        e_ind = M - yte
        e_av = float(np.mean(e_ind ** 2))
        e_com = mse(M.mean(0), yte)
        R = np.corrcoef(e_ind - e_ind.mean(1, keepdims=True))
        rho = float(R[np.triu_indices(len(R), 1)].mean())
        oob = f.oob_prediction_
        ok = ~np.isnan(oob)
        res[name] = {"test": reg_metrics(yte, M.mean(0)), "train": reg_metrics(y, f.predict(X)),
                     "oob": reg_metrics(y[ok], oob[ok]), "oob_coverage": f.oob_coverage_,
                     "E_AV_test_mse": e_av, "E_COM_test_mse": e_com,
                     "mean_member_error_corr": rho,
                     "importance": dict(zip(FEATURES, f.feature_importances()))}
        PRED_R[f"{name} (own)"] = M.mean(0)
    # error vs number of trees
    curves = {}
    for name, f in [("Random forest", rf), ("Bagged trees", bag)]:
        M = f.member_predictions(Xte)
        cs = np.cumsum(M, 0) / np.arange(1, len(M) + 1)[:, None]
        curves[name] = np.sqrt(np.mean((cs - yte) ** 2, 1))
    skrf = RandomForestRegressor(500, max_features=int(best.max_features),
                                 min_samples_leaf=int(best.min_leaf), random_state=SEED).fit(X, y)
    res["sklearn_rf_test"] = reg_metrics(yte, skrf.predict(Xte))
    m["R2_forests"] = res

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for (name, cvals), col in zip(curves.items(), [BLUE, ORANGE]):
        ax[0].plot(np.arange(1, len(cvals) + 1), cvals, color=col, label=name)
    ax[0].set_xscale("log"); ax[0].set_xlabel("number of trees M (log)")
    ax[0].set_ylabel("test RMSE"); ax[0].set_title("U5_02a  Averaging more trees")
    ax[0].legend(fontsize=8)
    names = ["Bagged trees", "Random forest"]
    eav = [res[n]["E_AV_test_mse"] for n in names]; ecom = [res[n]["E_COM_test_mse"] for n in names]
    xx = np.arange(2)
    ax[1].bar(xx - .2, eav, .4, color=LIGHT, edgecolor=MUTED, label="E_AV (average single tree)")
    ax[1].bar(xx + .2, ecom, .4, color=BLUE, label="E_COM (committee)")
    for i, n in enumerate(names):
        ax[1].text(i, max(eav[i], ecom[i]) + 8, f"ρ = {res[n]['mean_member_error_corr']:.2f}",
                   ha="center", fontsize=8)
    ax[1].set_xticks(xx, names); ax[1].set_ylabel("test MSE")
    ax[1].set_title("U5_02b  Committee vs its members"); ax[1].legend(fontsize=8)
    save(fig, FIG, "U5_02_committees")
    md += ["## R2 Bagging and random forests (own, 500 trees)", "",
           md_table(g.assign(cv_mse=g.cv_mse.map(f4))), "",
           md_table(pd.DataFrame([
               metric_row("Bagged trees", res["Bagged trees"]["test"],
                          f"min leaf {res['bag_min_leaf']}; OOB R² {f4(res['Bagged trees']['oob']['R2'])}"),
               metric_row("Random forest", res["Random forest"]["test"],
                          f"m = {res['rf_max_features']} of 6, min leaf {res['rf_min_leaf']}; "
                          f"OOB R² {f4(res['Random forest']['oob']['R2'])}"),
               metric_row("scikit-learn RF (same settings)", res["sklearn_rf_test"], "cross-check")])), "",
           "| committee | E_AV (mean single-tree MSE) | E_COM | E_COM / E_AV | mean error corr ρ |",
           "|---|---|---|---|---|"] + [
           f"| {n} | {res[n]['E_AV_test_mse']:.2f} | {res[n]['E_COM_test_mse']:.2f} | "
           f"{res[n]['E_COM_test_mse'] / res[n]['E_AV_test_mse']:.3f} | {res[n]['mean_member_error_corr']:.3f} |"
           for n in names] + ["",
           f"- OOB coverage: {res['Random forest']['oob_coverage']:.4f} of training pairs. OOB is "
           "optimistic here: the other 14 pairs of a requirement are usually in-bag.",
           "- RF importance: " + ", ".join(f"{k} {v:.3f}" for k, v in res["Random forest"]["importance"].items()), ""]
    return res


# ======================================================================== R4 gradient boosting
def r4_boosting(r, md, m):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    folds = gfolds(r)
    MMAX = 600
    grid, curves = [], {}
    for depth in [1, 2, 3, 4]:
        for nu in [0.03, 0.1, 0.3]:
            S = np.zeros(MMAX)
            for tr, va in folds:
                gb = GradientBoosting("squared", MMAX, nu, depth, min_samples_leaf=5).fit(X[tr], y[tr])
                S += np.mean((gb.staged_raw(X[va]) - y[va]) ** 2, 1) / len(folds)
            kbest = int(np.argmin(S))
            grid.append({"max_depth": depth, "learning_rate": nu, "best_M": kbest + 1,
                         "cv_mse": float(S[kbest])})
            curves[(depth, nu)] = S
    g = pd.DataFrame(grid)
    b = g.loc[g.cv_mse.idxmin()]
    depth, nu, M = int(b.max_depth), float(b.learning_rate), int(b.best_M)
    gb = GradientBoosting("squared", MMAX, nu, depth, min_samples_leaf=5).fit(X, y)
    st_te = gb.staged_raw(Xte)
    st_tr = gb.staged_raw(X)
    pred = st_te[M - 1]
    PRED_R["Gradient boosting (own)"] = pred
    d1 = g[g.max_depth == 1].sort_values("cv_mse").iloc[0]
    gb1 = GradientBoosting("squared", int(d1.best_M), float(d1.learning_rate), 1,
                           min_samples_leaf=5).fit(X, y)
    PRED_R["Boosted stumps (additive)"] = gb1.predict(Xte)
    sk = GradientBoostingRegressor(n_estimators=M, learning_rate=nu, max_depth=depth,
                                   min_samples_leaf=5, random_state=0).fit(X, y)
    res = {"grid": grid, "best": {"max_depth": depth, "learning_rate": nu, "M": M},
           "train": reg_metrics(y, st_tr[M - 1]), "test": reg_metrics(yte, pred),
           "stumps": {"learning_rate": float(d1.learning_rate), "M": int(d1.best_M),
                      "test": reg_metrics(yte, gb1.predict(Xte))},
           "sklearn_test": reg_metrics(yte, sk.predict(Xte)),
           "sklearn_max_abs_diff_test": float(np.max(np.abs(sk.predict(Xte) - pred)))}
    # interpretation: permutation importance (test) and partial dependence
    class Wrap:
        def fit(self, *a):
            return self

        def predict(self, Z):
            return gb.staged_raw(Z)[M - 1]

    # permutation importance on test: shuffle one column, measure the rise in MSE (30 repeats)
    prng = np.random.default_rng(SEED)
    base_mse = mse(Wrap().predict(Xte), yte)
    pim = []
    for j in range(len(FEATURES)):
        rises = []
        for _ in range(30):
            Z = Xte.copy(); Z[:, j] = prng.permutation(Z[:, j])
            rises.append(mse(Wrap().predict(Z), yte) - base_mse)
        pim.append(float(np.mean(rises)))
    res["permutation_importance_test_mse_increase"] = dict(zip(FEATURES, pim))
    imp = np.zeros(len(FEATURES))
    for t in gb.trees[:M]:
        imp += t.feature_importances() * t.weighted_n_[0] * t.impurity_[0]
    res["impurity_importance"] = dict(zip(FEATURES, imp / imp.sum()))

    def pd_curve(j, grid_vals):
        out = []
        for v in grid_vals:
            Z = X.copy(); Z[:, j] = v
            out.append(float(Wrap().predict(Z).mean()))
        return np.array(out)

    dgrid = np.r_[np.linspace(0, 3, 13), np.linspace(3.5, 48, 20)]
    pd_dist = pd_curve(FI["distance_km"], dgrid)
    pd2 = np.zeros((2, 2))
    for a in (0, 1):
        for bb in (0, 1):
            Z = X.copy(); Z[:, FI["category_match"]] = a; Z[:, FI["within_radius"]] = bb
            pd2[a, bb] = Wrap().predict(Z).mean()
    res["partial_dependence_2d"] = {"cat0_rad0": pd2[0, 0], "cat0_rad1": pd2[0, 1],
                                    "cat1_rad0": pd2[1, 0], "cat1_rad1": pd2[1, 1],
                                    "interaction_contrast": (pd2[1, 1] - pd2[1, 0]) - (pd2[0, 1] - pd2[0, 0])}
    res["partial_dependence_distance"] = {"at_0km": pd_dist[0], "at_3km": float(pd_curve(FI["distance_km"], [3.0])[0]),
                                          "at_48km": pd_dist[-1]}
    m["R4_boosting"] = res

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for (dd, col) in zip([1, 2, 3, 4], [MUTED, BLUE, ORANGE, PURPLE]):
        ax[0].plot(np.arange(1, MMAX + 1), np.sqrt(curves[(dd, 0.1)]), color=col, label=f"depth {dd}")
    ax[0].plot(np.arange(1, MMAX + 1), np.sqrt(curves[(depth, 0.3)]), color=YELLOW, ls="--",
               label=f"depth {depth}, ν = 0.3")
    ax[0].set_xscale("log"); ax[0].set_ylim(12, 25)
    ax[0].set_xlabel("boosting rounds M (log)"); ax[0].set_ylabel("grouped CV RMSE (ν = 0.1)")
    ax[0].set_title("U5_04a  Depth = order of interactions"); ax[0].legend(fontsize=7.5)
    ax[1].plot(np.arange(1, MMAX + 1), np.sqrt(np.mean((st_tr - y) ** 2, 1)), color=MUTED, label="train")
    ax[1].plot(np.arange(1, MMAX + 1), np.sqrt(np.mean((st_te - yte) ** 2, 1)), color=BLUE, label="test")
    ax[1].axvline(M, color=ORANGE, ls="--", label=f"M chosen by CV = {M}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("boosting rounds M (log)"); ax[1].set_ylabel("RMSE")
    ax[1].set_title(f"U5_04b  Chosen model (depth {depth}, ν = {nu})"); ax[1].legend(fontsize=8)
    save(fig, FIG, "U5_04_gradient_boosting")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
    ax[0].plot(dgrid, pd_dist, color=BLUE, lw=2)
    ax[0].set_xlabel("distance_km"); ax[0].set_ylabel("partial dependence (score)")
    ax[0].set_title("U5_05a  PD on distance")
    for a, col, lab in [(0, MUTED, "other category"), (1, BLUE, "needed category")]:
        ax[1].plot([0, 1], pd2[a], "o-", color=col, lw=2, label=lab)
    ax[1].set_xticks([0, 1], ["outside", "within radius"]); ax[1].legend(fontsize=8)
    ax[1].set_title("U5_05b  2-D PD: the interaction is learned")
    pim = pd.Series(res["permutation_importance_test_mse_increase"]).sort_values()
    ax[2].barh(pim.index, pim.values, color=BLUE)
    ax[2].set_xlabel("test MSE increase when permuted"); ax[2].set_title("U5_05c  Permutation importance")
    save(fig, FIG, "U5_05_boosting_interpretation")
    md += ["## R4 Gradient boosting (own, squared error, min leaf 5)", "",
           md_table(g.assign(cv_mse=g.cv_mse.map(f4))), "",
           md_table(pd.DataFrame([
               metric_row(f"GB depth {depth}, ν {nu}, M {M}", res["test"], f"train R² {f4(res['train']['R2'])}"),
               metric_row(f"boosted stumps (depth 1), ν {d1.learning_rate}, M {int(d1.best_M)}",
                          res["stumps"]["test"], "additive: no interactions"),
               metric_row("scikit-learn GB (same settings)", res["sklearn_test"],
                          f"max |Δ| {res['sklearn_max_abs_diff_test']:.2e}")])), "",
           "- Permutation importance (test MSE increase): " + ", ".join(
               f"{k} {v:.1f}" for k, v in res["permutation_importance_test_mse_increase"].items()),
           f"- 2-D partial dependence (category × radius): {pd2[0, 0]:.2f}, {pd2[0, 1]:.2f} | "
           f"{pd2[1, 0]:.2f}, {pd2[1, 1]:.2f}; interaction contrast "
           f"{res['partial_dependence_2d']['interaction_contrast']:.2f} (EDA cell contrast 28.15).", ""]
    return res


# ======================================================================== R3 bias-variance
def r3_bias_variance(r, md, m, a_best, rfres, gbres):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    groups = r["groups"]
    ureq = np.unique(groups)
    rng = np.random.default_rng(SEED)
    R = 30
    gbb = gbres["best"]
    makers = {
        "OLS": lambda Xb, yb: LinearRegression().fit(Xb, yb).predict(Xte),
        "full-depth tree": lambda Xb, yb: CART("regression").fit(Xb, yb).predict(Xte),
        "pruned tree": lambda Xb, yb: CART("regression").fit(Xb, yb).predict(Xte, alpha=a_best),
        "bagged trees (50)": lambda Xb, yb: Forest("regression", 50, seed=int(rng.integers(1e9)))
            .fit(Xb, yb).predict(Xte),
        "random forest (50)": lambda Xb, yb: Forest("regression", 50, max_features=rfres["rf_max_features"],
                                                    min_samples_leaf=rfres["rf_min_leaf"],
                                                    seed=int(rng.integers(1e9))).fit(Xb, yb).predict(Xte),
        "gradient boosting": lambda Xb, yb: GradientBoosting("squared", gbb["M"], gbb["learning_rate"],
                                                             gbb["max_depth"], 5).fit(Xb, yb).predict(Xte),
    }
    P = {k: [] for k in makers}
    for _ in range(R):
        pick = rng.choice(ureq, len(ureq), replace=True)
        rows = np.concatenate([np.where(groups == q)[0] for q in pick])
        for k, f in makers.items():
            P[k].append(f(X[rows], y[rows]))
    rows = []
    for k, v in P.items():
        v = np.array(v)
        var = float(np.mean(v.var(0)))
        b2n = mse(v.mean(0), yte)          # bias^2 + irreducible noise
        rows.append({"model": k, "variance": var, "bias2_plus_noise": b2n, "expected_mse": var + b2n})
    df = pd.DataFrame(rows)
    m["R3_bias_variance"] = {"replicates": R, "resampling": "bootstrap of the 24 training requirements",
                             "rows": rows}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(df.model, df.bias2_plus_noise, color=LIGHT, edgecolor=MUTED, label="bias² + noise")
    ax.bar(df.model, df.variance, bottom=df.bias2_plus_noise, color=ORANGE, label="variance")
    ax.set_ylabel("expected test MSE"); ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.set_title(f"U5_03  Bias–variance over {R} bootstrap training sets"); ax.legend(fontsize=8)
    save(fig, FIG, "U5_03_bias_variance")
    md += [f"## R3 Bias–variance ({R} bootstrap resamples of the 24 training requirements)", "",
           md_table(df.assign(**{c: df[c].map(lambda v: f"{v:.2f}") for c in
                                 ["variance", "bias2_plus_noise", "expected_mse"]})), "",
           "- variance = mean over test pairs of Var(prediction across training sets); "
           "bias² + noise = MSE of the average prediction (noise cannot be separated from bias "
           "without repeated scores of the same pair).", ""]


# ======================================================================== R5 GAM
def gam_terms(kind="additive"):
    t = [gam.Term(FI["distance_km"], n_segments=12, name="s(distance_km)"),
         gam.Term(FI["n_categories"], n_segments=4, name="s(n_categories)"),
         gam.Term(FI["within_radius"], "linear", name="within_radius"),
         gam.Term(FI["category_match"], "linear", name="category_match"),
         gam.Term(FI["keyword_in_title"], "linear", name="keyword_in_title"),
         gam.Term(FI["is_chain"], "linear", name="is_chain")]
    if kind == "interaction":
        t += [gam.Term(FI["within_radius"], "linear", by=FI["category_match"],
                       name="within_radius × category_match"),
              gam.Term(FI["distance_km"], n_segments=12, by=FI["category_match"],
                       name="s(distance_km) × category_match")]
    return t


def r5_gam(r, md, m):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    folds = gfolds(r)
    lams = np.logspace(-2, 4, 25)
    res = {}
    fits = {}
    for kind in ["additive", "interaction"]:
        gcv, cv = [], []
        for lam in lams:
            gcv.append(gam.GAM(gam_terms(kind), lam).fit(X, y).gcv)
            cv.append(np.mean([mse(gam.GAM(gam_terms(kind), lam).fit(X[tr], y[tr]).predict(X[va]), y[va])
                               for tr, va in folds]))
        lam_cv, lam_gcv = float(lams[int(np.argmin(cv))]), float(lams[int(np.argmin(gcv))])
        model = gam.GAM(gam_terms(kind), lam_cv).fit(X, y)
        bf = gam.GAM(gam_terms(kind), lam_cv)
        hist = bf.backfit(X, y, tol=1e-9, max_iter=5000)
        fits[kind] = (model, gcv, cv)
        res[kind] = {"lambda_grouped_cv": lam_cv, "lambda_gcv": lam_gcv, "edf": model.edf,
                     "term_edf": {t.name: e for t, e in zip(model.terms, model.term_edf)},
                     "train": reg_metrics(y, model.predict(X)), "test": reg_metrics(yte, model.predict(Xte)),
                     "backfit_sweeps": len(hist),
                     "backfit_vs_direct_max_abs_diff": float(np.max(np.abs(bf.predict(X) - model.predict(X))))}
        PRED_R["GAM (additive)" if kind == "additive" else "GAM + interaction terms"] = model.predict(Xte)
    m["R5_gam"] = res
    model = fits["additive"][0]
    mi = fits["interaction"][0]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    xs = np.linspace(0, X[:, FI["distance_km"]].max(), 200)
    Z = np.tile(np.median(X, 0), (200, 1)); Z[:, FI["distance_km"]] = xs
    ax[0].plot(xs, model.partial(Z, 0), color=BLUE, lw=2, label="additive GAM")
    for cmv, col, lab in [(0, MUTED, "GA²M, other category"), (1, ORANGE, "GA²M, needed category")]:
        Z2 = Z.copy(); Z2[:, FI["category_match"]] = cmv
        ax[0].plot(xs, mi.partial(Z2, 0) + mi.partial(Z2, 7), color=col, lw=1.6, ls="--", label=lab)
    ax[0].set_xlabel("distance_km"); ax[0].set_ylabel("f(distance), centred")
    ax[0].set_title("U5_06a  Smooth distance effect"); ax[0].legend(fontsize=7)
    nc = np.arange(1, 11)
    Z = np.tile(np.median(X, 0), (10, 1)); Z[:, FI["n_categories"]] = nc
    ax[1].plot(nc, model.partial(Z, 1), "o-", color=BLUE)
    ax[1].set_xlabel("n_categories"); ax[1].set_ylabel("f(n_categories), centred")
    ax[1].set_title("U5_06b  Smooth n_categories effect")
    for kind, col in [("additive", BLUE), ("interaction", ORANGE)]:
        _, gcv, cv = fits[kind]
        ax[2].plot(lams, np.sqrt(cv), color=col, label=f"{kind}: grouped CV")
        ax[2].plot(lams, np.sqrt(gcv), color=col, ls=":", label=f"{kind}: GCV")
    ax[2].set_xscale("log"); ax[2].set_xlabel("λ (roughness penalty)"); ax[2].set_ylabel("RMSE")
    ax[2].set_title("U5_06c  Choosing λ"); ax[2].legend(fontsize=7)
    save(fig, FIG, "U5_06_gam")
    md += ["## R5 Generalised additive model (own P-splines, backfitting)", "",
           md_table(pd.DataFrame([
               metric_row("GAM (additive)", res["additive"]["test"],
                          f"λ = {res['additive']['lambda_grouped_cv']:.3g} (GCV: {res['additive']['lambda_gcv']:.3g}), "
                          f"edf {res['additive']['edf']:.2f}"),
               metric_row("GAM + interaction terms (GA²M)", res["interaction"]["test"],
                          f"λ = {res['interaction']['lambda_grouped_cv']:.3g}, edf {res['interaction']['edf']:.2f}")])), "",
           "- Effective degrees of freedom per term (additive): " + ", ".join(
               f"{k} {v:.2f}" for k, v in res["additive"]["term_edf"].items()),
           "- Effective degrees of freedom per term (interaction): " + ", ".join(
               f"{k} {v:.2f}" for k, v in res["interaction"]["term_edf"].items()),
           f"- Backfitting reaches the direct penalised-LS solution: additive {res['additive']['backfit_sweeps']} sweeps, "
           f"max |Δ| {res['additive']['backfit_vs_direct_max_abs_diff']:.1e}; interaction "
           f"{res['interaction']['backfit_sweeps']} sweeps, max |Δ| {res['interaction']['backfit_vs_direct_max_abs_diff']:.1e}.", ""]


# ======================================================================== R6 BMA
def r6_bma(r, md, m):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    gp = bma.LinearBMA(evidence="gprior").fit(X, y, FEATURES)
    bic = bma.LinearBMA(evidence="bic").fit(X, y, FEATURES)
    top = gp.order[:8]
    rows = [{"rank": i + 1, "inputs": ", ".join(FEATURES[j] for j in gp.models[k]["subset"]) or "(none)",
             "k": gp.models[k]["k"], "R²": f4(gp.models[k]["r2"]), "p(M|D) g-prior": f4(gp.post[k]),
             "p(M|D) BIC": f4(bic.post[k])} for i, k in enumerate(top)]
    mu = gp.predict(Xte)
    lo, hi = gp.predictive_interval(Xte)
    ols = LinearRegression().fit(X, y)
    uniform = gp.model_predictions(Xte)[0].mean(0)
    top1 = gp.predict(Xte, top=1)
    PRED_R["BMA (g-prior, 64 models)"] = mu
    res = {"n_models": len(gp.models), "g": gp.g_,
           "inclusion_gprior": dict(zip(FEATURES, gp.inclusion)),
           "inclusion_bic": dict(zip(FEATURES, bic.inclusion)),
           "top_models": rows,
           "post_top1_gprior": float(gp.post[gp.order[0]]),
           "post_top1_bic": float(bic.post[bic.order[0]]),
           "n_models_for_99pct": int(np.searchsorted(np.cumsum(np.sort(gp.post)[::-1]), 0.99) + 1),
           "test_bma": reg_metrics(yte, mu), "test_bma_bic": reg_metrics(yte, bic.predict(Xte)),
           "test_top_model": reg_metrics(yte, top1), "test_full_ols": reg_metrics(yte, ols.predict(Xte)),
           "test_uniform_average_64": reg_metrics(yte, uniform),
           "interval95_coverage_test": float(np.mean((yte >= lo) & (yte <= hi))),
           "interval95_mean_width": float(np.mean(hi - lo))}
    # concentration of the posterior as the number of training requirements grows
    rng = np.random.default_rng(SEED)
    groups = r["groups"]
    ureq = np.unique(groups)
    conc = []
    for k in [2, 3, 4, 6, 8, 12, 16, 20, 24]:
        tops, effs, incl_kw = [], [], []
        for _ in range(30 if k < 24 else 1):
            pick = rng.choice(ureq, k, replace=False)
            rows_ = np.isin(groups, pick)
            b = bma.LinearBMA().fit(X[rows_], y[rows_], FEATURES)
            pst = b.post[b.post > 0]
            tops.append(b.post.max())
            effs.append(float(np.exp(-np.sum(pst * np.log(pst)))))
        conc.append({"requirements": k, "pairs": 15 * k, "top_model_post": float(np.mean(tops)),
                     "effective_n_models": float(np.mean(effs))})
    res["concentration"] = conc
    m["R6_bma"] = res
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    inc = pd.Series(res["inclusion_gprior"]).sort_values()
    ax[0].barh(inc.index, inc.values, color=BLUE, label="g-prior")
    ax[0].scatter(pd.Series(res["inclusion_bic"])[inc.index], inc.index, color=ORANGE, zorder=3,
                  label="BIC")
    ax[0].set_xlabel("posterior inclusion probability"); ax[0].legend(fontsize=8)
    ax[0].set_title("U5_07a  Which inputs does the data need?")
    cdf = pd.DataFrame(conc)
    ax[1].plot(cdf.pairs, cdf.top_model_post, "o-", color=BLUE, label="p(top model | D)")
    ax2 = ax[1].twinx()
    ax2.plot(cdf.pairs, cdf.effective_n_models, "s--", color=ORANGE, label="effective no. of models")
    ax2.set_ylabel("exp(entropy of p(M|D))", color=ORANGE); ax2.grid(False)
    ax[1].set_xlabel("training pairs (15 per requirement)"); ax[1].set_ylabel("p(top model | D)", color=BLUE)
    ax[1].set_title("U5_07b  BMA concentrates as N grows")
    save(fig, FIG, "U5_07_bma")
    md += ["## R6 Bayesian model averaging over all 2⁶ = 64 linear models", "",
           f"Zellner g-prior with g = n = {len(y)} (unit information); uniform prior over models.", "",
           md_table(pd.DataFrame(rows)), "",
           "- Posterior inclusion probabilities (g-prior): " + ", ".join(
               f"{k} {v:.4f}" for k, v in res["inclusion_gprior"].items()),
           f"- {res['n_models_for_99pct']} models hold 99% of the posterior mass; top model "
           f"{res['post_top1_gprior']:.4f} (BIC: {res['post_top1_bic']:.4f}).", "",
           md_table(pd.DataFrame([
               metric_row("BMA, g-prior (posterior mean)", res["test_bma"]),
               metric_row("BMA, BIC weights", res["test_bma_bic"]),
               metric_row("top model only", res["test_top_model"]),
               metric_row("full OLS (Phase 1)", res["test_full_ols"]),
               metric_row("uniform average of all 64 models", res["test_uniform_average_64"], "not BMA")])), "",
           f"- BMA 95% predictive intervals (Student-t mixture) cover {res['interval95_coverage_test']:.4f} "
           f"of test scores, mean width {res['interval95_mean_width']:.2f} points.", "",
           md_table(cdf.assign(top_model_post=cdf.top_model_post.map(f4),
                               effective_n_models=cdf.effective_n_models.map(lambda v: f"{v:.2f}"))), ""]


# ======================================================================== R7 mixtures
def r7_mixtures(r, md, m):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    Phi = np.column_stack([np.ones(len(X)), X]); Phi_te = np.column_stack([np.ones(len(Xte)), Xte])
    sc = StandardScaler().fit(X)
    G = np.column_stack([np.ones(len(X)), sc.transform(X)])
    G_te = np.column_stack([np.ones(len(Xte)), sc.transform(Xte)])
    ols = LinearRegression().fit(X, y)
    s2 = mse(ols.predict(X), y)
    ll_ols_te = float(np.mean(-0.5 * (np.log(2 * np.pi * s2) + (yte - ols.predict(Xte)) ** 2 / s2)))
    rows, mods = [], {}
    for K in [1, 2, 3, 4]:
        mm = mixture.mix_linreg(Phi, y, K, n_init=10, seed=SEED)
        mods[K] = mm
        mu = Phi_te @ mm["W"].T
        lp = np.log(mm["pi"]) - 0.5 * (np.log(2 * np.pi * mm["var"]) + (yte[:, None] - mu) ** 2 / mm["var"])
        rows.append({"K": K, "train loglik": mm["loglik"], "params": mm["n_params"], "BIC": mm["bic"],
                     "test loglik / pair": float(np.mean(logsumexp(lp, 1))),
                     "test RMSE of mean": reg_metrics(yte, mixture.mix_linreg_predict(mm, Phi_te))["RMSE"],
                     "EM iterations": len(mm["history"]),
                     "monotone": bool(np.all(np.diff(mm["history"]) > -1e-6))})
    df = pd.DataFrame(rows)
    Kb = int(df.loc[df.BIC.idxmin(), "K"])
    me = mixture.mixture_of_experts(Phi, G, y, n_init=10, seed=SEED)
    pred_me, g_te = mixture.moe_predict(me, Phi_te, G_te)
    mu_me = Phi_te @ me["W"].T
    lp_me = (np.log(np.column_stack([g_te, 1 - g_te]) + 1e-300)
             - 0.5 * (np.log(2 * np.pi * me["var"]) + (yte[:, None] - mu_me) ** 2 / me["var"]))
    PRED_R["Mixture of experts (2 linear experts)"] = pred_me
    m2 = mods[2]
    comp = []
    tr = r["train"].assign(r1=m2["resp"][:, 0])
    for k in range(2):
        comp.append({"component": k + 1, "π": f4(m2["pi"][k]), "s.d.": f"{np.sqrt(m2['var'][k]):.2f}",
                     **{f: f"{w:.2f}" for f, w in zip(["intercept"] + FEATURES, m2["W"][k])}})
    cellr = tr.groupby(["category_match", "within_radius"]).r1.mean()
    me_tr, g_tr = mixture.moe_predict(me, Phi, G)
    gate_by_cell = r["train"].assign(g=g_tr).groupby(["category_match", "within_radius"]).g.mean()
    res = {"mix_linreg": rows, "K_bic": Kb, "components_K2": comp,
           "K2_resp_comp1_by_cell": {str(k): float(v) for k, v in cellr.items()},
           "ols_test_loglik_per_pair": ll_ols_te,
           "moe": {"train_loglik": me["loglik"], "bic": me["bic"], "n_params": me["n_params"],
                   "test": reg_metrics(yte, pred_me),
                   "test_loglik_per_pair": float(np.mean(logsumexp(lp_me, 1))),
                   "expert_sd": [float(np.sqrt(v)) for v in me["var"]],
                   "experts": [{f: float(w) for f, w in zip(["intercept"] + FEATURES, me["W"][k])} for k in range(2)],
                   "gate_coef_standardised": {f: float(v) for f, v in zip(["intercept"] + FEATURES, me["v"])},
                   "gate_mean_by_cell": {str(k): float(v) for k, v in gate_by_cell.items()}}}
    m["R7_mixtures"] = res
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for K, col in zip([2, 3, 4], [BLUE, ORANGE, PURPLE]):
        h = mods[K]["history"]
        ax[0].plot(np.arange(1, len(h) + 1), h, color=col, label=f"K = {K}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("EM iteration"); ax[0].set_ylabel("training log-likelihood")
    ax[0].set_title("U5_08a  EM never decreases ln p"); ax[0].legend(fontsize=8)
    labs = [f"cat={'Y' if a else 'N'}\nrad={'Y' if b else 'N'}" for a, b in gate_by_cell.index]
    xx = np.arange(len(labs))
    ax[1].bar(xx - .2, cellr.values, .4, color=LIGHT, edgecolor=MUTED, label="mixture K=2: γ(comp. 1)")
    ax[1].bar(xx + .2, gate_by_cell.values, .4, color=BLUE, label="experts: gate π₁(x)")
    ax[1].set_xticks(xx, labs); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].set_title("U5_08b  Who explains each cell?")
    save(fig, FIG, "U5_08_mixtures")
    md += ["## R7 Conditional mixture models (own EM; residual s.d. floored at 1 point)", "",
           md_table(df.assign(**{"train loglik": df["train loglik"].map(lambda v: f"{v:.1f}"),
                                 "BIC": df.BIC.map(lambda v: f"{v:.1f}"),
                                 "test loglik / pair": df["test loglik / pair"].map(f4),
                                 "test RMSE of mean": df["test RMSE of mean"].map(f4)})), "",
           f"- BIC picks K = {Kb}. OLS (Gaussian MLE) test log-likelihood per pair: {ll_ols_te:.4f}.",
           "- Mixture of 2 linear regressions, components:", "", md_table(pd.DataFrame(comp)), "",
           "- Mean responsibility of component 1 by cell: " + ", ".join(f"{k}: {v:.3f}" for k, v in cellr.items()),
           "", f"- Mixture of experts (2 linear experts, logistic gate on standardised inputs): test "
           f"RMSE {f4(res['moe']['test']['RMSE'])}, R² {f4(res['moe']['test']['R2'])}, test loglik / pair "
           f"{res['moe']['test_loglik_per_pair']:.4f}; expert s.d. "
           f"{res['moe']['expert_sd'][0]:.2f} / {res['moe']['expert_sd'][1]:.2f}; BIC {me['bic']:.1f}.",
           "- Gate π₁(x) by cell: " + ", ".join(f"{k}: {v:.3f}" for k, v in gate_by_cell.items()), ""]


# ======================================================================== R8 stacking
def r8_stacking(r, md, m, a_best, rfres, gbres, lam_add):
    X, y, Xte, yte = r["Xn_tr"], r["y_tr"], r["Xn_te"], r["y_te"]
    folds = gfolds(r)
    gbb = gbres["best"]
    base = {
        "OLS": lambda A, b: LinearRegression().fit(A, b).predict,
        "Kernel ridge RBF": lambda A, b: make_pipeline(StandardScaler(), KernelRidge(
            kernel="rbf", alpha=0.1, gamma=0.0316227766017)).fit(A, b).predict,
        "GAM": lambda A, b: gam.GAM(gam_terms(), lam_add).fit(A, b).predict,
        "CART pruned": lambda A, b: (lambda t: (lambda Z: t.predict(Z, alpha=a_best)))(CART("regression").fit(A, b)),
        "Random forest": lambda A, b: Forest("regression", 200, max_features=rfres["rf_max_features"],
                                             min_samples_leaf=rfres["rf_min_leaf"], seed=SEED).fit(A, b).predict,
        "Gradient boosting": lambda A, b: GradientBoosting("squared", gbb["M"], gbb["learning_rate"],
                                                           gbb["max_depth"], 5).fit(A, b).predict,
    }
    names = list(base)
    Z = np.zeros((len(y), len(names)))
    for tr, va in folds:
        for j, k in enumerate(names):
            Z[va, j] = base[k](X[tr], y[tr])(X[va])
    Zte = np.column_stack([base[k](X, y)(Xte) for k in names])
    PRED_R["Kernel ridge RBF (Phase 1)"] = Zte[:, 1]
    PRED_R["OLS (Phase 1)"] = Zte[:, 0]
    w_nnls, _ = nnls(Z, y)
    lin = LinearRegression().fit(Z, y)
    simple = Zte.mean(1)
    stack = Zte @ w_nnls
    PRED_R["Stacking (NNLS)"] = stack
    PRED_R["Simple average (6 models)"] = simple
    res = {"base_models": names,
           "oof_rmse": {k: float(np.sqrt(mse(Z[:, j], y))) for j, k in enumerate(names)},
           "nnls_weights": dict(zip(names, map(float, w_nnls))), "nnls_weight_sum": float(w_nnls.sum()),
           "unconstrained_weights": dict(zip(names, map(float, lin.coef_))),
           "unconstrained_intercept": float(lin.intercept_),
           "test_stack_nnls": reg_metrics(yte, stack),
           "test_stack_unconstrained": reg_metrics(yte, lin.predict(Zte)),
           "test_simple_average": reg_metrics(yte, simple),
           "test_base": {k: reg_metrics(yte, Zte[:, j]) for j, k in enumerate(names)},
           "oof_stack_nnls_rmse": float(np.sqrt(mse(Z @ w_nnls, y)))}
    m["R8_stacking"] = res
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(names[::-1], w_nnls[::-1], color=BLUE, label="NNLS stacking weight")
    ax.scatter(lin.coef_[::-1], names[::-1], color=ORANGE, zorder=3, label="unconstrained LS weight")
    ax.axvline(0, color=INK2, lw=.8)
    ax.set_xlabel("weight on out-of-fold prediction"); ax.legend(fontsize=8)
    ax.set_title("U5_10  Stacked regression: who gets the weight?")
    save(fig, FIG, "U5_10_stacking")
    md += ["## R8 Stacking (level-0 out-of-fold predictions by grouped 4-fold CV; level-1 NNLS)", "",
           md_table(pd.DataFrame({"base model": names,
                                  "OOF RMSE": [f4(res["oof_rmse"][k]) for k in names],
                                  "test R²": [f4(res["test_base"][k]["R2"]) for k in names],
                                  "NNLS weight": [f4(res["nnls_weights"][k]) for k in names],
                                  "unconstrained weight": [f4(res["unconstrained_weights"][k]) for k in names]})), "",
           md_table(pd.DataFrame([
               metric_row("stacking, NNLS (Breiman 1996)", res["test_stack_nnls"], f"weights sum {w_nnls.sum():.3f}"),
               metric_row("stacking, unconstrained LS", res["test_stack_unconstrained"]),
               metric_row("simple average of the 6", res["test_simple_average"])])), ""]


# ======================================================================== classification
def c1_cart(c, md, m):
    X, y, Xte, yte = c["Xn_tr"], c["y_tr"], c["Xn_te"], c["y_te"]
    full = CART("classification").fit(X, y)
    alphas, _, masks = full.pruning_path()
    cand = np.unique(np.r_[0.0, np.sqrt(alphas[1:-1] * alphas[2:])])
    folds = sfolds(c)
    fts = [CART("classification").fit(X[tr], y[tr]) for tr, _ in folds]
    cv = np.array([[np.mean(t.predict(X[va], alpha=a) != y[va]) for (_, va), t in zip(folds, fts)]
                   for a in cand])
    cvm, cvse = cv.mean(1), cv.std(1, ddof=1) / np.sqrt(cv.shape[1])
    k = int(np.argmin(cvm))
    k1 = int(np.max(np.where(cvm <= cvm[k] + cvse[k])[0]))
    a_best, a_1se = float(cand[k]), float(cand[k1])
    p_full = full.predict_proba(Xte)[:, 1]
    p_best = full.predict_proba(Xte, alpha=a_best)[:, 1]
    p_1se = full.predict_proba(Xte, alpha=a_1se)[:, 1]
    PRED_C["CART (pruned, CV)"] = p_best
    sk = DecisionTreeClassifier(random_state=0).fit(X, y)
    res = {"full": {"leaves": full.n_leaves(), "depth": int(full.depth_.max()), "test": clf_metrics(yte, p_full)},
           "alpha_best": a_best, "alpha_1se": a_1se,
           "leaves_best": full.n_leaves(full._mask_for(a_best)), "leaves_1se": full.n_leaves(full._mask_for(a_1se)),
           "test_best": clf_metrics(yte, p_best), "test_1se": clf_metrics(yte, p_1se),
           "sklearn_full_tree_nodes": int(sk.tree_.node_count), "own_full_tree_nodes": int(full.n_nodes),
           "sklearn_full_test_prediction_agreement": float(np.mean(sk.predict(Xte) == full.predict(Xte)))}
    text = full.describe(c["names"], max_depth=3, collapsed=full._mask_for(a_best))
    m["C1_cart"] = res
    md += ["# Pharmacy classifier (3,689 train / 923 test vendors, 441 binary columns)", "",
           "## C1 CART classification tree (own, Gini), pruned by 5-fold CV misclassification", "",
           md_table(pd.DataFrame([
               clf_row("full-depth tree", res["full"]["test"], f"{res['full']['leaves']} leaves, depth {res['full']['depth']}"),
               clf_row("pruned, min CV", res["test_best"], f"{res['leaves_best']} leaves"),
               clf_row("pruned, 1-SE", res["test_1se"], f"{res['leaves_1se']} leaves")])), "",
           f"- Own vs scikit-learn full tree: {res['own_full_tree_nodes']} vs {res['sklearn_full_tree_nodes']} nodes; "
           f"test predictions agree on {res['sklearn_full_test_prediction_agreement']:.4f}.",
           "", "Top of the pruned tree:", "", "```", text, "```", ""]


def c2_forest(c, md, m):
    X, y, Xte, yte = c["Xn_tr"], c["y_tr"], c["Xn_te"], c["y_te"]
    p = X.shape[1]
    grid = []
    for mf in [int(np.sqrt(p)), int(0.1 * p), int(0.3 * p), p]:
        f = Forest("classification", 200, max_features=mf, seed=SEED).fit(X, y)
        o = f.oob_prediction_
        ok = ~np.isnan(o)
        grid.append({"max_features": mf, "oob_log_loss": logloss(y[ok], o[ok]),
                     "oob_f1": clf_metrics(y[ok], o[ok])["f1"]})
    g = pd.DataFrame(grid)
    mbest = int(g.loc[g[g.max_features < p].oob_log_loss.idxmin(), "max_features"])
    rf = Forest("classification", 500, max_features=mbest, seed=SEED).fit(X, y)
    bag = Forest("classification", 300, max_features=None, seed=SEED).fit(X, y)
    p_rf, p_bag = rf.predict(Xte), bag.predict(Xte)
    PRED_C["Random forest (own)"] = p_rf
    PRED_C["Bagged trees (own)"] = p_bag
    imp = rf.feature_importances()
    top = np.argsort(-imp)[:15]
    sk = RandomForestClassifier(500, max_features=mbest, random_state=SEED).fit(X, y)
    res = {"grid": grid, "max_features": mbest, "test_rf": clf_metrics(yte, p_rf),
           "test_bag": clf_metrics(yte, p_bag), "sklearn_rf_test": clf_metrics(yte, sk.predict_proba(Xte)[:, 1]),
           "top_importance": {c["names"][k]: float(imp[k]) for k in top}}
    m["C2_forest"] = res
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([c["names"][k].replace("tok=", "") for k in top][::-1], imp[top][::-1], color=BLUE)
    ax.set_xlabel("mean decrease in Gini impurity"); ax.set_title("U5_11  Random forest: what decides 'pharmacy'")
    save(fig, FIG, "U5_11_rf_importance")
    md += ["## C2 Bagging and random forest (own)", "",
           md_table(g.assign(oob_log_loss=g.oob_log_loss.map(f4), oob_f1=g.oob_f1.map(f4))), "",
           md_table(pd.DataFrame([
               clf_row("bagged trees (m = 441)", res["test_bag"], "300 trees"),
               clf_row(f"random forest (m = {mbest})", res["test_rf"], "500 trees, m by OOB log loss"),
               clf_row("scikit-learn RF (same settings)", res["sklearn_rf_test"], "cross-check")])), "",
           "- Top importance: " + ", ".join(f"{k.replace('tok=', '')} {v:.3f}" for k, v in res["top_importance"].items()), ""]
    return mbest


def c3_adaboost(c, md, m):
    X, y, Xte, yte = c["Xn_tr"], c["y_tr"], c["Xn_te"], c["y_te"]
    MMAX = 1000
    cv = np.zeros(MMAX)
    cvl = np.zeros(MMAX)
    for tr, va in sfolds(c):
        ab = AdaBoost(MMAX).fit(X[tr], y[tr])
        S = ab.staged_f(X[va])
        S = np.vstack([S, np.repeat(S[-1:], MMAX - len(S), 0)]) if len(S) < MMAX else S
        cv += np.mean((S > 0).astype(int) != y[va], 1) / 5
        cvl += np.array([logloss(y[va], 1 / (1 + np.exp(-2 * s))) for s in S]) / 5
    M = int(np.argmin(cv) + 1)
    ab = AdaBoost(MMAX).fit(X, y)
    St = ab.staged_f(Xte)
    p = 1 / (1 + np.exp(-2 * St[M - 1]))
    PRED_C["AdaBoost, stumps (own)"] = p
    used = sorted({c["names"][j] for j, _, _ in ab.stumps[:M] if j >= 0})
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=M, random_state=SEED).fit(X, y)
    res = {"M_cv_error": M, "cv_error_at_M": float(cv[M - 1]), "M_cv_logloss": int(np.argmin(cvl) + 1),
           "test": clf_metrics(yte, p), "distinct_words_used": len(used),
           "first_10_stumps": [{"column": c["names"][j], "alpha": float(a), "eps": float(e), "sign": int(s)}
                               for (j, _, s), a, e in zip(ab.stumps[:10], ab.alphas[:10], ab.eps[:10])],
           "train_error_at_M": ab.train_error[M - 1], "exp_loss_at_M": ab.exp_loss[M - 1],
           "sklearn_samme_test": clf_metrics(yte, sk.predict_proba(Xte)[:, 1])}
    m["C3_adaboost"] = res
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ms = np.arange(1, len(ab.exp_loss) + 1)
    ax[0].plot(ms, ab.exp_loss, color=BLUE, label="exponential error E/N = Π 2√(ε(1−ε))")
    ax[0].plot(ms, ab.train_error, color=MUTED, label="training error")
    ax[0].plot(ms, np.mean((St > 0).astype(int) != yte, 1), color=ORANGE, label="test error")
    ax[0].plot(np.arange(1, MMAX + 1), cv, color=AQUA, ls="--", label="5-fold CV error")
    ax[0].axvline(M, color=INK2, ls=":", label=f"M by CV = {M}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("rounds M (log)")
    ax[0].set_title("U5_09a  AdaBoost with decision stumps"); ax[0].legend(fontsize=7)
    ax[1].plot(ms, ab.eps, color=PURPLE, lw=.8)
    ax[1].axhline(0.5, color=ORANGE, ls="--", label="ε = 0.5 (no better than chance)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("round m (log)"); ax[1].set_ylabel("weighted error ε_m")
    ax[1].set_title("U5_09b  Later stumps face harder weights"); ax[1].legend(fontsize=8)
    save(fig, FIG, "U5_09_adaboost")
    md += ["## C3 AdaBoost (own, decision stumps minimising weighted error; Bishop 14.3)", "",
           md_table(pd.DataFrame([clf_row(f"AdaBoost, M = {M} (5-fold CV error)", res["test"],
                                          f"{len(used)} distinct columns used"),
                                  clf_row("scikit-learn AdaBoost (SAMME, Gini stumps)", res["sklearn_samme_test"],
                                          "cross-check")])), "",
           f"- CV picks M = {M} by error (CV error {cv[M - 1]:.4f}) and M = {res['M_cv_logloss']} by log loss.",
           "- First 10 rounds: " + "; ".join(f"{d['column'].replace('tok=', '')} (ε {d['eps']:.3f}, α {d['alpha']:.2f})"
                                             for d in res["first_10_stumps"]), ""]
    return M


def c4_gboost(c, md, m):
    X, y, Xte, yte = c["Xn_tr"], c["y_tr"], c["Xn_te"], c["y_te"]
    MMAX = 500
    grid, curves = [], {}
    for depth in [1, 2, 3]:
        S = np.zeros(MMAX)
        for tr, va in sfolds(c):
            gb = GradientBoosting("logistic", MMAX, 0.1, depth).fit(X[tr], y[tr])
            F = gb.staged_raw(X[va])
            S += np.mean(np.logaddexp(0, F) - y[va] * F, 1) / 5
        kb = int(np.argmin(S))
        grid.append({"max_depth": depth, "best_M": kb + 1, "cv_log_loss": float(S[kb])})
        curves[depth] = S
    g = pd.DataFrame(grid)
    b = g.loc[g.cv_log_loss.idxmin()]
    depth, M = int(b.max_depth), int(b.best_M)
    gb = GradientBoosting("logistic", M, 0.1, depth).fit(X, y)
    p = gb.predict(Xte)
    PRED_C["Gradient boosting (own)"] = p
    sk = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1, max_depth=depth, random_state=0).fit(X, y)
    psk = sk.predict_proba(Xte)[:, 1]
    res = {"grid": grid, "max_depth": depth, "M": M, "test": clf_metrics(yte, p),
           "sklearn_test": clf_metrics(yte, psk), "sklearn_max_abs_prob_diff": float(np.max(np.abs(p - psk)))}
    m["C4_gboost"] = res
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for d, col in zip([1, 2, 3], [MUTED, BLUE, ORANGE]):
        ax.plot(np.arange(1, MMAX + 1), curves[d], color=col, label=f"depth {d}")
    ax.set_xscale("log"); ax.set_xlabel("rounds M (log)"); ax.set_ylabel("5-fold CV log loss")
    ax.set_title("U5_12  Logistic gradient boosting (ν = 0.1)"); ax.legend(fontsize=8)
    save(fig, FIG, "U5_12_gb_classification")
    md += ["## C4 Gradient boosting, logistic loss (own, Newton leaf values, ν = 0.1)", "",
           md_table(g.assign(cv_log_loss=g.cv_log_loss.map(f4))), "",
           md_table(pd.DataFrame([clf_row(f"GB depth {depth}, M {M}", res["test"]),
                                  clf_row("scikit-learn GB (same settings)", res["sklearn_test"],
                                          f"max |Δp| {res['sklearn_max_abs_prob_diff']:.2e}")])), ""]
    return depth, M


def c5_bma(c, md, m):
    tr, te = c["train"], c["test"]
    tokens = c["small"]["tokens"]
    X = L.token_indicators(tr, tokens); Xte = L.token_indicators(te, tokens)
    y, yte = c["y_tr"], c["y_te"]
    lb = bma.LogisticBMA(lambda A, b: L.irls(A, b, 0.0, 100)[0]).fit(X, y, tokens)
    p = lb.predict_proba(Xte)
    PRED_C["Logistic BMA (1,024 small models)"] = p
    top = lb.order[:6]
    rows = [{"rank": i + 1, "words": ", ".join(tokens[j] for j in lb.models[k]["subset"]),
             "k": lb.models[k]["k"], "p(M|D)": f4(lb.post[k])} for i, k in enumerate(top)]
    full = [mm for mm in lb.models if mm["k"] == len(tokens)][0]
    res = {"n_models": len(lb.models), "inclusion": dict(zip(tokens, lb.inclusion)),
           "top_models": rows, "post_top1": float(lb.post[lb.order[0]]),
           "test_bma": clf_metrics(yte, p),
           "test_top_model": clf_metrics(yte, lb.model_probs(Xte)[lb.order[0]]),
           "test_full_small_mle": clf_metrics(yte, lb.model_probs(Xte)[lb.models.index(full)])}
    m["C5_logistic_bma"] = res
    md += ["## C5 Logistic BMA over all 2¹⁰ = 1,024 subsets of the 10-word MLE model (BIC evidence)", "",
           md_table(pd.DataFrame(rows)), "",
           "- Inclusion probabilities: " + ", ".join(f"{k} {v:.3f}" for k, v in res["inclusion"].items()), "",
           md_table(pd.DataFrame([clf_row("logistic BMA", res["test_bma"]),
                                  clf_row("top model only", res["test_top_model"]),
                                  clf_row("10-word MLE (Phase 1 small model)", res["test_full_small_mle"])])), ""]


def c6_stacking(c, md, m, mbest, M_ab, gbd, gbM):
    X, y, Xte, yte = c["Xn_tr"], c["y_tr"], c["Xn_te"], c["y_te"]
    makers = {
        "Logistic L2": lambda A, b: (lambda mdl: lambda Z: mdl.predict_proba(Z)[:, 1])(
            L.sk_logreg(c["best_c"], max_iter=10000, tol=1e-10).fit(A, b)),   # Phase 1 settings
        "Naive Bayes": lambda A, b: (lambda mdl: lambda Z: mdl.predict_proba(Z)[:, 1])(BernoulliNB(alpha=1.0).fit(A, b)),
        "Random forest": lambda A, b: Forest("classification", 200, max_features=mbest, seed=SEED).fit(A, b).predict,
        "AdaBoost": lambda A, b: (lambda ab: lambda Z: 1 / (1 + np.exp(-2 * ab.staged_f(Z)[min(M_ab, len(ab.alphas)) - 1])))(
            AdaBoost(M_ab).fit(A, b)),
        "Gradient boosting": lambda A, b: GradientBoosting("logistic", gbM, 0.1, gbd).fit(A, b).predict,
    }
    names = list(makers)
    Z = np.zeros((len(y), len(names)))
    for tr, va in sfolds(c):
        for j, k in enumerate(names):
            Z[va, j] = makers[k](X[tr], y[tr])(X[va])
    Zte = np.column_stack([makers[k](X, y)(Xte) for k in names])
    PRED_C["Logistic L2 (Phase 1)"] = Zte[:, 0]
    PRED_C["Naive Bayes (Phase 1)"] = Zte[:, 1]
    logit = lambda P: np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))  # noqa: E731
    meta = LogisticRegression(C=1.0, max_iter=5000).fit(logit(Z), y)
    p_stack = meta.predict_proba(logit(Zte))[:, 1]
    p_vote = Zte.mean(1)
    PRED_C["Stacking (logistic meta)"] = p_stack
    PRED_C["Soft voting (5 models)"] = p_vote
    res = {"base_models": names, "meta_coef": dict(zip(names, map(float, meta.coef_.ravel()))),
           "meta_intercept": float(meta.intercept_[0]),
           "oof_log_loss": {k: logloss(y, Z[:, j]) for j, k in enumerate(names)},
           "test_base": {k: clf_metrics(yte, Zte[:, j]) for j, k in enumerate(names)},
           "test_stack": clf_metrics(yte, p_stack), "test_vote": clf_metrics(yte, p_vote),
           "oof_error_corr": {f"{names[i]} | {names[j]}": float(np.corrcoef(y - Z[:, i], y - Z[:, j])[0, 1])
                              for i in range(len(names)) for j in range(i + 1, len(names))}}
    m["C6_stacking"] = res
    md += ["## C6 Stacking (5-fold OOF; logistic regression on base-model logits) and soft voting", "",
           md_table(pd.DataFrame({"base model": names,
                                  "OOF log loss": [f4(res["oof_log_loss"][k]) for k in names],
                                  "test F1": [f4(res["test_base"][k]["f1"]) for k in names],
                                  "meta weight (logit)": [f4(res["meta_coef"][k]) for k in names]})), "",
           md_table(pd.DataFrame([clf_row("stacking", res["test_stack"]),
                                  clf_row("soft voting (mean probability)", res["test_vote"])])), ""]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    r = regression_data()
    c = classification_data()
    m = {"note": "Regression target: AI-judged expected_score (not ground truth). Classification "
                 "target: weak is_pharmacy label. Seed 42; all hyperparameters by CV on train."}
    md = ["# Unit V numbers (generated by `python src/pbl3_unit5.py`; do not edit by hand)", "",
          f"Match score: {len(r['y_tr'])} train / {len(r['y_te'])} test pairs (6 unseen requirements). "
          f"Pharmacy: {len(c['y_tr'])} / {len(c['y_te'])} vendors. Hyperparameters by CV on the training "
          "split only (grouped by requirement for the score). Seed 42.", "",
          "# Match-score regression", ""]
    tick("R1 CART"); a_best = r1_cart(r, md, m)
    tick("R2 forests"); rfres = r2_forests(r, md, m)
    tick("R4 boosting"); gbres = r4_boosting(r, md, m)
    tick("R3 bias-variance"); r3_bias_variance(r, md, m, a_best, rfres, gbres)
    tick("R5 GAM"); r5_gam(r, md, m)
    tick("R6 BMA"); r6_bma(r, md, m)
    tick("R7 mixtures"); r7_mixtures(r, md, m)
    tick("R8 stacking"); r8_stacking(r, md, m, a_best, rfres, gbres, m["R5_gam"]["additive"]["lambda_grouped_cv"])
    tick("C1 CART"); c1_cart(c, md, m)
    tick("C2 forest"); mbest = c2_forest(c, md, m)
    tick("C3 AdaBoost"); M_ab = c3_adaboost(c, md, m)
    tick("C4 GB"); gbd, gbM = c4_gboost(c, md, m)
    tick("C5 BMA"); c5_bma(c, md, m)
    tick("C6 stacking"); c6_stacking(c, md, m, mbest, M_ab, gbd, gbM)
    m["test_summary_regression"] = {k: reg_metrics(r["y_te"], v) for k, v in PRED_R.items()}
    m["test_summary_classification"] = {k: clf_metrics(c["y_te"], v) for k, v in PRED_C.items()}
    m["runtime_seconds"] = time.time() - T0
    write_json(METRICS, m)
    np.savez(PRED, reg_names=np.array(list(PRED_R)), reg_pred=np.array(list(PRED_R.values())),
             reg_y=r["y_te"], reg_req=r["test"].requirement_id.to_numpy().astype(str),
             reg_pair=r["test"].pair_id.to_numpy().astype(str),
             clf_names=np.array(list(PRED_C)), clf_prob=np.array(list(PRED_C.values())),
             clf_y=c["y_te"], clf_vendor=c["test"].vendor_id.to_numpy().astype(str))
    (OUT / "unit5_numbers.md").write_text("\n".join(md), encoding="utf-8")
    tick(f"wrote {METRICS}, {PRED}, {OUT / 'unit5_numbers.md'}")
    for k, v in m["test_summary_regression"].items():
        print(f"  {k:40s} RMSE {v['RMSE']:.4f} R2 {v['R2']:.4f}")
    for k, v in m["test_summary_classification"].items():
        print(f"  {k:40s} F1 {v['f1']:.4f} AUC {v['roc_auc']:.4f} PR {v['pr_auc']:.4f} LL {v['log_loss']:.4f}")


if __name__ == "__main__":
    main()
