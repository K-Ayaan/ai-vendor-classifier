"""PBL-3 EDA: is there anything for combined models (Unit V) to gain on this data?

Usage: python src/pbl3_eda.py
Writes pbl-3/02_eda_insights/figures/E01-E08.png and pbl-3/02_eda_insights/eda_numbers.md.
Every target-related number uses the TRAINING split only (360 pairs / 3,689 vendors).

E01 interaction: category_match x within_radius cell means vs the additive (OLS) fit
E02 thresholds: score against distance / radius, in and out of category
E03 structured residuals: out-of-fold OLS residuals by cell and by sampling stratum
E04 diversity: correlation of out-of-fold errors of different model families
E05 instability: the same tree refitted on bootstrap resamples
E06 weak learners: how good is one word on its own? (single-word stumps)
E07 coverage: how many pharmacies does the best k words reach?
E08 noise: score distribution and spread per cell (one Gaussian or several?)
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble.boosting import fit_stump
from ensemble.tree import CART
from pbl3_style import (AQUA, BLUE, FEATURES, INK2, LIGHT, MUTED, ORANGE, PBL3, PURPLE, SEED,
                        YELLOW, classification_data, f4, md_table, plt, regression_data, save)

OUT = PBL3 / "02_eda_insights"
FIG = OUT / "figures"
N_BOOT = 200


def cells(df):
    return df.groupby(["category_match", "within_radius"]).y


# ---------------------------------------------------------------- E01 interaction
def e01(r, md):
    tr = r["train"]
    g = cells(tr).agg(["mean", "std", "count"]).reset_index()
    ols = LinearRegression().fit(r["Xn_tr"], r["y_tr"])
    tr = tr.assign(ols=ols.predict(r["Xn_tr"]))
    g["ols_mean"] = tr.groupby(["category_match", "within_radius"]).ols.mean().values
    m = {(bool(a), bool(b)): v for a, b, v in zip(g.category_match, g.within_radius, g["mean"])}
    radius_off = m[(False, True)] - m[(False, False)]
    radius_on = m[(True, True)] - m[(True, False)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for cm, col, lab in [(False, MUTED, "other category"), (True, BLUE, "needed category")]:
        sub = g[g.category_match == cm]
        ax[0].plot([0, 1], sub["mean"], "o-", color=col, lw=2.2, ms=8, label=f"{lab}: data")
        ax[0].plot([0, 1], sub["ols_mean"], "s--", color=col, alpha=.6, label=f"{lab}: additive OLS")
    ax[0].set_xticks([0, 1], ["outside radius", "within radius"])
    ax[0].set_ylabel("mean AI-judged score (train)")
    ax[0].set_title("E01a  Radius matters only for the right category")
    ax[0].legend(fontsize=8)
    ax[1].bar(["radius effect\nother category", "radius effect\nneeded category"],
              [radius_off, radius_on], color=[MUTED, BLUE])
    for i, v in enumerate([radius_off, radius_on]):
        ax[1].text(i, v + .8, f"+{v:.1f}", ha="center", fontweight="bold")
    ax[1].set_ylabel("score points")
    ax[1].set_title("E01b  Non-additive: one slope cannot fit both")
    save(fig, FIG, "E01_interaction")
    md += ["## E01 Interaction (train, 360 pairs)", "",
           md_table(pd.DataFrame({"category_match": g.category_match, "within_radius": g.within_radius,
                                  "n": g["count"], "mean score": g["mean"].map(lambda v: f"{v:.2f}"),
                                  "sd": g["std"].map(lambda v: f"{v:.2f}"),
                                  "additive OLS mean": g.ols_mean.map(lambda v: f"{v:.2f}")})), "",
           f"- Radius effect: +{radius_off:.2f} (other category) vs +{radius_on:.2f} (needed category). "
           f"Interaction contrast {radius_on - radius_off:.2f} points.", ""]
    return {"radius_effect_off": radius_off, "radius_effect_on": radius_on,
            "interaction_contrast": radius_on - radius_off}


# ---------------------------------------------------------------- E02 thresholds
def e02(r, md):
    tr = r["train"].copy()
    tr["ratio"] = tr.distance_km / tr.max_distance_km
    bins = [0, .25, .5, .75, 1, 1.5, 2, 4, 8, 100]
    tr["bin"] = pd.cut(tr.ratio, bins, include_lowest=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    rows = []
    for cm, col, lab in [(True, BLUE, "needed category"), (False, MUTED, "other category")]:
        sub = tr[tr.category_match == cm]
        ax[0].scatter(sub.distance_km, sub.y + np.random.default_rng(0).uniform(-1.5, 1.5, len(sub)),
                      s=10, color=col, alpha=.55, label=lab)
        b = sub.groupby("bin", observed=True).y.agg(["mean", "count"])
        mids = [min(iv.right, 10) for iv in b.index]
        ax[1].plot(mids, b["mean"], "o-", color=col, lw=2, label=lab)
        rows += [{"category": lab, "distance/radius": str(iv), "n": int(c), "mean": f"{mu:.2f}"}
                 for iv, mu, c in zip(b.index, b["mean"], b["count"])]
    ax[1].axvline(1, color=ORANGE, ls="--", lw=1.4)
    ax[1].text(1.05, 85, "radius", color=ORANGE)
    ax[0].set_xlabel("distance_km"); ax[0].set_ylabel("score (jittered)")
    ax[0].set_title("E02a  Score vs raw distance"); ax[0].legend(fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("distance / buyer's radius (log)")
    ax[1].set_ylabel("mean score"); ax[1].set_title("E02b  A step at the radius, not a line")
    ax[1].legend(fontsize=8)
    save(fig, FIG, "E02_thresholds")
    inside = tr[(tr.category_match) & (tr.ratio <= 1)].y.mean()
    just_out = tr[(tr.category_match) & (tr.ratio > 1) & (tr.ratio <= 2)].y.mean()
    far = tr[(tr.category_match) & (tr.ratio > 4)].y.mean()
    corr_lin = np.corrcoef(tr.distance_km, tr.y)[0, 1]
    md += ["## E02 Thresholds (train)", "", md_table(pd.DataFrame(rows)), "",
           f"- In category: mean {inside:.2f} inside the radius, {just_out:.2f} at 1–2× the radius, "
           f"{far:.2f} beyond 4×. corr(distance_km, score) = {corr_lin:.4f}.", ""]
    return {"in_cat_inside": inside, "in_cat_1_2x": just_out, "in_cat_beyond_4x": far,
            "corr_distance_score": corr_lin}


# ---------------------------------------------------------------- E03-E04 out-of-fold errors
def oof_regression(r):
    X, y, groups = r["Xn_tr"], r["y_tr"], r["groups"]
    models = {
        "OLS": lambda: LinearRegression(),
        "Huber": lambda: HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=5000),
        "Kernel ridge RBF": lambda: make_pipeline(StandardScaler(),
                                                  KernelRidge(kernel="rbf", alpha=0.1, gamma=0.0316)),
        "k-NN (k=10)": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
        "CART depth 3": None,
    }
    oof = {k: np.zeros(len(y)) for k in models}
    for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
        for k, mk in models.items():
            if mk is None:
                oof[k][va] = CART("regression", max_depth=3).fit(X[tr], y[tr]).predict(X[va])
            else:
                oof[k][va] = mk().fit(X[tr], y[tr]).predict(X[va])
    return oof


def e03(r, oof, md):
    tr = r["train"].assign(res=r["y_tr"] - oof["OLS"])
    by_cell = tr.groupby(["category_match", "within_radius"]).res.agg(["mean", "count"]).reset_index()
    by_str = tr.groupby("stratum").res.agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labs = [f"cat={'Y' if a else 'N'}, rad={'Y' if b else 'N'}"
            for a, b in zip(by_cell.category_match, by_cell.within_radius)]
    ax[0].bar(labs, by_cell["mean"], color=[MUTED, MUTED, BLUE, BLUE])
    ax[0].axhline(0, color=INK2, lw=.8)
    ax[0].set_ylabel("mean out-of-fold OLS residual")
    ax[0].set_title("E03a  Residuals are not centred in each cell")
    ax[0].tick_params(axis="x", labelsize=8)
    order = by_str.sort_values("mean").stratum
    data = [tr[tr.stratum == s].res for s in order]
    bp = ax[1].boxplot(data, tick_labels=[s.replace("_", "\n") for s in order], patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor(LIGHT)
    ax[1].axhline(0, color=ORANGE, lw=1)
    ax[1].set_ylabel("OLS residual"); ax[1].set_title("E03b  ...and differ by sampling stratum")
    ax[1].tick_params(axis="x", labelsize=8)
    save(fig, FIG, "E03_structured_residuals")
    md += ["## E03 Structured residuals (grouped 4-fold out-of-fold OLS, train)", "",
           md_table(pd.DataFrame({"category_match": by_cell.category_match,
                                  "within_radius": by_cell.within_radius, "n": by_cell["count"],
                                  "mean residual": by_cell["mean"].map(lambda v: f"{v:.2f}")})), "",
           md_table(pd.DataFrame({"stratum": by_str.stratum, "n": by_str["count"],
                                  "mean residual": by_str["mean"].map(lambda v: f"{v:.2f}"),
                                  "sd": by_str["std"].map(lambda v: f"{v:.2f}")})), ""]
    return {"cell_mean_residuals": dict(zip(labs, by_cell["mean"])),
            "stratum_mean_residuals": dict(zip(by_str.stratum, by_str["mean"]))}


def oof_classification(c):
    X, y = c["Xn_tr"], c["y_tr"]
    oof = {k: np.zeros(len(y)) for k in ["Logistic L2", "Naive Bayes", "CART depth 4"]}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        oof["Logistic L2"][va] = LogisticRegression(C=c["best_c"], max_iter=5000).fit(
            X[tr], y[tr]).predict_proba(X[va])[:, 1]
        oof["Naive Bayes"][va] = BernoulliNB(alpha=1.0).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        oof["CART depth 4"][va] = CART("classification", max_depth=4).fit(
            X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


def e04(r, oof_r, c, oof_c, md):
    y = r["y_tr"]
    names = list(oof_r)
    E = np.array([y - oof_r[k] for k in names])
    C = np.corrcoef(E)
    yc = c["y_tr"]
    cn = list(oof_c)
    Ec = np.array([yc - oof_c[k] for k in cn])
    Cc = np.corrcoef(Ec)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [5, 3]})
    for a, M, nm, t in [(ax[0], C, names, "E04a  Match score: error correlation"),
                        (ax[1], Cc, cn, "E04b  Pharmacy: error correlation")]:
        im = a.imshow(M, vmin=0, vmax=1, cmap="Blues")
        a.set_xticks(range(len(nm)), nm, rotation=35, ha="right", fontsize=8)
        a.set_yticks(range(len(nm)), nm, fontsize=8)
        a.grid(False)
        for i in range(len(nm)):
            for j in range(len(nm)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if M[i, j] > .6 else INK2)
        a.set_title(t)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(FIG / "E04_error_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    rmse = {k: float(np.sqrt(np.mean((y - oof_r[k]) ** 2))) for k in names}
    avg = np.mean([oof_r[k] for k in names], 0)
    rmse_avg = float(np.sqrt(np.mean((y - avg) ** 2)))
    off = C[np.triu_indices(len(names), 1)]
    md += ["## E04 Diversity of out-of-fold errors (train)", "",
           md_table(pd.DataFrame({"model": names, "OOF RMSE": [f4(rmse[k]) for k in names]})), "",
           f"- Plain average of the 5 OOF predictions: RMSE {rmse_avg:.4f} "
           f"(best single {min(rmse.values()):.4f}).",
           f"- Pairwise error correlation (regression): min {off.min():.3f}, mean {off.mean():.3f}, "
           f"max {off.max():.3f}.",
           f"- Classification: corr(L2, NB) {Cc[0, 1]:.3f}, corr(L2, tree) {Cc[0, 2]:.3f}, "
           f"corr(NB, tree) {Cc[1, 2]:.3f}.", ""]
    return {"oof_rmse": rmse, "oof_rmse_plain_average": rmse_avg,
            "reg_error_corr": {f"{names[i]} | {names[j]}": float(C[i, j])
                               for i in range(len(names)) for j in range(i + 1, len(names))},
            "clf_error_corr": {f"{cn[i]} | {cn[j]}": float(Cc[i, j])
                               for i in range(len(cn)) for j in range(i + 1, len(cn))}}


# ---------------------------------------------------------------- E05 instability
def e05(r, md):
    X, y = r["Xn_tr"], r["y_tr"]
    n = len(y)
    rng = np.random.default_rng(SEED)
    second = Counter()
    P_deep, P_ols = [], []
    for _ in range(N_BOOT):
        b = rng.integers(0, n, n)
        t2 = CART("regression", max_depth=2).fit(X[b], y[b])
        root = FEATURES[t2.feature_[0]]
        kids = tuple(FEATURES[t2.feature_[k]] if t2.feature_[k] >= 0 else "leaf"
                     for k in (t2.left_[0], t2.right_[0]))
        second[(root, *kids)] += 1
        P_deep.append(CART("regression").fit(X[b], y[b]).predict(X))
        P_ols.append(LinearRegression().fit(X[b], y[b]).predict(X))
    sd_deep, sd_ols = np.std(P_deep, 0), np.std(P_ols, 0)
    top = second.most_common(6)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    lab = [f"root {a}\nL: {b} | R: {c}" for (a, b, c), _ in top]
    ax[0].barh(range(len(top))[::-1], [v for _, v in top], color=BLUE)
    ax[0].set_yticks(range(len(top))[::-1], lab, fontsize=7)
    ax[0].set_xlabel(f"count out of {N_BOOT} bootstrap refits")
    ax[0].set_title("E05a  Depth-2 tree: the second level changes")
    bins = np.linspace(0, max(sd_deep.max(), sd_ols.max()), 30)
    ax[1].hist(sd_ols, bins, color=AQUA, alpha=.8, label=f"OLS (mean {sd_ols.mean():.2f})")
    ax[1].hist(sd_deep, bins, color=ORANGE, alpha=.7, label=f"full-depth tree (mean {sd_deep.mean():.2f})")
    ax[1].set_xlabel("s.d. of a pair's prediction across refits (points)")
    ax[1].set_ylabel("pairs"); ax[1].legend(fontsize=8)
    ax[1].set_title("E05b  Deep trees are high-variance")
    save(fig, FIG, "E05_tree_instability")
    roots = Counter(k[0] for k in second.elements())
    md += ["## E05 Tree instability (200 bootstrap refits of the 360 training pairs)", "",
           md_table(pd.DataFrame({"root / left child / right child": [" / ".join(k) for k, _ in top],
                                  "count": [v for _, v in top]})), "",
           f"- Root split variable: {dict(roots)}.",
           f"- Prediction s.d. across refits: full-depth tree mean {sd_deep.mean():.3f}, "
           f"OLS mean {sd_ols.mean():.3f} (ratio {sd_deep.mean() / sd_ols.mean():.2f}).", ""]
    return {"depth2_structures": {" | ".join(k): v for k, v in top}, "root_counts": dict(roots),
            "pred_sd_deep_tree_mean": float(sd_deep.mean()), "pred_sd_ols_mean": float(sd_ols.mean())}


# ---------------------------------------------------------------- E06-E07 weak learners
def e06_e07(c, md):
    X, y, names = c["Xn_tr"], c["y_tr"], c["names"]
    t = np.where(y > 0, 1.0, -1.0)
    w = np.full(len(y), 1 / len(y))
    base_err = min(y.mean(), 1 - y.mean())
    pos, neg = w * (t > 0), w * (t < 0)
    p1, q1 = X.T @ pos, X.T @ neg
    err_plus = (pos.sum() - p1) + q1
    err = np.minimum(err_plus, 1 - err_plus)
    df_tr = X.sum(0)
    j, _, s, best_err = fit_stump(X, t, w)
    order = np.argsort(err)[:15]
    n_pos_x = X[y == 1].sum(0); n_neg_x = X[y == 0].sum(0)
    pure_pos = int(np.sum((n_pos_x > 0) & (n_neg_x == 0)))
    pure_neg = int(np.sum((n_neg_x > 0) & (n_pos_x == 0)))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    ax[0].hist(1 - err, 40, color=LIGHT, edgecolor=MUTED)
    ax[0].axvline(1 - base_err, color=ORANGE, ls="--", label=f"always 'no' {1 - base_err:.3f}")
    ax[0].axvline(1 - best_err, color=BLUE, label=f"best word '{names[j]}' {1 - best_err:.3f}")
    ax[0].set_xlabel("training accuracy of a single-word stump")
    ax[0].set_ylabel("columns"); ax[0].set_title("E06a  Every word alone is a weak learner")
    ax[0].legend(fontsize=8)
    ax[1].barh(range(15)[::-1], (base_err - err[order]) * 100, color=BLUE)
    ax[1].set_yticks(range(15)[::-1], [names[k].replace("tok=", "") for k in order], fontsize=8)
    ax[1].set_xlabel("error reduction vs always 'no' (percentage points)")
    ax[1].set_title("E06b  The 15 best single-word stumps")
    save(fig, FIG, "E06_weak_learners")

    # E07: coverage of pharmacies by the k most precise positive words
    prec = n_pos_x / np.maximum(n_pos_x + n_neg_x, 1)
    pos_words = [k for k in np.argsort(-(prec * (n_pos_x >= 5) + 1e-6 * n_pos_x))
                 if n_pos_x[k] >= 5 and prec[k] > .5][:40]
    cover, fp_rate = [], []
    Xp, Xn = X[y == 1], X[y == 0]
    for k in range(1, len(pos_words) + 1):
        cols = pos_words[:k]
        cover.append(float(np.mean(Xp[:, cols].max(1) > 0)))
        fp_rate.append(float(np.mean(Xn[:, cols].max(1) > 0)))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ks = np.arange(1, len(cover) + 1)
    ax.plot(ks, cover, "o-", color=BLUE, ms=4, label="pharmacies hit (recall)")
    ax.plot(ks, fp_rate, "s-", color=ORANGE, ms=4, label="non-pharmacies hit")
    ax.set_xlabel("k most precise pharmacy words, OR-ed together")
    ax.set_ylabel("fraction of training vendors")
    ax.set_title("E07  No single rule reaches every pharmacy")
    ax.legend(fontsize=8)
    save(fig, FIG, "E07_rule_coverage")
    md += ["## E06 Weak learners (train, 3,689 vendors, 441 binary columns)", "",
           f"- Always 'not pharmacy': accuracy {1 - base_err:.4f}. Best single stump "
           f"'{names[j]}': {1 - best_err:.4f}.",
           f"- Columns whose stump beats the majority rule: {int(np.sum(err < base_err - 1e-12))} of "
           f"{len(names)}. Words seen only in pharmacies: {pure_pos}; only in non-pharmacies: {pure_neg}.",
           "", md_table(pd.DataFrame({"column": [names[k] for k in order],
                                      "train doc freq": [int(df_tr[k]) for k in order],
                                      "stump accuracy": [f4(1 - err[k]) for k in order]})), "",
           "## E07 Rule coverage", "",
           f"- Columns with >= 5 training pharmacies and precision > 0.5: {len(cover)}.",
           f"- The 1 most precise word reaches {cover[0]:.3f} of pharmacies; 5 words "
           f"{cover[min(4, len(cover) - 1)]:.3f}; all {len(cover)} words {cover[-1]:.3f} "
           f"(non-pharmacies hit: {fp_rate[-1]:.4f}).",
           f"- Words used, in order: {', '.join(names[k].replace('tok=', '') for k in pos_words)}.", ""]
    return {"majority_accuracy": 1 - base_err, "best_stump": names[j],
            "best_stump_accuracy": 1 - best_err,
            "n_stumps_beating_majority": int(np.sum(err < base_err - 1e-12)),
            "words_only_in_pharmacies": pure_pos, "words_only_in_non_pharmacies": pure_neg,
            "coverage_k1": cover[0], "coverage_k5": cover[min(4, len(cover) - 1)],
            "coverage_kmax": cover[-1], "kmax": len(cover), "fp_rate_kmax": fp_rate[-1]}


# ---------------------------------------------------------------- E08 noise
def e08(r, md):
    tr = r["train"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(tr.y, bins=np.arange(-2.5, 103, 5), color=BLUE, alpha=.85)
    for a in (0, 25, 50, 75, 100):
        ax[0].axvline(a, color=ORANGE, ls=":", lw=1)
    ax[0].set_xlabel("AI-judged score (train)"); ax[0].set_ylabel("pairs")
    ax[0].set_title("E08a  Scores pile up at the rubric anchors")
    g = cells(tr)
    labs, data = [], []
    for (a, b), s in g:
        labs.append(f"cat={'Y' if a else 'N'}\nrad={'Y' if b else 'N'}"); data.append(s.values)
    bp = ax[1].boxplot(data, tick_labels=labs, patch_artist=True)
    for bx, col in zip(bp["boxes"], [LIGHT, LIGHT, BLUE, BLUE]):
        bx.set_facecolor(col)
    sds = g.std()
    ax[1].set_ylabel("score")
    ax[1].set_title(f"E08b  Spread differs {sds.max() / sds.min():.1f}-fold between cells")
    save(fig, FIG, "E08_noise_structure")
    anchors = float(np.mean(tr.y.isin([0, 25, 50, 75, 100])))
    zeros = float(np.mean(tr.y == 0))
    md += ["## E08 Noise structure (train)", "",
           f"- Share of scores exactly on an anchor (0/25/50/75/100): {anchors:.4f}; exactly 0: {zeros:.4f}.",
           f"- Score s.d. per cell: " + ", ".join(f"{k}: {v:.2f}" for k, v in sds.items()) +
           f" (max/min ratio {sds.max() / sds.min():.2f}).", ""]
    return {"share_on_anchor": anchors, "share_zero": zeros,
            "sd_per_cell": {str(k): float(v) for k, v in sds.items()},
            "sd_ratio_max_min": float(sds.max() / sds.min())}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    r = regression_data()
    c = classification_data()
    md = ["# PBL-3 EDA numbers (generated by `python src/pbl3_eda.py`; do not edit by hand)", "",
          f"Regression: {len(r['y_tr'])} train / {len(r['y_te'])} test pairs (split by requirement). "
          f"Classification: {len(c['y_tr'])} train / {len(c['y_te'])} test vendors. "
          "All numbers below use the training split only. Seed 42.", ""]
    out = {"E01": e01(r, md), "E02": e02(r, md)}
    oof_r = oof_regression(r)
    out["E03"] = e03(r, oof_r, md)
    oof_c = oof_classification(c)
    out["E04"] = e04(r, oof_r, c, oof_c, md)
    out["E05"] = e05(r, md)
    out["E06_E07"] = e06_e07(c, md)
    out["E08"] = e08(r, md)
    (OUT / "eda_numbers.md").write_text("\n".join(md), encoding="utf-8")
    for k, v in out.items():
        print(k, {a: b for a, b in v.items() if not isinstance(b, dict)})
    print(f"wrote {FIG} and {OUT / 'eda_numbers.md'}")


if __name__ == "__main__":
    main()
