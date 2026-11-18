"""PBL-3 comparative analysis: single models vs combined models, with uncertainty.

Usage: python src/pbl3_comparison.py      (needs reports/pbl3_predictions.npz from pbl3_unit5.py)
Writes pbl-3/04_comparative_analysis/figures/C01-C06.png and .../comparison_numbers.md.

C01 match score: test R² of every model, 95% cluster-bootstrap CI over the 6 test requirements
C02 match score: paired difference in test MSE against OLS and against kernel ridge (Phase 1 best)
C03 match score: per-requirement RMSE (does the winner win everywhere?)
C04 pharmacy: F1 / PR-AUC / log loss with vendor-bootstrap CIs; exact McNemar against L2 logistic
C05 pharmacy: calibration (reliability) of the probability outputs
C06 pharmacy: which test vendors does every model get wrong, and are their labels right?
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from pbl3_style import (AQUA, BLUE, INK2, LIGHT, MUTED, ORANGE, PBL3, PURPLE, SEED, clf_metrics,
                        f4, md_table, plt, reg_metrics, save)

OUT = PBL3 / "04_comparative_analysis"
FIG = OUT / "figures"
PRED = Path("reports/pbl3_predictions.npz")
B = 2000
FAMILY = {  # colour by family: Phase 1 single models, single trees, committees, boosting, Bayesian/additive, combiners
    "OLS (Phase 1)": MUTED, "Kernel ridge RBF (Phase 1)": MUTED, "Logistic L2 (Phase 1)": MUTED,
    "Naive Bayes (Phase 1)": MUTED,
}


def colour(name):
    if name in FAMILY:
        return FAMILY[name]
    n = name.lower()
    if "stack" in n or "average" in n or "voting" in n:
        return PURPLE
    if "boost" in n or "adaboost" in n:
        return ORANGE
    if "forest" in n or "bagged" in n:
        return BLUE
    if "cart" in n:
        return LIGHT
    return AQUA          # BMA, GAM, mixtures


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def cluster_boot(y, P, req, rng):
    """Resample the 6 test requirements with replacement; returns (B, n_models) R² and MSE."""
    ureq = np.unique(req)
    idx = {q: np.where(req == q)[0] for q in ureq}
    R2, MSE = [], []
    for _ in range(B):
        rows = np.concatenate([idx[q] for q in rng.choice(ureq, len(ureq))])
        yy = y[rows]
        if np.var(yy) == 0:
            continue
        R2.append([r2(yy, p[rows]) for p in P])
        MSE.append([np.mean((yy - p[rows]) ** 2) for p in P])
    return np.array(R2), np.array(MSE)


def regression(d, md, rng):
    names, P, y, req = list(d["reg_names"]), d["reg_pred"], d["reg_y"], d["reg_req"]
    R2b, MSEb = cluster_boot(y, P, req, rng)
    rows = []
    for j, n in enumerate(names):
        mt = reg_metrics(y, P[j])
        lo, hi = np.percentile(R2b[:, j], [2.5, 97.5])
        rows.append({"model": n, "R2": mt["R2"], "RMSE": mt["RMSE"], "MAE": mt["MAE"], "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).sort_values("R2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(df.model, df.R2, color=[colour(n) for n in df.model], edgecolor=MUTED, lw=.4)
    ax.errorbar(df.R2, df.model, xerr=[df.R2 - df.lo, df.hi - df.R2], fmt="none", ecolor=INK2,
                lw=.8, capsize=2)
    ols = df[df.model == "OLS (Phase 1)"].R2.iloc[0]
    ax.axvline(ols, color=MUTED, ls="--", lw=1)
    ax.text(ols, -1.2, " OLS", color=MUTED, fontsize=8)
    ax.set_xlim(min(0.4, df.lo.min()), 1)
    ax.set_xlabel("test R² (90 pairs, 6 unseen requirements); bar = 95% cluster-bootstrap CI")
    ax.set_title("C01  Match score: single vs combined models")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, FIG, "C01_regression_leaderboard")

    base = {"OLS (Phase 1)": names.index("OLS (Phase 1)"),
            "Kernel ridge RBF (Phase 1)": names.index("Kernel ridge RBF (Phase 1)")}
    drows = []
    for bn, bj in base.items():
        for j, n in enumerate(names):
            if j == bj:
                continue
            diff = MSEb[:, j] - MSEb[:, bj]
            point = np.mean((y - P[j]) ** 2) - np.mean((y - P[bj]) ** 2)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            drows.append({"vs": bn, "model": n, "ΔMSE": point, "lo": lo, "hi": hi,
                          "P(better)": float(np.mean(diff < 0))})
    dd = pd.DataFrame(drows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2), sharey=False)
    for a, bn in zip(ax, base):
        s = dd[dd.vs == bn].sort_values("ΔMSE", ascending=False)
        a.barh(s.model, s["ΔMSE"], color=[colour(n) for n in s.model], edgecolor=MUTED, lw=.4)
        a.errorbar(s["ΔMSE"], s.model, xerr=[s["ΔMSE"] - s.lo, s.hi - s["ΔMSE"]], fmt="none",
                   ecolor=INK2, lw=.8, capsize=2)
        a.axvline(0, color=INK2, lw=.8)
        a.set_xlabel(f"test MSE − MSE of {bn.split(' (')[0]}  (< 0 = better)")
        a.set_title(f"C02  Paired difference vs {bn.split(' (')[0]}")
        a.tick_params(axis="y", labelsize=7.5)
    save(fig, FIG, "C02_paired_differences")

    ureq = np.unique(req)
    show = ["OLS (Phase 1)", "Kernel ridge RBF (Phase 1)", "GAM + interaction terms",
            "Random forest (own)", "Gradient boosting (own)", "Stacking (NNLS)"]
    pr = pd.DataFrame({n: [np.sqrt(np.mean((y[req == q] - P[names.index(n)][req == q]) ** 2)) for q in ureq]
                       for n in show}, index=ureq)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.13
    for k, (n, col) in enumerate(zip(show, [MUTED, INK2, AQUA, BLUE, ORANGE, PURPLE])):
        ax.bar(np.arange(len(ureq)) + (k - 2.5) * w, pr[n], w, color=col, label=n.replace(" (Phase 1)", ""))
    ax.set_xticks(range(len(ureq)), ureq); ax.set_ylabel("test RMSE")
    ax.set_title("C03  Per-requirement error (6 unseen buyers)"); ax.legend(fontsize=7, ncol=3)
    save(fig, FIG, "C03_per_requirement")
    wins = (pr.idxmin(axis=1)).value_counts().to_dict()
    md += ["## C01 Match score: test metrics, 95% CI by resampling the 6 test requirements", "",
           md_table(df[::-1].assign(**{"R²": df[::-1].R2.map(f4), "RMSE": df[::-1].RMSE.map(f4),
                                       "MAE": df[::-1].MAE.map(f4),
                                       "95% CI (R²)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(df[::-1].lo, df[::-1].hi)]})
                    [["model", "R²", "95% CI (R²)", "RMSE", "MAE"]]), "",
           "## C02 Paired ΔMSE (model − baseline) on the same test pairs, cluster-bootstrap 95% CI", ""]
    for bn in base:
        s = dd[dd.vs == bn].sort_values("ΔMSE")
        md += [f"### vs {bn}", "", md_table(pd.DataFrame({
            "model": s.model, "ΔMSE": s["ΔMSE"].map(lambda v: f"{v:.2f}"),
            "95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(s.lo, s.hi)],
            "P(ΔMSE < 0)": s["P(better)"].map(lambda v: f"{v:.3f}")})), ""]
    md += ["## C03 Per-requirement test RMSE", "",
           md_table(pr.map(lambda v: f"{v:.2f}").reset_index().rename(columns={"index": "requirement"})), "",
           f"- Lowest RMSE per requirement (count): {wins}.", ""]
    return df, dd


def mcnemar(y, a, b):
    """Exact McNemar test on the discordant pairs of two 0/1 prediction vectors."""
    ea, eb = a != y, b != y
    n01, n10 = int(np.sum(ea & ~eb)), int(np.sum(~ea & eb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def classification(d, md, rng):
    names, P, y, vid = list(d["clf_names"]), d["clf_prob"], d["clf_y"], d["clf_vendor"]
    n = len(y)
    boots = rng.integers(0, n, size=(B, n))
    ref = names.index("Logistic L2 (Phase 1)")
    rows = []
    for j, nm in enumerate(names):
        mt = clf_metrics(y, P[j])
        f1s = []
        for b in boots[:1000]:
            yy, pp = y[b], (P[j][b] >= .5)
            tp = np.sum(pp & (yy == 1)); fp = np.sum(pp & (yy == 0)); fn = np.sum(~pp & (yy == 1))
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        lo, hi = np.percentile(f1s, [2.5, 97.5])
        n01, n10, p = mcnemar(y, (P[j] >= .5).astype(int), (P[ref] >= .5).astype(int))
        rows.append({"model": nm, **mt, "f1_lo": lo, "f1_hi": hi, "worse_only": n01, "better_only": n10,
                     "mcnemar_p": p})
    df = pd.DataFrame(rows).sort_values("f1").reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].barh(df.model, df.f1, color=[colour(nm) for nm in df.model], edgecolor=MUTED, lw=.4)
    ax[0].errorbar(df.f1, df.model, xerr=[df.f1 - df.f1_lo, df.f1_hi - df.f1], fmt="none", ecolor=INK2,
                   lw=.8, capsize=2)
    ax[0].set_xlim(.5, 1); ax[0].set_xlabel("test F1 (923 vendors), 95% bootstrap CI")
    ax[0].set_title("C04a  Pharmacy: F1"); ax[0].tick_params(axis="y", labelsize=8)
    ax[1].scatter(df.pr_auc, df.log_loss, c=[colour(nm) for nm in df.model], edgecolor=INK2, s=45, zorder=3)
    for _, rr in df.iterrows():
        if rr.f1 > .85:
            ax[1].annotate(rr.model.replace(" (own)", "").replace(" (Phase 1)", ""), (rr.pr_auc, rr.log_loss),
                           fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax[1].set_xlabel("PR-AUC (higher better)"); ax[1].set_ylabel("log loss (lower better)")
    ax[1].set_xlim(.85, 1); ax[1].set_ylim(0, max(.2, df[df.f1 > .85].log_loss.max() * 1.1))
    ax[1].set_title("C04b  Ranking vs probability quality")
    save(fig, FIG, "C04_classification_leaderboard")

    show = ["Logistic L2 (Phase 1)", "Naive Bayes (Phase 1)", "Random forest (own)",
            "AdaBoost, stumps (own)", "Gradient boosting (own)", "Stacking (logistic meta)"]
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.plot([0, 1], [0, 1], color=INK2, ls=":", lw=1)
    ece = {}
    for nm, col in zip(show, [MUTED, INK2, BLUE, ORANGE, AQUA, PURPLE]):
        p = P[names.index(nm)]
        k = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys, e = [], [], 0.0
        for b in range(10):
            s = k == b
            if s.sum() >= 5:
                xs.append(p[s].mean()); ys.append(y[s].mean())
            if s.any():
                e += s.mean() * abs(p[s].mean() - y[s].mean())
        ece[nm] = e
        ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.3, label=f"{nm.replace(' (Phase 1)', '')} (ECE {e:.3f})")
    ax.set_xlabel("mean predicted P(pharmacy) in bin"); ax.set_ylabel("observed fraction of pharmacies")
    ax.set_title("C05  Calibration (bins with ≥ 5 vendors)"); ax.legend(fontsize=7)
    save(fig, FIG, "C05_calibration")

    # C06: errors shared across models + the AI label audit of test vendors
    good = [nm for nm in names if clf_metrics(y, P[names.index(nm)])["f1"] > .85]
    E = np.array([(P[names.index(nm)] >= .5).astype(int) != y for nm in good])
    k_wrong = E.sum(0)
    audit = pd.read_csv("data/annotation/label_audit_sample.csv", encoding="utf-8-sig")
    vend = pd.read_csv("data/processed/vendors.csv", encoding="utf-8")
    err = pd.DataFrame({"vendor_id": vid, "weak_label": y, "models_wrong": k_wrong,
                        "p_stack": P[names.index("Stacking (logistic meta)")]})
    err = err[err.models_wrong > 0].merge(vend[["vendor_id", "title", "categories"]], on="vendor_id")
    err = err.merge(audit[["vendor_id", "expected_is_pharmacy"]], on="vendor_id", how="left")
    err = err.sort_values(["models_wrong", "vendor_id"], ascending=[False, True])
    all_wrong = err[err.models_wrong == len(good)]
    audited = all_wrong.expected_is_pharmacy.notna()
    label_wrong = int(np.sum(all_wrong.expected_is_pharmacy[audited] != all_wrong.weak_label[audited]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cnt = pd.Series(k_wrong[k_wrong > 0]).value_counts().sort_index()
    ax.bar(cnt.index, cnt.values, color=[ORANGE if i == len(good) else BLUE for i in cnt.index])
    ax.set_xlabel(f"number of the {len(good)} strong classifiers that misclassify the vendor")
    ax.set_ylabel("test vendors"); ax.set_title("C06  Errors are shared: combining cannot fix them")
    save(fig, FIG, "C06_shared_errors")

    md += ["## C04 Pharmacy: test metrics (threshold 0.5), F1 95% bootstrap CI (1,000 vendor resamples), "
           "exact McNemar test against Phase 1 L2 logistic", "",
           md_table(pd.DataFrame({
               "model": df.model[::-1], "F1": df.f1[::-1].map(f4),
               "95% CI": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(df.f1_lo[::-1], df.f1_hi[::-1])],
               "PR-AUC": df.pr_auc[::-1].map(f4), "ROC-AUC": df.roc_auc[::-1].map(f4),
               "log loss": df.log_loss[::-1].map(f4), "FP/FN": [f"{a}/{b}" for a, b in zip(df.fp[::-1], df.fn[::-1])],
               "vs L2: worse-only / better-only": [f"{a} / {b}" for a, b in zip(df.worse_only[::-1], df.better_only[::-1])],
               "McNemar p": df.mcnemar_p[::-1].map(lambda v: f"{v:.3f}")})), "",
           "## C05 Expected calibration error (10 equal-width bins)", "",
           md_table(pd.DataFrame({"model": list(ece), "ECE": [f4(v) for v in ece.values()]})), "",
           f"## C06 Shared errors ({len(good)} classifiers with test F1 > 0.85)", "",
           f"- Test vendors misclassified by at least one: {int(np.sum(k_wrong > 0))}; by all {len(good)}: "
           f"{len(all_wrong)}.",
           f"- Of the {len(all_wrong)} that all models get wrong, {int(audited.sum())} are in the 200-vendor AI label "
           f"audit; for {label_wrong} of those the audit says the weak label itself is wrong.", "",
           md_table(err.head(25).assign(p_stack=err.head(25).p_stack.map(lambda v: f"{v:.3f}"),
                                        expected_is_pharmacy=err.head(25).expected_is_pharmacy.map(
                                            lambda v: "" if pd.isna(v) else str(int(v))))
                    [["vendor_id", "title", "categories", "weak_label", "expected_is_pharmacy",
                      "models_wrong", "p_stack"]]), ""]
    return df, {"n_good": len(good), "wrong_by_any": int(np.sum(k_wrong > 0)),
                "wrong_by_all": len(all_wrong), "all_wrong_audited": int(audited.sum()),
                "all_wrong_label_wrong": label_wrong, "ece": ece}


def main():
    d = np.load(PRED, allow_pickle=False)
    rng = np.random.default_rng(SEED)
    md = ["# PBL-3 comparison numbers (generated by `python src/pbl3_comparison.py`; do not edit by hand)",
          "", f"Test sets only. {B} cluster-bootstrap resamples of the 6 test requirements (score); "
          "1,000 vendor resamples (pharmacy). Seed 42.", ""]
    regression(d, md, rng)
    _, info = classification(d, md, rng)
    (OUT / "comparison_numbers.md").write_text("\n".join(md), encoding="utf-8")
    print(info)
    print(f"wrote {FIG} and {OUT / 'comparison_numbers.md'}")


if __name__ == "__main__":
    main()
