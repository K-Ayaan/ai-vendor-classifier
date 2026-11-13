"""Viva pack - graphs/ (numbered PNGs) and viva/00_key_numbers.md.

Usage: python src/make_viva_pack.py
Re-fits the Step 4 and Step 5 models (same seed/splits) so every graph and number comes
from code. Regression target = AI-judged expected_score (not human-annotated, not ground
truth). Classification target = weak is_pharmacy labels (source file membership).
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402
import logistic_regression_classification as L  # noqa: E402

GRAPHS = Path("graphs")
VIVA = Path("viva")
# validated categorical palette (light surface): blue, orange, aqua; aqua always has a legend
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TARGET = "expected_score"  # graph label
MD_TARGET = "AI-judged expected_score"  # viva/00_key_numbers.md text

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPHS / f"{name}.png", dpi=150)
    plt.close(fig)


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    GRAPHS.mkdir(exist_ok=True)
    VIVA.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = lr.load_annotations()
        r = lr.run(df)
        c = L.run()

    # ================= regression (target: AI-judged expected_score) =================
    names = ["intercept"] + lr.FEATURES
    X_tr, X_te, y_tr, y_te, beta = r["X_tr"], r["X_te"], r["y_tr"], r["y_te"], r["beta"]
    f_tr, f_te = X_tr @ beta, X_te @ beta
    res_tr, res_te = y_tr - f_tr, y_te - f_te
    n, p = X_tr.shape
    g, mt, me = r["mle"], r["train_metrics"], r["test_metrics"]
    s2 = g["s2_unbiased"]
    cov = s2 * np.linalg.inv(X_tr.T @ X_tr)          # Var(beta_hat) = s^2 (X^T X)^-1
    se = np.sqrt(np.diag(cov))

    # 01 actual vs predicted
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(f_tr, y_tr, s=16, alpha=0.55, color=BLUE, label=f"train (n={n})")
    ax.scatter(f_te, y_te, s=22, alpha=0.75, color=ORANGE, marker="^", label=f"test (n={len(y_te)})")
    lo, hi = min(f_tr.min(), f_te.min(), 0) - 5, 105
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="perfect prediction (y = ŷ)")
    ax.set(xlim=(lo, hi), ylim=(-5, 105), xlabel=f"predicted {TARGET} (ŷ = Xβ̂)",
           ylabel=f"actual {TARGET} (y)",
           title=f"01 Actual vs predicted (R² train {mt['R2']:.3f}, test {me['R2']:.3f})")
    ax.legend(loc="upper left")
    save(fig, "01_actual_vs_predicted")

    # 02 residuals vs fitted
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(f_tr, res_tr, s=16, alpha=0.55, color=BLUE, label="train")
    ax.scatter(f_te, res_te, s=22, alpha=0.75, color=ORANGE, marker="^", label="test")
    ax.axhline(0, color=INK2, lw=1)
    ax.set(xlabel=f"fitted {TARGET}", ylabel="residual  e = y − ŷ",
           title="02 Residuals vs fitted (should be a patternless band around 0)")
    ax.legend(loc="lower left")
    save(fig, "02_residuals_vs_fitted")

    # 03 residual histogram with the fitted Gaussian N(0, sigma2_MLE)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(res_tr, bins=30, density=True, color=BLUE, alpha=0.75, edgecolor=SURFACE,
            linewidth=1, label="train residuals")
    xs = np.linspace(res_tr.min() - 5, res_tr.max() + 5, 300)
    ax.plot(xs, stats.norm.pdf(xs, 0, g["sigma_mle"]), color=ORANGE,
            label=f"N(0, σ̂²_MLE = {g['sigma2_mle']:.1f})")
    ax.set(xlabel="residual", ylabel="density",
           title="03 Residual histogram vs the assumed Gaussian error")
    ax.legend(loc="upper left")
    save(fig, "03_residual_histogram")

    # 04 Q-Q plot
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    (osm, osr), (slope, icpt, _) = stats.probplot(res_tr, dist="norm")
    ax.scatter(osm, osr, s=14, color=BLUE, alpha=0.7, label="train residuals")
    ax.plot(osm, slope * osm + icpt, color=ORANGE, label="Gaussian reference line")
    ax.set(xlabel="theoretical normal quantiles", ylabel="ordered residuals",
           title=f"04 Normal Q-Q (Shapiro-Wilk p = {stats.shapiro(res_tr).pvalue:.1e})")
    ax.legend(loc="upper left")
    save(fig, "04_qq_plot")

    # 05 coefficients with approximate 95% CI
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, beta, height=0.55, color=BLUE)
    ax.errorbar(beta, ypos, xerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ypos, names)
    for yv, b, e in zip(ypos, beta, se):
        ax.text(max(b, 0) + 1.96 * e + 1.2, yv, f"{b:.2f}", va="center", ha="left",
                fontsize=9, color=INK2)
    ax.set_xlim(right=max(beta + 1.96 * se) + 9)
    ax.set(xlabel="β̂ (change in score per unit of the feature)",
           title="05 Coefficients β̂ with ±1.96·SE (SE from s²(XᵀX)⁻¹)")
    save(fig, "05_coefficients")

    # 06 profile log-likelihood in sigma^2 at beta_hat
    rss = mt["RSS"]
    s2_grid = np.linspace(0.5 * g["sigma2_mle"], 2.2 * g["sigma2_mle"], 400)
    ll = -n / 2 * np.log(2 * np.pi * s2_grid) - rss / (2 * s2_grid)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(s2_grid, ll, color=BLUE, label="ℓ(β̂, σ²) = −n/2·ln(2πσ²) − RSS/(2σ²)")
    ax.axvline(g["sigma2_mle"], color=ORANGE, lw=1.5,
               label=f"σ̂²_MLE = RSS/n = {g['sigma2_mle']:.2f} (maximum)")
    ax.axvline(s2, color=AQUA, lw=1.5, ls="--", label=f"s² = RSS/(n−p) = {s2:.2f}")
    ax.scatter([g["sigma2_mle"]], [g["loglik_formula"]], color=ORANGE, s=40, zorder=3)
    ax.annotate(f"ℓ_max = {g['loglik_formula']:.2f}", (g["sigma2_mle"], g["loglik_formula"]),
                textcoords="offset points", xytext=(10, -14), color=INK2)
    ax.set(xlabel="σ²", ylabel="log-likelihood (train)",
           title="06 Log-likelihood as a function of σ² (β fixed at β̂)")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "06_log_likelihood_sigma2")

    # 07 train vs test metrics (two panels: error scale and R2 scale - no dual axis)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.0), gridspec_kw={"width_ratios": [2, 1]})
    xm = np.arange(2)
    for off, (lab, m, col) in zip((-0.18, 0.18), (("train", mt, BLUE), ("test", me, ORANGE))):
        vals = [m["RMSE"], m["MAE"]]
        a1.bar(xm + off, vals, width=0.34, color=col, label=lab)
        for xv, v in zip(xm + off, vals):
            a1.text(xv, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
    a1.set_xticks(xm, ["RMSE", "MAE"])
    a1.set(ylabel="score points", title="07a Error (lower is better)")
    a1.legend(loc="upper right")
    r2s = [mt["R2"], me["R2"]]
    a2.bar([0, 1], r2s, width=0.55, color=[BLUE, ORANGE])
    for xv, v in zip([0, 1], r2s):
        a2.text(xv, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK2)
    a2.set_xticks([0, 1], ["train", "test"])
    a2.set(ylim=(0, 1), ylabel="R²", title="07b R² (higher is better)")
    save(fig, "07_train_vs_test_metrics")

    # 08 summary table
    cons = r["consistency"]
    blind = cons.get("blind") or cons["non_blind"]
    rows = [
        ("Target", TARGET + " (0–100)"),
        ("n train / n test (pairs)", f"{n} / {len(y_te)}"),
        ("p (intercept + 6 features)", f"{p}"),
        ("cond(X_train)", f"{r['ols']['cond_X']:.2f}"),
        ("RSS train / test", f"{mt['RSS']:.2f} / {me['RSS']:.2f}"),
        ("MSE train / test", f"{mt['MSE']:.4f} / {me['MSE']:.4f}"),
        ("RMSE train / test", f"{mt['RMSE']:.4f} / {me['RMSE']:.4f}"),
        ("MAE train / test", f"{mt['MAE']:.4f} / {me['MAE']:.4f}"),
        ("R² train / test", f"{mt['R2']:.4f} / {me['R2']:.4f}"),
        ("σ̂²_MLE = RSS/n", f"{g['sigma2_mle']:.4f}"),
        ("σ̂_MLE", f"{g['sigma_mle']:.4f}"),
        ("s² = RSS/(n−p)", f"{g['s2_unbiased']:.4f}"),
        ("log-likelihood (direct / closed form)",
         f"{g['loglik_direct_sum']:.4f} / {g['loglik_formula']:.4f}"),
        ("β̂ intercept (β₀)", f"{beta[0]:.4f}"),
        ("max|β̂_lstsq − β̂_sklearn|", f"{r['ols']['max_abs_diff_sklearn_vs_lstsq']:.2e}"),
        ("Blind re-score consistency (r / MAD)",
         f"{blind['pearson_r']:.4f} / {blind['mean_abs_difference']:.3f}"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 0.27 * len(rows) + 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["quantity", "value"], loc="center",
                   cellLoc="left", colWidths=[0.55, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#eef3fb" if ri == 0 else SURFACE)
        if ri == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("08 Linear regression summary (Gaussian MLE / least squares)", pad=6)
    save(fig, "08_summary_table")

    # ================= classification (target: weak is_pharmacy labels) =================
    y_c = c["y_te"]
    models = [("L2 (MAP, not MLE)", c["p_l2"], BLUE, "-"),
              ("Small-model MLE", c["small"]["p_test"], ORANGE, "--"),
              ("Full unpenalised (invalid MLE)", c["p_unpen"], AQUA, ":")]

    # 09 confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    cms = {}
    for ax, (name, pr, _, _) in zip(axes, models):
        cm = confusion_matrix(y_c, (pr >= 0.5).astype(int), labels=[0, 1])
        cms[name] = cm
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "seqblue", ["#eef3fb", "#2a78d6", "#0d366b"]))
        for i in range(2):
            for j in range(2):
                tag = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i, f"{tag}\n{cm[i, j]}", ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() * 0.5 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["actual 0", "actual 1"])
        ax.grid(False)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name}\naccuracy {acc:.4f}", fontsize=10)
    fig.suptitle("09 Confusion matrices: is_pharmacy, test set (weak labels), threshold 0.5",
                 fontweight="bold")
    save(fig, "09_confusion_matrices")

    # 10 ROC and PR curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    for name, pr, col, ls in models:
        fpr, tpr, _ = roc_curve(y_c, pr)
        a1.plot(fpr, tpr, color=col, ls=ls, label=name)
        prec, rec, _ = precision_recall_curve(y_c, pr)
        a2.plot(rec, prec, color=col, ls=ls, label=name)
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="chance")
    a2.axhline(y_c.mean(), color=MUTED, lw=1, ls="--", label=f"prevalence {y_c.mean():.3f}")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title="10a ROC curve (test)")
    a2.set(xlabel="recall", ylabel="precision", title="10b Precision-recall curve (test)")
    a1.legend(loc="lower right", fontsize=8.5)
    a2.legend(loc="lower left", fontsize=8.5)
    save(fig, "10_roc_pr_curves")

    # 11 MLE existence: IRLS on full model (separated) vs small model
    sm = c["small"]
    Xs = np.column_stack([np.ones(len(c["train"])), L.token_indicators(c["train"], sm["tokens"])])
    _, hist_small, _ = L.irls(Xs, c["y_tr"], l2=0.0, max_iter=100)
    hf, hs = pd.DataFrame(c["history"]), pd.DataFrame(hist_small)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.plot(hf.iter, hf.objective, color=BLUE, label="full model (442 cols, separated): flat ridge, no peak")
    a1.plot(hs.iter, hs.objective, color=ORANGE, ls="--", label="small model (11 cols)")
    a1.set(xlabel="IRLS iteration", ylabel="log-likelihood ℓ(β)", title="11a ℓ(β) per iteration")
    a2.plot(hf.iter, hf.beta_norm, color=BLUE,
            label="full: ‖β‖ runs away, then stalls only\nbecause probabilities hit 0/1 in float64")
    a2.plot(hs.iter, hs.beta_norm, color=ORANGE, ls="--", label="small: ‖β‖ settles (MLE exists)")
    a2.set(xlabel="IRLS iteration", ylabel="‖β‖", title="11b Size of β per iteration (no finite maximiser)")
    a1.legend(loc="lower right", fontsize=8.5)
    a2.legend(loc="center right", fontsize=8.5)
    save(fig, "11_mle_existence_irls")

    # 12 why not linear regression for a 0/1 target: linear probability model
    lpm_beta = np.linalg.lstsq(c["X_tr"], c["y_tr"].astype(float), rcond=None)[0]
    lpm_te = c["X_te"] @ lpm_beta
    below, above = int((lpm_te < 0).sum()), int((lpm_te > 1).sum())
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    w = 0.05  # bin edges aligned so 0 and 1 are exact edges
    bins = np.arange(np.floor(min(lpm_te.min(), 0) / w) * w,
                     np.ceil(max(lpm_te.max(), 1) / w) * w + w / 2, w)
    ax.axvspan(bins[0], 0, color=ORANGE, alpha=0.08)
    ax.axvspan(1, bins[-1], color=ORANGE, alpha=0.08)
    ax.hist(lpm_te, bins=bins, histtype="step", color=ORANGE, linewidth=2,
            label=f"linear regression ŷ: {below} below 0, {above} above 1")
    ax.hist(c["p_l2"], bins=bins, histtype="step", color=BLUE, linewidth=2, linestyle="--",
            label="logistic σ(Xβ̂): always inside [0, 1]")
    for v in (0, 1):
        ax.axvline(v, color=INK2, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set(xlabel="predicted value for is_pharmacy (test set)", ylabel="count (log scale)",
           title="12 Why logistic, not linear, regression for a 0/1 target")
    ax.text(bins[0] + 0.02, 400, "impossible\n(< 0)", color=INK2, fontsize=8.5)
    ax.text(1.03, 400, "impossible (> 1)", color=INK2, fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    save(fig, "12_why_not_linear_regression")

    # ================= key numbers file (all computed here) =================
    tss_tr = float(np.sum((y_tr - y_tr.mean()) ** 2))
    term1 = -n / 2 * np.log(2 * np.pi * g["sigma2_mle"])
    term2 = -rss / (2 * g["sigma2_mle"])
    coef_tbl = pd.DataFrame({"feature": names, "β̂": [f"{b:.4f}" for b in beta],
                             "SE": [f"{s:.4f}" for s in se],
                             "t = β̂/SE": [f"{b / s:.2f}" for b, s in zip(beta, se)]})
    clf_rows = []
    for name, pr, _, _ in models:
        m = L.evaluate(y_c, pr)
        cm = m["confusion_matrix"]
        clf_rows.append({"model": name, "TN": cm["tn"], "FP": cm["fp"], "FN": cm["fn"],
                         "TP": cm["tp"], "accuracy": f"{m['accuracy']:.4f}",
                         "precision": f"{m['precision']:.4f}", "recall": f"{m['recall']:.4f}",
                         "F1": f"{m['f1']:.4f}", "ROC-AUC": f"{m['roc_auc']:.4f}",
                         "PR-AUC": f"{m['pr_auc']:.4f}"})
    base = L.majority_baseline(c["y_tr"], y_c)
    sep = L.separation_lp(c["X_tr"], c["y_tr"])
    md = f"""# Key numbers (generated by `python src/make_viva_pack.py`; do not edit by hand)

Every number below was computed by re-fitting the models in this run.

## A. Linear regression (target: {MD_TARGET}, not ground truth)

* n_train = {n} pairs (24 requirements), n_test = {len(y_te)} pairs (6 requirements), p = {p}
  columns (intercept + {len(lr.FEATURES)} features).
* Condition number of X_train = {r['ols']['cond_X']:.2f}; rank = {r['ols']['rank']}.
* Solvers agree: |β̂_inverse − β̂_lstsq| ≤ {r['ols']['max_abs_diff_inverse_vs_lstsq']:.2e},
  |β̂_sklearn − β̂_lstsq| ≤ {r['ols']['max_abs_diff_sklearn_vs_lstsq']:.2e}.

{md_table(coef_tbl)}

SE = √diag(s²(XᵀX)⁻¹) (valid only if the Gaussian, constant-variance assumptions hold).

| quantity | train | test |
|---|---|---|
| RSS = Σ(y − ŷ)² | {mt['RSS']:.4f} | {me['RSS']:.4f} |
| MSE = RSS/n | {mt['MSE']:.4f} | {me['MSE']:.4f} |
| RMSE = √MSE | {mt['RMSE']:.4f} | {me['RMSE']:.4f} |
| MAE = mean abs(y − ŷ) | {mt['MAE']:.4f} | {me['MAE']:.4f} |
| R² = 1 − RSS/TSS | {mt['R2']:.4f} | {me['R2']:.4f} |

* TSS_train = Σ(y − ȳ)² = {tss_tr:.4f}, so R²_train = 1 − {mt['RSS']:.4f}/{tss_tr:.4f} = {1 - mt['RSS'] / tss_tr:.4f}.
* σ̂²_MLE = RSS/n = {mt['RSS']:.4f}/{n} = {g['sigma2_mle']:.4f};  σ̂_MLE = {g['sigma_mle']:.4f}.
* s² = RSS/(n−p) = {mt['RSS']:.4f}/{n - p} = {g['s2_unbiased']:.4f};  s = {g['s_unbiased']:.4f}.
* Bias factor (n−p)/n = {n - p}/{n} = {(n - p) / n:.6f}; σ̂²_MLE / s² = {g['sigma2_mle'] / g['s2_unbiased']:.6f} (identical, as theory says).
* Maximised log-likelihood, plugged in:
  ℓ = −(n/2)·ln(2πσ̂²) − RSS/(2σ̂²) = {term1:.4f} + ({term2:.4f}) = {term1 + term2:.4f}
  (the second term is always −n/2 = {-n / 2:.1f} at σ̂²_MLE).
  Closed form −(n/2)(ln(2πσ̂²) + 1) = {g['loglik_formula']:.4f}; direct Σ log φ(eᵢ; 0, σ̂) = {g['loglik_direct_sum']:.4f}.
* Train MSE = σ̂²_MLE: {np.isclose(mt['MSE'], g['sigma2_mle'])} (same number, different meaning).
* Residual diagnostics (train): skew {stats.skew(res_tr):.3f}, excess kurtosis
  {stats.kurtosis(res_tr):.3f}, Shapiro-Wilk p = {stats.shapiro(res_tr).pvalue:.2e};
  fitted values below 0: {int((f_tr < 0).sum())} train, {int((f_te < 0).sum())} test.
* AI judge consistency on 45 pairs: blind r = {blind['pearson_r']:.4f}, MAD = {blind['mean_abs_difference']:.3f}
  (non-blind r = {cons['non_blind']['pearson_r']:.4f}, MAD = {cons['non_blind']['mean_abs_difference']:.3f}).

## B. Logistic regression (target: is_pharmacy weak labels)

* Split: train {len(c['y_tr'])} vendors ({int(c['y_tr'].sum())} pharmacies), test {len(y_c)}
  ({int(y_c.sum())} pharmacies, prevalence {y_c.mean():.4f}).
* Full design: {c['X_tr'].shape[1]} columns, rank {np.linalg.matrix_rank(c['X_tr'])}.
  Single-feature separators: {len(c['separated'])}. Joint LP optimum {sep['lp_optimum']:g}
  (> 0 means separated, so the MLE does not exist).
* Log-likelihoods on train: null (intercept only) {c['ll_null']:.4f}; full unpenalised IRLS
  {c['ll_irls']:.4f} (not a maximum, it keeps creeping up); small-model MLE
  {sm['loglik_irls']:.4f}; L2 model {c['ll_l2']:.4f}.
* Full model: IRLS ‖β‖ after {len(c['history'])} iterations = {c['history'][-1]['beta_norm']:.2f};
  IRLS vs sklearn coefficients differ by up to
  {float(np.max(np.abs(c['coef'].beta_irls_unpenalised - c['coef'].beta_sklearn_unpenalised))):.2f}
  despite the same ℓ.
* Small model: tokens {sm['tokens']}; IRLS converged in {sm['irls_iterations']} iterations;
  |β̂_IRLS − β̂_sklearn| ≤ {sm['max_abs_coef_diff_irls_vs_sklearn']:.2e}.
* L2 model: C = {c['best_c']:g} (λ = {1 / c['best_c']:g}), chosen by 5-fold CV on train.

{md_table(pd.DataFrame(clf_rows))}

Majority baseline (always predict 0): accuracy {base['accuracy']:.4f}, recall 0, ROC-AUC 0.5,
PR-AUC = prevalence {base['pr_auc_equals_prevalence']:.4f}.

## C. Why not linear regression for the 0/1 target (linear probability model)

Fitting OLS to is_pharmacy with the same {c['X_tr'].shape[1]} columns gives test predictions
from {lpm_te.min():.3f} to {lpm_te.max():.3f}: {below} below 0 and {above} above 1 out of {len(lpm_te)},
which are impossible as probabilities. The logistic model's σ(Xβ̂) is always inside (0, 1).
"""
    md += extended_graphs_and_numbers()
    (VIVA / "00_key_numbers.md").write_text(md, encoding="utf-8")
    print(f"wrote {len(list(GRAPHS.glob('*.png')))} graphs to {GRAPHS}/ and {VIVA}/00_key_numbers.md")


def extended_graphs_and_numbers():
    """Graphs 13-19 and key-numbers section D for the extended Unit-II models."""
    import extended_models as E
    SIX = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300"]  # validated 6-slot order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = E.run_regression()
        clf = E.run_classification()
    rm, rd = reg["metrics"], reg["details"]
    cm_, cd = clf["metrics"], clf["details"]
    names = list(rm)

    # 13 regression model comparison (test), two panels
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    yp = np.arange(len(names))[::-1]
    for ax, key, xl in ((a1, "RMSE", "test RMSE (lower is better)"), (a2, "R2", "test R² (higher is better)")):
        vals = [rm[n]["test"][key] for n in names]
        ax.barh(yp, vals, height=0.55, color=BLUE)
        for y_, v in zip(yp, vals):
            ax.text(v, y_, f" {v:.3f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(yp, names if ax is a1 else [""] * len(names))
        ax.set(xlabel=xl)
    a1.set_xlim(0, max(rm[n]["test"]["RMSE"] for n in names) * 1.18)
    a2.set_xlim(0, 1)
    fig.suptitle("13 Regression models on held-out requirements (target: expected_score)",
                 fontweight="bold")
    save(fig, "13_regression_model_comparison")

    # 14 ridge shrinkage path
    lams, path = E.ridge_path(reg["X_tr"], reg["y_tr"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for j, (f, col) in enumerate(zip(rd["features"], SIX)):
        ax.plot(lams, path[:, j], color=col, label=f)
        ax.text(lams[0] * 0.8, path[0, j], f, ha="right", va="center", fontsize=8, color=INK2)
    ax.axvline(rd["ridge_alpha"], color=MUTED, ls="--", lw=1,
               label=f"λ chosen by grouped CV = {rd['ridge_alpha']:g}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(lams[0] * 0.02, lams[-1] * 1.5)
    ax.set(xlabel="ridge penalty λ (log scale)", ylabel="standardised coefficient",
           title="14 Ridge shrinkage path: larger λ pulls every β towards 0")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    save(fig, "14_ridge_shrinkage_path")

    # 15 OLS vs Huber coefficients
    feats = rd["features"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    yp = np.arange(len(feats))[::-1]
    ax.barh(yp + 0.19, [rd["ols_coef"][f] for f in feats], height=0.36, color=BLUE,
            label="OLS (Gaussian MLE)")
    ax.barh(yp - 0.19, [rd["huber_coef"][f] for f in feats], height=0.36, color=ORANGE,
            label=f"Huber (robust, ε = {rd['huber_epsilon']})")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(yp, feats)
    ax.set(xlabel="β̂", title=f"15 OLS vs robust Huber coefficients\n"
                              f"({rd['huber_n_outliers_train']} of {len(reg['y_tr'])} train pairs down-weighted by Huber)")
    ax.legend(loc="lower right")
    save(fig, "15_ols_vs_huber_coefficients")

    # 16 Bayesian predictive intervals on test
    mu, sd, yt = reg["bayes_mu_te"], reg["bayes_sd_te"], reg["y_te"]
    order = np.argsort(mu)
    xs = np.arange(len(mu))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(xs, (mu - 1.96 * sd)[order], (mu + 1.96 * sd)[order], color=BLUE, alpha=0.18,
                    label="95% predictive interval (mean ± 1.96·sd)")
    ax.plot(xs, mu[order], color=BLUE, label="posterior predictive mean")
    ax.scatter(xs, yt[order], s=16, color=ORANGE, zorder=3, label="actual expected_score")
    ax.set(xlabel="test pairs, sorted by predicted score", ylabel="expected_score",
           title=f"16 Bayesian linear regression: test coverage "
                 f"{rd['bayes_test_95pct_interval_coverage']:.1%} (target 95%)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "16_bayesian_predictive_intervals")

    # 17 classifier comparison
    cn = list(cm_)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    yp = np.arange(len(cn))[::-1] * 1.0
    for k, (met, col, lab) in enumerate((("f1", BLUE, "F1"), ("roc_auc", ORANGE, "ROC-AUC"),
                                         ("pr_auc", AQUA, "PR-AUC"))):
        ax.barh(yp + (1 - k) * 0.26, [cm_[n][met] for n in cn], height=0.24, color=col, label=lab)
    ax.set_yticks(yp, cn)
    ax.set_xlim(0, 1.08)
    ax.set(xlabel="score on the test set (weak is_pharmacy labels)",
           title="17 Classifier comparison on the test set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    save(fig, "17_classifier_comparison")

    # 18 Laplace: MAP plug-in vs Bayesian predictive probability
    bl = clf["laplace"]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.scatter(bl["p_map"], bl["p_pred"], s=14, color=BLUE, alpha=0.6, label="test vendors")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="no change")
    ax.set(xlabel="MAP plug-in probability σ(xᵀβ_MAP)",
           ylabel="Bayesian predictive probability (Laplace + probit)",
           title=f"18 Laplace moderation pulls probabilities towards 0.5\n"
                 f"log loss: plug-in {cd['laplace_test_logloss_map_plugin']:.4f}, "
                 f"predictive {cd['laplace_test_logloss_predictive']:.4f}")
    ax.legend(loc="upper left")
    save(fig, "18_laplace_moderation")

    # 19 confusion matrices of the added classifiers
    added = ["Fisher LDA (discriminant)", "Bernoulli naive Bayes (generative)",
             "Bayesian logistic (Laplace)", "SVM linear", "SVM RBF kernel"]
    fig, axes = plt.subplots(1, len(added), figsize=(16, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("sb", ["#eef3fb", BLUE, "#0d366b"])
    for ax, n in zip(axes, added):
        c = cm_[n]["confusion_matrix"]
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap=cmap)
        for i in range(2):
            for j in range(2):
                tag = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i, f"{tag}\n{mat[i, j]}", ha="center", va="center", fontsize=10,
                        color="white" if mat[i, j] > mat.max() * 0.5 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["actual 0", "actual 1"])
        ax.grid(False)
        ax.set_title(f"{n}\nF1 {cm_[n]['f1']:.4f}", fontsize=9)
    fig.suptitle("19 Confusion matrices of the added classifiers (test set, weak labels)",
                 fontweight="bold")
    save(fig, "19_confusion_matrices_extended")

    reg_tbl = pd.DataFrame([{"model": n, "train R²": f"{rm[n]['train']['R2']:.4f}",
                             "test R²": f"{rm[n]['test']['R2']:.4f}",
                             "test RMSE": f"{rm[n]['test']['RMSE']:.4f}",
                             "test MAE": f"{rm[n]['test']['MAE']:.4f}"} for n in names])
    clf_tbl = pd.DataFrame([{"model": n, **cm_[n]["confusion_matrix"],
                             **{k: f"{cm_[n][k]:.4f}" for k in
                                ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}}
                            for n in cn])
    bsd = rd["bayes_coef_sd"]
    return f"""
## D. Extended Unit-II models (`python src/extended_models.py`)

Same data, splits and seed as above; every hyperparameter chosen by CV on the training split.

### D1. Regression (target: {MD_TARGET}, not ground truth)

{md_table(reg_tbl)}

* **Ridge**: grouped 4-fold CV chose λ = {rd['ridge_alpha']:g}, the smallest value on the grid,
  so ridge ≈ OLS. With 6 well-conditioned features there is nothing to shrink.
* **Huber** (ε = {rd['huber_epsilon']}): {rd['huber_n_outliers_train']} of {len(reg['y_tr'])} training pairs
  have |residual| > ε·scale (scale = {rd['huber_scale']:.4f}) and get linear instead of squared loss.
  Intercept {rd['huber_intercept']:.4f} (OLS {rd['ols_intercept']:.4f}).
* **Bayesian linear regression** (evidence maximisation): noise precision α =
  {rd['bayes_noise_precision_alpha']:.6f} (noise variance 1/α = {rd['bayes_noise_variance_1_over_alpha']:.4f}),
  weight precision λ = {rd['bayes_weight_precision_lambda']:.6f} (a weak prior, so the posterior
  mean ≈ OLS). The closed-form posterior matches sklearn to {rd['bayes_closed_form_vs_sklearn_max_abs_diff']:.2e}.
  Posterior sd of β: category_match {bsd['category_match']:.4f}, within_radius {bsd['within_radius']:.4f}.
  Test: mean predictive sd {rd['bayes_test_mean_predictive_sd']:.4f}; 95% interval coverage
  {rd['bayes_test_95pct_interval_coverage']:.4f}.
* **Kernel ridge**: polynomial degree 2 (α = {rd['kpoly_alpha']:.4g}) and RBF (α = {rd['krbf_alpha']:.4g},
  γ = {rd['krbf_gamma']:.4g}); the kernels add feature interactions that the linear model cannot.

### D2. Classification (target: weak is_pharmacy labels, test set)

{md_table(clf_tbl)}

* **Fisher LDA**: shared-covariance Gaussian classes, {cd['lda_shrinkage']} shrinkage.
* **Bernoulli naive Bayes**: Laplace smoothing α = {cd['nb_alpha']}.
* **Bayesian logistic (Laplace)**: prior λ = {cd['laplace_lambda']:g}; MAP by penalised IRLS
  (converged {cd['laplace_irls_converged']}); posterior ≈ N(β_MAP, (XᵀWX + λP)⁻¹); probit-approximated
  predictive. Test log loss: plug-in {cd['laplace_test_logloss_map_plugin']:.4f}, predictive
  {cd['laplace_test_logloss_predictive']:.4f}; mean |shift| {cd['laplace_mean_abs_shift_map_to_predictive']:.4f},
  max {cd['laplace_max_abs_shift_map_to_predictive']:.4f}.
* **SVM**: linear C = {cd['svm_linear_C']:g} with {cd['svm_linear_n_support']} support vectors; RBF C =
  {cd['svm_rbf_C']:g} with {cd['svm_rbf_n_support']} support vectors (of {cd['n_train']} training vendors).
"""


if __name__ == "__main__":
    main()
