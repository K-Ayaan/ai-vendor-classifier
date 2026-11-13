"""Step 5 - diagnostic plots for the linear regression of the AI-judged expected_score:
residuals vs fitted, residual Q-Q.

Usage: python src/linear_regression_plots.py
Runs only when the annotation sheet is ready (same rule as linear_regression_mle.py).
Writes reports/figures/residuals_vs_fitted.png and reports/figures/residual_qq.png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402

FIG_DIR = Path("reports/figures")
PLOT_TARGET = "expected_score"  # label used on the figures


def residual_plots(r):
    """Return (fig_resid_vs_fitted, fig_qq) for a results dict from lr.run()."""
    fit_tr, fit_te = r["X_tr"] @ r["beta"], r["X_te"] @ r["beta"]
    res_tr, res_te = r["y_tr"] - fit_tr, r["y_te"] - fit_te

    f1, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fit_tr, res_tr, s=14, alpha=0.6, label=f"train (n={len(res_tr)})")
    ax.scatter(fit_te, res_te, s=14, alpha=0.6, marker="^", label=f"test (n={len(res_te)})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel=f"fitted {PLOT_TARGET}", ylabel="residual (y - fitted)",
           title=f"Residuals vs fitted\n(target: {PLOT_TARGET})")
    ax.legend()
    f1.tight_layout()

    f2, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(res_tr, dist="norm", plot=ax)
    ax.set_title(f"Normal Q-Q of training residuals\n(target: {PLOT_TARGET})")
    f2.tight_layout()
    return f1, f2


def main():
    df = lr.load_annotations()
    status = lr.readiness(df)
    if not status["ready"]:
        print(f"BLOCKED: {status['complete_rows']} rows with an {lr.TARGET_LABEL}, need "
              f">= {lr.MIN_COMPLETE_ROWS}. No plots produced.")
        return
    f1, f2 = residual_plots(lr.run(df))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f1.savefig(FIG_DIR / "residuals_vs_fitted.png", dpi=150)
    f2.savefig(FIG_DIR / "residual_qq.png", dpi=150)
    print(f"wrote {FIG_DIR / 'residuals_vs_fitted.png'} and {FIG_DIR / 'residual_qq.png'}")


if __name__ == "__main__":
    main()
