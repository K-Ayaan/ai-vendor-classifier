"""L2 logistic (MAP) confusion matrices for every category, one-vs-rest.

Usage: python src/l2_confusion_per_category.py
Writes "confusion matrix/l2_confusion_matrices_all_categories.png" and a CSV of the counts.

Same set-up as the Phase 1 is_pharmacy L2 model, repeated for each of the 23 categories:
same train/test split, FeatureBuilder fitted on train, C chosen by 5-fold CV on train
(log-loss), threshold 0.5 on the test set. Labels are the weak category labels.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logistic_regression_classification as L  # noqa: E402

OUT = Path("confusion matrix")
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seqblue", ["#eef3fb", "#2a78d6", "#0d366b"])
NCOLS = 6


def labels(df, category):
    return df.categories.str.split(";").map(lambda c: category in c).astype(int).to_numpy()


def main():
    v = L.load_vendors()
    train, test = L.split(v)
    fb = L.FeatureBuilder().fit(train)
    X_tr, X_te = fb.transform(train), fb.transform(test)
    cats = v.categories.str.split(";").explode().value_counts().index.tolist()

    rows = []
    for cat in cats:
        y_tr, y_te = labels(train, cat), labels(test, cat)
        best_c, _ = L.choose_c(X_tr, y_tr)
        m = L.sk_logreg(best_c, max_iter=10000, tol=1e-10).fit(X_tr[:, 1:], y_tr)
        pred = (m.predict_proba(X_te[:, 1:])[:, 1] >= 0.5).astype(int)
        (tn, fp), (fn, tp) = confusion_matrix(y_te, pred, labels=[0, 1])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"category": cat, "C": best_c, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "accuracy": (tn + tp) / len(y_te), "precision": prec, "recall": rec,
                     "f1": f1})
        print(f"{cat:22s} C={best_c:<8.4g} TN={tn:4d} FP={fp:3d} FN={fn:3d} TP={tp:3d} F1={f1:.3f}")

    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "l2_confusion_matrices_all_categories.csv", index=False,
               float_format="%.4f", lineterminator="\n")

    nrows = -(-len(res) // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.0 * NCOLS, 3.1 * nrows),
                             facecolor=SURFACE)
    for ax, r in zip(axes.flat, res.itertuples()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        # colour by row share so the small positive row stays readable next to TN
        share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap=CMAP, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                tag = [["TN", "FP"], ["FN", "TP"]][i][j]
                ax.text(j, i, f"{tag}\n{cm[i, j]}", ha="center", va="center", fontsize=10,
                        color="white" if share[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8, color=MUTED)
        ax.set_yticks([0, 1], ["actual 0", "actual 1"], fontsize=8, color=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{r.category}\nF1 {r.f1:.3f} · P {r.precision:.2f} · R {r.recall:.2f}",
                     fontsize=9.5, fontweight="bold", color=INK)
    for ax in axes.flat[len(res):]:
        ax.axis("off")
    fig.suptitle("L2 logistic regression (MAP), one-vs-rest per category — test set "
                 f"(n = {len(test)}, weak labels), threshold 0.5\n"
                 "cell colour = share of the actual row; C chosen per category by 5-fold CV",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "l2_confusion_matrices_all_categories.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
